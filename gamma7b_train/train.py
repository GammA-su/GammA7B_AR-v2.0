from __future__ import annotations
import os, time, math, random

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn.utils import clip_grad_norm_

from .config import ModelCfg, TrainCfg, DataCfg, RunCfg, ContractCfg, contract_hash, write_run_metadata, sha256_file
from .attn_plan import AttnPlan
from .model_gamma7b import Gamma7B
from .data_mmap import open_pack, MMapRowDataset, apply_domain_prefix, apply_fim, to_torch_batch
from .optim import wsd_lr, wd_schedule, z_loss as zloss_fn


def setup_dist():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank_env = os.environ.get("RANK")
    local_rank_env = os.environ.get("LOCAL_RANK")
    if world > 1 or rank_env is not None:
        if dist.is_available() and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = int(rank_env or "0")
        local_rank = int(local_rank_env or "0")
    else:
        rank = 0
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world


def is_main(rank: int) -> bool:
    return rank == 0


class EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=(1 - self.decay))

    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if n in getattr(self, "backup", {}):
                p.copy_(self.backup[n])


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tokenizer-model", required=True)

    ap.add_argument("--pack-2048", default=None)
    ap.add_argument("--pack-4096", default=None)
    ap.add_argument("--pack-8192", default=None)
    ap.add_argument("--mix-2048", type=float, default=0.55)
    ap.add_argument("--mix-4096", type=float, default=0.35)
    ap.add_argument("--mix-8192", type=float, default=0.10)
    ap.add_argument("--use-index-domains", action="store_true")

    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=200)

    return ap.parse_args()


def choose_pack(rng: random.Random, packs, probs):
    r = rng.random()
    acc = 0.0
    for p, pr in zip(packs, probs):
        acc += pr
        if r <= acc:
            return p
    return packs[-1]


def main():
    args = parse_args()
    rank, local_rank, world = setup_dist()
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    is_main_rank = is_main(rank)

    domain_tokens = {"code_docs": "<|code|>", "math": "<|math|>", "paper_teasers": "<|paper|>", "default": "<|text|>"}

    model_cfg = ModelCfg()
    train_cfg = TrainCfg(seed=args.seed, micro_batch=args.micro_batch, grad_accum=args.grad_accum, lr=args.lr, domain_tokens=domain_tokens)
    data_cfg = DataCfg(
        pack_2048=args.pack_2048, pack_4096=args.pack_4096, pack_8192=args.pack_8192,
        mix_2048=args.mix_2048, mix_4096=args.mix_4096, mix_8192=args.mix_8192,
        tokenizer_model=args.tokenizer_model,
        use_index_domains=args.use_index_domains,
    )
    run_cfg = RunCfg(out_dir=args.out_dir, steps=args.steps, log_every=args.log_every, eval_every=args.eval_every, save_every=args.save_every)
    cfg = ContractCfg(model=model_cfg, train=train_cfg, data=data_cfg, run=run_cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    if is_main_rank:
        print(f"[init] rank={rank} local_rank={local_rank} world={world} device={device}", flush=True)
        print(f"[init] out_dir={args.out_dir}", flush=True)
        print(f"[init] packs: 2048={data_cfg.pack_2048} 4096={data_cfg.pack_4096} 8192={data_cfg.pack_8192}", flush=True)
        print(f"[init] mix: 2048={data_cfg.mix_2048} 4096={data_cfg.mix_4096} 8192={data_cfg.mix_8192}", flush=True)
        print(f"[init] steps={run_cfg.steps} micro_batch={train_cfg.micro_batch} grad_accum={train_cfg.grad_accum} lr={train_cfg.lr}", flush=True)

    extra = {
        "tokenizer_model_sha256": sha256_file(args.tokenizer_model) if os.path.exists(args.tokenizer_model) else "",
        "world_size": world,
    }
    write_run_metadata(os.path.join(args.out_dir, "run_meta.json"), cfg, extra)

    attn_plan = AttnPlan(swa_window=model_cfg.swa_window, global_layers=set(model_cfg.global_layers), sink_tokens=model_cfg.sink_tokens)
    model = Gamma7B(
        vocab_size=model_cfg.vocab_size, d_model=model_cfg.d_model, layers=model_cfg.layers,
        n_heads=model_cfg.n_heads, n_kv_heads=model_cfg.n_kv_heads, ffn_hidden=model_cfg.ffn_intermediate,
        attn_plan=attn_plan, qkv_bias=model_cfg.qkv_bias, qk_norm=model_cfg.qk_norm, tie_embeddings=model_cfg.tie_embeddings
    ).to(device)

    wrap_policy = size_based_auto_wrap_policy(min_num_params=1_000_000)
    model = FSDP(model, auto_wrap_policy=wrap_policy, device_id=device, use_orig_params=True)

    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, betas=(0.9, 0.95), weight_decay=train_cfg.weight_decay_start, fused=True)

    packs = []
    probs = []
    rng = random.Random(args.seed)

    def add_pack(path, prob):
        if path:
            pk = open_pack(path, use_index_domains=data_cfg.use_index_domains)
            packs.append(pk)
            probs.append(prob)

    add_pack(data_cfg.pack_2048, data_cfg.mix_2048)
    add_pack(data_cfg.pack_4096, data_cfg.mix_4096)
    add_pack(data_cfg.pack_8192, data_cfg.mix_8192)

    if not packs:
        raise SystemExit("Need at least one --pack-*")

    s = sum(probs)
    probs = [p/s for p in probs]

    datasets = {pk.root: MMapRowDataset(pk, seed=args.seed + rank) for pk in packs}
    if is_main_rank:
        for pk in packs:
            print(f"[pack] root={pk.root} seq_len={pk.seq_len} num_seqs={pk.num_seqs} has_domains={pk.row_domain is not None}", flush=True)

    # Optional tokenizer-driven ids for domain prefix + FIM
    dom_id_map = {}
    fim_ids = {}
    if train_cfg.domain_prefix_enable or train_cfg.fim_frac > 0:
        try:
            import sentencepiece as spm

            sp = spm.SentencePieceProcessor()
            sp.Load(data_cfg.tokenizer_model)
            for dom, tok in domain_tokens.items():
                tid = sp.PieceToId(tok)
                if tid >= 0:
                    dom_id_map[dom] = tid
            fim_ids = {
                "fim_prefix": sp.PieceToId("<|fim_prefix|>"),
                "fim_suffix": sp.PieceToId("<|fim_suffix|>"),
                "fim_middle": sp.PieceToId("<|fim_middle|>"),
            }
        except Exception:
            dom_id_map = {}
            fim_ids = {}
    if is_main_rank:
        print(f"[tok] domain_prefix_enable={train_cfg.domain_prefix_enable} dom_id_map={dom_id_map}", flush=True)
        print(f"[tok] fim_frac={train_cfg.fim_frac} fim_ids={fim_ids}", flush=True)

    ema = EMA(model, decay=train_cfg.ema_decay) if train_cfg.ema_enable else None

    loss_hist = []
    grad_hist = []

    model.train()
    t0 = time.time()
    last_log_time = t0
    seen_samples = 0
    pack_counts = {pk.root: 0 for pk in packs}
    for step in range(run_cfg.steps):
        lr = wsd_lr(step, run_cfg.steps, train_cfg.lr, train_cfg.warmup_frac, train_cfg.stable_frac)
        for pg in opt.param_groups:
            pg["lr"] = lr
            pg["weight_decay"] = wd_schedule(step, run_cfg.steps, train_cfg.weight_decay_start, train_cfg.weight_decay_end)

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(train_cfg.grad_accum):
            pk = choose_pack(rng, packs, probs)
            ds = datasets[pk.root]
            rows = []
            for _ in range(train_cfg.micro_batch):
                arr, dom = ds.sample_row()
                if train_cfg.fim_frac > 0 and fim_ids:
                    if rng.random() < train_cfg.fim_frac:
                        arr = apply_fim(arr, rng, fim_ids)
                if train_cfg.domain_prefix_enable and dom_id_map and dom:
                    dom_id = dom_id_map.get(dom, dom_id_map.get("default"))
                    if dom_id is not None:
                        arr = apply_domain_prefix(arr, dom_id)
                rows.append(arr)
            x, y = to_torch_batch(rows)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(x, y, logit_softcap=train_cfg.logit_softcap)
                total = loss + zloss_fn(logits) * train_cfg.z_loss

            total.backward()
            accum_loss += float(loss.detach().cpu())
            seen_samples += len(rows)
            pack_counts[pk.root] += len(rows)

        gn = clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        gn_val = float(gn.detach().cpu()) if torch.is_tensor(gn) else float(gn)

        loss_hist.append(accum_loss / train_cfg.grad_accum)
        grad_hist.append(gn_val)
        if len(loss_hist) > 200:
            loss_hist.pop(0)
            grad_hist.pop(0)

        do_step = True
        if len(loss_hist) >= 50:
            mean = sum(loss_hist)/len(loss_hist)
            var = sum((x-mean)**2 for x in loss_hist)/len(loss_hist)
            sigma = math.sqrt(var)
            med_g = sorted(grad_hist)[len(grad_hist)//2]
            if (loss_hist[-1] > mean + train_cfg.skip_spike_sigma * sigma) or (gn_val > train_cfg.skip_grad_mult * max(1e-9, med_g)):
                do_step = False

        if do_step:
            opt.step()

        frac = (step + 1) / run_cfg.steps
        if ema and frac >= train_cfg.ema_start_frac and do_step:
            ema.update(model)

        if is_main_rank and (step + 1) % run_cfg.log_every == 0:
            dt = time.time() - t0
            step_time = time.time() - last_log_time
            last_log_time = time.time()
            samples_per_s = seen_samples / max(1e-9, dt)
            pack_mix = {os.path.basename(k): v for k, v in pack_counts.items()}
            mem = ""
            if device.type == "cuda":
                mem_alloc = torch.cuda.memory_allocated(device) / (1024**2)
                mem_reserved = torch.cuda.memory_reserved(device) / (1024**2)
                mem = f" cuda_mem={mem_alloc:.0f}MiB/{mem_reserved:.0f}MiB"
            print(
                f"step={step+1}/{run_cfg.steps} loss={loss_hist[-1]:.4f} gn={gn_val:.2f} "
                f"lr={lr:.3e} do_step={do_step} dt={dt:.1f}s step_dt={step_time:.2f}s "
                f"samples={seen_samples} samples/s={samples_per_s:.1f} pack_counts={pack_mix}{mem}",
                flush=True,
            )

        if is_main_rank and (step + 1) % run_cfg.save_every == 0:
            ckpt = {"step": step+1, "cfg_hash": contract_hash(cfg)}
            path = os.path.join(run_cfg.out_dir, f"ckpt_step{step+1}.pt")
            state = model.state_dict()
            torch.save({"model": state, "opt": opt.state_dict(), "meta": ckpt}, path)
            print(f"saved {path}", flush=True)

    if is_main_rank:
        print("DONE", flush=True)


if __name__ == "__main__":
    main()

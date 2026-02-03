from __future__ import annotations
import argparse
import math
import os

import torch

from .attn_plan import AttnPlan
from .config import ModelCfg
from .data_mmap import open_pack, MMapRowDataset, to_torch_batch
from .model_gamma7b import Gamma7B


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="Pack root (with meta.json + input_ids.mmap)")
    ap.add_argument("--tokenizer-model", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model_cfg = ModelCfg()
    attn_plan = AttnPlan(swa_window=model_cfg.swa_window, global_layers=set(model_cfg.global_layers), sink_tokens=model_cfg.sink_tokens)
    model = Gamma7B(
        vocab_size=model_cfg.vocab_size,
        d_model=model_cfg.d_model,
        layers=model_cfg.layers,
        n_heads=model_cfg.n_heads,
        n_kv_heads=model_cfg.n_kv_heads,
        ffn_hidden=model_cfg.ffn_intermediate,
        attn_plan=attn_plan,
        qkv_bias=model_cfg.qkv_bias,
        qk_norm=model_cfg.qk_norm,
        tie_embeddings=model_cfg.tie_embeddings,
    ).to(device)
    model.eval()

    pack = open_pack(args.pack, use_index_domains=False)
    ds = MMapRowDataset(pack, seed=args.seed)

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(args.steps):
            rows = []
            for _ in range(args.micro_batch):
                arr, _ = ds.sample_row()
                rows.append(arr)
            x, y = to_torch_batch(rows)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
                logits, loss = model(x, y)
            total_loss += float(loss.detach().cpu())
            total_tokens += x.numel()

    avg_loss = total_loss / max(1, args.steps)
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    print(f"loss={avg_loss:.4f} ppl={ppl:.2f} steps={args.steps} tokens={total_tokens}")


if __name__ == "__main__":
    main()

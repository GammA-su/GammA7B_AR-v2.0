from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import hashlib, json, os, subprocess, time

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def git_state() -> Dict[str, Any]:
    def _cmd(args):
        try:
            return subprocess.check_output(args, text=True).strip()
        except Exception:
            return ""
    commit = _cmd(["git", "rev-parse", "HEAD"])
    dirty = _cmd(["git", "status", "--porcelain"])
    return {"git_commit": commit, "git_dirty": bool(dirty)}

@dataclass(frozen=True)
class ModelCfg:
    layers: int = 32
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    ffn_intermediate: int = 18432
    rope_base: float = 10000.0
    qkv_bias: bool = True
    qk_norm: bool = True
    vocab_size: int = 44032
    tie_embeddings: bool = True

    swa_window: int = 4096
    global_layers: List[int] = (5, 11, 17, 23, 29, 31)
    sink_tokens: int = 64

@dataclass(frozen=True)
class TrainCfg:
    seed: int = 1234
    micro_batch: int = 1
    grad_accum: int = 8
    lr: float = 3e-4
    weight_decay_start: float = 0.08
    weight_decay_end: float = 0.02
    warmup_frac: float = 0.015
    stable_frac: float = 0.80
    z_loss: float = 1e-4
    logit_softcap: float = 30.0
    grad_clip: float = 1.0
    skip_spike_sigma: float = 6.0
    skip_grad_mult: float = 5.0

    ema_enable: bool = True
    ema_start_frac: float = 0.90
    ema_decay: float = 0.999

    fim_frac: float = 0.05
    causal_frac: float = 0.95

    domain_prefix_enable: bool = True
    domain_prefix_start_frac: float = 0.60
    domain_tokens: Dict[str, str] = None

@dataclass(frozen=True)
class DataCfg:
    pack_2048: Optional[str] = None
    pack_4096: Optional[str] = None
    pack_8192: Optional[str] = None
    mix_2048: float = 0.55
    mix_4096: float = 0.35
    mix_8192: float = 0.10

    tokenizer_model: str = ""
    use_index_domains: bool = True

@dataclass(frozen=True)
class RunCfg:
    out_dir: str
    steps: int = 1000
    log_every: int = 20
    eval_every: int = 200
    save_every: int = 200
    eval_pack: Optional[str] = None

@dataclass(frozen=True)
class ContractCfg:
    model: ModelCfg
    train: TrainCfg
    data: DataCfg
    run: RunCfg

def contract_hash(cfg: ContractCfg) -> str:
    b = json.dumps(asdict(cfg), sort_keys=True, ensure_ascii=True).encode("utf-8")
    return _sha256_bytes(b)

def write_run_metadata(path: str, cfg: ContractCfg, extra: Dict[str, Any]) -> None:
    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "contract_hash": contract_hash(cfg),
        **git_state(),
        **extra,
        "config": asdict(cfg),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

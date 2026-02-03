from __future__ import annotations
import os, json, random, subprocess
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch

@dataclass
class Pack:
    root: str
    seq_len: int
    num_seqs: int
    mmap_path: str
    row_domain: Optional[List[str]] = None


def _load_meta(pack_root: str) -> Dict:
    p = os.path.join(pack_root, "meta.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_domains(pack_root: str) -> Optional[List[str]]:
    idx = os.path.join(pack_root, "index.jsonl.zst")
    if not os.path.exists(idx):
        return None
    cmd = f"zstdcat '{idx}' | jq -r '.domain'"
    out = subprocess.check_output(cmd, shell=True, text=True)
    doms = out.splitlines()
    return doms


def open_pack(pack_root: str, use_index_domains: bool = True) -> Pack:
    meta = _load_meta(pack_root)
    seq_len = int(meta["seq_len"])
    num_seqs = meta.get("num_seqs")
    if num_seqs is None:
        num_seqs = int(meta.get("n_seqs"))
    else:
        num_seqs = int(num_seqs)

    mmap_path = os.path.join(pack_root, "input_ids.mmap")
    if not os.path.exists(mmap_path):
        raise FileNotFoundError(mmap_path)

    row_domain = _load_domains(pack_root) if use_index_domains else None
    if row_domain is not None and len(row_domain) != num_seqs:
        row_domain = None

    return Pack(root=pack_root, seq_len=seq_len, num_seqs=num_seqs, mmap_path=mmap_path, row_domain=row_domain)


class MMapRowDataset:
    """
    Reads fixed-length rows from input_ids.mmap (int32).
    Produces input_ids and labels (next-token shift), both int64 tensors on CPU.
    """

    def __init__(self, pack: Pack, seed: int):
        self.pack = pack
        self.rng = random.Random(seed)
        self.fd = open(pack.mmap_path, "rb", buffering=0)
        self.row_bytes = pack.seq_len * 4

    def __len__(self):
        return self.pack.num_seqs

    def read_row(self, row: int) -> np.ndarray:
        off = row * self.row_bytes
        self.fd.seek(off)
        b = self.fd.read(self.row_bytes)
        arr = np.frombuffer(b, dtype=np.uint32)
        return arr

    def sample_row(self) -> Tuple[np.ndarray, Optional[str]]:
        row = self.rng.randrange(self.pack.num_seqs)
        dom = self.pack.row_domain[row] if self.pack.row_domain else None
        return self.read_row(row), dom


def apply_domain_prefix(tokens: np.ndarray, dom_id: int) -> np.ndarray:
    out = tokens.copy()
    out[1:] = out[:-1]
    out[0] = np.uint32(dom_id)
    return out


def apply_fim(tokens: np.ndarray, rng: random.Random, fim_ids: Dict[str, int]) -> np.ndarray:
    """
    Apply a simple FIM transform: <fim_prefix> prefix <fim_suffix> suffix <fim_middle> middle.
    Keeps sequence length by truncation.
    """
    fp = fim_ids.get("fim_prefix")
    fs = fim_ids.get("fim_suffix")
    fm = fim_ids.get("fim_middle")
    if fp is None or fs is None or fm is None:
        return tokens

    # Pick cut points with a minimum segment size.
    L = len(tokens)
    if L < 8:
        return tokens
    a = rng.randrange(1, max(2, L // 3))
    b = rng.randrange(a + 1, max(a + 2, 2 * L // 3))
    prefix = tokens[:a]
    middle = tokens[a:b]
    suffix = tokens[b:]

    out = np.concatenate(
        [
            np.array([fp], dtype=tokens.dtype),
            prefix,
            np.array([fs], dtype=tokens.dtype),
            suffix,
            np.array([fm], dtype=tokens.dtype),
            middle,
        ]
    )
    if len(out) >= L:
        return out[:L].copy()
    # If shorter, pad by repeating end (rare); keep length fixed.
    pad = tokens[: L - len(out)]
    return np.concatenate([out, pad]).copy()


def to_torch_batch(rows: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(np.stack(rows).astype(np.int64))
    y = x.clone()
    y[:, :-1] = x[:, 1:]
    y[:, -1] = -100
    return x, y

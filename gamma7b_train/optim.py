from __future__ import annotations
import math
import torch

def wsd_lr(step: int, total_steps: int, lr: float, warmup_frac: float, stable_frac: float) -> float:
    warm = int(total_steps * warmup_frac)
    stable = int(total_steps * stable_frac)
    if step < warm:
        return lr * (step + 1) / max(1, warm)
    if step < warm + stable:
        return lr
    rem = total_steps - (warm + stable)
    t = step - (warm + stable)
    if rem <= 0:
        return lr
    return lr * 0.5 * (1.0 + math.cos(math.pi * t / rem))

def wd_schedule(step: int, total_steps: int, wd0: float, wd1: float) -> float:
    t = step / max(1, total_steps - 1)
    return wd0 + (wd1 - wd0) * t

def z_loss(logits: torch.Tensor) -> torch.Tensor:
    z = torch.logsumexp(logits.float(), dim=-1)
    return (z ** 2).mean()

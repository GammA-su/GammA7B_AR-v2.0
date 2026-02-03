from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_plan import AttnPlan

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight

class RoPE:
    def __init__(self, head_dim: int, base: float = 10000.0):
        self.head_dim = head_dim
        self.base = base

    def freqs(self, seqlen: int, device, dtype):
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
        t = torch.arange(seqlen, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        return freqs

    def apply(self, xq, xk):
        B, H, T, D = xq.shape
        freqs = self.freqs(T, xq.device, xq.dtype)
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]

        def _rot(x):
            x1 = x[..., : D//2]
            x2 = x[..., D//2:]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return _rot(xq), _rot(xk)

class QKNorm(nn.Module):
    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight

class MHA_GQA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, attn_plan: AttnPlan, qkv_bias: bool, qk_norm: bool):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.attn_plan = attn_plan
        self.rope = RoPE(self.head_dim)

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=qkv_bias)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=qkv_bias)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=qkv_bias)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.qknorm = QKNorm(self.head_dim) if qk_norm else None

    def forward(self, x, layer_idx: int):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k_rope = self.rope.apply(q, k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1))
        k = k_rope
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        if self.qknorm is not None:
            q = self.qknorm(q)
            k = self.qknorm(k)

        is_global = self.attn_plan.is_global(layer_idx)
        if is_global:
            attn_mask = None
        else:
            device = x.device
            i = torch.arange(T, device=device)[:, None]
            j = torch.arange(T, device=device)[None, :]
            causal = j <= i
            window = (i - j) < self.attn_plan.swa_window
            sink = j < self.attn_plan.sink_tokens
            allowed = causal & (window | sink)
            attn_mask = ~allowed

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class Block(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, ffn_hidden, attn_plan, qkv_bias, qk_norm):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MHA_GQA(d_model, n_heads, n_kv_heads, attn_plan, qkv_bias=qkv_bias, qk_norm=qk_norm)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, ffn_hidden)

    def forward(self, x, layer_idx: int):
        x = x + self.attn(self.norm1(x), layer_idx=layer_idx)
        x = x + self.ff(self.norm2(x))
        return x

class Gamma7B(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, layers: int, n_heads: int, n_kv_heads: int, ffn_hidden: int,
                 attn_plan: AttnPlan, qkv_bias: bool, qk_norm: bool, tie_embeddings: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.layers = layers

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, n_kv_heads, ffn_hidden, attn_plan, qkv_bias=qkv_bias, qk_norm=qk_norm)
            for _ in range(layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, logit_softcap: float = 0.0):
        x = self.tok_emb(input_ids)
        for i, blk in enumerate(self.blocks):
            x = blk(x, layer_idx=i)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        if logit_softcap and logit_softcap > 0:
            logits = logit_softcap * torch.tanh(logits / logit_softcap)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return logits, loss

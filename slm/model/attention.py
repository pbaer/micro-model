"""Grouped-query causal self-attention on top of PyTorch fused SDPA, with an optional KV cache.

Design notes
- Fused QKV projection, no biases.
- Optional QK-norm (RMSNorm over head_dim, applied before RoPE) for LR stability.
- GQA is handed to SDPA via `enable_gqa=True`; no explicit KV-head repeat in the hot path.
- Backend selection is the caller's job (see `slm.utils.sdpa`). On Windows the flash kernel is
  unavailable; cuDNN is the fast fused backend there.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.config import ModelConfig
from slm.model.rope import apply_rope


class KVCache:
    """Preallocated per-layer key/value cache for incremental decoding.

    Shapes: k/v [n_layers, B, n_kv_heads, max_len, head_dim]. `pos` is the number of tokens
    already written (shared across layers; call `advance()` once per forward).
    """

    def __init__(self, cfg: ModelConfig, batch_size: int, max_len: int, device, dtype) -> None:
        shape = (cfg.n_layers, batch_size, cfg.n_kv_heads, max_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.pos = 0
        self.max_len = max_len

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = k.shape[2]
        assert self.pos + t <= self.max_len, "KV cache overflow"
        self.k[layer, :, :, self.pos : self.pos + t] = k
        self.v[layer, :, :, self.pos : self.pos + t] = v
        return self.k[layer, :, :, : self.pos + t], self.v[layer, :, :, : self.pos + t]

    def advance(self, t: int) -> None:
        self.pos += t

    def reset(self) -> None:
        self.pos = 0


def causal_mask_with_offset(q_len: int, kv_len: int, device) -> torch.Tensor:
    """Boolean [q_len, kv_len] mask (True = attend) for queries at positions kv_len-q_len .. kv_len-1."""
    offset = kv_len - q_len
    qi = torch.arange(q_len, device=device)[:, None] + offset
    kj = torch.arange(kv_len, device=device)[None, :]
    return kj <= qi


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.wqkv = nn.Linear(cfg.d_model, cfg.q_dim + 2 * cfg.kv_dim, bias=False)
        self.wo = nn.Linear(cfg.q_dim, cfg.d_model, bias=False)
        if cfg.qk_norm:
            self.q_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.norm_eps)
            self.k_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.norm_eps)
        else:
            self.q_norm = self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.wqkv(x).split([self.cfg.q_dim, self.cfg.kv_dim, self.cfg.kv_dim], dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, Hq, T, D]
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, Hkv, T, D]
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is None:
            if attn_mask is None:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            else:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=True)
        else:
            k, v = cache.update(self.layer_idx, k, v)
            kv_len = k.shape[2]
            if attn_mask is not None:
                mask = attn_mask
            elif T == 1:
                mask = None  # single query attends to everything cached
            elif kv_len == T:
                mask = None
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
                return self.wo(out.transpose(1, 2).reshape(B, T, self.cfg.q_dim))
            else:
                mask = causal_mask_with_offset(T, kv_len, x.device)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)

        return self.wo(out.transpose(1, 2).reshape(B, T, self.cfg.q_dim))

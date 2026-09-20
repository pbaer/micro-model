"""Decoder-only Transformer: pre-norm blocks, GQA attention, SwiGLU, RMSNorm, RoPE, tied head."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from slm.config import ModelConfig, RopeScaling
from slm.model.attention import Attention, KVCache
from slm.model.loss import chunked_cross_entropy
from slm.model.mlp import SwiGLU
from slm.model.rope import rope_cos_sin
from slm.eval.sampling import sample_next


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.mlp_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, attn_mask=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, attn_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        if cfg.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.set_rope(cfg.max_seq_len, cfg.rope_theta, cfg.rope_scaling)
        self.apply(self._init_weights)
        # Scaled init for residual-writing projections (GPT-2 / nanoGPT convention).
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))

    # ------------------------------------------------------------------ setup
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
        elif isinstance(m, nn.RMSNorm):
            nn.init.ones_(m.weight)

    def set_rope(self, max_seq_len: int, theta: float | None = None, scaling: RopeScaling | None = None) -> None:
        """(Re)build RoPE tables; used at construction and for context extension."""
        theta = self.cfg.rope_theta if theta is None else theta
        scaling = self.cfg.rope_scaling if scaling is None else scaling
        device = self.embed.weight.device
        cos, sin = rope_cos_sin(max_seq_len, self.cfg.head_dim, theta, scaling, device=device)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.cfg.max_seq_len = max_seq_len
        self.cfg.rope_theta = theta
        self.cfg.rope_scaling = scaling

    @property
    def output_weight(self) -> torch.Tensor:
        return self.embed.weight if self.lm_head is None else self.lm_head.weight

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
            if self.lm_head is not None:
                n -= self.lm_head.weight.numel()
        return n

    # ---------------------------------------------------------------- forward
    def hidden_states(
        self,
        idx: torch.Tensor,
        cache: KVCache | None = None,
        attn_mask: torch.Tensor | None = None,
        return_all: bool = False,
    ):
        """Run the trunk. Returns final-normed hidden [B, T, d] (and per-layer residuals if asked)."""
        B, T = idx.shape
        start = cache.pos if cache is not None else 0
        assert start + T <= self.rope_cos.shape[0], f"sequence {start + T} exceeds RoPE table {self.rope_cos.shape[0]}"
        cos = self.rope_cos[start : start + T]
        sin = self.rope_sin[start : start + T]
        x = self.embed(idx)
        residuals = [x] if return_all else None
        for block in self.blocks:
            if self.cfg.grad_checkpointing and self.training and cache is None:
                x = checkpoint(block, x, cos, sin, None, attn_mask, use_reentrant=False)
            else:
                x = block(x, cos, sin, cache, attn_mask)
            if return_all:
                residuals.append(x)
        if cache is not None:
            cache.advance(T)
        h = self.final_norm(x)
        return (h, residuals) if return_all else h

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
        attn_mask: torch.Tensor | None = None,
        last_only: bool = False,
    ):
        """With targets: returns (loss_sum, n_valid_tokens). Without: returns logits [B, T, V]."""
        h = self.hidden_states(idx, cache, attn_mask)
        if targets is not None:
            return chunked_cross_entropy(h, self.output_weight, targets, self.cfg.loss_chunk_size)
        if last_only:
            h = h[:, -1:, :]
        return h @ self.output_weight.t()

    # ------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        stop_ids: tuple[int, ...] = (),
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Greedy/sampled decoding with a KV cache. idx: [B, T0] (same length prompts)."""
        B, T0 = idx.shape
        # Under autocast the K/V written to the cache are bf16 anyway; storing them in the fp32 master dtype doubles
        # the cache (3.2 GB for 8 rows at 4K) and, inside a training process whose allocator sits at the VRAM edge,
        # forces new segments and WDDM paging for the whole eval (M8 phase 2: 421 s and 622 s evals).
        cache_dtype = torch.bfloat16 if idx.is_cuda and torch.is_autocast_enabled() else self.output_weight.dtype
        cache = KVCache(self.cfg, B, T0 + max_new_tokens, idx.device, cache_dtype)
        out = idx
        cur = idx
        done = torch.zeros(B, dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            logits = self(cur, cache=cache, last_only=True)[:, -1, :].float()
            nxt = sample_next(logits, temperature, top_p, top_k, generator)
            if stop_ids:
                for s in stop_ids:
                    done |= nxt == s
            out = torch.cat([out, nxt[:, None]], dim=1)
            cur = nxt[:, None]
            if bool(done.all()):
                break
        return out

"""Rotary position embeddings with configurable context-extension scaling.

Supported `RopeScaling.type`:
  none   - standard RoPE
  linear - position interpolation (Chen et al. 2023): positions divided by `factor`
  ntk    - NTK-aware base scaling: theta' = theta * factor^(D/(D-2))
  yarn   - YaRN (Peng et al. 2023): per-frequency blend of interpolation and extrapolation
           plus an attention temperature (mscale) folded into cos/sin.
"""

from __future__ import annotations

import math

import torch

from slm.config import RopeScaling


def _yarn_correction_dim(n_rot: float, dim: int, base: float, orig_max: int) -> float:
    return (dim * math.log(orig_max / (n_rot * 2 * math.pi))) / (2 * math.log(base))


def rope_inv_freq(head_dim: int, theta: float, scaling: RopeScaling | None = None) -> tuple[torch.Tensor, float]:
    """Return (inv_freq [head_dim/2] in fp32, mscale) for the given scaling config."""
    scaling = scaling or RopeScaling()
    ar = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    mscale = 1.0
    t = scaling.type
    if t == "none" or scaling.factor == 1.0:
        inv_freq = 1.0 / (theta**ar)
    elif t == "linear":
        inv_freq = 1.0 / (theta**ar) / scaling.factor
    elif t == "ntk":
        base = theta * (scaling.factor ** (head_dim / (head_dim - 2)))
        inv_freq = 1.0 / (base**ar)
    elif t == "yarn":
        assert scaling.original_max_seq_len > 0, "yarn needs original_max_seq_len"
        pos_freqs = theta**ar
        inv_extra = 1.0 / pos_freqs
        inv_inter = 1.0 / (scaling.factor * pos_freqs)
        low = math.floor(_yarn_correction_dim(scaling.beta_fast, head_dim, theta, scaling.original_max_seq_len))
        high = math.ceil(_yarn_correction_dim(scaling.beta_slow, head_dim, theta, scaling.original_max_seq_len))
        low = max(low, 0)
        high = min(high, head_dim // 2 - 1)
        if low == high:
            high += 0.001
        ramp = (torch.arange(head_dim // 2, dtype=torch.float32) - low) / (high - low)
        ramp = ramp.clamp(0.0, 1.0)
        extrapolation_mask = 1.0 - ramp  # 1 for high-frequency dims: keep as-is
        inv_freq = inv_inter * (1 - extrapolation_mask) + inv_extra * extrapolation_mask
        if scaling.mscale_enabled:
            mscale = 0.1 * math.log(scaling.factor) + 1.0
    else:
        raise ValueError(f"unknown rope scaling type {t!r}")
    return inv_freq, mscale


def rope_cos_sin(
    seq_len: int,
    head_dim: int,
    theta: float,
    scaling: RopeScaling | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape [seq_len, head_dim] (fp32, halves duplicated for rotate_half)."""
    inv_freq, mscale = rope_inv_freq(head_dim, theta, scaling)
    inv_freq = inv_freq.to(device)
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv_freq)  # [T, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, D]
    return emb.cos() * mscale, emb.sin() * mscale


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, D]; cos/sin: [T, D]. Rotation is computed in fp32 and cast back."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    xf = x.float()
    out = xf * cos + rotate_half(xf) * sin
    return out.to(x.dtype)

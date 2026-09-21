"""Causal LM loss that never materializes the full [B*T, V] fp32 logit tensor when chunked.

Returns (loss_sum, n_valid) so that gradient accumulation can normalize by the *global* number
of valid tokens, making the result invariant to the accumulation factor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IGNORE_INDEX = -100


def _chunk_loss(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = F.linear(h, weight).float()
    # Do not hand IGNORE_INDEX to cross_entropy: under torch.compile the chunked path (activation checkpointing)
    # lowers to a kernel that gathers at the target index *before* applying the ignore mask, and inductor's bounds
    # check fires on -100 ("index out of bounds: 0 <= tmp4 < 32768"). Pretraining never hit it because every token
    # is a target; the first SFT run with loss_chunk_size set crashed at its first eval (M9 stage A, 2026-09-20).
    # Masking the per-token losses instead is arithmetically identical and index-safe.
    valid = targets != IGNORE_INDEX
    safe = torch.where(valid, targets, torch.zeros_like(targets))
    per_token = F.cross_entropy(logits, safe, reduction="none")
    return (per_token * valid).sum()


def chunked_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """hidden [B, T, d] (already final-normed), weight [V, d], targets [B, T] with IGNORE_INDEX.

    chunk_size == 0 -> single full-size chunk (fastest when memory allows).
    Chunks are recomputed in backward via activation checkpointing, so peak memory is
    O(chunk_size * V) instead of O(B * T * V).
    """
    h = hidden.reshape(-1, hidden.shape[-1])
    t = targets.reshape(-1)
    n_valid = (t != IGNORE_INDEX).sum()
    if chunk_size <= 0 or chunk_size >= h.shape[0]:
        return _chunk_loss(h, weight, t), n_valid
    loss = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.shape[0], chunk_size):
        loss = loss + checkpoint(_chunk_loss, h[i : i + chunk_size], weight, t[i : i + chunk_size], use_reentrant=False)
    return loss, n_valid

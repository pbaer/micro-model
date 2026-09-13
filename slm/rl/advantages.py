"""Group-relative advantages."""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, normalize_std: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """rewards [G] for one prompt's group -> advantages [G].

    normalize_std=True is GRPO's (R - mean) / (std + eps); False is the Dr. GRPO variant (R - mean),
    which avoids over-weighting prompts whose group happens to have tiny reward variance.
    A group with identical rewards yields zero advantages (no learning signal, by design).
    """
    r = rewards.float()
    a = r - r.mean()
    if normalize_std:
        a = a / (r.std(unbiased=False) + eps)
    return a

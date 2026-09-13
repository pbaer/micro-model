"""Token-indexed learning-rate schedules (progress is measured in tokens, not steps)."""

from __future__ import annotations

import math

from slm.train.config import ScheduleConfig


def lr_at(tokens: int, cfg: ScheduleConfig, peak_lr: float) -> float:
    min_lr = peak_lr * cfg.min_lr_ratio
    if cfg.warmup_tokens > 0 and tokens < cfg.warmup_tokens:
        return peak_lr * (tokens + 1) / cfg.warmup_tokens
    if cfg.type == "constant":
        return peak_lr
    total = max(cfg.total_tokens, cfg.warmup_tokens + 1)
    if cfg.type == "cosine":
        progress = min(1.0, (tokens - cfg.warmup_tokens) / (total - cfg.warmup_tokens))
        return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    if cfg.type == "wsd":
        decay_start = total * (1.0 - cfg.decay_frac)
        if tokens < decay_start:
            return peak_lr
        progress = min(1.0, (tokens - decay_start) / max(1, total - decay_start))
        return peak_lr + (min_lr - peak_lr) * progress
    raise ValueError(f"unknown schedule {cfg.type}")

"""SwiGLU feed-forward block with a fused gate/up projection and no biases."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.d_ff = cfg.d_ff
        self.w13 = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=False)  # [gate | up]
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)

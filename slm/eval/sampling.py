"""Next-token sampling shared by Transformer.generate, the evaluation suite, RL rollouts, and the portal harness."""

from __future__ import annotations

import torch


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """logits [B, V] (float) -> token ids [B]. temperature <= 0 means greedy."""
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(-1)
        remove = cum - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)

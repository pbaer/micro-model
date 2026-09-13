"""Policy-gradient objectives on token log-probs.

All functions take per-token tensors over the *completion* positions only (prompt tokens are
excluded by the caller's mask) and return (loss, stats). `loss` is a sum over tokens so the trainer
can normalize by a global token count, exactly like pretraining/SFT.
"""

from __future__ import annotations

import torch


def policy_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float = 0.2,
    use_ratio: bool = True,
) -> tuple[torch.Tensor, dict]:
    """logp/old_logp [B, T] token log-probs under current/old policy; advantages [B] per sequence;
    mask [B, T] 1 on completion tokens. Returns summed loss over masked tokens.

    use_ratio=False: plain REINFORCE  -A * logp
    use_ratio=True : PPO/GRPO clipped  -min(r*A, clip(r,1-eps,1+eps)*A) with r = exp(logp - old_logp)
    """
    adv = advantages[:, None].expand_as(logp)
    if use_ratio:
        ratio = torch.exp(logp - old_logp)
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
        per_tok = -torch.min(unclipped, clipped)
        clip_frac = (((ratio - 1).abs() > clip_eps) & (mask > 0)).float().sum() / mask.sum().clamp_min(1)
    else:
        per_tok = -adv * logp
        ratio = torch.ones_like(logp)
        clip_frac = torch.zeros((), device=logp.device)
    loss = (per_tok * mask).sum()
    stats = {"clip_frac": float(clip_frac), "ratio_mean": float((ratio * mask).sum() / mask.sum().clamp_min(1))}
    return loss, stats


def kl_penalty(logp: torch.Tensor, ref_logp: torch.Tensor, mask: torch.Tensor, kind: str = "k3") -> torch.Tensor:
    """Per-token KL estimator between policy and frozen reference, summed over masked tokens.
    k1: logp - ref_logp (unbiased, high variance). k3: exp(ref-logp) - (ref-logp) - 1 (low variance, >= 0)."""
    d = ref_logp - logp
    if kind == "k1":
        kl = -d
    elif kind == "k3":
        kl = torch.exp(d) - d - 1
    else:
        raise ValueError(kind)
    return (kl * mask).sum()


def sequence_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits [B, T, V] (float), targets [B, T] -> per-token log-prob of the target [B, T]."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, targets[..., None]).squeeze(-1)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)

"""Illustration payloads for hyper-parameters, computed with the real training functions."""

from __future__ import annotations

import math
from pathlib import Path

from slm.config import ModelConfig, RopeScaling, from_dict, load_yaml
from slm.model.rope import rope_inv_freq
from slm.train.config import TrainConfig, load_train_config
from slm.train.schedule import lr_at


def lr_curve(cfg: TrainConfig, n_points: int = 300) -> dict:
    total = cfg.schedule.total_tokens
    xs = [int(total * i / (n_points - 1)) for i in range(n_points)]
    # make sure the warmup corner is sampled
    xs = sorted(set(xs + [cfg.schedule.warmup_tokens, max(0, cfg.schedule.warmup_tokens - 1)]))
    return {"tokens": xs, "lr": [lr_at(x, cfg.schedule, cfg.optim.lr) for x in xs], "peak": cfg.optim.lr, "min_lr": cfg.optim.lr * cfg.schedule.min_lr_ratio,
            "warmup_tokens": cfg.schedule.warmup_tokens, "total_tokens": total, "type": cfg.schedule.type, "decay_frac": cfg.schedule.decay_frac}


def batch_diagram(cfg: TrainConfig) -> dict:
    mb, T = cfg.batch.microbatch, cfg.data.seq_len
    accum = cfg.grad_accum
    return {"microbatch": mb, "seq_len": T, "grad_accum": accum, "tokens_per_micro": mb * T, "tokens_per_update": cfg.batch.tokens_per_update,
            "updates_total": cfg.schedule.total_tokens // cfg.batch.tokens_per_update,
            "note": "loss_sum over every micro-batch is divided by the total valid tokens of the whole update, so the gradient does not depend on how many micro-batches it was split into"}


def rope_curves(mcfg: ModelConfig, positions: int | None = None) -> dict:
    D = mcfg.head_dim
    base, _ = rope_inv_freq(D, mcfg.rope_theta)
    wavelengths = (2 * math.pi / base).tolist()  # tokens per full rotation, per frequency pair
    curves = {"none": base.tolist()}
    factor = max(2.0, mcfg.rope_scaling.factor if mcfg.rope_scaling.factor > 1 else 4.0)
    for t in ("linear", "ntk", "yarn"):
        sc = RopeScaling(type=t, factor=factor, original_max_seq_len=mcfg.max_seq_len)
        inv, ms = rope_inv_freq(D, mcfg.rope_theta, sc)
        curves[t] = inv.tolist()
        if t == "yarn":
            curves["yarn_mscale"] = ms
    positions = positions or mcfg.max_seq_len
    # angle of the lowest-frequency pair across the context, to show whether it completes a rotation
    return {"head_dim": D, "theta": mcfg.rope_theta, "pairs": list(range(D // 2)), "inv_freq": curves, "wavelength_tokens": wavelengths,
            "max_seq_len": mcfg.max_seq_len, "illustrated_factor": factor, "configured": {"type": mcfg.rope_scaling.type, "factor": mcfg.rope_scaling.factor},
            "pairs_exceeding_context": int(sum(1 for w in wavelengths if w > mcfg.max_seq_len))}


def cadence(cfg: TrainConfig) -> dict:
    total = cfg.schedule.total_tokens
    return {"total_tokens": total, "eval_every": cfg.eval.every_tokens, "gen_every": cfg.eval.gen_every_tokens, "milestone_every": cfg.milestone_tokens,
            "n_evals": total // max(1, cfg.eval.every_tokens), "n_milestones": total // max(1, cfg.milestone_tokens), "ckpt_every_minutes": cfg.ckpt.every_minutes}


def illustrations(train_cfg_path: str) -> dict:
    cfg = load_train_config(train_cfg_path)
    mcfg = cfg.model_config()
    return {"config": train_cfg_path, "lr": lr_curve(cfg), "batch": batch_diagram(cfg), "rope": rope_curves(mcfg), "cadence": cadence(cfg),
            "mixture": cfg.data.mixture, "seq_len": cfg.data.seq_len, "model_file": cfg.model_file}


def model_config_from(spec: str, runs_root: Path, overrides: list[str] | None = None) -> tuple[ModelConfig, str]:
    """spec: path to a model yaml, 'run:<name>' (stored model_config), or 'train:<train yaml>'."""
    if spec.startswith("run:"):
        import json

        meta = json.loads((Path(runs_root) / spec[4:] / "run.json").read_text(encoding="utf-8"))
        d = meta["model_config"]
        label = spec
    elif spec.startswith("train:"):
        cfg = load_train_config(spec[6:])
        d = load_yaml(cfg.model_file)
        d.update(cfg.model or {})
        label = cfg.model_file
    else:
        d = load_yaml(spec)
        label = spec
    from slm.config import apply_overrides

    apply_overrides(d, overrides or [])
    return from_dict(ModelConfig, d), label

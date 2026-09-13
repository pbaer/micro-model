"""Checkpointing: full resumable state (latest) and lightweight model snapshots (milestones, best).

Files under runs/<run>/checkpoints/:
    latest.pt          full state: model, optimizer, loader, counters, RNG, config, metadata
    latest.prev.pt     previous latest (safety against a corrupt write)
    snap_<tokens>.pt   bf16 model-only snapshot at milestones (name includes trained tokens)
    best.pt            bf16 model-only snapshot with the best validation loss
Writes are atomic (tmp file + replace).
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from slm.utils.logging import fmt_tokens


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def env_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git_commit": git_commit(),
    }


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(s: dict[str, Any]) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch_cpu"])
    if s.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["torch_cuda"])


def _replace_with_retry(src: Path, dst: Path, attempts: int = 10, wait_s: float = 1.0) -> None:
    """os.replace fails on Windows while another process holds `dst` open (e.g. a viewer loading
    the checkpoint); retry briefly instead of crashing a multi-hour run."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(wait_s)


def _atomic_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    _replace_with_retry(tmp, path)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def save_full(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader_state: dict,
    counters: dict,
    config: dict,
    meta: dict,
    keep_prev: bool = True,
) -> None:
    if keep_prev and path.exists():
        _replace_with_retry(path, path.with_name(path.stem + ".prev" + path.suffix))
    _atomic_save(
        {
            "model": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "loader": loader_state,
            "counters": counters,
            "rng": rng_state(),
            "config": config,
            "meta": meta,
        },
        path,
    )


def load_full(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None, device="cuda") -> dict:
    ck = torch.load(path, map_location=device, weights_only=False)
    unwrap(model).load_state_dict(ck["model"])
    if optimizer is not None and "optimizer" in ck:
        optimizer.load_state_dict(ck["optimizer"])
    if "rng" in ck:
        set_rng_state(ck["rng"])
    return ck


def save_snapshot(path: Path, model: torch.nn.Module, config: dict, meta: dict, dtype=torch.bfloat16) -> None:
    sd = {k: v.to(dtype) if v.is_floating_point() else v for k, v in unwrap(model).state_dict().items()}
    _atomic_save({"model": sd, "config": config, "meta": meta}, path)


def load_snapshot(path: Path, model: torch.nn.Module, device="cuda") -> dict:
    ck = torch.load(path, map_location=device, weights_only=False)
    unwrap(model).load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
    return ck


def snapshot_name(tokens: int) -> str:
    return f"snap_{fmt_tokens(tokens).replace('.', 'p')}.pt"

"""Which kind of model a run produces, from its run.json. Shared by the portal and the quality eval."""

from __future__ import annotations

import json
from pathlib import Path


def run_stage(meta: dict) -> str:
    """base (pretraining / context extension) | sft | reasoning (SFT with think spans) | rl."""
    cfg = meta.get("config") or {}
    if meta.get("stage") == "grpo" or cfg.get("group_size"):
        return "rl"
    data = cfg.get("data") or {}
    if isinstance(data, dict) and data.get("kind") == "sft":
        mix = " ".join((data.get("mixture") or {}).keys())
        return "reasoning" if "reasoning" in mix or "tools" in mix or "gsm8k" in mix else "sft"
    return "base"


def run_tools(meta: dict) -> bool:
    """Whether the run's model was trained to call the Python tool (RL `tools: true`, or an SFT mixture of *-tools sets)."""
    cfg = meta.get("config") or {}
    if cfg.get("tools"):
        return True
    data = cfg.get("data") or {}
    return isinstance(data, dict) and any("tools" in name for name in (data.get("mixture") or {}))


def run_meta(run_dir: Path) -> dict:
    p = Path(run_dir) / "run.json"
    if not p.exists():
        return {"run_name": Path(run_dir).name}
    return json.loads(p.read_text(encoding="utf-8"))


def stage_of_run(run_dir: Path) -> str:
    return run_stage(run_meta(run_dir))

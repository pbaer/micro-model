"""Needle retrieval on every milestone snapshot of a run, at one sample count and one fixed haystack draw.

The in-run tracker (`eval.needle_*`) scores whatever `n` the run was configured with at the time, so a run whose
settings changed mid-way has a curve of mixed precision. This sweep re-measures every snapshot the same way and
writes runs/<run>/needle_sweep.json, which the run page overlays on the needle chart.

    python -m slm.eval.needle_sweep --run m8_base_stable_336m --n 64 --lengths 1024 2048 4096

Resumable: snapshots already in the file are skipped unless --force. About 35 s per 336M snapshot on the GPU at
n=64 with 32K-token batches (640 prompts for two lengths); do not run it beside a training job on the same GPU.
On CPU (--device cpu --threads 24) it runs beside training; see docs/results.md for the measured rate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from slm.eval.quality import load_model, load_tokenizer, selected_checkpoints

DEFAULT_TOKENIZED_ROOT = r"C:\slm-data\tokenized\v1"


def sweep_path(run_dir: Path) -> Path:
    return Path(run_dir) / "needle_sweep.json"


def read_sweep(run_dir: Path) -> dict | None:
    p = sweep_path(run_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write(run_dir: Path, data: dict) -> None:
    p = sweep_path(run_dir)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def sweep(run_dir: Path, lengths: list[int], depths: list[float], n: int, seed: int = 0, source: str = "fineweb-edu-b", tokenized_root: str = DEFAULT_TOKENIZED_ROOT,
          batch_tokens: int = 32768, device: str = "cuda", checkpoints: list[str] | None = None, every: int = 1, force: bool = False, threshold: float = 0.8,
          haystack=None, log=print) -> dict:
    import torch

    from slm.eval.long_context import make_haystack, run_needle

    run_dir = Path(run_dir)
    settings = {"n": n, "lengths": lengths, "depths": depths, "seed": seed, "haystack": "real", "source": source, "threshold": threshold}
    data = None if force else read_sweep(run_dir)
    if data is None or any(data.get(k) != v for k, v in settings.items()):
        if data is not None:
            log(f"[{run_dir.name}] existing sweep has different settings; starting over")
        data = {"run": run_dir.name, **settings, "checkpoints": []}
    done = {c["checkpoint"] for c in data["checkpoints"]}
    picks = selected_checkpoints(run_dir, checkpoints, every)
    todo = [(name, tokens) for name, tokens in picks if name not in done]
    log(f"[{run_dir.name}] needle sweep n={n} lengths={lengths} depths={depths}: {len(picks)} checkpoints, {len(todo)} to do, device={device}")
    if not todo:
        return data
    tok = load_tokenizer(run_dir)
    haystack = haystack or make_haystack(tok, "real", Path(tokenized_root) / source / "val")
    for name, tokens in todo:
        t0 = time.time()
        model, _ = load_model(run_dir / "checkpoints" / name, device)
        res = run_needle(model, tok, lengths, depths, n, seed=seed, haystack=haystack, threshold=threshold, max_batch_tokens=batch_tokens)
        del model
        if device != "cpu":
            torch.cuda.empty_cache()
        entry = {"tokens": tokens, "checkpoint": name, "seconds": round(time.time() - t0, 1), "effective": res["effective_context"],
                 "summary": {str(L): {"mean": round(s["mean"], 4), "min": round(s["min"], 4)} for L, s in res["summary"].items()},
                 "cells": [{"length": r["length"], "depth": r["depth"], "accuracy": r["accuracy"]} for r in res["results"] if "accuracy" in r]}
        data["checkpoints"] = sorted([c for c in data["checkpoints"] if c["checkpoint"] != name] + [entry], key=lambda c: c["tokens"])
        data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _write(run_dir, data)
        log(f"  {name:16s} {tokens / 1e9:6.2f}B  {entry['seconds']:5.1f}s  " + "  ".join(f"{L}:{s['mean'] * 100:3.0f}%/{s['min'] * 100:3.0f}%" for L, s in entry["summary"].items()) + f"  effective {entry['effective']}")
    return data


def series(data: dict) -> dict:
    """Chart-ready arrays: tokens plus mean/min per length (None where a length was skipped)."""
    cks = data.get("checkpoints", [])
    lengths = [str(L) for L in data.get("lengths", [])]
    return {"n": data.get("n"), "lengths": lengths, "tokens": [c["tokens"] for c in cks],
            "mean": {L: [c["summary"].get(L, {}).get("mean") for c in cks] for L in lengths},
            "min": {L: [c["summary"].get(L, {}).get("min") for c in cks] for L in lengths},
            "effective": [c.get("effective") for c in cks]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", default="fineweb-edu-b")
    ap.add_argument("--tokenized-root", default=DEFAULT_TOKENIZED_ROOT)
    ap.add_argument("--batch-tokens", type=int, default=32768)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoints", nargs="*", default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default); also drops the process to below-normal priority")
    a = ap.parse_args()
    if a.device == "cpu" and a.threads:
        import torch

        from slm.eval.quality import _lower_priority

        torch.set_num_threads(a.threads)
        _lower_priority()
    sweep(Path(a.runs_root) / a.run, a.lengths, a.depths, a.n, a.seed, a.source, a.tokenized_root, a.batch_tokens, a.device, a.checkpoints, a.every, a.force)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()

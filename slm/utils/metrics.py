"""Run metrics reading shared by the HTML report and the portal: incremental JSONL tailing,
status detection, series extraction/thinning, checkpoint listing (torch-free)."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

TRAIN_FIELDS = ["loss", "lr", "grad_norm", "tok_s", "tok_s_ema", "step_ms", "fwd_ms", "bwd_ms", "opt_ms", "data_ms", "vram_gib", "elapsed_s", "eta_s"]
EVAL_FIELDS = ["val_loss", "val_ppl"]
EVENT_KINDS = ("start", "resume", "stop", "finish", "checkpoint")


class JsonlTail:
    """Incrementally reads complete lines appended to a JSONL file (safe while a writer appends)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.offset = 0
        self.records: list[dict] = []

    def refresh(self) -> list[dict]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self.offset:  # truncated/rewritten
            self.offset, self.records = 0, []
        if size == self.offset:
            return []
        new: list[dict] = []
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            chunk = f.read(size - self.offset)
        last_nl = chunk.rfind(b"\n")
        if last_nl < 0:
            return []
        for line in chunk[: last_nl + 1].splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                new.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        self.offset += last_nl + 1
        self.records.extend(new)
        return new


def run_status(records: list[dict], stale_after_s: float = 900.0) -> str:
    if any(r["kind"] == "finish" for r in records):
        return "finished"
    if records and records[-1]["kind"] == "stop":
        return "stopped"
    if not records:
        return "empty"
    if time.time() - records[-1]["time"] > stale_after_s:
        return "stale"
    return "running"


def active_seconds(records: list[dict]) -> float:
    total, seg_start, prev_t = 0.0, None, None
    for r in records:
        if r["kind"] in ("start", "resume"):
            if seg_start is not None and prev_t is not None:
                total += prev_t - seg_start
            seg_start = r["time"]
        prev_t = r["time"]
    if seg_start is not None and prev_t is not None:
        total += prev_t - seg_start
    return total


def thin(xs: list, ys: list, max_points: int) -> tuple[list, list]:
    if len(xs) <= max_points:
        return xs, ys
    step = len(xs) // max_points + 1
    return xs[::step], ys[::step]


def series(records: list[dict], max_points: int = 1500) -> dict[str, Any]:
    train = [r for r in records if r["kind"] == "train"]
    evals = [r for r in records if r["kind"] == "eval"]
    out: dict[str, Any] = {"train": {}, "eval": {}, "milestones": [r for r in records if r["kind"] == "milestone"]}
    tx = [r["tokens"] for r in train]
    tu = [r.get("update", 0) for r in train]
    tt = [r["time"] for r in train]
    out["train"]["tokens"], _ = thin(tx, tx, max_points)
    out["train"]["update"], _ = thin(tu, tu, max_points)
    out["train"]["time"], _ = thin(tt, tt, max_points)
    for f in TRAIN_FIELDS:
        _, out["train"][f] = thin(tx, [r.get(f) for r in train], max_points)
    out["eval"]["tokens"] = [r["tokens"] for r in evals]
    out["eval"]["time"] = [r["time"] for r in evals]
    for f in EVAL_FIELDS:
        out["eval"][f] = [r.get(f) for r in evals]
    return out


def summary(records: list[dict], meta: dict) -> dict[str, Any]:
    cfg = meta.get("config", {})
    train = [r for r in records if r["kind"] == "train"]
    evals = [r for r in records if r["kind"] == "eval"]
    last = train[-1] if train else {}
    total = cfg.get("schedule", {}).get("total_tokens", 0)
    tokens = last.get("tokens", 0)
    tps = last.get("tok_s_ema") or last.get("tok_s") or 0.0
    elapsed = active_seconds(records)
    eta = (total - tokens) / tps if tps > 0 and total > tokens else 0.0
    return {
        "run_name": meta.get("run_name"), "stage": meta.get("stage", "pretrain"), "status": run_status(records),
        "tokens": tokens, "total_tokens": total, "progress": tokens / total if total else 0.0,
        "update": last.get("update", 0), "loss": last.get("loss"), "lr": last.get("lr"), "grad_norm": last.get("grad_norm"),
        "tok_s": tps, "tok_s_avg": tokens / elapsed if elapsed > 0 else 0.0, "elapsed_s": elapsed, "eta_s": eta,
        "initial_estimate_s": meta.get("initial_estimate_s"),
        "val_loss": evals[-1]["val_loss"] if evals else None, "val_ppl": evals[-1].get("val_ppl") if evals else None,
        "best_val": min((e["val_loss"] for e in evals), default=None),
        "vram_gib": last.get("vram_gib"), "step_ms": last.get("step_ms"), "fwd_ms": last.get("fwd_ms"), "bwd_ms": last.get("bwd_ms"),
        "opt_ms": last.get("opt_ms"), "data_ms": last.get("data_ms"),
        "n_params": meta.get("n_params"), "started": meta.get("started"), "git_commit": (meta.get("env") or {}).get("git_commit"),
        "gpu": (meta.get("env") or {}).get("gpu"), "last_record_time": records[-1]["time"] if records else None,
    }


_SNAP_RE = re.compile(r"^snap_(\d+)(p(\d+))?([KMB])\.pt$")


def parse_snapshot_tokens(name: str) -> int | None:
    m = _SNAP_RE.match(name)
    if not m:
        return None
    val = float(m.group(1) + ("." + m.group(3) if m.group(3) else ""))
    return int(val * {"K": 1e3, "M": 1e6, "B": 1e9}[m.group(4)])


def list_checkpoints(run_dir: Path, records: list[dict]) -> list[dict]:
    d = Path(run_dir) / "checkpoints"
    if not d.exists():
        return []
    evals = [r for r in records if r["kind"] == "eval"]
    out = []
    for p in sorted(d.glob("*.pt")):
        st = p.stat()
        kind, tokens, val = "other", None, None
        if p.name == "latest.pt":
            kind = "latest"
            tokens = next((r["tokens"] for r in reversed(records) if r["kind"] == "checkpoint"), None)
        elif p.name == "latest.prev.pt":
            kind = "latest_prev"
        elif p.name == "best.pt":
            kind = "best"
            best = min(evals, key=lambda e: e["val_loss"], default=None)
            if best:
                tokens, val = best["tokens"], best["val_loss"]
        elif p.name == "final.pt":
            kind = "final"
            if evals:
                tokens, val = evals[-1]["tokens"], evals[-1]["val_loss"]
        elif (t := parse_snapshot_tokens(p.name)) is not None:
            kind, tokens = "snapshot", t
            before = [e for e in evals if e["tokens"] <= t]
            val = before[-1]["val_loss"] if before else None
        out.append({"name": p.name, "path": str(p), "kind": kind, "tokens": tokens, "val_loss": val, "bytes": st.st_size, "mtime": st.st_mtime})
    return out


def list_samples(run_dir: Path) -> list[dict]:
    d = Path(run_dir) / "samples"
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.txt")):
        if p.name == "latest.txt":
            continue
        try:
            out.append({"tokens": int(p.stem), "path": str(p)})
        except ValueError:
            continue
    return out


def parse_samples(text: str) -> dict:
    header, _, body = text.partition("\n")
    items = []
    for block in body.split("=" * 80):
        block = block.strip("\n")
        if not block.startswith("PROMPT:"):
            continue
        first, _, rest = block.partition("\n")
        prompt = first[len("PROMPT:") :].strip()
        greedy, sampled = None, None
        if "--- greedy:\n" in rest:
            g, _, rest = rest.partition("--- sampled:\n")
            greedy = g.replace("--- greedy:\n", "", 1).rstrip("\n")
            sampled = rest.rstrip("\n")
        elif "--- sampled:\n" in rest:
            sampled = rest.replace("--- sampled:\n", "", 1).rstrip("\n")
        items.append({"prompt": prompt, "greedy": greedy, "sampled": sampled})
    return {"header": header, "items": items}

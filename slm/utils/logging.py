"""JSONL metrics log + console lines. One record per event, all keyed by tokens and wall time."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


class MetricsLogger:
    def __init__(self, run_dir: Path, filename: str = "metrics.jsonl") -> None:
        self.path = Path(run_dir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")  # noqa: SIM115
        self.t0 = time.time()

    def log(self, kind: str, **fields: Any) -> dict:
        rec = {"kind": kind, "time": time.time(), **fields}
        self._f.write(json.dumps(rec, default=_default) + "\n")
        self._f.flush()
        return rec

    def close(self) -> None:
        self._f.close()

    def truncate_after(self, key: str, value: int) -> int:
        """Drop every record from the first `train` record whose `key` exceeds `value` to the end, and reopen
        for append. Called on resume, so a run that crashed after its last checkpoint does not keep the records
        of the steps it is about to replay: without this the token counter runs backwards mid-file and every
        chart draws the line back over itself (M9 stage C run 5, 2026-09-26; also m9_sft_336m and m1).
        Returns the number of records dropped."""
        self._f.close()
        try:
            recs = self.read(self.path)
            cut = next((i for i, r in enumerate(recs) if r.get("kind") == "train" and r.get(key, -1) > value), len(recs))
            dropped = len(recs) - cut
            if dropped:
                with open(self.path, "w", encoding="utf-8") as f:
                    for r in recs[:cut]:
                        f.write(json.dumps(r, default=_default) + "\n")
            return dropped
        finally:
            self._f = open(self.path, "a", encoding="utf-8")  # noqa: SIM115

    @classmethod
    def repair(cls, path: Path, key: str) -> int:
        """Offline repair of a finished run's log: keep the LAST occurrence of each `key` (the replayed segment)
        and drop the earlier crashed one, preserving order. For live runs use truncate_after via the trainer."""
        recs = cls.read(path)
        seen, keep = {}, []
        for i, r in enumerate(recs):
            if r.get("kind") == "train" and key in r:
                seen[r[key]] = i
        for i, r in enumerate(recs):
            if r.get("kind") == "train" and key in r and seen[r[key]] != i:
                continue
            keep.append(r)
        dropped = len(recs) - len(keep)
        if dropped:
            with open(path, "w", encoding="utf-8") as f:
                for r in keep:
                    f.write(json.dumps(r, default=_default) + "\n")
        return dropped

    @staticmethod
    def read(path: Path) -> list[dict]:
        out = []
        if not Path(path).exists():
            return out
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # partial last line while writing
        return out


def _default(o: Any):
    try:
        return float(o)
    except Exception:  # noqa: BLE001
        return str(o)


def console(msg: str) -> None:
    print(time.strftime("%H:%M:%S") + " " + msg, flush=True)
    sys.stdout.flush()


def fmt_duration(seconds: float) -> str:
    if seconds != seconds or seconds == float("inf"):
        return "?"
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d {h:02d}h {m:02d}m"
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def fmt_tokens(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    return f"{n / 1e3:.0f}K"

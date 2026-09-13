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

"""Run discovery and incremental metrics reading for the portal (torch-free)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from slm.utils import metrics as M


class RunReader:
    def __init__(self, run_dir: Path) -> None:
        self.dir = Path(run_dir)
        self.name = self.dir.name
        self.tail = M.JsonlTail(self.dir / "metrics.jsonl")
        self._meta: dict | None = None
        self._meta_mtime = 0.0
        self.lock = threading.Lock()

    @property
    def records(self) -> list[dict]:
        return self.tail.records

    def refresh(self) -> list[dict]:
        with self.lock:
            return self.tail.refresh()

    def meta(self) -> dict:
        p = self.dir / "run.json"
        if not p.exists():
            return {"run_name": self.name}
        mt = p.stat().st_mtime
        if self._meta is None or mt != self._meta_mtime:
            try:
                self._meta = json.loads(p.read_text(encoding="utf-8"))
                self._meta_mtime = mt
            except (OSError, json.JSONDecodeError):
                return self._meta or {"run_name": self.name}
        return self._meta

    def summary(self) -> dict:
        self.refresh()
        s = M.summary(self.records, self.meta())
        s["run_name"] = self.name
        s["has_report"] = (self.dir / "report.html").exists()
        return s

    def series(self, max_points: int = 1500) -> dict:
        self.refresh()
        return M.series(self.records, max_points)

    def events(self, limit: int = 50) -> list[dict]:
        self.refresh()
        return [r for r in self.records if r["kind"] in M.EVENT_KINDS][-limit:]

    def checkpoints(self) -> list[dict]:
        self.refresh()
        return M.list_checkpoints(self.dir, self.records)

    def samples(self) -> list[dict]:
        return M.list_samples(self.dir)

    def sample(self, tokens: int) -> dict:
        p = self.dir / "samples" / f"{tokens:012d}.txt"
        if not p.exists():
            raise FileNotFoundError(p)
        return M.parse_samples(p.read_text(encoding="utf-8"))


class RunIndex:
    def __init__(self, runs_root: Path) -> None:
        self.root = Path(runs_root)
        self._readers: dict[str, RunReader] = {}
        self.lock = threading.Lock()

    def names(self) -> list[str]:
        if not self.root.exists():
            return []
        out = []
        for d in self.root.iterdir():
            if d.is_dir() and ((d / "metrics.jsonl").exists() or (d / "run.json").exists()):
                out.append(d.name)
        return sorted(out)

    def get(self, name: str) -> RunReader:
        with self.lock:
            if name not in self._readers:
                d = self.root / name
                if not d.is_dir() or name != d.name or "/" in name or "\\" in name:
                    raise KeyError(name)
                self._readers[name] = RunReader(d)
            return self._readers[name]

    def summaries(self) -> list[dict]:
        out = [self.get(n).summary() for n in self.names()]
        out.sort(key=lambda s: s.get("last_record_time") or 0, reverse=True)
        return out

    def live_runs(self, within_s: float = 120.0) -> list[str]:
        import time

        now = time.time()
        return [s["run_name"] for s in self.summaries() if s["status"] == "running" and s["last_record_time"] and now - s["last_record_time"] < within_s]

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

    def quality(self) -> dict | None:
        """runs/<run>/quality/summary.json (judged-quality means per checkpoint), or None."""
        p = self.dir / "quality" / "summary.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def quality_detail(self, tokens: int) -> dict:
        from slm.eval.quality import checkpoint_detail  # torch-free at import

        return checkpoint_detail(self.dir, tokens)


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
        by_name = {s["run_name"]: s for s in out}
        for s in out:
            s["cumulative_tokens"] = self._cumulative_tokens(s, by_name, depth=0)
        out.sort(key=lambda s: s.get("last_record_time") or 0, reverse=True)
        return out

    def _cumulative_tokens(self, s: dict, by_name: dict, depth: int) -> int:
        """Tokens seen by the weights: this run's tokens plus those of the checkpoint it started from
        (exact per-file counts from that run's checkpoints/index.json), followed recursively."""
        own = int(s.get("tokens") or 0)
        init = s.get("init_from") or ""
        if not init or depth > 8:
            return own
        p = Path(init)
        parent_name = p.parent.parent.name if p.parent.name == "checkpoints" else None
        if parent_name not in by_name:
            return own
        idx_path = self.root / parent_name / "checkpoints" / "index.json"
        base = None
        if idx_path.exists():
            try:
                base = json.loads(idx_path.read_text(encoding="utf-8")).get(p.name, {}).get("tokens")
            except json.JSONDecodeError:
                base = None
        if base is None:
            base = by_name[parent_name].get("tokens") or 0
        parent_cum = self._cumulative_tokens(by_name[parent_name], by_name, depth + 1) - int(by_name[parent_name].get("tokens") or 0)
        return own + int(base) + max(0, parent_cum)

    def live_runs(self, within_s: float = 900.0) -> list[str]:  # pretraining logs every ~3 min at 524K-token updates; keep a generous window
        import time

        now = time.time()
        return [s["run_name"] for s in self.summaries() if s["status"] == "running" and s["last_record_time"] and now - s["last_record_time"] < within_s]

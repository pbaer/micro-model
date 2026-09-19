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
        q = self.quality()
        judged = [c for c in (q or {}).get("checkpoints", []) if c.get("overall") is not None]
        s["quality"] = {"overall": judged[-1]["overall"], "tokens": judged[-1]["tokens"], "n_judged": len(judged), "judge": ", ".join(q.get("judges", []))} if judged else None
        return s

    def series(self, max_points: int = 1500) -> dict:
        self.refresh()
        out = M.series(self.records, max_points)
        sw = self.needle_sweep()
        if sw and sw.get("checkpoints"):
            from slm.eval.needle_sweep import series as sweep_series  # torch-free at import

            out["needle_sweep"] = sweep_series(sw)
        return out

    def needle_sweep(self) -> dict | None:
        """runs/<run>/needle_sweep.json (slm.eval.needle_sweep: every snapshot at one n and one haystack draw)."""
        p = self.dir / "needle_sweep.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def stream_sources(self) -> dict | None:
        """Per-source loader state ({name: {tokens, epoch}}) from the last `checkpoint` record that
        carries it. Runs started before the trainers logged it fall back to weight x tokens."""
        self.refresh()
        for r in reversed(self.records):
            if r.get("kind") == "checkpoint" and isinstance(r.get("sources"), dict):
                return r["sources"]
        return None

    def rollout_steps(self) -> list[int]:
        d = self.dir / "rollouts"
        return sorted(int(p.stem.split("_")[1]) for p in d.glob("step_*.jsonl")) if d.is_dir() else []

    def rollouts(self, step: int | None = None, offset: int = 0, limit: int = 20) -> dict:
        """One RL step's samples: what the model produced and what it was rewarded for.

        `old_logprobs` / `ref_logprobs` are one float per token and never displayed, so they are dropped
        before the payload is built. The file is opened read-only per request and closed immediately.
        """
        steps = self.rollout_steps()
        if not steps:
            return {"steps": [], "step": None, "n": 0, "offset": offset, "rollouts": []}
        step = steps[-1] if step is None or step not in steps else step
        drop = ("old_logprobs", "ref_logprobs")
        rows = []
        with (self.dir / "rollouts" / f"step_{step:05d}.jsonl").open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < offset:
                    continue
                if len(rows) >= limit:
                    break
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append({k: v for k, v in r.items() if k not in drop})
        n = sum(1 for _ in (self.dir / "rollouts" / f"step_{step:05d}.jsonl").open(encoding="utf-8"))
        return {"steps": steps, "step": step, "n": n, "offset": offset, "rollouts": rows}

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
            s["cumulative_tokens"] = sum(e["tokens_used"] for e in self.chain(s["run_name"], by_name))
        out.sort(key=lambda s: s.get("last_record_time") or 0, reverse=True)
        return out

    def _checkpoint_tokens(self, run: str, ckpt: str) -> int | None:
        """Exact token count of one checkpoint file, from that run's checkpoints/index.json."""
        p = self.root / run / "checkpoints" / "index.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8")).get(ckpt, {}).get("tokens")
        except (OSError, json.JSONDecodeError):
            return None

    def chain(self, name: str, by_name: dict | None = None) -> list[dict]:
        """The `init_from` chain ending at `name`, root first.

        `tokens_used` is the part of each run the weights at the end of the chain actually saw: the run's
        own tokens for the last entry, and for an ancestor the token count of the checkpoint its child
        loaded (exact, from that run's checkpoints/index.json; its own total when the index is missing).
        Summing the column gives the cumulative tokens shown on the overview.
        """
        if by_name is None:
            by_name = {s["run_name"]: s for s in (self.get(n).summary() for n in self.names())}
        out: list[dict] = []
        cur, used, ckpt = name, None, None
        for _ in range(10):
            s = by_name.get(cur)
            if s is None:
                break
            own = int(s.get("tokens") or 0)
            out.append({"run": cur, "stage": s.get("stage") or "pretrain", "own_tokens": own,
                        "tokens_used": own if used is None else int(used), "checkpoint": ckpt,
                        "init_from": s.get("init_from") or "", "status": s.get("status")})
            p = Path(s.get("init_from") or "")
            parent = p.parent.parent.name if str(p) and p.parent.name == "checkpoints" else None
            if not parent or parent not in by_name or any(e["run"] == parent for e in out):
                break
            base = self._checkpoint_tokens(parent, p.name)
            used = base if base is not None else (by_name[parent].get("tokens") or 0)
            ckpt, cur = p.name, parent
        out.reverse()
        return out

    def live_runs(self, within_s: float = 900.0) -> list[str]:  # pretraining logs every ~3 min at 524K-token updates; keep a generous window
        import time

        now = time.time()
        return [s["run_name"] for s in self.summaries() if s["status"] == "running" and s["last_record_time"] and now - s["last_record_time"] < within_s]

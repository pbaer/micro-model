"""Arena episodes on disk and the world without a model (torch-free; the API process imports it).

The Arena tab's viewer mode replays an episode from a JSON file instead of a live run. Two shapes exist and both carry
the same transcript records (`slm.arena.world.Runner.run`):

- the CLI's (`python -m slm.arena ... --out x.json`): `{task, n, agents, seed, turns, score, events, seconds,
  transcript}` -- no start state, no per-turn states, no messages;
- the portal's download: `{request, meta, robots, start, turns: [turn events], result}`.

`list_episodes` lists `<runs_root>/arena/*.json` with a peek at each (kind, task, size of the world, verdict);
`read_episode` returns one file, refusing anything that is not a plain file name inside that directory. `world_view`
builds the world a CLI episode was played in (World(task, n, agents, seed) is deterministic: the same objects, robot
positions, tools and briefings), so the viewer can draw the start state the CLI file does not carry."""

from __future__ import annotations

import json
import re
from pathlib import Path

MAX_EPISODE_BYTES = 64 * 2**20
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.json$")
_peek_cache: dict[str, tuple[float, int, dict]] = {}


def robots(world) -> list[dict]:
    """Every robot's id, name, briefing and declared tools as the Arena tab shows them (the harness's `start` event)."""
    return [{"id": a.id, "name": a.name, "system_prompt": world.system_prompt(a),
             "tools": [{"name": d.name, "signature": d.signature, "comment": d.comment} for d in world.decls[a.id]]} for a in world.agents]


def world_view(task: str, n: int, n_agents: int, seed: int, max_history: int = 1) -> dict:
    """The start of an episode without a model: the state, the robots and the task's comm range and sight."""
    from slm.arena.world import World

    w = World(task, int(n), int(n_agents), int(seed), int(max_history))
    return {"state": w.state(), "robots": robots(w),
            "meta": {"task": task, "n": int(n), "n_agents": int(n_agents), "seed": int(seed), "comm_range": w.task.comm_range, "sight": w.task.sight}}


def arena_dir(runs_root: Path) -> Path:
    return Path(runs_root) / "arena"


def _peek(d) -> dict:
    """What kind of file this is and the few fields the dropdown shows."""
    if not isinstance(d, dict):
        return {"kind": "unknown"}
    if "turns" in d and isinstance(d.get("turns"), list) and "start" in d:  # the portal's download
        m = d.get("meta") or d.get("request") or {}
        res = d.get("result") or {}
        return {"kind": "portal", "task": m.get("task"), "n": m.get("n"), "agents": m.get("n_agents"), "seed": m.get("seed"),
                "turns": res.get("turns", len(d["turns"])), "success": (res.get("score") or {}).get("success"), "checkpoint": m.get("checkpoint")}
    if isinstance(d.get("transcript"), list):  # the CLI's --out
        return {"kind": "cli", "task": d.get("task"), "n": d.get("n"), "agents": d.get("agents"), "seed": d.get("seed"), "turns": d.get("turns"),
                "success": (d.get("score") or {}).get("success")}
    if isinstance(d.get("results"), list):  # scripts/arena_eval.py writes summaries, not episodes
        return {"kind": "summary", "checkpoint": d.get("checkpoint")}
    return {"kind": "unknown"}


def list_episodes(runs_root: Path) -> list[dict]:
    """`<runs_root>/arena/*.json`, newest first, each with its size, mtime and peek (parsed once per mtime and size)."""
    root = arena_dir(runs_root)
    if not root.is_dir():
        return []
    out = []
    for p in root.glob("*.json"):
        if not p.is_file() or not NAME_RE.match(p.name):
            continue
        st = p.stat()
        hit = _peek_cache.get(str(p))
        if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
            peek = hit[2]
        elif st.st_size > MAX_EPISODE_BYTES:
            peek = {"kind": "too_large"}
        else:
            try:
                peek = _peek(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                peek = {"kind": "unreadable"}
            _peek_cache[str(p)] = (st.st_mtime, st.st_size, peek)
        out.append({"name": p.name, "size": st.st_size, "mtime": st.st_mtime, **peek})
    out.sort(key=lambda e: -e["mtime"])
    return out


def episode_path(runs_root: Path, name: str) -> Path:
    """The file `name` inside `<runs_root>/arena`, or ValueError: a bare file name ending in .json, no separators, no
    leading dot, and resolving to a direct child of that directory (a symlink pointing elsewhere is refused too)."""
    if not isinstance(name, str) or not NAME_RE.match(name) or ".." in name:
        raise ValueError(f"not an episode file name: {name!r}")
    root = arena_dir(runs_root).resolve()
    p = (root / name).resolve()
    if p.parent != root:
        raise ValueError(f"not an episode file name: {name!r}")
    return p


def read_episode(runs_root: Path, name: str) -> dict:
    """One episode file, parsed. ValueError for a bad name, FileNotFoundError, OverflowError above the size limit."""
    p = episode_path(runs_root, name)
    if not p.is_file():
        raise FileNotFoundError(name)
    if p.stat().st_size > MAX_EPISODE_BYTES:
        raise OverflowError(f"{name} is larger than {MAX_EPISODE_BYTES // 2**20} MiB")
    return json.loads(p.read_text(encoding="utf-8"))

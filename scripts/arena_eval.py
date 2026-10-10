"""The arena as an eval: tasks x seeds x robot counts, success rate and turns to completion, one batched generation
per turn (so a 32-robot episode costs about what a 4-robot one does per turn -- docs/results.md §24).

    .venv/Scripts/python.exe -u scripts/arena_eval.py --checkpoint runs/m10_rl_336m/checkpoints/final.pt \
        --tasks key_door,relay,triangulate --agents 4,16,32 --seeds 5 --turns 16 --out runs/arena/eval_m10.json
    (team_relay, tag, gold_rush: the competitive tasks; their own measures are printed and stored under `games`)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=r"C:\slm-data\tokenizer\v1")
    ap.add_argument("--tasks", default="key_door,relay,triangulate")
    ap.add_argument("--agents", default="4,16,32")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--turns", type=int, default=16, help="turn budget; relay gets max(this, hops + 3) since the chain needs one turn per hop")
    ap.add_argument("--n", type=int, default=0, help="grid size; 0 = sized to the robot count")
    ap.add_argument("--opponent", default=None, help="side B of the competitive tasks: a checkpoint path or external:<name> "
                    "(e.g. external:smollm2-360m-instruct); the checkpoint plays side A")
    ap.add_argument("--batch-size", type=int, default=None, help="max robots per generation call (lower = less peak VRAM)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from slm.arena.world import Runner, World, load_player

    players = {"A": load_player(a.checkpoint, a.tokenizer)}
    if a.opponent:
        players["B"] = load_player(a.opponent, a.tokenizer)
    print("players: " + ", ".join(f"{k} = {p.label}{'' if p.native else ' (external, plain-text calls)'}" for k, p in players.items()), flush=True)
    rows = []
    for task in a.tasks.split(","):
        for k in (int(x) for x in a.agents.split(",")):
            n = a.n or (max(8, 2 * k + 2) if task == "relay" else max(8, k) if task == "team_relay" else max(8, int(k ** 0.5 * 3)))
            eps = []
            for seed in range(a.seeds):
                w = World(task, n, k, seed)
                r = Runner(w, players=players if (a.opponent and task in ("team_relay", "tag", "gold_rush")) else {"A": players["A"]},
                           batch_size=a.batch_size)
                res = r.run(max(a.turns, k + 2) if task == "relay" else max(a.turns, k // 2 + 3) if task == "team_relay" else a.turns)
                calls = sum(x["n_calls"] for x in res["transcript"]); turns_rows = len(res["transcript"])
                errors = sum(1 for x in res["transcript"] for c in x["calls"] if str(c[1]).startswith("error"))
                by_player = {}
                for x in res["transcript"]:
                    bp = by_player.setdefault(x.get("player", "A"), {"turns": 0, "calls": 0})
                    bp["turns"] += 1
                    bp["calls"] += int(bool(x["calls"]))
                eps.append({"seed": seed, "success": bool(res["score"].get("success")), "progress": res["score"].get("progress"), "done_turn": res["score"].get("done_turn"), "turns": res["turns"],
                            "score": res["score"], "calls_per_turn": round(calls / max(1, turns_rows), 2), "call_errors": errors, "seconds": res["seconds"],
                            "vram_peak_gib": r.vram_peak_gib, "acted": {pk: round(v["calls"] / max(1, v["turns"]), 2) for pk, v in by_player.items()}})
                print(f"{task:12s} k={k:2d} n={n:2d} seed={seed}: {'OK ' if eps[-1]['success'] else '-- '} turns {res['turns']:2d} done {res['score'].get('done_turn')} "
                      f"calls/turn {eps[-1]['calls_per_turn']} errors {errors} [{res['seconds']}s]", flush=True)
                import torch; torch.cuda.empty_cache()
            done = [e["done_turn"] if e["done_turn"] is not None else e["turns"] for e in eps if e["success"]]
            games = {}  # the competitive tasks' own measures, averaged over the seeds that have them
            for key in ("rival_code_rate", "chaser_good_moves", "runner_good_moves", "nearest_coin_moves", "any_coin_moves"):
                vals = [e["score"].get(key) for e in eps if e["score"].get(key) is not None]
                if vals:
                    games[key] = round(statistics.fmean(vals), 2)
            verdicts = [e["score"].get("verdict") or e["score"].get("winner") for e in eps if "verdict" in e["score"]]
            if verdicts:
                games["verdicts"] = verdicts
            for key in ("chaser_chance", "runner_chance", "nearest_chance"):
                vals = [e["score"].get(key) for e in eps if e["score"].get(key) is not None]
                if vals:
                    games[key] = round(statistics.fmean(vals), 2)
            if any("winner_side" in e["score"] for e in eps):
                games["wins"] = {sd: sum(e["score"].get("winner_side") == sd for e in eps) for sd in ("A", "B", "tie")}
            if any("coins_A" in e["score"] for e in eps):
                games["coins"] = {"A": sum(e["score"]["coins_A"] for e in eps), "B": sum(e["score"]["coins_B"] for e in eps)}
            games["acted"] = {pk: round(statistics.fmean(e["acted"].get(pk, 0) for e in eps), 2) for pk in sorted({x for e in eps for x in e["acted"]})}
            games["vram_peak_gib"] = max(e["vram_peak_gib"] for e in eps)
            rows.append({"task": task, "agents": k, "n": n, "episodes": eps, "games": games, "success_rate": round(sum(e["success"] for e in eps) / len(eps), 2),
                         "mean_progress": round(statistics.fmean(e["progress"] or 0 for e in eps), 2),
                         "mean_done_turn": round(statistics.fmean(done), 1) if done else None, "mean_seconds": round(statistics.fmean(e["seconds"] for e in eps), 1)})
            print(f"== {task} k={k}: success {rows[-1]['success_rate']} progress {rows[-1]['mean_progress']} mean done turn {rows[-1]['mean_done_turn']} mean s/episode {rows[-1]['mean_seconds']}"
                  + (f" | {games}" if games else ""), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"checkpoint": a.checkpoint, "opponent": a.opponent, "batch_size": a.batch_size, "turns": a.turns,
                                           "players": {k: {"label": p.label, "native": p.native} for k, p in players.items()}, "results": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

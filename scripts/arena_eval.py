"""The arena as an eval: tasks x seeds x robot counts, success rate and turns to completion, one batched generation
per turn (so a 32-robot episode costs about what a 4-robot one does per turn -- docs/results.md §24).

    .venv/Scripts/python.exe -u scripts/arena_eval.py --checkpoint runs/m10_rl_336m/checkpoints/final.pt \
        --tasks key_door,relay,triangulate --agents 4,16,32 --seeds 5 --turns 16 --out runs/arena/eval_m10.json
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
    ap.add_argument("--turns", type=int, default=16)
    ap.add_argument("--n", type=int, default=0, help="grid size; 0 = sized to the robot count")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from slm.arena.world import Runner, World
    from slm.data.tokenizer import SlmTokenizer
    from slm.eval.quality import load_model

    model, _ = load_model(Path(a.checkpoint), "cuda")
    tok = SlmTokenizer.load(a.tokenizer)
    rows = []
    for task in a.tasks.split(","):
        for k in (int(x) for x in a.agents.split(",")):
            n = a.n or (max(8, 2 * k + 2) if task == "relay" else max(8, int(k ** 0.5 * 3)))
            eps = []
            for seed in range(a.seeds):
                w = World(task, n, k, seed)
                r = Runner(w, model, tok)
                res = r.run(a.turns)
                calls = sum(x["n_calls"] for x in res["transcript"]); turns_rows = len(res["transcript"])
                errors = sum(1 for x in res["transcript"] for c in x["calls"] if str(c[1]).startswith("error"))
                eps.append({"seed": seed, "success": bool(res["score"].get("success")), "done_turn": res["score"].get("done_turn"), "turns": res["turns"],
                            "score": res["score"], "calls_per_turn": round(calls / max(1, turns_rows), 2), "call_errors": errors, "seconds": res["seconds"]})
                print(f"{task:12s} k={k:2d} n={n:2d} seed={seed}: {'OK ' if eps[-1]['success'] else '-- '} turns {res['turns']:2d} done {res['score'].get('done_turn')} "
                      f"calls/turn {eps[-1]['calls_per_turn']} errors {errors} [{res['seconds']}s]", flush=True)
                import torch; torch.cuda.empty_cache()
            done = [e["done_turn"] for e in eps if e["success"]]
            rows.append({"task": task, "agents": k, "n": n, "episodes": eps, "success_rate": round(sum(e["success"] for e in eps) / len(eps), 2),
                         "mean_done_turn": round(statistics.fmean(done), 1) if done else None, "mean_seconds": round(statistics.fmean(e["seconds"] for e in eps), 1)})
            print(f"== {task} k={k}: success {rows[-1]['success_rate']} mean done turn {rows[-1]['mean_done_turn']} mean s/episode {rows[-1]['mean_seconds']}", flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"checkpoint": a.checkpoint, "turns": a.turns, "results": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Greedy held-out score of RL task families on a checkpoint, with more prompts than the trainer's eval carries.

The RL trainer's held-out set is ~128 prompts over every family, so a family's score there rests on a handful of
prompts (constraints: ~13 in m11_rl_arena_336m). This draws `--n` held-out prompts per family (`make_tasks` with the
"heldout" split, the trainer's own) and decodes them the way the trainer's eval does (`greedy_accuracy`, tools on).

    python -u scripts/rl_family_eval.py --checkpoint runs/m10_rl_336m/checkpoints/final.pt --families constraints,chat --n 200 --out x.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=r"C:\slm-data\tokenizer\v1")
    ap.add_argument("--families", default="constraints")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch

    from slm.arena.world import load_player
    from slm.rl.rollout import greedy_accuracy
    from slm.rl.tasks import make_tasks, set_task_corpus
    from slm.utils.sdpa import sdpa_context

    pl = load_player(a.checkpoint, a.tokenizer)
    set_task_corpus(pl.tok)
    out = {"checkpoint": a.checkpoint, "n": a.n, "seed": a.seed, "families": {}}
    for fam in a.families.split(","):
        t0 = time.time()
        tasks = make_tasks([fam], a.n, "heldout", a.seed)
        with torch.no_grad(), sdpa_context("decode"):
            r = greedy_accuracy(pl.model, pl.tok, tasks, a.max_new, True, tools=True, max_tool_calls=6)
        out["families"][fam] = {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} | {"seconds": round(time.time() - t0, 1)}
        print(f"{fam}: score {r['score']:.3f} acc {r['accuracy']:.3f} malformed {r['malformed_rate']:.3f} n {r['n']}", flush=True)
        torch.cuda.empty_cache()
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

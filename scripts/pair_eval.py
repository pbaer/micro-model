"""Pairwise selection accuracy on a pair pool: the tournament's atomic decision, measured on its own.

    .venv/Scripts/python.exe scripts/pair_eval.py --checkpoint runs/m9_rl5_336m/checkpoints/step_00200.pt \
        --pool C:/slm-data/sft/v1/pair-sft/pair_pool_rl.jsonl --n 200 --out runs/m9_rl5_336m/pair_eval_s200.json

Reports accuracy (chance 0.5), the share of parsable picks, and the position bias (share of 'A' picks; the pool is
balanced, so 0.5 is unbiased). A bracket of these decisions can only be as good as this number.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=r"C:\slm-data\tokenizer\v1")
    ap.add_argument("--pool", default=r"C:\slm-data\sft\v1\pair-sft\pair_pool_rl.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch

    from slm.data.chat import format_chat, parse_assistant
    from slm.data.tokenizer import SlmTokenizer
    from slm.eval.quality import load_model
    from slm.swarm import parse_pick
    from slm.tools.loop import sample_with_tools
    from slm.utils.sdpa import sdpa_context

    rows = [json.loads(l) for l in Path(a.pool).read_text(encoding="utf-8").splitlines() if l.strip()]
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.n]
    model, _ = load_model(Path(a.checkpoint), "cuda")
    tok = SlmTokenizer.load(a.tokenizer)
    t0 = time.time()
    picks: list[int | None] = []
    with torch.no_grad(), sdpa_context("decode"):
        for i in range(0, len(rows), a.batch):
            batch = rows[i : i + a.batch]
            ids = [format_chat(tok, [{"role": "user", "content": r["prompt"]}], add_generation_prompt=True, think_required=True).ids for r in batch]
            gen = torch.Generator(device="cuda"); gen.manual_seed(0)
            tcs = sample_with_tools(model, tok, ids, a.max_new, 1.0, 1.0, 1, gen, max_calls=0)
            picks += [parse_pick(parse_assistant(tok, tc.ids)["answer"]) for tc in tcs]
            torch.cuda.empty_cache()
    gold = [0 if r["gold"] == "A" else 1 for r in rows]
    n = len(rows)
    correct = sum(p == g for p, g in zip(picks, gold))
    parsed = sum(p is not None for p in picks)
    by_src: dict[str, list[int]] = {}
    for r, p, g in zip(rows, picks, gold):
        by_src.setdefault(r.get("source") or "?", []).append(int(p == g))
    res = {"checkpoint": a.checkpoint, "pool": a.pool, "n": n, "accuracy": round(correct / n, 3), "parsed": round(parsed / n, 3),
           "a_share": round(sum(p == 0 for p in picks) / max(1, parsed), 3), "gold_a_share": round(sum(g == 0 for g in gold) / n, 3),
           "by_source": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_src.items())}, "seconds": round(time.time() - t0, 1)}
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Reasoning benchmark for any checkpoint: greedy accuracy on our verifiable task generators
(held-out split) and on GSM8K test, with malformed-rate and length stats.

    python -m slm.eval.reasoning --checkpoint runs/m5_reasoning_149m/checkpoints/final.pt --tasks arith1 arith2 algebra word --n 200 --gsm8k 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

from slm.config import ModelConfig, from_dict
from slm.data.sources import SOURCES
from slm.data.tokenizer import SlmTokenizer
from slm.model import Transformer
from slm.rl.rollout import greedy_accuracy
from slm.rl.tasks import Task, make_tasks
from slm.utils.sdpa import sdpa_context


def load_model(path: str) -> Transformer:
    ck = torch.load(path, map_location="cuda", weights_only=False)
    mcfg = from_dict(ModelConfig, ck.get("meta", {}).get("model_config") or ck["config"])
    m = Transformer(mcfg).cuda()
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
    return m.eval()


def gsm8k_tasks(n: int) -> list[Task]:
    src = SOURCES["gsm8k"]
    files = [p for p in src.local_dir.rglob("*.parquet") if "test" in p.name]
    if not files:
        return []
    rows = pq.read_table(files[0]).to_pylist()
    rows = rows if n < 0 else rows[:n]
    out = []
    for i, r in enumerate(rows):
        gold = r["answer"].rpartition("####")[2].strip()
        out.append(Task(id=f"gsm8k-test-{i}", prompt=r["question"].strip(), answer=gold, task="gsm8k"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=r"C:\slm-data\tokenizer\v1")
    ap.add_argument("--tasks", nargs="+", default=["arith1", "arith2", "arith2mul", "arith_multi", "algebra", "word"])
    ap.add_argument("--n", type=int, default=100, help="held-out problems per task")
    ap.add_argument("--gsm8k", type=int, default=0, help="number of GSM8K test problems (0 = skip, -1 = all 1319)")
    ap.add_argument("--tools", action="store_true", help="calculator tool available during generation")
    ap.add_argument("--max-tool-calls", type=int, default=8)
    ap.add_argument("--dump", default=None, help="jsonl of every problem with the model's trace, answer and tool stats")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tok = SlmTokenizer.load(a.tokenizer)
    model = load_model(a.checkpoint)
    res = {"checkpoint": a.checkpoint, "tools": a.tools, "per_task": {}}
    keep: list = []
    t0 = time.time()

    def line(name, r):
        tools_s = f"  calls {r['tool_calls_mean']:.1f} err {r['tool_error_rate']:.2f}" if a.tools else ""
        print(f"{name:12s} acc {r['accuracy']:.3f}  malformed {r['malformed_rate']:.2f}  len {r['mean_len']:.0f}{tools_s}  (n={r['n']})", flush=True)

    with sdpa_context("decode"):
        for name in a.tasks:
            tasks = make_tasks([name], a.n, "heldout", a.seed)
            r = greedy_accuracy(model, tok, tasks, a.max_new, tools=a.tools, max_tool_calls=a.max_tool_calls, keep=keep)
            res["per_task"][name] = r
            line(name, r)
        if a.gsm8k:
            r = greedy_accuracy(model, tok, gsm8k_tasks(a.gsm8k), a.max_new, tools=a.tools, max_tool_calls=a.max_tool_calls, keep=keep)
            res["per_task"]["gsm8k_test"] = r
            line("gsm8k_test", r)
    if a.dump:
        Path(a.dump).parent.mkdir(parents=True, exist_ok=True)
        with open(a.dump, "w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps({"task": r.task, "id": r.prompt_id, "prompt": r.prompt, "gold": r.gold, "text": r.text, "answer": r.parsed, "correct": r.correct,
                                    "malformed": r.malformed, "tool_calls": r.tool_calls, "tool_errors": r.tool_errors, "n_tokens": r.n_tokens}, ensure_ascii=False) + "\n")
    res["seconds"] = time.time() - t0
    res["mean_accuracy"] = sum(v["accuracy"] for v in res["per_task"].values()) / max(1, len(res["per_task"]))
    print(f"mean accuracy {res['mean_accuracy']:.3f} in {res['seconds']:.0f}s")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())

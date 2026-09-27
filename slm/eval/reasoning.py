"""Reasoning benchmark for any checkpoint: greedy accuracy on our verifiable task generators
(held-out split) and on GSM8K test, with malformed-rate and length stats.

    python -m slm.eval.reasoning --checkpoint runs/m5_reasoning_149m/checkpoints/final.pt --tasks arith1 arith2 algebra word --n 200 --gsm8k 200
    python -m slm.eval.reasoning --external qwen2.5-0.5b-instruct --n 100 --gsm8k 200 --svamp 300     # slm.eval.external

An external chat model gets the same tasks and the same user turn (`slm.rl.tasks.prompt_messages`: the problem plus the
`#### <number>` instruction, SUFFIX) through its own chat template, greedy, the same budget and `verify_answer`. It
has no think span and no tool, so `malformed` means only "never stopped" and the tool-use fields are null. Base
models are n/a (the eval is a chat prompt).
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


def load_model(path: str, device: str = "cuda") -> Transformer:
    ck = torch.load(path, map_location=device, weights_only=False)
    mcfg = from_dict(ModelConfig, ck.get("meta", {}).get("model_config") or ck["config"])
    m = Transformer(mcfg).to(device)
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


def svamp_tasks(n: int) -> list[Task]:
    """SVAMP test problems (300): one-step arithmetic word problems. The resolution band between our synthetic
    word problems, which post-training saturates, and GSM8K, where a 336M model sits near the floor."""
    src = SOURCES["svamp"]
    files = [p for p in src.local_dir.rglob("*.parquet") if "test" in p.name]
    if not files:
        return []
    rows = pq.read_table(files[0]).to_pylist()
    rows = rows if n < 0 else rows[:n]
    out = []
    for i, r in enumerate(rows):
        gold = str(r["Answer"]).strip()
        gold = gold[:-2] if gold.endswith(".0") else gold
        out.append(Task(id=f"svamp-test-{i}", prompt=r["question_concat"].strip(), answer=gold, task="svamp"))
    return out


def greedy_accuracy_external(model, tasks: list[Task], max_new_tokens: int = 256, batch: int = 32, keep: list | None = None) -> dict:
    """`slm.rl.rollout.greedy_accuracy` for an `slm.eval.external.HfChatModel` (no tool, no think span)."""
    from types import SimpleNamespace

    from slm.rl.rewards import verify_answer
    from slm.rl.tasks import prompt_messages

    if any(t.meta.get("functions") for t in tasks):
        raise SystemExit("declared-function tasks use our tool protocol: n/a for an external model")
    gens = model.batch_generate_chat([prompt_messages(t) for t in tasks], max_new_tokens, 0.0, batch_size=batch)
    correct = malformed = lenient = 0
    lengths, scores = [], []
    for t, g in zip(tasks, gens):
        v = verify_answer(g.text, t.answer, t.meta.get("verifier", "auto"))
        correct += int(v.correct)
        # not the score: the same answer without the '####' marker requirement (last number in the reply), so a write-up
        # can separate "did not follow our answer format" from "got it wrong"
        lenient += int(verify_answer(g.text, t.answer, t.meta.get("verifier", "auto"), strict=False).correct)
        malformed += int(not g.terminated)
        lengths.append(g.n_tokens)
        scores.append(v.fraction if v.fraction is not None else float(v.correct))
        if keep is not None:
            keep.append(SimpleNamespace(task=t.task, prompt_id=t.id, prompt=t.prompt, gold=t.answer, text=g.text, parsed=v.parsed, correct=v.correct,
                                        malformed=not g.terminated, tool_calls=None, tool_errors=None, answer_from_tool=None, tool_results=[], n_tokens=g.n_tokens))
    n = max(1, len(tasks))
    return {"accuracy": correct / n, "score": sum(scores) / n, "malformed_rate": malformed / n, "mean_len": sum(lengths) / n, "n": len(tasks),
            "tool_calls_mean": None, "tool_error_rate": None, "tool_use_rate": None, "answer_from_tool_rate": None, "accuracy_lenient": lenient / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--external", default=None, help="a registered external chat model (slm.eval.external) instead of a checkpoint")
    ap.add_argument("--tokenizer", default=r"C:\slm-data\tokenizer\v1")
    ap.add_argument("--tasks", nargs="+", default=["arith1", "arith2", "arith2mul", "arith_multi", "algebra", "word"])
    ap.add_argument("--n", type=int, default=100, help="held-out problems per task")
    ap.add_argument("--gsm8k", type=int, default=0, help="number of GSM8K test problems (0 = skip, -1 = all 1319)")
    ap.add_argument("--svamp", type=int, default=0, help="number of SVAMP test problems (0 = skip, -1 = all 300)")
    ap.add_argument("--tools", action="store_true", help="calculator tool available during generation")
    ap.add_argument("--max-tool-calls", type=int, default=8)
    ap.add_argument("--dump", default=None, help="jsonl of every problem with the model's trace, answer and tool stats")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.external:
        return main_external(a)
    if not a.checkpoint:
        ap.error("--checkpoint is required (or --external)")
    tok = SlmTokenizer.load(a.tokenizer)
    model = load_model(a.checkpoint)
    res = {"checkpoint": a.checkpoint, "tools": a.tools, "per_task": {}}
    keep: list = []
    t0 = time.time()

    def line(name, r):
        tools_s = f"  calls {r['tool_calls_mean']:.1f} err {r['tool_error_rate']:.2f} use {r['tool_use_rate']:.2f} from-tool {r['answer_from_tool_rate']:.2f}" if a.tools else ""
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
        if a.svamp:
            r = greedy_accuracy(model, tok, svamp_tasks(a.svamp), a.max_new, tools=a.tools, max_tool_calls=a.max_tool_calls, keep=keep)
            res["per_task"]["svamp_test"] = r
            line("svamp_test", r)
    if a.dump:
        Path(a.dump).parent.mkdir(parents=True, exist_ok=True)
        with open(a.dump, "w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps({"task": r.task, "id": r.prompt_id, "prompt": r.prompt, "gold": r.gold, "text": r.text, "answer": r.parsed, "correct": r.correct,
                                    "malformed": r.malformed, "tool_calls": r.tool_calls, "tool_errors": r.tool_errors, "answer_from_tool": r.answer_from_tool,
                                    "tool_results": r.tool_results, "n_tokens": r.n_tokens}, ensure_ascii=False) + "\n")
    res["seconds"] = time.time() - t0
    res["mean_accuracy"] = sum(v["accuracy"] for v in res["per_task"].values()) / max(1, len(res["per_task"]))
    print(f"mean accuracy {res['mean_accuracy']:.3f} in {res['seconds']:.0f}s")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


def main_external(a: argparse.Namespace) -> None:
    from slm.eval.external import chat_only, load_external, result_header

    if a.tools:
        raise SystemExit("--tools is our sandboxed Python tool protocol: n/a for an external model")
    chat_only(a.external, "reasoning")
    model = load_external(a.external)
    res = {**result_header(a.external), "tools": False, "per_task": {}}
    keep: list = []
    t0 = time.time()

    def run(name, tasks):
        r = greedy_accuracy_external(model, tasks, a.max_new, keep=keep)
        res["per_task"][name] = r
        print(f"{name:12s} acc {r['accuracy']:.3f}  malformed {r['malformed_rate']:.2f}  len {r['mean_len']:.0f}  (n={r['n']})"
              f"  [no-marker last-number acc {r['accuracy_lenient']:.3f}]", flush=True)

    for name in a.tasks:
        run(name, make_tasks([name], a.n, "heldout", a.seed))
    if a.gsm8k:
        run("gsm8k_test", gsm8k_tasks(a.gsm8k))
    if a.svamp:
        run("svamp_test", svamp_tasks(a.svamp))
    if a.dump:
        Path(a.dump).parent.mkdir(parents=True, exist_ok=True)
        with open(a.dump, "w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps({"task": r.task, "id": r.prompt_id, "prompt": r.prompt, "gold": r.gold, "text": r.text, "answer": r.parsed, "correct": r.correct,
                                    "malformed": r.malformed, "tool_calls": r.tool_calls, "tool_errors": r.tool_errors, "answer_from_tool": r.answer_from_tool,
                                    "tool_results": r.tool_results, "n_tokens": r.n_tokens}, ensure_ascii=False) + "\n")
    res["seconds"] = time.time() - t0
    res["mean_accuracy"] = sum(v["accuracy"] for v in res["per_task"].values()) / max(1, len(res["per_task"]))
    print(f"mean accuracy {res['mean_accuracy']:.3f} in {res['seconds']:.0f}s")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())

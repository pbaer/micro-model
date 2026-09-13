"""Synthetic reasoning traces for our verifiable tasks (no teacher model): templated step-by-step
solutions that are correct by construction. Written straight into SFT shards for reasoning SFT.

    python -m slm.rl.synth --tokenizer C:/slm-data/tokenizer/v1 --n 40000 --tasks arith1 arith2 arith2mul arith_multi algebra word
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

from slm.data.chat import format_chat
from slm.data.sft import SFT_DIR, SftShardWriter
from slm.data.tokenizer import SlmTokenizer
from slm.rl.tasks import Task, make_tasks, prompt_messages


def _add_trace(a: int, b: int) -> str:
    tens = (b // 10) * 10
    ones = b % 10
    if b < 10 or tens == 0:
        return f"{a} + {b} = {a + b}."
    return f"{a} + {b}: {a} + {tens} = {a + tens}, {a + tens} + {ones} = {a + b}."


def _sub_trace(a: int, b: int) -> str:
    tens = (b // 10) * 10
    ones = b % 10
    if b < 10 or tens == 0:
        return f"{a} - {b} = {a - b}."
    return f"{a} - {b}: {a} - {tens} = {a - tens}, {a - tens} - {ones} = {a - b}."


def _mul_trace(a: int, b: int) -> str:
    if a < 10 and b < 10:
        return f"{a} * {b} = {a * b}."
    tens = (b // 10) * 10
    ones = b % 10
    if tens == 0:
        return f"{a} * {b} = {a * b}."
    return f"{a} * {b}: {a} * {tens} = {a * tens}, {a} * {ones} = {a * ones}, {a * tens} + {a * ones} = {a * b}."


def trace_for(t: Task, rng: random.Random) -> str:
    if t.task.startswith("arith") and "expr" in t.meta and t.task != "arith_multi":
        a, op, b = t.meta["expr"].split()
        a, b = int(a), int(b)
        return {"+": _add_trace, "-": _sub_trace, "*": _mul_trace}[op](a, b)
    if t.task == "arith_multi":
        toks = t.meta["expr"].split()
        nums = [int(x) for x in toks[0::2]]
        ops = toks[1::2]
        # left-to-right with precedence handled by evaluating * first, then +/- (explain both stages)
        steps = []
        vals, opsl = nums[:], ops[:]
        i = 0
        while i < len(opsl):
            if opsl[i] == "*":
                v = vals[i] * vals[i + 1]
                steps.append(f"{vals[i]} * {vals[i + 1]} = {v}")
                vals[i : i + 2] = [v]
                opsl.pop(i)
            else:
                i += 1
        acc = vals[0]
        for o, v in zip(opsl, vals[1:]):
            new = acc + v if o == "+" else acc - v
            steps.append(f"{acc} {o} {v} = {new}")
            acc = new
        return "First multiply, then add and subtract from left to right. " + " ".join(s + "." for s in steps)
    if t.task == "algebra":
        m = re.match(r"(-?\d+)x ([+-]) (\d+) = (-?\d+)", t.meta["eq"])
        a, sign, b, c = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        b_signed = b if sign == "+" else -b
        rhs = c - b_signed
        return f"Move the constant: {a}x = {c} {'-' if b_signed >= 0 else '+'} {abs(b_signed)} = {rhs}. Divide both sides by {a}: x = {rhs} / {a} = {rhs // a}."
    if t.task == "word":
        nums = [int(x) for x in re.findall(r"\d+", t.prompt)]
        if "buys" in t.prompt:
            return f"Start with {nums[0]}, add {nums[1]}: {nums[0]} + {nums[1]} = {nums[0] + nums[1]}."
        if "gives away" in t.prompt:
            return f"Start with {nums[0]}, take away {nums[1]}: {nums[0]} - {nums[1]} = {nums[0] - nums[1]}."
        return f"{nums[0]} boxes with {nums[1]} each: {nums[0]} * {nums[1]} = {nums[0] * nums[1]}."
    raise ValueError(t.task)


def build(tok: SlmTokenizer, tasks: list[str], n: int, out_root: Path, name: str = "synthetic-reasoning", seed: int = 0) -> dict:
    rng = random.Random(seed)
    items = make_tasks(tasks, n, "train", seed)
    val_items = make_tasks(tasks, max(200, n // 50), "heldout", seed + 1)
    out = out_root / name
    stats = {}
    for split, its in (("train", items), ("val", val_items)):
        w = SftShardWriter(out / split)
        for t in its:
            msgs = prompt_messages(t) + [{"role": "assistant", "think": trace_for(t, rng), "content": "#### " + t.answer}]
            enc = format_chat(tok, msgs, think_required=True)
            w.add(enc.ids, enc.loss_mask)
        w.flush()
        stats[split] = {"examples": w.total_examples, "tokens": w.total_tokens, "targets": w.total_targets}
    import json

    (out / "manifest.json").write_text(json.dumps({"name": name, "tasks": tasks, "tokenizer_sha256": tok.sha256, "train_examples": stats["train"]["examples"],
                                                   "train_tokens": stats["train"]["tokens"], "train_targets": stats["train"]["targets"], "val_examples": stats["val"]["examples"],
                                                   "val_tokens": stats["val"]["tokens"], "val_targets": stats["val"]["targets"], "think_required": True}, indent=1), encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--tasks", nargs="+", default=["arith1", "arith2", "arith2mul", "arith_multi", "algebra", "word"])
    ap.add_argument("--name", default="synthetic-reasoning")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tok = SlmTokenizer.load(a.tokenizer)
    st = build(tok, a.tasks, a.n, SFT_DIR / Path(a.tokenizer).name, a.name, a.seed)
    print(st)


if __name__ == "__main__":
    sys.exit(main())

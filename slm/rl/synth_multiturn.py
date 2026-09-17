"""Templated multi-turn tool conversations (no model involved): a first question solved with a Python call that
stores the result in a variable, then 1-3 follow-ups ("Now add 5 to that", "Double it") solved by reusing the
session variable. Correct by construction; the conversion runs every call in one PySession per conversation, so
the recorded results are exactly what the harness inserts at inference.

    python -m slm.rl.synth_multiturn --tokenizer C:/slm-data/tokenizer/v1 --n 20000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from slm.data.answers import SUFFIX, apply_style  # noqa: F401 (SUFFIX re-exported for older callers)
from slm.data.chat import format_chat
from slm.data.sft import SFT_DIR, SftShardWriter
from slm.data.tokenizer import SlmTokenizer
from slm.rl.tasks import _split_of

NAMES = ["Ada", "Ben", "Cleo", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivan", "Jo", "Kai", "Lena", "Mo", "Nia", "Omar", "Pia"]
THINGS = ["apples", "books", "coins", "marbles", "stickers", "cards", "pencils", "shells", "cookies", "tickets"]


def first_turn(rng: random.Random) -> tuple[str, str, int]:
    """(user text, think text with a program that sets `total`, value)."""
    n, thing = rng.choice(NAMES), rng.choice(THINGS)
    k = rng.randrange(4)
    if k == 0:
        a, b = rng.randint(3, 60), rng.randint(2, 40)
        return f"{n} has {a} {thing} and buys {b} more. How many {thing} does {n} have now?", f"Start with {a}, add {b}. <<<total = {a} + {b}\ntotal>>>", a + b
    if k == 1:
        a, b = rng.randint(20, 90), rng.randint(2, 19)
        return f"{n} has {a} {thing} and gives away {b}. How many {thing} are left?", f"Start with {a}, take away {b}. <<<total = {a} - {b}\ntotal>>>", a - b
    if k == 2:
        boxes, each = rng.randint(2, 9), rng.randint(3, 25)
        return f"{n} has {boxes} boxes with {each} {thing} each. How many {thing} in total?", f"{boxes} boxes of {each}. <<<total = {boxes} * {each}\ntotal>>>", boxes * each
    price, qty = rng.randint(2, 15), rng.randint(2, 12)
    return f"Each of the {qty} {thing} costs {price} dollars. How much do they cost together?", f"{qty} times {price} dollars. <<<total = {qty} * {price}\ntotal>>>", price * qty


def follow_up(rng: random.Random, cur: int) -> tuple[str, str, int]:
    k = rng.randrange(6)
    if k == 0:
        m = rng.randint(2, 30)
        return f"Now add {m} to that.", f"Add {m} to the previous total. <<<total = total + {m}\ntotal>>>", cur + m
    if k == 1:
        m = rng.randint(1, max(1, cur - 1)) if cur > 1 else 1
        return f"Then subtract {m}.", f"Subtract {m} from the previous total. <<<total = total - {m}\ntotal>>>", cur - m
    if k == 2:
        return "Double it.", "Twice the previous total. <<<total = total * 2\ntotal>>>", cur * 2
    if k == 3:
        m = rng.randint(2, 5)
        return f"What is that times {m}?", f"Multiply the previous total by {m}. <<<total = total * {m}\ntotal>>>", cur * m
    if k == 4:
        d = next((x for x in (2, 3, 4, 5) if cur % x == 0), None)
        if d is not None:
            return f"Split that evenly among {d} people. How many each?", f"Divide the previous total by {d}. <<<total = total // {d}\ntotal>>>", cur // d
        m = rng.randint(2, 30)
        return f"Now add {m} to that.", f"Add {m} to the previous total. <<<total = total + {m}\ntotal>>>", cur + m
    m = rng.randint(10, 99)
    return f"How much more is that than {m}?", f"Difference between the previous total and {m}. <<<total = total - {m}\ntotal>>>", cur - m


def conversation(rng: random.Random) -> list[dict]:
    u, think, val = first_turn(rng)
    msgs = [{"role": "user", "content": u}, {"role": "assistant", "think": think, "content": f"#### {val}"}]
    for _ in range(rng.choice([1, 1, 2, 2, 3])):
        u, think, val = follow_up(rng, val)
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "think": think, "content": f"#### {val}"}]
    return apply_style(msgs, rng, 0.5)


def build(tok: SlmTokenizer, n: int, out_root: Path, name: str = "synthetic-multiturn-tools", seed: int = 0) -> dict:
    rng = random.Random(seed)
    out = out_root / name
    writers = {"train": SftShardWriter(out / "train"), "val": SftShardWriter(out / "val")}
    counts = {"train": 0, "val": 0}
    seen: set[str] = set()
    while counts["train"] + counts["val"] < n:
        msgs = conversation(rng)
        key = msgs[0]["content"]
        if key in seen:
            continue
        seen.add(key)
        split = _split_of(key, 50)  # 5% held out by first-question hash, disjoint from the RL split rule
        enc = format_chat(tok, msgs, think_required=True, tools=True)
        writers[split if split == "train" else "val"].add(enc.ids, enc.loss_mask)
        counts["train" if split == "train" else "val"] += 1
    for w in writers.values():
        w.flush()
    m = {"name": name, "tokenizer_sha256": tok.sha256, "think_required": True, "tools": True, "multiturn": True,
         "train_examples": writers["train"].total_examples, "train_tokens": writers["train"].total_tokens, "train_targets": writers["train"].total_targets,
         "val_examples": writers["val"].total_examples, "val_tokens": writers["val"].total_tokens, "val_targets": writers["val"].total_targets}
    (out / "manifest.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--name", default="synthetic-multiturn-tools")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tok = SlmTokenizer.load(a.tokenizer)
    print(json.dumps(build(tok, a.n, SFT_DIR / Path(a.tokenizer).name, a.name, a.seed), indent=1))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()

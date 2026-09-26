"""Multi-turn chat: does the model remember what the user said two turns ago, and does it hold the format?

    python -m slm.eval.multiturn --checkpoint runs/m9_rl2_336m/checkpoints/best.pt --n 64

Every other eval in this project is single-turn. This one runs scripted three-turn conversations:

    turn 1  the user states a fact            "My cat is called Biscuit."
    turn 2  an unrelated distractor           "Why is the sky blue?"
    turn 3  a question that needs the fact    "What is my cat called?"

and scores, deterministically and without a judge:

    recall     the turn-3 answer contains the fact
    format     every assistant turn terminated properly (<|end|> reached, not cut off)
    misfire    share of assistant turns that called the Python tool -- these are chat turns, so any call is one
    templated  share of turn-3 answers that are a bare verifier template ("So the answer is Biscuit.")

Facts, names and distractors are drawn from small tables with a seeded RNG, so a run is reproducible and no
conversation here can appear in training data (the tables are ours). Decoding is greedy (top_k=1) so the
numbers are stable across runs. The tool loop is used for generation so a model that does call the tool gets
a real result back, exactly as it would in the harness -- the score measures the model, not a dead result.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

NAMES = ["Biscuit", "Marlowe", "Pepper", "Juniper", "Atlas", "Clementine", "Bramble", "Nimbus", "Saffron", "Wilbur",
         "Odette", "Fennel", "Rowan", "Thistle", "Quincy", "Maple"]
CITIES = ["Oslo", "Lisbon", "Nairobi", "Kyoto", "Bogota", "Tallinn", "Auckland", "Marrakesh", "Ljubljana", "Hanoi",
          "Reykjavik", "Valparaiso", "Tbilisi", "Porto", "Adelaide", "Cusco"]
JOBS = ["beekeeper", "glassblower", "cartographer", "lighthouse keeper", "piano tuner", "florist", "locksmith",
        "sommelier", "falconer", "typesetter", "ferry captain", "archivist"]
COLOURS = ["teal", "maroon", "mustard", "lavender", "olive", "crimson", "turquoise", "amber", "indigo", "coral"]
NUMBERS = [str(n) for n in (7, 13, 21, 34, 42, 58, 63, 77, 81, 96, 108, 117, 144, 233)]

# (statement template, question template, answer table, hint for matching)
FACTS = [
    ("My cat is called {x}.", "What is my cat called?", NAMES),
    ("My dog's name is {x}.", "What is my dog's name?", NAMES),
    ("I live in {x}.", "Which city do I live in?", CITIES),
    ("I grew up in {x}.", "Where did I grow up?", CITIES),
    ("I work as a {x}.", "What do I do for a living?", JOBS),
    ("My car is {x}.", "What colour is my car?", COLOURS),
    ("My favourite number is {x}.", "What is my favourite number?", NUMBERS),
    ("My sister is called {x}.", "What is my sister called?", NAMES),
]
DISTRACTORS = [
    "Why is the sky blue?", "Write one sentence about rain.", "Name a primary colour.",
    "What is the capital of France?", "Give me a word that rhymes with cat.", "Describe a forest in one sentence.",
    "What do bees make?", "Is a tomato a fruit or a vegetable?", "Name a musical instrument.",
    "What season comes after winter?",
]
_TEMPLATE_OPENERS = ("so the answer is", "the answer is", "that would be", "it is", "so it's", "that makes", "that gives")


def max_conversations() -> int:
    """Distinct (statement, value, distractor) triples the tables allow: the hard cap on `n`."""
    return sum(len(table) for _, _, table in FACTS) * len(DISTRACTORS)


def make_conversations(n: int, seed: int) -> list[dict]:
    """`n` distinct conversations. Distinct on the (statement, value, distractor) triple, and bounded: asking
    for more than the tables can supply raises instead of spinning -- the first version of this deduped on
    (statement, value), 116 combinations, and a test asking for 400 looped forever on two CPUs."""
    cap = max_conversations()
    if n > cap:
        raise ValueError(f"only {cap} distinct conversations are possible from the tables; asked for {n}")
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        stmt, q, table = rng.choice(FACTS)
        x = rng.choice(table)
        d = rng.choice(DISTRACTORS)
        key = (stmt, x, d)
        if key in seen:
            continue
        seen.add(key)
        out.append({"fact": x, "turns": [stmt.format(x=x) + " Please remember that.", d, q]})
    return out


def _generate_turn(model, tok, histories, max_new, max_calls):
    """One assistant turn for every conversation, batched; returns the tool-loop results in order."""
    import torch

    from slm.data.chat import format_chat
    from slm.tools.loop import sample_with_tools

    prompts = [format_chat(tok, h, add_generation_prompt=True, think_required=True).ids for h in histories]
    gen = torch.Generator(device="cuda")
    gen.manual_seed(0)
    return sample_with_tools(model, tok, prompts, max_new, 1.0, 1.0, 1, gen, max_calls=max_calls)  # top_k=1: greedy


def run(checkpoint: str, tokenizer: str, n: int, seed: int, max_new: int, max_calls: int, batch: int) -> dict:
    import torch

    from slm.data.chat import parse_assistant
    from slm.data.tokenizer import SlmTokenizer
    from slm.eval.quality import load_model
    from slm.utils.sdpa import sdpa_context

    model, _ = load_model(Path(checkpoint), "cuda")
    tok = SlmTokenizer.load(tokenizer)
    convs = make_conversations(n, seed)
    t0 = time.time()
    with torch.no_grad(), sdpa_context("decode"):
        for lo in range(0, n, batch):
            chunk = convs[lo:lo + batch]
            histories = [[] for _ in chunk]
            for turn in range(3):
                for h, c in zip(histories, chunk):
                    h.append({"role": "user", "content": c["turns"][turn]})
                tcs = _generate_turn(model, tok, histories, max_new, max_calls)
                for h, c, tc in zip(histories, chunk, tcs):
                    p = parse_assistant(tok, tc.ids)
                    c.setdefault("assistant", []).append({
                        "answer": p["answer"], "think": p["think"], "terminated": bool(p["terminated"]) and not p["malformed"],
                        "tool_calls": int(tc.n_calls), "n_tokens": len(tc.ids)})
                    h.append({"role": "assistant", "ids": list(tc.ids)})
            torch.cuda.empty_cache()
    for c in convs:
        final = c["assistant"][2]["answer"].strip()
        c["recall"] = c["fact"].lower() in final.lower()
        c["format_ok"] = all(a["terminated"] for a in c["assistant"])
        c["misfires"] = sum(1 for a in c["assistant"] if a["tool_calls"])
        c["templated"] = final.lower().startswith(_TEMPLATE_OPENERS) and len(final.split()) <= 8
    summary = {
        "n": n, "seed": seed,
        "recall": round(statistics.fmean(c["recall"] for c in convs), 3),
        "format": round(statistics.fmean(c["format_ok"] for c in convs), 3),
        "misfire": round(sum(c["misfires"] for c in convs) / (3 * n), 3),
        "templated": round(statistics.fmean(c["templated"] for c in convs), 3),
        "mean_answer_tokens": round(statistics.fmean(a["n_tokens"] for c in convs for a in c["assistant"]), 1),
        "seconds": round(time.time() - t0, 1),
    }
    return {"checkpoint": checkpoint, "summary": summary, "conversations": convs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=r"C:\slm-data\tokenizer\v1")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--max-tool-calls", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run(a.checkpoint, a.tokenizer, a.n, a.seed, a.max_new, a.max_tool_calls, a.batch)
    s = res["summary"]
    print(f"multiturn n={s['n']}: recall {s['recall']:.3f}  format {s['format']:.3f}  misfire {s['misfire']:.3f}"
          f"  templated {s['templated']:.3f}  mean tokens/turn {s['mean_answer_tokens']}  [{s['seconds']:.0f}s]")
    for c in res["conversations"][:4]:
        print(f"  fact={c['fact']!r:<14} recall={str(c['recall']):<5} turn3={c['assistant'][2]['answer'][:70]!r}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

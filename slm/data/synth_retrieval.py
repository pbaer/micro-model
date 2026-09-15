"""Templated long-document retrieval data for context extension (no model involved; correct by construction).

Two document types, written straight to tokenized shards as a normal pretraining source:

  needle docs   Continuous real training text (document boundaries stripped) with 1-4 templated facts
                inserted at random depths, followed by a question per fact and its answer:
                    ... The secret number is 482913. ... <text> ...
                    Question: What is the secret number?\\nAnswer: 482913
  ledger docs   A long list of "key -> value" lines followed by queries for a few random keys.

Lengths are log-uniform between --min-len and --max-len tokens so every context bucket gets examples.
The point is to give the model many training signals that require attending far back, which natural
long documents provide only weakly; the eval (slm.eval.long_context) uses *validation* text and
different fact values, so this is training the skill, not the test items.

    python -m slm.data.synth_retrieval --tokenizer C:/slm-data/tokenizer/v1 --tokens 150000000 --max-len 16384
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from slm.data.prepare import ShardWriter
from slm.data.sources import DATA_ROOT
from slm.data.tokenizer import SlmTokenizer

CITIES = ["Lisbon", "Oslo", "Nairobi", "Kyoto", "Denver", "Havana", "Perth", "Quito", "Zagreb", "Muscat", "Tallinn", "Lima", "Cardiff", "Tunis", "Bergen", "Osaka"]
NAMES = ["Marta", "Elias", "Noor", "Tomasz", "Priya", "Jonas", "Amara", "Felix", "Ingrid", "Rafael", "Sofia", "Kenji", "Leila", "Bram", "Zara", "Owen"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
WORDS = ["falcon", "granite", "harbor", "lantern", "meadow", "orchid", "quartz", "saffron", "tundra", "velvet", "willow", "zephyr", "cobalt", "ember", "juniper", "marble"]


N_TEMPLATES = 5


def make_fact(rng: random.Random, k: int | None = None) -> tuple[str, str, str]:
    """(sentence to insert, question, answer) for template k (random if None)."""
    k = rng.randrange(N_TEMPLATES) if k is None else k
    if k == 0:
        v = rng.randint(100000, 999999)
        return f"The secret number is {v}.", "What is the secret number?", str(v)
    if k == 1:
        code = f"{rng.choice(WORDS).upper()}-{rng.randint(1000, 9999)}"
        return f"The access code for the archive is {code}.", "What is the access code for the archive?", code
    if k == 2:
        n, c = rng.choice(NAMES), rng.choice(CITIES)
        return f"For the record, {n}'s favorite city is {c}.", f"What is {n}'s favorite city?", c
    if k == 3:
        d = f"{rng.choice(MONTHS)} {rng.randint(1, 28)}, {rng.randint(1990, 2035)}"
        return f"The final review meeting is scheduled for {d}.", "When is the final review meeting scheduled?", d
    v = rng.randint(10, 999)
    unit = rng.choice(["kilograms", "liters", "meters", "boxes", "pages"])
    return f"The shipment contained exactly {v} {unit}.", "How many items did the shipment contain, and in what unit?", f"{v} {unit}"


class Backdrop:
    """Continuous text windows from a tokenized train split (boundary specials removed)."""

    def __init__(self, tok: SlmTokenizer, split_dir: Path) -> None:
        self.tok = tok
        self.paths = sorted(Path(split_dir).glob("shard_*.bin"))
        assert self.paths, f"no shards under {split_dir}"
        self.mm = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.paths]
        self.drop = np.array(sorted({tok.bos_id, tok.eos_id, tok.pad_id}), dtype=np.uint16)

    def text(self, n_tokens: int, rng: random.Random) -> str:
        mm = self.mm[rng.randrange(len(self.mm))]
        want = int(n_tokens * 1.05) + 64
        start = rng.randrange(0, max(1, len(mm) - want))
        chunk = np.asarray(mm[start : start + want])
        chunk = chunk[~np.isin(chunk, self.drop)][:n_tokens]
        return self.tok.decode(chunk.astype(np.int64).tolist())


def needle_doc(tok: SlmTokenizer, backdrop: Backdrop, length: int, rng: random.Random, early_frac: float = 0.0) -> list[int]:
    """early_frac: probability that a fact is placed in the first 15% of the document (the hardest cells
    at full context are needles near the start, so stage 1b oversamples them)."""
    n_facts = rng.choice([1, 1, 2, 2, 3, 4])
    facts = [make_fact(rng, k) for k in rng.sample(range(N_TEMPLATES), n_facts)]  # distinct templates: every question has one answer
    qa_text = "".join(f"\n\nQuestion: {q}\nAnswer: {a}" for _, q, a in facts)
    overhead = len(tok.encode(qa_text)) + sum(len(tok.encode(" " + f + " ")) for f, _, _ in facts) + 2
    body = backdrop.text(max(64, length - overhead), rng)
    # insert each fact at a sentence boundary near a random depth (0 = start, 1 = end)
    sents = body.split(". ")
    for f, _, _ in facts:
        depth = rng.random() * 0.15 if rng.random() < early_frac else rng.random()
        pos = int(depth * len(sents))
        sents.insert(pos, f.rstrip("."))
    text = ". ".join(sents) + qa_text
    return [tok.bos_id, *tok.encode(text), tok.eos_id]


def ledger_doc(tok: SlmTokenizer, length: int, rng: random.Random) -> list[int]:
    n_pairs = max(4, length // 12)
    keys = rng.sample(range(1000, 99999), n_pairs)
    vals = [rng.randint(100000, 999999) for _ in keys]
    lines = [f"Record {k}: value {v}" for k, v in zip(keys, vals)]
    rng.shuffle(lines)
    q_idx = rng.sample(range(n_pairs), min(3, n_pairs))
    qa = "".join(f"\n\nQuestion: What is the value of record {keys[i]}?\nAnswer: {vals[i]}" for i in q_idx)
    text = "Reference ledger.\n" + "\n".join(lines) + qa
    return [tok.bos_id, *tok.encode(text), tok.eos_id]


def build(tok: SlmTokenizer, backdrop_dir: Path, out_root: Path, total_tokens: int, min_len: int, max_len: int, ledger_frac: float, val_tokens: int,
          name: str = "synth-retrieval", seed: int = 0, shard_tokens: int = 50_000_000, early_frac: float = 0.0) -> dict:
    rng = random.Random(seed)
    backdrop = Backdrop(tok, backdrop_dir)
    out = out_root / name
    writers = {"train": ShardWriter(out / "train", shard_tokens), "val": ShardWriter(out / "val", shard_tokens)}
    t0 = time.time()
    counts = {"needle": 0, "ledger": 0}
    for split, budget in (("val", val_tokens), ("train", total_tokens)):
        w = writers[split]
        while w.total_tokens < budget:
            length = int(min_len * (max_len / min_len) ** rng.random())  # log-uniform
            if rng.random() < ledger_frac:
                ids = ledger_doc(tok, length, rng)
                counts["ledger"] += 1
            else:
                ids = needle_doc(tok, backdrop, length, rng, early_frac)
                counts["needle"] += 1
            w.add(ids[: max_len + 64])
        w.flush()
    m = {"name": name, "tokenizer_sha256": tok.sha256, "train_tokens": writers["train"].total_tokens, "train_docs": writers["train"].total_docs,
         "val_tokens": writers["val"].total_tokens, "val_docs": writers["val"].total_docs, "min_len": min_len, "max_len": max_len, "ledger_frac": ledger_frac,
         "backdrop": str(backdrop_dir), "seed": seed, "early_frac": early_frac, "seconds": time.time() - t0, **counts}
    (out / "manifest.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tokenized-root", default=str(DATA_ROOT / "tokenized"))
    ap.add_argument("--backdrop", default="fineweb-edu-b", help="tokenized source whose TRAIN split provides the surrounding text")
    ap.add_argument("--tokens", type=int, default=150_000_000)
    ap.add_argument("--val-tokens", type=int, default=2_000_000)
    ap.add_argument("--min-len", type=int, default=1024)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--ledger-frac", type=float, default=0.3)
    ap.add_argument("--name", default="synth-retrieval")
    ap.add_argument("--out-root", default=None, help="where to write <name>/{train,val} (default: the tokenized root)")
    ap.add_argument("--early-frac", type=float, default=0.0, help="fraction of facts placed in the first 15%% of the document")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tok = SlmTokenizer.load(a.tokenizer)
    root = Path(a.tokenized_root) / Path(a.tokenizer).name
    out_root = Path(a.out_root) if a.out_root else root
    m = build(tok, root / a.backdrop / "train", out_root, a.tokens, a.min_len, a.max_len, a.ledger_frac, a.val_tokens, a.name, a.seed, early_frac=a.early_frac)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()

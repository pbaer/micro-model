"""Templated retrieval documents: every question has exactly one answer, the answer text appears in the
body, lengths follow the requested range, and shards are written in the normal pretraining layout."""

import json
import random
from pathlib import Path

import numpy as np

from slm.data.synth_retrieval import N_TEMPLATES, Backdrop, build, ledger_doc, make_fact, needle_doc
from slm.data.tokenizer import SlmTokenizer, train_bpe


def _tok() -> SlmTokenizer:
    return SlmTokenizer(train_bpe(["The secret number is 123456. Question: What is the value of record 42? Answer: 987654 " * 40, "the market opened early and the vendors arranged goods " * 40], vocab_size=320))


def _split(tmp_path: Path, tok: SlmTokenizer) -> Path:
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 200, size=30000, dtype=np.uint16)
    ids[::40] = tok.eos_id
    d = tmp_path / "back" / "train"
    d.mkdir(parents=True)
    ids.tofile(d / "shard_00000.bin")
    np.save(d / "shard_00000.idx.npy", np.arange(0, 30000, 40, dtype=np.int64))
    return d


def test_facts_are_unambiguous_and_present(tmp_path):
    tok = _tok()
    rng = random.Random(1)
    for k in range(N_TEMPLATES):
        sent, q, a = make_fact(rng, k)
        assert a in sent and q.endswith("?")
    back = Backdrop(tok, _split(tmp_path, tok))
    for _ in range(20):
        ids = needle_doc(tok, back, 600, rng)
        text = tok.decode(ids)
        assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id and tok.bos_id not in ids[1:-1]
        qa = text.split("\n\nQuestion: ")[1:]
        questions = [x.split("\nAnswer:")[0] for x in qa]
        assert len(set(questions)) == len(questions), "duplicate question in one document"
        for x in qa:
            ans = x.split("\nAnswer: ")[1].split("\n")[0].strip("<|eos|>").strip()
            assert ans in text.split("\n\nQuestion:")[0], f"answer {ans!r} not in body"
    led = tok.decode(ledger_doc(tok, 400, rng))
    assert led.startswith("<|bos|>Reference ledger.") and led.count("Question:") == 3


def test_build_writes_shards(tmp_path):
    tok = _tok()
    back = _split(tmp_path, tok)
    m = build(tok, back, tmp_path / "out", total_tokens=20000, min_len=128, max_len=1024, ledger_frac=0.3, val_tokens=3000, shard_tokens=8000)
    assert m["train_tokens"] >= 20000 and m["val_tokens"] >= 3000 and m["needle"] + m["ledger"] == m["train_docs"] + m["val_docs"]
    tr = tmp_path / "out" / "synth-retrieval" / "train"
    shards = sorted(tr.glob("shard_*.bin"))
    assert len(shards) >= 3  # 8000-token shards
    idx = np.load(shards[0].with_name(shards[0].name.replace(".bin", ".idx.npy")))
    mm = np.memmap(shards[0], dtype=np.uint16, mode="r")
    assert idx[0] == 0 and all(mm[i] == tok.bos_id for i in idx)
    assert json.loads((tmp_path / "out" / "synth-retrieval" / "manifest.json").read_text())["min_len"] == 128

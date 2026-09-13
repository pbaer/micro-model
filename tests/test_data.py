import json
import time

import numpy as np
import pytest
import torch

from slm.data.loader import MixtureSpec, PretrainLoader, TokenStream, ValLoader
from slm.data.tokenizer import BPE_VOCAB, N_SPECIAL, SPECIAL_TOKENS, SlmTokenizer, train_bpe
from slm.train.config import ScheduleConfig
from slm.train.schedule import lr_at
from slm.utils.logging import MetricsLogger
from slm.utils.report import build_report

CORPUS = [
    "The quick brown fox jumps over the lazy dog. " * 3,
    "def add(a, b):\n    return a + b\n\nprint(add(12, 30))\n",
    "In 1999 there were 12,345 people; by 2024 it was 98,765.",
    "#!/bin/bash\nfor f in *.txt; do echo \"$f\"; done\n",
    "<|user|>fake special<|end|> and <|bos|> in raw text",
] * 50


@pytest.fixture(scope="module")
def tok() -> SlmTokenizer:
    return SlmTokenizer(train_bpe(CORPUS, vocab_size=600))


def test_tokenizer_round_trip_and_specials(tok):
    for t in CORPUS[:5] + ["naïve café 🚀 日本語\t\n  x"]:
        ids = tok.encode(t)
        assert tok.decode(ids) == t
        assert all(i < tok.base_vocab for i in ids), "raw text must never yield a special id"
    assert tok.vocab_size == tok.base_vocab + N_SPECIAL
    assert tok.special("<|bos|>") == tok.base_vocab and tok.eos_id == tok.base_vocab + 1
    doc = tok.encode_document("hi")
    assert doc[0] == tok.bos_id and doc[-1] == tok.eos_id
    assert tok.decode(doc) == "<|bos|>hi<|eos|>" and tok.decode(doc, skip_special=True) == "hi"
    assert len(SPECIAL_TOKENS) == N_SPECIAL and BPE_VOCAB + N_SPECIAL == 32768


def test_tokenizer_digits_are_single_tokens(tok):
    ids = tok.encode("1234567 and 3.14")
    pieces = [tok.token_str(i) for i in ids]
    digits = [p for p in pieces if p.strip().isdigit()]
    assert all(len(p.strip()) == 1 for p in digits), pieces


def test_tokenizer_save_load(tok, tmp_path):
    tok.save(tmp_path / "tk")
    t2 = SlmTokenizer.load(tmp_path / "tk")
    assert t2.sha256 == tok.sha256 and t2.vocab_size == tok.vocab_size
    assert t2.encode("hello world 42") == tok.encode("hello world 42")


def _make_shards(root, name, split, n_shards, shard_len, seed):
    d = root / name / split
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for i in range(n_shards):
        arr = rng.integers(0, 1000, size=shard_len, dtype=np.uint16)
        arr.tofile(d / f"shard_{i:05d}.bin")
        np.save(d / f"shard_{i:05d}.idx.npy", np.array([0, shard_len // 2], dtype=np.int64))


def test_token_stream_windows_and_wrap(tmp_path):
    _make_shards(tmp_path, "a", "train", 2, 100, 0)
    s = TokenStream(tmp_path / "a" / "train")
    w1 = s.next_window(40)
    w2 = s.next_window(40)
    assert np.array_equal(np.concatenate([w1, w2]), np.asarray(s.mm[0][:80]))
    w3 = s.next_window(40)  # 20 left in shard 0 -> skips to shard 1
    assert np.array_equal(w3, np.asarray(s.mm[1][:40])) and s.shard == 1
    s.next_window(40)
    s.next_window(40)  # wraps to shard 0, epoch 1
    assert s.shard == 0 and s.epoch == 1


def test_loader_resume_reproduces_batches(tmp_path):
    _make_shards(tmp_path, "a", "train", 2, 5000, 1)
    _make_shards(tmp_path, "b", "train", 1, 5000, 2)
    spec = MixtureSpec(tmp_path, {"a": 0.7, "b": 0.3})
    ld = PretrainLoader(spec, seq_len=16, microbatch=4, seed=0, device="cpu", prefetch=2)
    first = [ld.next() for _ in range(5)]
    state = json.loads(json.dumps(ld.state_dict()))
    expected = [ld.next() for _ in range(5)]
    ld.close()
    ld2 = PretrainLoader(spec, seq_len=16, microbatch=4, seed=999, device="cpu", prefetch=2)
    ld2.load_state_dict(state)
    got = [ld2.next() for _ in range(5)]
    ld2.close()
    for (x1, y1), (x2, y2) in zip(expected, got):
        assert torch.equal(x1, x2) and torch.equal(y1, y2)
    x, y = first[0]
    assert x.shape == (4, 16) and torch.equal(x[:, 1:], y[:, :-1])


def test_val_loader_fixed(tmp_path):
    _make_shards(tmp_path, "a", "val", 1, 2000, 3)
    spec = MixtureSpec(tmp_path, {"a": 1.0}, "val")
    v1 = list(ValLoader(spec, 16, 2, 160, device="cpu"))
    v2 = list(ValLoader(spec, 16, 2, 160, device="cpu"))
    assert len(v1) == 5 and all(torch.equal(a[0], b[0]) for a, b in zip(v1, v2))


def test_schedules():
    c = ScheduleConfig(type="cosine", total_tokens=1000, warmup_tokens=100, min_lr_ratio=0.1)
    assert lr_at(0, c, 1.0) == pytest.approx(0.01) and lr_at(99, c, 1.0) == pytest.approx(1.0)
    assert lr_at(100, c, 1.0) == pytest.approx(1.0) and lr_at(1000, c, 1.0) == pytest.approx(0.1)
    assert lr_at(550, c, 1.0) == pytest.approx(0.55, abs=1e-6)
    w = ScheduleConfig(type="wsd", total_tokens=1000, warmup_tokens=100, min_lr_ratio=0.0, decay_frac=0.2)
    assert lr_at(500, w, 1.0) == 1.0 and lr_at(900, w, 1.0) == pytest.approx(0.5) and lr_at(1000, w, 1.0) == pytest.approx(0.0)


def test_report_builds_from_metrics(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"run_name": "r", "config": {"schedule": {"total_tokens": 1000}, "milestone_tokens": 100}, "n_params": 1}))
    lg = MetricsLogger(run)
    lg.log("start", msg="go")
    for i in range(1, 6):
        lg.log("train", tokens=i * 100, update=i, loss=5 - i * 0.1, lr=1e-4, grad_norm=1.0, tok_s=1000.0, tok_s_ema=1000.0,
               step_ms=100.0, fwd_ms=30.0, bwd_ms=60.0, opt_ms=10.0, data_ms=1.0, vram_gib=3.0, elapsed_s=i, eta_s=5)
        time.sleep(0.001)
    lg.log("eval", tokens=500, update=5, val_loss=4.5, val_ppl=90.0, best=True, eval_s=1)
    lg.log("milestone", tokens=500, update=5, segment_s=5.0, elapsed_s=5.0, tok_s=1000.0, loss=4.6, val_loss=4.5)
    lg.close()
    h = build_report(run)
    assert "<html" in h and "4.5000" in h and "milestone" in h.lower() and "ETA" in h

"""Needle-in-a-haystack eval: haystack construction (CPU), the eval loop on the tiny model, and needle
tracking inside the trainer (CUDA)."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from slm.config import ModelConfig, load_config
from slm.data.tokenizer import SlmTokenizer, train_bpe
from slm.eval.long_context import DEFAULT_DEPTHS, FillerHaystack, RealHaystack, build_haystack, format_table, run_needle
from slm.model import Transformer


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    t = SlmTokenizer(train_bpe(["The secret number is 123456. Question: What is the secret number? Answer: " * 30, "the village market opened early " * 30], vocab_size=320))
    return t


def _fake_split(tmp_path: Path, tok: SlmTokenizer, n: int = 20000) -> Path:
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 200, size=n, dtype=np.uint16)
    ids[::50] = tok.eos_id  # document boundaries every 50 tokens
    ids[1::50] = tok.bos_id
    d = tmp_path / "src" / "val"
    d.mkdir(parents=True)
    ids.tofile(d / "shard_00000.bin")
    np.save(d / "shard_00000.idx.npy", np.arange(1, n, 50, dtype=np.int64))
    return d


def test_haystacks_hit_budget_and_drop_boundaries(tmp_path, tok):
    rng = random.Random(0)
    f = FillerHaystack(tok).tokens(500, rng)
    assert len(f) == 500
    real = RealHaystack(tok, _fake_split(tmp_path, tok))
    r = real.tokens(3000, rng)
    assert len(r) == 3000 and tok.bos_id not in r and tok.eos_id not in r
    for depth in (0.0, 0.5, 1.0):
        ids, pos = build_haystack(tok, 256, depth, "The secret number is 424242.", "What is the secret number?", rng, real)
        assert len(ids) == 256 - 16 and ids[0] == tok.bos_id
        needle = tok.encode(" The secret number is 424242. ")
        assert ids[pos : pos + len(needle)] == needle


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_run_needle_tiny_model(tmp_path, tok):
    cfg = load_config(ModelConfig, "configs/model/tiny.yaml")
    cfg.vocab_size = tok.vocab_size
    cfg.max_seq_len = 256
    model = Transformer(cfg).cuda().eval()
    real = RealHaystack(tok, _fake_split(tmp_path, tok))
    res = run_needle(model, tok, [128, 256, 512], depths=[0.0, 1.0], n=2, haystack=real)
    assert res["haystack"] == "real" and res["effective_context"] in (0, 128, 256)
    assert [r for r in res["results"] if r.get("skipped")][0]["length"] == 512
    assert set(res["summary"]) == {128, 256} and all(0 <= s["mean"] <= 1 for s in res["summary"].values())
    assert "effective context" in format_table(res)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_trainer_tracks_needle(tmp_path):
    from test_train import _setup

    from slm.train.pretrain import Trainer
    from slm.utils.logging import MetricsLogger

    cfg = _setup(tmp_path, "needle", 6)
    cfg.eval.needle_lengths = [64, 4096]  # 4096 exceeds tiny.yaml's RoPE table and must be skipped
    cfg.eval.needle_depths = [0.0, 1.0]
    cfg.eval.needle_n = 1
    cfg.eval.needle_source = "syn"
    Trainer(cfg).train()
    evals = [r for r in MetricsLogger.read(cfg.run_dir / "metrics.jsonl") if r["kind"] == "eval"]
    assert evals and all("needle_64" in e and "needle_min_64" in e and "needle_effective" in e and "needle_4096" not in e for e in evals)
    from slm.utils import metrics as M

    ser = M.series(MetricsLogger.read(cfg.run_dir / "metrics.jsonl"))
    assert ser["needle_keys"] == ["needle_64", "needle_min_64"] and len(ser["eval"]["needle_64"]) == len(evals)
    assert M.summary(MetricsLogger.read(cfg.run_dir / "metrics.jsonl"), {})["needle"]["64"] is not None
    assert "c_needle" in (cfg.run_dir / "report.html").read_text(encoding="utf-8")

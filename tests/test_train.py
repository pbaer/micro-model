"""End-to-end trainer tests on synthetic data (CUDA): checkpoint/resume equivalence and stop file."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from slm.data.tokenizer import SlmTokenizer, train_bpe
from slm.train.config import TrainConfig, from_dict
from slm.train.pretrain import Trainer
from slm.utils.logging import MetricsLogger

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _setup(tmp_path: Path, run_name: str, total_updates: int) -> TrainConfig:
    tok_dir = tmp_path / "tok"
    if not tok_dir.exists():
        SlmTokenizer(train_bpe(["hello world " * 20, "abc def ghi " * 20], vocab_size=300)).save(tok_dir)
        rng = np.random.default_rng(0)
        for split, n in (("train", 200_000), ("val", 5_000)):
            d = tmp_path / "tokd" / "syn" / split
            d.mkdir(parents=True)
            rng.integers(0, 364, size=n, dtype=np.uint16).tofile(d / "shard_00000.bin")
            np.save(d / "shard_00000.idx.npy", np.array([0], dtype=np.int64))
    seq, mb, accum = 64, 2, 2
    tpu = seq * mb * accum
    return from_dict(
        TrainConfig,
        {
            "run_name": run_name, "runs_root": str(tmp_path / "runs"), "tokenizer_dir": str(tok_dir),
            "model_file": "configs/model/tiny.yaml", "model": {"vocab_size": 364},
            "data": {"tokenized_root": str(tmp_path / "tokd"), "mixture": {"syn": 1.0}, "seq_len": seq, "val_tokens": 2048, "prefetch": 1},
            "optim": {"lr": 1e-3}, "schedule": {"total_tokens": tpu * total_updates, "warmup_tokens": tpu * 2},
            "batch": {"microbatch": mb, "tokens_per_update": tpu},
            "runtime": {"compile": False, "sdpa_backend": "auto", "log_every_updates": 1, "min_free_vram_gib": 0.5},
            "ckpt": {"every_minutes": 1e9}, "eval": {"every_tokens": tpu * 5, "gen_every_tokens": 10**12, "prompts": [], "report_every_minutes": 1e9},
            "milestone_tokens": tpu * 4,
        },
    )


def _params(t: Trainer) -> list[torch.Tensor]:
    return [p.detach().clone() for p in t.model.parameters()]


def test_resume_equivalence(tmp_path):
    torch.use_deterministic_algorithms(False)
    n = 12
    cfg_a = _setup(tmp_path, "cont", n)
    ta = Trainer(cfg_a)
    ta.train()
    pa = _params(ta)
    recs_a = [r for r in MetricsLogger.read(cfg_a.run_dir / "metrics.jsonl") if r["kind"] == "train"]

    cfg_b = _setup(tmp_path, "split", n)
    tb = Trainer(cfg_b)
    (cfg_b.run_dir / "STOP").touch()  # stops after the first update...
    tb.train()
    # ...then drive a few more updates by lowering total and resuming repeatedly
    tokens_first = tb.counters["tokens"]
    assert tokens_first == cfg_b.batch.tokens_per_update
    tb2 = Trainer(cfg_b)  # resumes from latest.pt
    assert tb2.counters["tokens"] == tokens_first
    tb2.train()
    pb = _params(tb2)
    recs_b = [r for r in MetricsLogger.read(cfg_b.run_dir / "metrics.jsonl") if r["kind"] == "train"]

    assert len(recs_a) == len(recs_b) == n
    for ra, rb in zip(recs_a, recs_b):
        assert ra["tokens"] == rb["tokens"]
        assert abs(ra["loss"] - rb["loss"]) < 2e-3, (ra["tokens"], ra["loss"], rb["loss"])
    for a, b in zip(pa, pb):
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-3)
    assert (cfg_b.run_dir / "report.html").exists()
    kinds = [r["kind"] for r in MetricsLogger.read(cfg_b.run_dir / "metrics.jsonl")]
    assert "resume" in kinds and "stop" in kinds and "finish" in kinds and "milestone" in kinds and "eval" in kinds
    meta = json.loads((cfg_b.run_dir / "run.json").read_text())
    assert meta["n_params"] > 0


def test_pretrain_continuation_without_init_loader_from_warns(tmp_path, capsys):
    """Continuing a pretraining run without continuing its data streams re-reads what the parent trained on; the
    trainer must say so (loudly, not fatally: a fresh mixture is a legitimate reason to start at token 0)."""
    from slm.train.pretrain import Trainer
    from slm.utils.logging import MetricsLogger

    parent = _setup(tmp_path, "parent", 4)
    Trainer(parent).train()
    child = _setup(tmp_path, "child", 4)
    child.init_from = str(parent.run_dir / "checkpoints" / "final.pt")
    Trainer(child)
    assert "init_loader_from is not" in capsys.readouterr().out
    warns = [r for r in MetricsLogger.read(child.run_dir / "metrics.jsonl") if r["kind"] == "warn"]
    assert warns and "re-read from the beginning" in warns[0]["msg"]
    # with it set, no warning and the streams continue
    child2 = _setup(tmp_path, "child2", 4)
    child2.init_from = str(parent.run_dir / "checkpoints" / "final.pt")
    child2.init_loader_from = str(parent.run_dir / "checkpoints" / "latest.pt")
    t = Trainer(child2)
    assert "init_loader_from is not" not in capsys.readouterr().out
    assert any(v["tokens"] > 0 for v in t.loader.consumed().values())

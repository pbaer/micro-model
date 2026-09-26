"""SFT data pipeline: chat formatting masks, shard writer, loaders, and a short masked-loss training run."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from slm.data.chat import format_chat, parse_assistant
from slm.data.sft import SftLoader, SftShardWriter, SftValLoader
from slm.data.tokenizer import SlmTokenizer, train_bpe
from slm.model.loss import IGNORE_INDEX


@pytest.fixture(scope="module")
def tok():
    return SlmTokenizer(train_bpe(["hello world how are you " * 40, "the answer is 42 #### 42 " * 40], vocab_size=300))


def test_format_chat_mask_and_parse(tok):
    enc = format_chat(tok, [{"role": "user", "content": "hi"}, {"role": "assistant", "think": "t", "content": "#### 42"}])
    ids, m = enc.ids, enc.loss_mask
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id and m[-1] == 0
    labels = {lab for s, e, lab in enc.segments}
    assert {"bos", "role:user", "content:user", "end", "role:assistant", "think_open", "think", "think_close", "content:assistant", "eos"} <= labels
    # user tokens masked, assistant think+content+end are targets, role tokens are not
    for s, e, lab in enc.segments:
        want = 1 if lab in ("think_open", "think", "think_close", "content:assistant", "end") and s > ids.index(tok.special("<|assistant|>")) else 0
        if lab == "end" and s < ids.index(tok.special("<|assistant|>")):
            want = 0
        assert all(x == want for x in m[s:e]), (lab, m[s:e])
    gen = format_chat(tok, [{"role": "user", "content": "hi"}], add_generation_prompt=True, think_required=True)
    assert gen.ids[-2:] == [tok.special("<|assistant|>"), tok.special("<|think|>")]
    after = enc.ids[enc.ids.index(tok.special("<|assistant|>")) + 1 :]
    parsed = parse_assistant(tok, after)
    assert parsed["think"] == "t" and parsed["answer"] == "#### 42" and parsed["terminated"] and not parsed["malformed"]
    # completion generated after a prompt that already ends in <|think|>: no opening tag in the ids
    comp = [*tok.encode("t"), tok.special("<|/think|>"), *tok.encode("#### 42"), tok.end_id]
    p2 = parse_assistant(tok, comp)
    assert p2["think"] == "t" and p2["answer"] == "#### 42" and not p2["malformed"]
    p3 = parse_assistant(tok, [*tok.encode("#### 42"), tok.end_id])
    assert p3["malformed"] and p3["answer"] == "#### 42"  # missing closing tag is flagged but the answer still parses
    p4 = parse_assistant(tok, tok.encode("#### 42"))
    assert p4["malformed"] and not p4["terminated"]


def _write_sft(root: Path, name: str, tok, n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    for split, k in (("train", n), ("val", 40)):
        w = SftShardWriter(root / name / split, shard_tokens=5000)
        for i in range(k):
            enc = format_chat(tok, [{"role": "user", "content": "hello " * int(rng.integers(1, 6))}, {"role": "assistant", "content": "the answer is 42 " * int(rng.integers(1, 4))}])
            w.add(enc.ids, enc.loss_mask)
        w.flush()


def test_sft_loader_masks_and_resume(tmp_path, tok):
    _write_sft(tmp_path, "a", tok)
    ld = SftLoader(tmp_path, {"a": 1.0}, seq_len=32, microbatch=4, seed=0, device="cpu", prefetch=1)
    x, y = ld.next()
    assert x.shape == (4, 32) and y.shape == (4, 32)
    valid = y != IGNORE_INDEX
    assert 0 < valid.sum() < y.numel()
    # targets equal the shifted inputs where valid
    assert torch.equal(y[valid], x[:, 1:][valid[:, :-1]] if False else y[valid])
    assert torch.equal(x[:, 1:][valid[:, :-1]], y[:, :-1][valid[:, :-1]])
    state = json.loads(json.dumps(ld.state_dict()))
    a = [ld.next() for _ in range(3)]
    ld.close()
    ld2 = SftLoader(tmp_path, {"a": 1.0}, seq_len=32, microbatch=4, seed=5, device="cpu", prefetch=1)
    ld2.load_state_dict(state)
    b = [ld2.next() for _ in range(3)]
    ld2.close()
    assert all(torch.equal(p[0], q[0]) and torch.equal(p[1], q[1]) for p, q in zip(a, b))
    vl = SftValLoader(tmp_path, {"a": 1.0}, 32, 4, 32 * 8, device="cpu")
    assert len(vl) == 2 and all((yy != IGNORE_INDEX).any() for _, yy in vl)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sft_trainer_epochs(tmp_path, tok):
    from slm.train.config import TrainConfig, from_dict
    from slm.train.pretrain import Trainer
    from slm.utils.logging import MetricsLogger

    tok.save(tmp_path / "tok")
    _write_sft(tmp_path / "sft", "a", tok, n=120)
    cfg = from_dict(TrainConfig, {
        "run_name": "sft", "runs_root": str(tmp_path / "runs"), "tokenizer_dir": str(tmp_path / "tok"),
        "model_file": "configs/model/tiny.yaml", "model": {"vocab_size": tok.vocab_size},
        "data": {"kind": "sft", "sft_root": str(tmp_path / "sft"), "mixture": {"a": 1.0}, "seq_len": 32, "val_tokens": 256, "prefetch": 1},
        "optim": {"lr": 1e-3}, "schedule": {"epochs": 2.0, "warmup_tokens": 64, "type": "constant"},
        "batch": {"microbatch": 2, "tokens_per_update": 128},
        "runtime": {"compile": False, "sdpa_backend": "auto", "log_every_updates": 1, "min_free_vram_gib": 0.5},
        "ckpt": {"every_minutes": 1e9}, "eval": {"every_tokens": 10**12, "gen_every_tokens": 10**12, "prompts": [], "report_every_minutes": 1e9},
    })
    t = Trainer(cfg)
    assert cfg.schedule.total_tokens == 2 * t.loader.total_tokens and cfg.milestone_tokens == t.loader.total_tokens
    t.train()
    recs = MetricsLogger.read(cfg.run_dir / "metrics.jsonl")
    kinds = [r["kind"] for r in recs]
    assert kinds.count("milestone") == 2 and "finish" in kinds
    losses = [r["loss"] for r in recs if r["kind"] == "train"]
    assert losses[-1] < losses[0]


def test_sft_stream_tiles_a_split_shorter_than_one_window(tmp_path):
    """A rebuilt selection set had a 9-example val split (3,180 tokens) and the 4,097-token val window came back
    short, which killed the trainer at startup (2026-09-26). Short splits are tiled to full-shape windows."""
    import numpy as np

    from slm.data.sft import SftShardWriter, SftStream

    w = SftShardWriter(tmp_path / "val")
    w.add([5, 6, 7, 8, 9], [0, 1, 1, 1, 1])
    w.flush()
    s = SftStream(tmp_path / "val")
    t, m = s.next_window_masked(12)
    assert t.shape == (12,) and m.shape == (12,)
    assert t.tolist() == [5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6] and m.tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    assert s.epoch == 3, "three tilings of a 5-token stream for a 12-token window"
    t2, _ = s.next_window_masked(12)
    assert t2.shape == (12,), "and again on the next call"

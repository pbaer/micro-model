"""End-to-end GRPO trainer smoke test on CUDA with the tiny model: two steps, eval, checkpoint, rollouts."""

import json

import pytest
import torch

from slm.config import ModelConfig, load_config, to_dict
from slm.data.tokenizer import SlmTokenizer, train_bpe
from slm.model import Transformer
from slm.train.rl import RlConfig, RlTrainer
from slm.utils.checkpoint import save_snapshot
from slm.utils.logging import MetricsLogger

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_rl_trainer_two_steps(tmp_path):
    tok = SlmTokenizer(train_bpe(["What is 3 + 4? Think step by step #### 7 " * 60], vocab_size=300))
    tok.save(tmp_path / "tok")
    cfg_m = load_config(ModelConfig, "configs/model/tiny.yaml")
    cfg_m.vocab_size = tok.vocab_size
    save_snapshot(tmp_path / "init.pt", Transformer(cfg_m), to_dict(cfg_m), {"tokens": 0, "tokenizer_sha256": tok.sha256})
    cfg = RlConfig(run_name="rl", runs_root=str(tmp_path / "runs"), tokenizer_dir=str(tmp_path / "tok"), model_file="configs/model/tiny.yaml",
                   model={"vocab_size": tok.vocab_size}, init_from=str(tmp_path / "init.pt"), tasks=["arith1"], n_train_prompts=40, n_heldout_prompts=8,
                   group_size=4, prompts_per_step=2, max_new_tokens=12, total_steps=2, eval_every_steps=1, eval_max_new_tokens=12,
                   ckpt_every_minutes=1e9, report_every_minutes=1e9, microbatch=4)
    t = RlTrainer(cfg)
    t.train()
    recs = MetricsLogger.read(cfg.run_dir / "metrics.jsonl")
    kinds = [r["kind"] for r in recs]
    assert kinds.count("train") == 2 and kinds.count("eval") == 3 and "finish" in kinds  # pre-RL eval + one per step
    tr = [r for r in recs if r["kind"] == "train"][0]
    for k in ("reward_mean", "success_rate", "kl", "entropy", "clip_frac", "grad_norm", "len_mean", "malformed_rate"):
        assert k in tr, k
    assert (cfg.run_dir / "checkpoints" / "final.pt").exists() and (cfg.run_dir / "checkpoints" / "latest.pt").exists()
    idx = json.loads((cfg.run_dir / "checkpoints" / "index.json").read_text())
    assert "final.pt" in idx and "step_00002.pt" in idx
    roll = [json.loads(line) for line in (cfg.run_dir / "rollouts" / "step_00001.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(roll) == 8 and all("old_logprobs" in r and "ref_logprobs" in r and r["reward"] in (0.0, 1.0) for r in roll)
    # resume: a second trainer picks up from latest.pt at step 2 and stops immediately
    cfg.total_steps = 2
    t2 = RlTrainer(cfg)
    assert t2.step == 2


def test_rl_collapse_guard_stops_and_keeps_best(tmp_path, monkeypatch):
    """Entropy above entropy_stop for a full `guard_window` makes the guard fire: warn event, an eval, a stop
    with a checkpoint, and best.pt (the pre-RL policy) plus its index entry exist. (The tiny random model
    yields no reward signal, so optimize() would be skipped; its entropy is forced here.)

    guard_window=2 keeps the test short; the shipped configs use 10. See
    test_rl_collapse_guard_ignores_a_single_spike for the half of the behaviour that matters more."""
    orig = RlTrainer.optimize

    def hot(self, rollouts):
        out = orig(self, rollouts)
        out["entropy"] = 10.0
        return out

    monkeypatch.setattr(RlTrainer, "optimize", hot)
    tok = SlmTokenizer(train_bpe(["What is 3 + 4? Think step by step #### 7 " * 60], vocab_size=300))
    tok.save(tmp_path / "tok")
    cfg_m = load_config(ModelConfig, "configs/model/tiny.yaml")
    cfg_m.vocab_size = tok.vocab_size
    save_snapshot(tmp_path / "init.pt", Transformer(cfg_m), to_dict(cfg_m), {"tokens": 0, "tokenizer_sha256": tok.sha256})
    cfg = RlConfig(run_name="guard", runs_root=str(tmp_path / "runs"), tokenizer_dir=str(tmp_path / "tok"), model_file="configs/model/tiny.yaml",
                   model={"vocab_size": tok.vocab_size}, init_from=str(tmp_path / "init.pt"), tasks=["arith1"], n_train_prompts=40, n_heldout_prompts=8,
                   group_size=4, prompts_per_step=2, max_new_tokens=12, total_steps=5, eval_every_steps=100, eval_max_new_tokens=12,
                   ckpt_every_minutes=1e9, report_every_minutes=1e9, microbatch=4, entropy_stop=2.5, guard_window=2)
    t = RlTrainer(cfg)
    t.train()
    recs = MetricsLogger.read(cfg.run_dir / "metrics.jsonl")
    kinds = [r["kind"] for r in recs]
    assert kinds.count("train") == 2 and "warn" in kinds and "stop" in kinds and "finish" not in kinds
    assert (cfg.run_dir / "checkpoints" / "best.pt").exists() and (cfg.run_dir / "checkpoints" / "latest.pt").exists()
    idx = json.loads((cfg.run_dir / "checkpoints" / "index.json").read_text())
    assert idx["best.pt"]["kind"] == "best" and "heldout_acc" in idx["best.pt"] and "step_00002.pt" in idx


def test_rl_collapse_guard_ignores_a_single_spike(tmp_path, monkeypatch):
    """One outlier step must NOT stop the run.

    M9 stage C try 1 was killed at step 69 by a single step touching entropy 2.54 against a 2.5 limit. Over
    its 69 steps entropy ran 0.00-2.54 with stdev 0.63 and no trend (corr with step -0.13) while KL to the
    reference sat at 0.0003 against a 0.15 budget -- a policy that has not moved cannot have collapsed. A step
    is only prompts_per_step x group_size rollouts, so per-step entropy is extremely noisy, and the guard now
    tests the window mean. The numbers below are that incident's: a 2.54 spike against a 2.5 limit, with the
    other steps at the run's 1.32 average, which a window mean absorbs and a per-step test does not."""
    orig = RlTrainer.optimize
    calls = {"n": 0}

    def spike_once(self, rollouts):
        out = orig(self, rollouts)
        calls["n"] += 1
        # the real numbers: one step at 2.54 against a 2.5 limit, the rest around the run's 1.32 mean
        out["entropy"] = 2.54 if calls["n"] == 1 else 1.3
        return out

    monkeypatch.setattr(RlTrainer, "optimize", spike_once)
    tok = SlmTokenizer(train_bpe(["What is 3 + 4? Think step by step #### 7 " * 60], vocab_size=300))
    tok.save(tmp_path / "tok")
    cfg_m = load_config(ModelConfig, "configs/model/tiny.yaml")
    cfg_m.vocab_size = tok.vocab_size
    save_snapshot(tmp_path / "init.pt", Transformer(cfg_m), to_dict(cfg_m), {"tokens": 0, "tokenizer_sha256": tok.sha256})
    cfg = RlConfig(run_name="spike", runs_root=str(tmp_path / "runs"), tokenizer_dir=str(tmp_path / "tok"), model_file="configs/model/tiny.yaml",
                   model={"vocab_size": tok.vocab_size}, init_from=str(tmp_path / "init.pt"), tasks=["arith1"], n_train_prompts=40, n_heldout_prompts=8,
                   group_size=4, prompts_per_step=2, max_new_tokens=12, total_steps=4, eval_every_steps=100, eval_max_new_tokens=12,
                   ckpt_every_minutes=1e9, report_every_minutes=1e9, microbatch=4, entropy_stop=2.5, guard_window=3)
    RlTrainer(cfg).train()
    kinds = [r["kind"] for r in MetricsLogger.read(cfg.run_dir / "metrics.jsonl")]
    assert "stop" not in kinds, "a single spike stopped the run"
    assert kinds.count("train") == 4 and "finish" in kinds

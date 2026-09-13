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

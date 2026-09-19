"""RL (GRPO) configuration, kept in its own torch-free module.

`slm.train.rl` imports torch at module level, so anything that only wants to *read* an RL yaml
(the portal's Data page, the Architecture page) would drag CUDA into the process. This module
holds the dataclass and the loader; `rl.py` re-exports both, so its public names are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from slm.config import ModelConfig, apply_overrides, from_dict, load_yaml


@dataclass
class RlConfig:
    run_name: str = "rl_debug"
    runs_root: str = "runs"
    tokenizer_dir: str = r"C:\slm-data\tokenizer\v1"
    model_file: str = "configs/model/base_149m.yaml"
    model: dict = field(default_factory=dict)
    init_from: str = ""  # reasoning-SFT checkpoint (policy and reference start here)
    gpu_sample_s: float = 2.0  # nvidia-smi telemetry period (s); temperature/power go into every train record
    tools: bool = False  # Python tool: rollouts pause at <|/python_call|>, results are inserted and excluded from the objective
    max_tool_calls: int = 8
    # Collapse guards (0 = off): stop the run (checkpoint + report) when the rollout entropy or the KL to the
    # reference exceeds these, which is what a diverging policy looks like before it produces garbage.
    entropy_stop: float = 0.0
    kl_stop: float = 0.0
    gpu_warn_temp_c: float = 80.0
    # tasks / curriculum
    tasks: list[str] = field(default_factory=lambda: ["arith1", "arith2"])
    n_train_prompts: int = 4000
    n_heldout_prompts: int = 200
    # generation
    group_size: int = 8
    prompts_per_step: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 0
    think_required: bool = True
    reward_scheme: str = "binary"
    # objective
    lr: float = 2e-6
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    clip_eps: float = 0.2
    use_ratio: bool = True
    kl_coef: float = 0.01
    kl_kind: str = "k3"
    normalize_std: bool = True
    ppo_epochs: int = 1
    microbatch: int = 8  # rollouts per forward
    total_steps: int = 300
    eval_every_steps: int = 25
    eval_max_new_tokens: int = 256
    ckpt_every_minutes: float = 15.0
    report_every_minutes: float = 5.0
    seed: int = 0
    sdpa_backend: str = "auto"
    notes: str = ""

    @property
    def run_dir(self) -> Path:
        return Path(self.runs_root) / self.run_name

    def model_config(self) -> ModelConfig:
        d = load_yaml(self.model_file)
        d.update(self.model or {})
        return from_dict(ModelConfig, d)


def load_rl_config(path: str | None, overrides: list[str] | None = None) -> RlConfig:
    d = load_yaml(path) if path else {}
    apply_overrides(d, overrides or [])
    cfg = from_dict(RlConfig, d)
    if isinstance(cfg.betas, list):
        cfg.betas = tuple(cfg.betas)
    return cfg


# Rule text per reward scheme, shown next to an RL recipe (mirrors slm.rl.rewards.reward_from_verdict).
REWARD_RULES = {
    "binary": "1.0 if the parsed final answer is correct, else 0.0",
    "tool": "1.0 correct AND the number came out of a Python call · 0.5 correct without one · 0.0 otherwise (malformed = 0.0)",
    "signed": "+1.0 correct, -1.0 wrong",
    "shaped": "1.0 correct · 0.0 wrong but parsable · -0.5 malformed/unparsable",
}

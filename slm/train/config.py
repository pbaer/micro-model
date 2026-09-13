"""Training configuration for pretraining (SFT/RL extend these)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slm.config import ModelConfig, apply_overrides, from_dict, load_yaml


@dataclass
class DataConfig:
    tokenized_root: str = r"C:\slm-data\tokenized\v1"
    mixture: dict[str, float] = field(default_factory=lambda: {"fineweb-edu": 1.0})
    seq_len: int = 2048
    val_tokens: int = 2_000_000
    prefetch: int = 4


@dataclass
class OptimConfig:
    lr: float = 5e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    fused: bool = True
    # Apply weight decay only to >=2-D tensors (matrices, embeddings); never to norm gains.
    decay_only_matrices: bool = True


@dataclass
class ScheduleConfig:
    type: str = "cosine"  # cosine | wsd (warmup-stable-decay) | constant
    total_tokens: int = 1_000_000_000
    warmup_tokens: int = 10_000_000
    min_lr_ratio: float = 0.1
    decay_frac: float = 0.2  # wsd: final fraction of tokens spent decaying linearly to min_lr


@dataclass
class BatchConfig:
    microbatch: int = 8
    tokens_per_update: int = 262_144


@dataclass
class RuntimeConfig:
    compile: bool = True
    sdpa_backend: str = "cudnn"  # cudnn | efficient | flash | auto
    seed: int = 0
    log_every_updates: int = 10
    # Assume no other GPU use; abort if less than this much VRAM is free at start (GiB).
    min_free_vram_gib: float = 12.0


@dataclass
class CheckpointConfig:
    every_minutes: float = 15.0
    keep_prev_latest: bool = True
    snapshot_at_milestones: bool = True  # bf16 model-only snapshot every milestone


@dataclass
class EvalConfig:
    every_tokens: int = 50_000_000
    gen_every_tokens: int = 100_000_000
    gen_max_new_tokens: int = 96
    gen_temperature: float = 0.8
    gen_top_p: float = 0.95
    prompts: list[str] = field(default_factory=list)
    report_every_minutes: float = 10.0


@dataclass
class TrainConfig:
    run_name: str = "debug"
    runs_root: str = "runs"
    tokenizer_dir: str = r"C:\slm-data\tokenizer\v1"
    model_file: str = "configs/model/base_149m.yaml"
    model: dict[str, Any] = field(default_factory=dict)  # overrides on top of model_file
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    ckpt: CheckpointConfig = field(default_factory=CheckpointConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    milestone_tokens: int = 100_000_000
    notes: str = ""

    @property
    def run_dir(self) -> Path:
        return Path(self.runs_root) / self.run_name

    @property
    def grad_accum(self) -> int:
        per_micro = self.batch.microbatch * self.data.seq_len
        assert self.batch.tokens_per_update % per_micro == 0, (
            f"tokens_per_update {self.batch.tokens_per_update} must be a multiple of microbatch*seq_len {per_micro}"
        )
        return self.batch.tokens_per_update // per_micro

    def model_config(self) -> ModelConfig:
        d = load_yaml(self.model_file) if self.model_file else {}
        d.update(self.model or {})
        return from_dict(ModelConfig, d)


def load_train_config(path: str | None, overrides: list[str] | None = None) -> TrainConfig:
    d = load_yaml(path) if path else {}
    apply_overrides(d, overrides or [])
    cfg = from_dict(TrainConfig, d)
    if isinstance(cfg.optim.betas, list):
        cfg.optim.betas = tuple(cfg.optim.betas)
    return cfg

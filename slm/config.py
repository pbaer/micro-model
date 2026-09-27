"""Config primitives: nested dataclasses loaded from YAML with `key.sub=value` overrides.

Every experiment is driven by an explicit config. Numeric hyperparameters live here, never
hard-coded in training code. Sub-configs for training/data/RL are added by their own modules
and composed into a top-level config per stage.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class RopeScaling:
    """RoPE context-extension settings. `type` in {"none", "linear", "ntk", "yarn"}."""

    type: str = "none"
    factor: float = 1.0
    # Original (pre-extension) trained context, needed by ntk/yarn.
    original_max_seq_len: int = 0
    # YaRN ramp parameters (Peng et al. 2023).
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    # YaRN attention temperature: cos/sin are scaled by mscale = 0.1*ln(factor)+1 when > 0.
    mscale_enabled: bool = True


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    n_layers: int = 18
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    head_dim: int = 64
    d_ff: int = 2304
    max_seq_len: int = 8192
    rope_theta: float = 100_000.0
    rope_scaling: RopeScaling = field(default_factory=RopeScaling)
    qk_norm: bool = True
    tie_embeddings: bool = True
    norm_eps: float = 1e-6
    init_std: float = 0.02
    # Recompute each block in backward to save activation memory (costs ~30% compute).
    grad_checkpointing: bool = False
    # Tokens per chunk for the chunked cross-entropy (0 = materialize full logits).
    loss_chunk_size: int = 0

    def __post_init__(self) -> None:
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        assert self.d_ff % 2 == 0
        if isinstance(self.rope_scaling, dict):
            self.rope_scaling = RopeScaling(**self.rope_scaling)

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim


# ----------------------------------------------------------------------------------------------
# Generic dataclass <-> dict / YAML plumbing
# ----------------------------------------------------------------------------------------------


def to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def from_dict(cls: type[T], d: dict[str, Any]) -> T:
    """Build a (nested) dataclass from a dict, erroring on unknown keys."""
    if d is None:
        d = {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(known)
    if unknown:
        raise KeyError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in d:
            continue
        v = d[name]
        ftype = _resolve_type(f)
        if is_dataclass(ftype) and isinstance(v, dict):
            v = from_dict(ftype, v)
        kwargs[name] = v
    return cls(**kwargs)


def _resolve_type(f: dataclasses.Field) -> Any:
    t = f.type
    if isinstance(t, str):  # from __future__ annotations
        import slm.config as me

        return getattr(me, t, None) or _lookup_global(t)
    return t


def _lookup_global(name: str) -> Any:
    import builtins
    import sys

    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        try:
            obj = getattr(mod, name, None)
        except Exception:  # noqa: BLE001 -- a lazy module (transformers) imports on attribute access and can fail (no torchvision)
            continue
        if is_dataclass(obj):
            return obj
    return getattr(builtins, name, None)


def _parse_scalar(s: str) -> Any:
    """Parse a CLI override value with YAML rules (so 1e-4, true, null, [1,2] all work)."""
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def apply_overrides(d: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `a.b.c=value` overrides to a nested dict in place."""
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must look like key=value, got {ov!r}")
        key, val = ov.split("=", 1)
        parts = key.split(".")
        cur = d
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = _parse_scalar(val)
    return d


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(cls: type[T], path: str | Path | None, overrides: list[str] | None = None) -> T:
    d = load_yaml(path) if path else {}
    d = apply_overrides(d, overrides or [])
    return from_dict(cls, d)


def dump_yaml(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(obj), f, sort_keys=False)

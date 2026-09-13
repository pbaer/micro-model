"""FLOP accounting, MFU, and CUDA timing helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from slm.config import ModelConfig

# RTX 4080 SUPER: 80 SMs @ ~2.55 GHz. Dense BF16 tensor throughput with FP32 accumulate on
# GeForce Ada is half the FP16-accumulate rate: ~104.5 TFLOPS (4090 = 165.2 at 128 SMs).
# Override per GPU via `peak_tflops` if benchmarking elsewhere. `measured_gemm_tflops` gives
# the practically reachable ceiling on this machine (see bench_throughput --gemm).
PEAK_BF16_TFLOPS = {
    "NVIDIA GeForce RTX 4080 SUPER": 104.5,
    "NVIDIA GeForce RTX 4090": 165.2,
}


def peak_tflops(device_name: str | None = None) -> float:
    name = device_name or (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    return PEAK_BF16_TFLOPS.get(name, 100.0)


def flops_per_token(cfg: ModelConfig, seq_len: int, n_nonembed: int, causal: bool = False) -> float:
    """Training FLOPs per token (fwd + bwd = 3x fwd).

    PaLM-style: 6*N_nonembed + 6*V*d (LM head) + 12*L*d_attn*T (attention scores/values).
    `causal=True` halves the attention term to approximate what a fused causal kernel executes.
    """
    lm_head = 6 * cfg.vocab_size * cfg.d_model
    attn = 12 * cfg.n_layers * cfg.q_dim * seq_len
    if causal:
        attn /= 2
    return 6 * n_nonembed + lm_head + attn


def mfu(tokens_per_sec: float, cfg: ModelConfig, seq_len: int, n_nonembed: int, causal: bool = False) -> float:
    return tokens_per_sec * flops_per_token(cfg, seq_len, n_nonembed, causal) / (peak_tflops() * 1e12)


@dataclass
class StepTimes:
    data_ms: float = 0.0
    fwd_ms: float = 0.0
    bwd_ms: float = 0.0
    opt_ms: float = 0.0
    total_ms: float = 0.0


class CudaTimer:
    """Records CUDA events around named phases; call `elapsed()` after a sync."""

    def __init__(self) -> None:
        self.events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self._wall: dict[str, float] = {}

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.events[name] = (e, None)
        self._wall[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        if torch.cuda.is_available():
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.events[name] = (self.events[name][0], e)
        self._wall[name] = time.perf_counter() - self._wall[name]

    def elapsed_ms(self, name: str) -> float:
        if torch.cuda.is_available() and name in self.events and self.events[name][1] is not None:
            s, e = self.events[name]
            return s.elapsed_time(e)
        return self._wall.get(name, 0.0) * 1000


def gemm_tflops(n: int = 8192, iters: int = 20) -> float:
    """Measured bf16 GEMM throughput: the practical ceiling for MFU on this GPU."""
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    return 2 * n**3 * iters / dt / 1e12

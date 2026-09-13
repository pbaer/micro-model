"""SDPA backend selection and verification.

On Windows builds of PyTorch the flash kernel is not compiled in; cuDNN attention is the fast
fused backend there and memory-efficient attention is the fallback. The math backend is ~20x
slower and materializes the full attention matrix, so we never allow it silently for training.
"""

from __future__ import annotations

import contextlib
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}


def probe_backends(seq_len: int = 2048, n_heads: int = 12, n_kv_heads: int = 4, head_dim: int = 64) -> dict[str, dict]:
    """Try each backend on a causal GQA bf16 problem; report ok/error and fwd+bwd ms."""
    if not torch.cuda.is_available():
        return {}
    res: dict[str, dict] = {}
    dev = "cuda"
    q = torch.randn(1, n_heads, seq_len, head_dim, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, n_kv_heads, seq_len, head_dim, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    for name, be in BACKENDS.items():
        try:
            with sdpa_kernel(be):
                for _ in range(2):
                    F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True).sum().backward()
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(5):
                    F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True).sum().backward()
                torch.cuda.synchronize()
            res[name] = {"ok": True, "ms": (time.perf_counter() - t) / 5 * 1000}
        except Exception as e:  # noqa: BLE001
            res[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:100]}"}
    return res


def best_backend(exclude_math: bool = True) -> str:
    """Fastest working backend by a quick probe (never 'math' unless nothing else works)."""
    res = probe_backends()
    ok = {k: v["ms"] for k, v in res.items() if v.get("ok") and (k != "math" or not exclude_math)}
    if not ok:
        if exclude_math and res.get("math", {}).get("ok"):
            raise RuntimeError("Only the math SDPA backend works; refusing to train on it")
        raise RuntimeError(f"No SDPA backend works: {res}")
    return min(ok, key=ok.get)


@contextlib.contextmanager
def sdpa_context(backend: str | None):
    """Context manager restricting SDPA to one backend ('auto'/None = PyTorch default)."""
    if backend in (None, "auto"):
        yield
        return
    with sdpa_kernel(BACKENDS[backend]):
        yield

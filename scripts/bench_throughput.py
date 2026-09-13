"""M0 throughput benchmark: seq len x microbatch x compile, with fwd/bwd/opt breakdown.

    python scripts/bench_throughput.py --config configs/model/base_149m.yaml --seq 2048 4096 8192 \
        --compile 0 1 --out artifacts/bench/base_149m.json

Reports tokens/sec, MFU (PaLM-style and causal-adjusted), peak VRAM, and per-phase timings.
Microbatch search doubles until OOM (or --max-mb) and keeps every successful point.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slm.config import ModelConfig, load_config  # noqa: E402
from slm.model import Transformer  # noqa: E402
from slm.utils.profiling import flops_per_token, gemm_tflops, peak_tflops  # noqa: E402
from slm.utils.sdpa import probe_backends, sdpa_context  # noqa: E402


def bench_point(cfg: ModelConfig, seq: int, mb: int, compile_: bool, steps: int, backend: str, chunk: int) -> dict:
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg.loss_chunk_size = chunk
    model = Transformer(cfg).cuda()
    n_ne = model.num_params(non_embedding=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    fwd = torch.compile(model) if compile_ else model
    x = torch.randint(0, cfg.vocab_size, (mb, seq), device="cuda")
    y = torch.randint(0, cfg.vocab_size, (mb, seq), device="cuda")
    ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731

    def step():
        e0, e1, e2, e3 = ev(), ev(), ev(), ev()
        e0.record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, n = fwd(x, y)
        e1.record()
        (loss / n).backward()
        e2.record()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        e3.record()
        return e0, e1, e2, e3

    t_compile = time.perf_counter()
    with sdpa_context(backend):
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        t_compile = time.perf_counter() - t_compile
        evs = []
        t0 = time.perf_counter()
        for _ in range(steps):
            evs.append(step())
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    f = sum(a.elapsed_time(b) for a, b, _, _ in evs) / steps
    b = sum(b.elapsed_time(c) for _, b, c, _ in evs) / steps
    o = sum(c.elapsed_time(d) for _, _, c, d in evs) / steps
    tps = mb * seq * steps / wall
    peak = peak_tflops()
    res = {
        "seq": seq, "microbatch": mb, "compile": compile_, "backend": backend, "loss_chunk": chunk,
        "tokens_per_sec": tps,
        "step_ms": wall / steps * 1000, "fwd_ms": f, "bwd_ms": b, "opt_ms": o,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "mfu_palm": tps * flops_per_token(cfg, seq, n_ne) / (peak * 1e12),
        "mfu_causal": tps * flops_per_token(cfg, seq, n_ne, causal=True) / (peak * 1e12),
        "warmup_s": t_compile,
    }
    del model, opt, fwd, x, y
    torch.cuda.empty_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model/base_149m.yaml")
    ap.add_argument("--seq", type=int, nargs="+", default=[2048, 4096, 8192])
    ap.add_argument("--compile", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--mb", type=int, nargs="*", default=None, help="explicit microbatches (default: double until OOM)")
    ap.add_argument("--max-mb", type=int, default=64)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--chunk", type=int, default=0, help="loss chunk size in tokens (0 = full logits)")
    ap.add_argument("--gemm", action="store_true", help="also measure raw bf16 GEMM TFLOPS")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(ModelConfig, args.config)
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  peak {peak_tflops():.1f} TFLOPS (spec)")
    if args.gemm:
        print(f"measured bf16 GEMM: {gemm_tflops():.1f} TFLOPS")
    print("SDPA backends:", {k: (f"{v['ms']:.2f}ms" if v["ok"] else "FAIL") for k, v in probe_backends().items()})
    results = []
    hdr = f"{'seq':>5} {'mb':>3} {'cmp':>3} {'tok/s':>8} {'step ms':>8} {'fwd':>6} {'bwd':>6} {'opt':>5} {'VRAM':>6} {'MFU':>5} {'MFUc':>5}"
    print(hdr)
    for seq in args.seq:
        for c in args.compile:
            mbs = args.mb or [1 << i for i in range(0, 10) if (1 << i) <= args.max_mb]
            for mb in mbs:
                if mb * seq > 1 << 20:
                    break
                try:
                    r = bench_point(cfg, seq, mb, bool(c), args.steps, args.backend, args.chunk)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"{seq:5d} {mb:3d} {c:3d}   OOM")
                    break
                if r["peak_vram_gib"] > 14.5:  # WDDM spills to host memory instead of raising OOM: treat as failure
                    print(f"{seq:5d} {mb:3d} {c:3d}   SPILL (peak {r['peak_vram_gib']:.1f} GiB > VRAM, {r['tokens_per_sec']:.0f} tok/s)")
                    break
                results.append(r)
                print(
                    f"{seq:5d} {mb:3d} {c:3d} {r['tokens_per_sec']:8.0f} {r['step_ms']:8.1f} {r['fwd_ms']:6.1f} "
                    f"{r['bwd_ms']:6.1f} {r['opt_ms']:5.1f} {r['peak_vram_gib']:6.2f} {r['mfu_palm']:5.1%} {r['mfu_causal']:5.1%}"
                )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"config": args.config, "gpu": torch.cuda.get_device_name(0), "results": results}, f, indent=1)


if __name__ == "__main__":
    main()

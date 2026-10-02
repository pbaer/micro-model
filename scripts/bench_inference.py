"""Inference throughput of a checkpoint on this GPU: prefill (input tokens/s) and decode (output tokens/s) measured
separately, for several prompt lengths and batch sizes, each configuration repeated N times.

    .venv/Scripts/python.exe -u scripts/bench_inference.py --checkpoint runs/m10_rl_336m/checkpoints/final.pt \
        --batches 1,16,32,64 --prompt-lens 32,256,1024 --ctx 4096 --repeats 10 --out artifacts/bench/infer_m10_rl.json

Each run decodes up to the full context window (ctx - prompt length new tokens) with the model's own KV-cache
`generate` path under the decode SDPA backends, greedy, with no stop token so every row produces the same number
of tokens. Prompts are real text (a tokenized val shard), one window per row. Timing uses CUDA events around the
prefill forward and around the decode loop; the first run of each configuration is a warm-up and not counted.
Peak allocated and reserved VRAM are recorded per configuration -- on Windows a reservation over ~14.5 GiB spills
into host memory and shows up as a collapse in output tokens/s rather than an error (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=r"C:\slm-data\tokenizer\v1")
    ap.add_argument("--shard", default=r"C:\slm-data\tokenized\v1\fineweb-edu-b\val")
    ap.add_argument("--batches", default="1,16,32,64")
    ap.add_argument("--prompt-lens", default="32,256,1024")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import numpy as np
    import torch

    from slm.eval.quality import load_model
    from slm.model.attention import KVCache
    from slm.model.transformer import sample_next
    from slm.utils.sdpa import sdpa_context

    model, meta = load_model(Path(a.checkpoint), "cuda")
    model.cfg.max_seq_len = max(model.cfg.max_seq_len, a.ctx)
    shard = sorted(Path(a.shard).glob("shard_*.bin"))[0]
    mm = np.memmap(shard, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    batches = [int(x) for x in a.batches.split(",")]
    plens = [int(x) for x in a.prompt_lens.split(",")]
    results = []
    dev = torch.device("cuda")

    def run_once(B: int, L: int) -> tuple[float, float, int]:
        new = a.ctx - L
        starts = rng.integers(0, len(mm) - L - 1, size=B)
        idx = torch.tensor(np.stack([np.asarray(mm[s : s + L]).astype(np.int64) for s in starts]), device=dev)
        cache = KVCache(model.cfg, B, a.ctx, dev, torch.bfloat16)
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), sdpa_context("decode"):
            e0.record()
            logits = model(idx, cache=cache, last_only=True)[:, -1, :].float()
            e1.record()
            cur = sample_next(logits, 1.0, 1.0, 1, None)[:, None]
            for _ in range(new - 1):
                logits = model(cur, cache=cache, last_only=True)[:, -1, :].float()
                cur = sample_next(logits, 1.0, 1.0, 1, None)[:, None]
            e2.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / 1000, e1.elapsed_time(e2) / 1000, new

    for L in plens:
        for B in batches:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                run_once(B, L)  # warm-up (cuDNN plans, allocator)
                pre, dec = [], []
                t0 = time.time()
                for _ in range(a.repeats):
                    tp, td, new = run_once(B, L)
                    pre.append(B * L / tp)
                    dec.append(B * new / td)
                peak = torch.cuda.max_memory_allocated() / 2**30
                reserved = torch.cuda.memory_reserved() / 2**30
                row = {"batch": B, "prompt_len": L, "new_tokens": new, "repeats": a.repeats,
                       "input_tok_s": {"mean": statistics.fmean(pre), "stdev": statistics.pstdev(pre), "min": min(pre), "max": max(pre)},
                       "output_tok_s": {"mean": statistics.fmean(dec), "stdev": statistics.pstdev(dec), "min": min(dec), "max": max(dec)},
                       "output_tok_s_per_row": statistics.fmean(dec) / B, "peak_alloc_gib": round(peak, 2), "reserved_gib": round(reserved, 2),
                       "seconds_per_run": round((time.time() - t0) / a.repeats, 2)}
                print(f"B={B:3d} L={L:5d} new={new:5d}: input {row['input_tok_s']['mean']:9.0f} tok/s (+-{row['input_tok_s']['stdev']:.0f})  "
                      f"output {row['output_tok_s']['mean']:7.0f} tok/s (+-{row['output_tok_s']['stdev']:.0f}, {row['output_tok_s_per_row']:.1f}/row)  "
                      f"peak {peak:.2f} GiB reserved {reserved:.2f} GiB  {row['seconds_per_run']:.1f} s/run", flush=True)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                row = {"batch": B, "prompt_len": L, "error": "OOM"}
                print(f"B={B:3d} L={L:5d}: OOM", flush=True)
            results.append(row)
    out = {"checkpoint": a.checkpoint, "ctx": a.ctx, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "results": results}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

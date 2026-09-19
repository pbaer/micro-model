# Project conventions

From-scratch small language model research stack (decoder-only Transformer; first base 149M, second base 336M
`configs/model/base_336m.yaml`, single RTX 4080 SUPER, native Windows). Package name is `slm`; the project's public name is a
placeholder and may change, so don't bake it into code.

## Documentation (keep current)
- `README.md` = stable high-level description (approach, structure, pointers); no run statistics in it.
- `docs/design.md` internals · `docs/runbook.md` operations · `docs/results.md` numbers (update when a run
  or eval finishes) · `docs/log.md` dated incidents/decisions · `docs/roadmap.md` remaining work ·
  `docs/command_center.md` the web UI.

## Environment
- `.venv` (uv, Python 3.13) with torch 2.14.0+cu130 and triton-windows 3.8.0. Run everything with
  `.venv/Scripts/python.exe`. Reinstall torch only from the cu130 index.
- SDPA flash backend is NOT available on Windows; use cuDNN (fast) or efficient (fallback).
  Never train on the math backend.
- torch.compile works (Triton via triton-windows).

## Layout
- Code, configs, checkpoints (`runs/<run>/`), reports, and benchmark artifacts live in this repo dir.
- Training data (raw downloads and tokenized shards) lives on `C:\slm-data` (faster, more space).
  Both roots are config values, never hard-coded.

## Scope decisions (from Peter)
- Data: English prose and Python only (bash dropped 2026-09-19: Python is the only code the model should write; the
  first base saw 1.5% shell and that knowledge is allowed to fade). Filter out other languages.
- No teacher models for our own data generation. Public datasets that were model-written (SmolTalk, MetaMathQA) are
  allowed, and chat/tool conversations may be mixed into pretraining (decided 2026-09-16).
- Own 32K BPE tokenizer with a reserved block of 64 special tokens (chat roles, think, tools).
  Special tokens are never produced from raw text; only the chat formatter inserts them.
- Chat format: `<|bos|><|user|>...<|end|><|assistant|><|think|>...<|/think|>answer<|end|><|eos|>`.
  `<|end|>` is a loss target (stop token); `<|eos|>` is masked in SFT. Final answers for
  verifiable tasks use the `#### <answer>` text convention. Think span is mandatory after
  reasoning SFT.
- Python tool: `<|python_call|>code<|/python_call|><|python_result|>out<|/python_result|>` inside the think span;
  the result span is never a loss/policy target; one `PySession` per conversation (state persists across calls and
  turns); code runs only in the sandboxed subset interpreter `slm/tools/pysandbox.py` (never exec/eval).

## Engineering rules
- Loss functions return `(loss_sum, n_valid_tokens)`; the trainer divides by the global token count
  so gradient accumulation is invariant to the accumulation factor.
- Every numeric hyperparameter is config-driven (`slm/config.py` dataclasses + YAML + `k=v` overrides).
- Long runs must: estimate wall-clock up front, checkpoint frequently, resume by default, save a
  checkpoint on Ctrl-C, and refresh a self-contained `runs/<run>/report.html` (inline charts, ETA,
  per-100M-token milestone timings) at least every 30 minutes and at every milestone.
- Tests live in `tests/`; run `.venv/Scripts/python.exe -m pytest`.
- Long context is claimed only where needle retrieval holds: `slm.eval.long_context` with the real-text haystack
  (never the filler control) and n >= 16; extension runs set `eval.needle_lengths` so the gate (min over depths
  >= 0.8 at every length) is visible in the run page before the next stage starts.
- Priorities (2026-09-16): strength per parameter first; context only as far as it costs <= 5% on short-context evals.
- GPU telemetry: both trainers run `slm.utils.gpu.GpuSampler` (nvidia-smi on a daemon thread) and put
  `gpu_temp_c`/`gpu_power_w`/`gpu_util` in every train record; above `runtime.gpu_warn_temp_c` (80 C) they print a
  console warning and log a `warn` event (rate-limited to one per 5 min). Runs started before 2026-09-13 (M3a) lack these fields.

## Windows gotchas learned
- A process holding a memmap or open handle on a file blocks overwriting/renaming it (prepare.py
  failed with EINVAL while the portal had a shard mapped). The portal opens memmaps per request
  and never caches them; stop viewers before re-tokenizing; checkpoint writes retry on PermissionError.
- Long background jobs: run `python -u ... > log 2>&1`; never pipe through grep/tail (buffered output
  is lost if the pipeline dies, which is how the first 149M benchmark's results were lost).
- Verify a GPU job is really gone (nvidia-smi compute apps + process list) before assuming the GPU is
  free: a benchmark left running silently shared the GPU with M1 for 45 minutes.
- KV-cache decoding must not run under the cuDNN-only SDPA restriction (`sdpa_context("decode")`
  = efficient+math); cuDNN re-plans per KV length and generation crawls.
- On Windows (WDDM) PyTorch does NOT OOM at 16 GiB: it spills into shared host memory and runs
  ~60x slower (bench showed "20-38 GiB peak" at ~1k tok/s). Treat peak VRAM > ~14.5 GiB as a
  failure; keep training configs around 11 GiB.
- What crosses that limit is *reserved* memory, not peak allocated, and an in-run eval can raise it
  permanently: a generation KV cache needs large contiguous segments the training blocks cannot supply,
  so the allocator takes new ones from the driver and keeps them. The mild form is not a 60x collapse but
  a steady ~8% loss with 100% util, lower power and lower clocks (M8, 2026-09-17). Trainers log
  `vram_gib`/`vram_reserved_gib` (console `peak/reserved`); call `torch.cuda.empty_cache()` after any
  eval that generates. Benchmark table: artifacts/bench/base_149m_full.log.
- Measured 149M throughput (compiled, cuDNN SDPA): 2K x mb8 = 58k tok/s (69% MFU), 4K x mb4 = 43k,
  eager 2K x mb8 = 9.5k (torch.compile is a 6x win here, not optional). 336M: 2K x mb4 = 28.6k, 4K x mb2 = 25.4k
  (12.5 GiB bench / 13.8 GiB live; mb up spills). Size sweep table: docs/results.md §1.

## Run naming convention
`m<milestone>_<role>_<params>`, e.g. `m2_base_149m`, `m3_base_stable_149m`, `m3_base_8k_149m`, `m4_sft_149m`,
`m6_rl_arith_149m`, `m7_ctx16k_149m`, `m8_base_stable_336m`, `m8_base_4k_336m`. Roles say what the run produces (`base_stable` = flat-LR continuation with no
finished model; `base_8k` = 8K phase + decay that yields the base checkpoint). Token counts never go in names:
exact counts live in `checkpoints/index.json` and the overview shows cumulative tokens along the init_from chain.
Config files are named after the run they define (`configs/train/<run_name>.yaml`).

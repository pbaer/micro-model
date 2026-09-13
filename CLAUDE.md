# Project conventions

From-scratch small language model research stack (decoder-only Transformer, ~149M params,
single RTX 4080 SUPER, native Windows). Package name is `slm`; the project's public name is a
placeholder and may change, so don't bake it into code.

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
- Data: English prose, Python, Linux bash only. Filter out other languages.
- No teacher models / model-generated synthetic data for now.
- Own 32K BPE tokenizer with a reserved block of 64 special tokens (chat roles, think, tools).
  Special tokens are never produced from raw text; only the chat formatter inserts them.
- Chat format: `<|bos|><|user|>...<|end|><|assistant|><|think|>...<|/think|>answer<|end|><|eos|>`.
  `<|end|>` is a loss target (stop token); `<|eos|>` is masked in SFT. Final answers for
  verifiable tasks use the `#### <answer>` text convention. Think span is mandatory after
  reasoning SFT.

## Engineering rules
- Loss functions return `(loss_sum, n_valid_tokens)`; the trainer divides by the global token count
  so gradient accumulation is invariant to the accumulation factor.
- Every numeric hyperparameter is config-driven (`slm/config.py` dataclasses + YAML + `k=v` overrides).
- Long runs must: estimate wall-clock up front, checkpoint frequently, resume by default, save a
  checkpoint on Ctrl-C, and refresh a self-contained `runs/<run>/report.html` (inline charts, ETA,
  per-100M-token milestone timings) at least every 30 minutes and at every milestone.
- Tests live in `tests/`; run `.venv/Scripts/python.exe -m pytest`.

## Windows gotchas learned
- A process holding a memmap or open handle on a file blocks overwriting/renaming it (prepare.py
  failed with EINVAL while the portal had a shard mapped). The portal opens memmaps per request
  and never caches them; stop viewers before re-tokenizing; checkpoint writes retry on PermissionError.
- Long background jobs: run `python -u ... > log 2>&1`; never pipe through grep/tail (buffered output
  is lost if the pipeline dies, which is how the first 149M benchmark's results were lost).

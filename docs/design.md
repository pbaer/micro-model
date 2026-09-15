# Design and implementation internals

How the pieces work, with the formulas and file formats an engineer needs to modify them safely.
Read the README first for the big picture. File references are relative to the repo root.

## 1. Configuration system (`slm/config.py`, `slm/train/config.py`)

- Nested dataclasses are the schema. `from_dict` builds them from YAML; `apply_overrides` applies
  `key.sub=value` strings from the command line (`python -m slm.train.pretrain --config x.yaml
  optim.lr=3e-4 batch.microbatch=4`). `to_dict` serializes them into checkpoints and `run.json`.
- `ModelConfig` (vocab, layers, widths, heads, `rope_theta`, `rope_scaling`, `qk_norm`,
  `tie_embeddings`, `grad_checkpointing`, `loss_chunk_size`) is loaded from `model_file` and then
  overridden by the training config's `model:` mapping, so a run can change context length or RoPE
  scaling without a new model file.
- `TrainConfig` composes `DataConfig`, `OptimConfig`, `ScheduleConfig`, `BatchConfig`, `RuntimeConfig`,
  `CheckpointConfig`, `EvalConfig`, plus `init_from`, `milestone_tokens`, `notes`. Derived value:
  `grad_accum = tokens_per_update / (microbatch × seq_len)` (must divide exactly).
- `RlConfig` (`slm/train/rl.py`) is flat: tasks, prompt counts, generation settings, objective
  settings, step budget.

## 2. Model (`slm/model/`)

**Block.** `x = x + Attn(RMSNorm(x)); x = x + SwiGLU(RMSNorm(x))`. Final RMSNorm, then logits from the
tied embedding matrix.

**Attention** (`attention.py`). One fused QKV projection to `[q: H·D, k: Hkv·D, v: Hkv·D]`; optional
QK-norm (RMSNorm over head_dim, before RoPE) which keeps logits bounded at high LR; RoPE applied to q
and k; `F.scaled_dot_product_attention(..., is_causal=True, enable_gqa=True)` so KV heads are shared
without an explicit repeat; output projection. `KVCache` preallocates `[B, Hkv, T_max, D]` per layer
and `cache.pos` tracks the fill; `Transformer.generate` decodes one token at a time with it.

**RoPE** (`rope.py`). Inverse frequencies `θ_i = base^(-2i/D)`. Scaling variants for context
extension: linear (positions divided by `factor`), NTK-aware (`base' = base · factor^(D/(D-2))`), YaRN
(per-frequency blend between interpolation and extrapolation using the `beta_fast`/`beta_slow` ramp,
plus the attention temperature `mscale = 0.1·ln(factor) + 1` folded into cos/sin). `set_rope` rebuilds
the tables on an existing model, which is how a checkpoint trained at 8K is extended to 16K.

**Loss** (`loss.py`). `chunked_cross_entropy(h, W, targets, chunk)` computes logits chunk by chunk in
fp32 and returns `(loss_sum, n_valid)` with `IGNORE_INDEX = -100` targets skipped. Returning sums (not
means) is what makes gradient accumulation exact: the trainer divides by the global valid-token count.

**Initialization.** All linear and embedding weights N(0, 0.02); `wo` and `w2` (the projections that
write into the residual stream) use std 0.02/√(2L).

**Introspection** (`introspect.py`). Walks the module tree, runs a probe forward with shape hooks
(B = 3, T = 7 so dimensions are unambiguous), rewrites those to symbols, and adds teaching-level nodes
(split, reshape, RoPE, SDPA with GQA groups, residual adds). Params and FLOPs per node reconcile with
`num_params()` and `profiling.flops_per_token` in tests.

**FLOPs and MFU** (`utils/profiling.py`). PaLM-style training FLOPs per token
`6·N_nonembed + 6·V·d + 12·L·d_attn·T`; MFU divides by the RTX 4080 SUPER dense bf16 peak
(104.5 TFLOPS with fp32 accumulate). `mfu_causal` halves the attention term.

## 3. Tokenizer (`slm/data/tokenizer.py`, `scripts/train_tokenizer.py`)

- HF `tokenizers` byte-level BPE with the GPT-4 pre-tokenization regex modified so `\p{N}` matches one
  digit at a time. Vocabulary 32,704; ids 32,704–32,767 are specials known only to `SlmTokenizer`.
- `encode_document(text)` → `<|bos|> ids <|eos|>`; `decode` renders specials as their literal names.
- Training: sample documents per source up to a character budget matching the pretraining mix
  (`--mix fineweb-edu:0.72 cosmopedia:0.10 finemath:0.06 python-edu:0.09 stack-edu-shell:0.03`).
- `meta.json` stores the specials list, regex and sha256; `scripts/eval_tokenizer.py` reports
  compression per source, digit/code fragmentation and round trips.

## 4. Data pipeline (`slm/data/`)

**Registry** (`sources.py`). `Source(name, repo, pattern, text_col, kind, license, content_via_swh)`.
Kinds: prose, math, code, chat, math_cot, math_qa. Code datasets ship only Software Heritage blob ids;
`swh.py` fetches contents from `s3://softwareheritage/content/<id>` (anonymous, gzip) with a thread
pool and writes text parquet next to the ids.

**Prepare** (`prepare.py`). Streams parquet row groups, applies `keep_doc` (≥ 64 chars; `language ==
"en"` when the column exists and the source is not code; ASCII-letter ratio ≥ 0.95 for prose, ASCII
ratio ≥ 0.9 for code), drops docs shorter than `--min-doc-tokens` (default 16) or longer than 65,536
tokens, and writes:

    tokenized/<tag>/<name>/train/shard_00000.bin      uint16 tokens, 100M per shard, docs = <|bos|> … <|eos|>
    tokenized/<tag>/<name>/train/shard_00000.idx.npy  int64 document start offsets within the shard
    tokenized/<tag>/<name>/val/…                       10 ‰ of documents by sha1(text[:2048]); never trained on
    tokenized/<tag>/<name>/manifest.json               counts, tokenizer sha256, min_doc_tokens

`--name` creates derived sources from the same raw files (e.g. `fineweb-edu-long` with
`--min-doc-tokens 4096`).

**Loader** (`loader.py`). `TokenStream` memory-maps one split of one source and hands out consecutive
windows of `seq_len + 1` tokens (windows never straddle shards; the tail of a shard shorter than one
window is skipped; wrapping increments `epoch`). `PretrainLoader` samples a source per row from the
mixture weights with a seeded `numpy` RNG, prefetches on a thread, pins memory, and snapshots
`{streams, rng, n_batches, tokens_served}` after each consumed batch. Resume restores that state and
the next batch is bit-identical (tested). `ValLoader` materializes a fixed set of windows once.

**Chat and SFT** (`chat.py`, `sft.py`). `format_chat(tok, messages, think_required)` returns ids and a
loss mask; `parse_assistant` splits a generated assistant turn into think/final spans on the closing
`<|/think|>` (the opening tag is part of the prompt). `rows_to_messages` normalizes chat rows, MetaMathQA
("The answer is: X" → think trace + `#### X`) and GSM8K (`#### X`). `prepare_sft` writes
`tokens_*.bin` / `mask_*.bin` / `idx_*.npy` per split; examples longer than `max_len` are dropped rather
than truncated (a cut-off answer teaches bad endings). `SftLoader` packs examples back to back like
pretraining and turns the mask into `IGNORE_INDEX` targets; cross-example attention inside a window is
the accepted packing compromise.

**Synthetic traces** (`rl/synth.py`). Templated step-by-step solutions for the RL task generators,
correct by construction, written straight into SFT shards (`synthetic-reasoning`).

## 5. Trainer (`slm/train/pretrain.py`)

Startup: seed, TF32 on, refuse to start with less than `runtime.min_free_vram_gib` free, build model
(fp32) and `torch.compile(model, dynamic=False)`, AdamW with two param groups (decay on ≥2-D tensors),
loaders, optional `extra_val_mixture` loader, then either resume from `latest.pt`, or `init_from` a
checkpoint (bf16 snapshot or full; weights only unless `init_optimizer`) and start fresh. `run.json`
records config, model config, param counts, tokenizer hash, environment, git commit and start time.

Update loop (per optimizer step): `grad_accum` microbatches under `autocast(bf16)` and the configured
SDPA backend; each microbatch's `(loss_sum, n_valid)` is accumulated and `loss_sum / n_valid_global`
is back-propagated; clip to 1.0; fused AdamW step; LR from `schedule.lr_at(tokens)` set before the step.

Schedules (`schedule.py`), all in tokens: linear warmup to `lr` over `warmup_tokens`, then cosine to
`min_lr_ratio·lr`; or WSD (flat, then linear decay over the last `decay_frac`); or constant. With
`epochs > 0`, `total_tokens = epochs × tokens in the training data` and one milestone = one epoch.

Cadence: log every `log_every_updates` (tokens/sec, EMA, step/fwd/bwd/opt/data ms, peak VRAM, GPU
temperature/power/utilization, ETA); validation every `eval.every_tokens`; samples every
`gen_every_tokens` under `sdpa_context("decode")`; milestone every `milestone_tokens` (bf16 snapshot,
segment timing); `latest.pt` every `ckpt.every_minutes`; `report.html` every `report_every_minutes`.
The initial wall-clock estimate is taken from the first post-warmup window and written into `run.json`.
Non-finite loss stops the run without checkpointing. A `warn` event is logged and printed when the GPU
temperature exceeds `runtime.gpu_warn_temp_c` (rate-limited to one per 5 minutes).

Stopping: Ctrl-C sets `stop_requested`; a `STOP` file in the run dir does the same and is deleted.
Either finishes the current update, saves `latest.pt`, logs `stop`, writes the report. `finish` is
logged when `tokens ≥ total_tokens`, followed by a final eval, samples, `final.pt` and index updates.

### Checkpoint files (`slm/utils/checkpoint.py`)

| File | Contents | Written |
|---|---|---|
| `latest.pt` | model fp32, optimizer, loader state, counters, RNG (CPU/CUDA), config, meta | periodically, on stop, on finish (atomic tmp + replace, with retry on Windows locks; `latest.prev.pt` kept) |
| `snap_<tokens>.pt` | bf16 model only + meta (`snap_800M.pt`, `snap_1p20B.pt`) | every milestone |
| `best.pt` | bf16 model at the best validation loss | on eval improvement |
| `final.pt` | bf16 model at the end | on finish |
| `index.json` | `{name: {kind, tokens, update, val_loss, time}}` | on every write (backfill script for old runs) |

### `metrics.jsonl` schema

One JSON object per line, always with `kind`, `time` (epoch seconds), and usually `tokens`.

| kind | fields |
|---|---|
| `start`, `resume`, `stop`, `finish` | `msg`; `finish`/`stop` carry `elapsed_s` |
| `checkpoint` | `msg` (file, seconds) |
| `train` | `update, loss, lr, grad_norm, tok_s, tok_s_ema, step_ms, fwd_ms, bwd_ms, opt_ms, data_ms, vram_gib, gpu_temp_c, gpu_power_w, gpu_util, elapsed_s, eta_s` (+ RL fields below) |
| `eval` | `val_loss, val_ppl, best, eval_s, val_pt_loss` (extra validation), RL: `heldout_acc, train_acc, heldout_malformed, heldout_len` |
| `milestone` | `tokens, segment_s, elapsed_s, tok_s, loss, val_loss` |
| `warn` | `msg` (GPU temperature) |

RL `train` records add `reward_mean, success_rate, group_std_mean, groups_no_signal, adv_abs_mean,
len_mean, len_correct, len_wrong, malformed_rate, length_term_rate, kl, entropy, clip_frac, ratio_mean`.
`slm/utils/metrics.py` (torch-free) reads this incrementally for both the HTML report and the portal;
`summary()` derives status, progress (tokens, or steps for RL), ETA, cumulative tokens along the
`init_from` chain, and the latest GPU state.

## 6. RL trainer (`slm/train/rl.py`, `slm/rl/`)

**Tasks** (`tasks.py`): generators `arith1`, `arith2`, `arith2mul`, `arith_multi`, `algebra`, `word`
yield `{id, prompt, answer, meta}`. `make_tasks` splits train/held-out by disjoint seeds and a hash of
the canonical prompt text, so overlapping generators cannot leak.

**Rollouts** (`rollout.py`): `rollout_group` samples G completions for one prompt (same prompt length,
no padding), stops on `<|end|>`, parses think/final spans, computes the reward, and recomputes old and
reference log-probs teacher-forced. `greedy_accuracy` batches prompts by length for evaluation.

**Rewards** (`rewards.py`): `parse_final_answer` takes the LAST `#### …` line and the first number-like
token after it (commas and `$` stripped, fractions allowed); `binary` reward = exact numeric match;
a malformed completion (no think span when required, no answer line) scores 0 and is counted.

**Advantages** (`advantages.py`): `A_i = r_i − mean(r)` and, with `normalize_std`, divided by
`std(r) + ε`. Groups with zero reward variance carry no signal and are skipped in the update
(`groups_no_signal` tracks how often).

**Objective** (`objectives.py`), per completion token with `ρ = exp(logp_new − logp_old)`:
`L = −Σ min(ρ·A, clip(ρ, 1−ε, 1+ε)·A)` (or REINFORCE `−Σ logp·A` with `use_ratio: false`), plus
`kl_coef · Σ k3(logp_new, logp_ref)` where `k3 = exp(ref − new) − (ref − new) − 1`. Sums are divided by
the total number of live completion tokens in the step (accumulation-invariant like pretraining).

**Step**: sample `prompts_per_step` prompts → rollouts → advantages → `ppo_epochs` passes of
`microbatch`-sized minibatches → clip, AdamW step. Every rollout is persisted to
`rollouts/step_<n>.jsonl` with prompt, completion, reward, advantage, and both log-prob vectors.
Evaluation every `eval_every_steps` on held-out prompts (greedy) and an equal-size train sample;
`val_loss` for RL runs is `1 − heldout_acc` so `best.pt` semantics carry over. Checkpoints:
`step_<n>.pt`, `latest.pt`, `final.pt`.

## 7. Context extension

`RopeScaling` in the training config's `model:` block rebuilds the RoPE tables when the model is
constructed; `init_from` then loads the 8K weights unchanged. The training row length comes from
`data.seq_len`; memory is handled with `grad_checkpointing: true` and `loss_chunk_size`. The
long-context eval (`slm/eval/long_context.py`) inserts a needle sentence at given depths in filler
text of a given total length, reserves 16 tokens for the answer within the RoPE table, and reports
retrieval accuracy per (length, depth); multi-needle mode asks for several facts at once. Extension
runs also keep `extra_val_mixture` pointed at the ordinary 2K validation mixture so short-context loss
is tracked next to the long-context metric.

## 8. Evaluation tools (`slm/eval/`)

- `generation.py` / `sampling.py`: fixed-seed greedy and sampled completions; one `sample_next`
  (temperature, top-p, top-k) shared by the trainer, RL rollouts, the evals and the portal.
- `diagnostics.py`: residual-stream RMS per layer and relative block update size; dead/rare SwiGLU
  units; per-head attention entropy, BOS-sink mass, mean distance, local mass; head and layer ablation
  deltas on validation loss; effective rank and spectral norms; embedding usage. `scripts/diagnose.py`
  writes `diagnostics.json` and `diagnostics.html`.
- `reasoning.py`: greedy accuracy per task on held-out prompts and on GSM8K test with the chat
  format and a mandatory think span.
- `lm_eval_wrapper.py`: an `lm_eval` `LM` subclass implementing `loglikelihood` (context/continuation
  split that respects BPE merges), `loglikelihood_rolling` and greedy `generate_until`; run from the
  CLI with `--tasks` and `--limit`.

## 9. SDPA backends and GPU telemetry (`slm/utils/sdpa.py`, `slm/utils/gpu.py`)

`sdpa_context(name)` restricts PyTorch's SDPA dispatcher: `cudnn` for training (fast fused backend on
Windows), `efficient` as fallback, `auto`, and the special `"decode"` context (efficient + math) used
for every KV-cache decoding path because cuDNN re-plans per KV length and crawls. The math backend is
never allowed silently for training. `GpuSampler` polls `nvidia-smi` every 2 s on a daemon thread and
exposes the latest temperature, power, utilization, clocks and throttle reasons; the portal's
`/api/system/gpu` uses the same query.

## 10. Testing (`tests/`)

- Model: causal masking, GQA equals manual KV repeat, KV-cache decode equals full forward, RoPE
  relative-position property and scaling variants, chunked CE equals full CE and respects ignore
  index, gradient checkpointing matches, gradient accumulation invariance, generation determinism,
  a tiny overfit, parameter counts of shipped configs.
- Data: tokenizer round trips and single-digit tokens, token stream windows/wrap, loader resume
  reproduces batches, fixed val loader, chat formatting masks and parsing, SFT shards and loaders.
- Training: resume equivalence (a run stopped and resumed matches an uninterrupted run to 2e-3 in
  loss and 1e-4 in weights), SFT epochs, RL objective clipping and REINFORCE equivalence, task split
  disjointness, verifier and reward schemes, synthetic-trace correctness, a two-step GRPO run.
- Portal: API routes on synthetic runs, SSE live tail against a real server, worker harness on CPU,
  JS module parse check, and a Playwright suite that clicks every control on every page.

## Tool use (sandboxed Python with REPL sessions)

Protocol on the reserved tokens (`slm/tools/protocol.py`); the call body is plain Python:

    ... 120 - 36 = <|python_call|>print(120-36)<|/python_call|><|python_result|>84<|/python_result|>84 pages left ...

The model generates through `<|/python_call|>`; the harness runs the code and appends the result span; generation
resumes. Result tokens are environment-written: loss mask 0 in SFT (`format_chat(tools=True)`), `gen_mask` 0 in RL
so they are excluded from the policy gradient and the KL term. One `PySession` per conversation keeps variables and
functions across calls and across turns (REPL semantics); future tools are Python functions exposed in that
namespace, not new token types. Text form for datasets and display: GSM8K's own `<<expr=result>>` for one expression
and `<<<code>>>` for a short program; `split_markup` runs the code while converting so the recorded result is exactly
what inference would insert, and `render_tools` turns generated ids back into the same markup.

Generation with tools (`slm/tools/loop.py`): batched rows diverge in length after a result is inserted, so decoding
runs in rounds, grouping active rows by current length; each returned token carries a gen_mask bit and each row keeps
its session. Rollouts record every (code, result) pair, tool_calls / tool_errors, and `answer_from_tool`: whether the
final `#### N` equals a number produced by a non-trivial call (`print(42)` does not count). The RL reward scheme
`tool` pays 1.0 for a correct answer that came out of a call and 0.5 for a correct answer computed in the head, which
is the incentive to use Python whenever possible; the eval reports tool_use_rate and answer_from_tool_rate.

Sandbox (`slm/tools/pysandbox.py`), chosen over a subprocess or container because it is small enough to audit and
leaves no side-effect surface at all: model code is parsed with `ast.parse` (inert) and executed by a tree-walking
interpreter for a Python subset. CPython's exec/eval/compile never see model output. No imports, no attribute access
except `math.<whitelisted>` and an explicit table of list/str/dict methods on the sandbox's own values, no dunders,
classes, lambdas, with/try/global/yield/async, no I/O builtins. Limits enforced by the interpreter: operation budget,
loop-iteration cap, call depth, integer bit length, string and sequence length, output length, exponent size,
session namespace size, wall-clock backstop. Every refusal is a `ToolError` with a line number, the offending source
line and a hint about what to use instead, returned to the model as `error: ...` so it can correct itself; a bug in
the interpreter can only raise, never escape. Residual risks: CPU time inside a single bounded call (~2 s worst
case), and interpreter bugs; both are contained to a wrong tool result, not to the host.
`tests/test_pysandbox.py` holds the refusal, limit and message cases.

Data: `slm.data.sft --tools` converts GSM8K's calculator annotations into calls and drops rows without any;
`slm.rl.synth --tools` writes templated traces whose every step is a call (programs for multi-step tasks).

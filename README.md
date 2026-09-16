# Small language model, from scratch

A learning project: train a small language model (SLM; a 149M first model, now a 336M second base) **from random
initialization on one consumer GPU**, using the same state-of-the-art recipe that frontier LLMs use, in order to build a
solid, hands-on intuition for how those models work. Every stage of the modern pipeline is implemented
in-house and instrumented for inspection:

    pretraining (2K → 8K context) → instruction SFT → reasoning SFT (think spans) →
    RL with verifiable rewards (GRPO) → context extension (8K → 16K → 32K, RoPE scaling)

The code is deliberately small (about 10K lines of Python plus a no-build web UI), config-driven,
tested, and written to be read. The Python package is `slm`; the project name is a placeholder and is
not baked into the code.

**Hardware and constraints.** One RTX 4080 SUPER (16 GB), native Windows 11, Python 3.13, PyTorch
2.14 + CUDA 13 with `triton-windows` so `torch.compile` works. Training data lives on `C:\slm-data`;
code, configs, checkpoints and reports live in this repo. Data policy: English prose, Python and Linux
shell only. No teacher models and no model-generated training data (synthetic reasoning traces are
templated and correct by construction). No paid APIs.

**Where things stand** is deliberately kept out of this file. Current numbers live in
[docs/results.md](docs/results.md), the narrative in [docs/log.md](docs/log.md), remaining work in
[docs/roadmap.md](docs/roadmap.md), and the live state in the command center (`python -m slm.portal`).

---

## 1. The stages and what each one teaches

| Stage | Run(s) | What it is | What you learn by doing it |
|---|---|---|---|
| M0 | `scripts/bench_throughput.py` | Environment, model, tests, throughput benchmark | Tokens/sec vs model-FLOP utilization, memory budget, `torch.compile`, SDPA backends, why microbatch size matters |
| M1 | `m1_tinystories_26m` | 26M model on TinyStories | The whole loop on a toy: loss curve shape, packing, checkpoint/resume, sampling, diagnostics |
| M2 | `m2_base_149m` | 149M base, first 1B tokens at 2K context, WSD schedule | Scaling up: data mixtures, LR schedules, what the first billion tokens buy |
| M3 | `m3_base_stable_149m` → `m3_base_8k_149m` | Continue the stable phase for several billion tokens, then an 8K-context phase that carries the LR decay | Warmup-stable-decay in practice, long-document upsampling, memory at 8K |
| M4 | `m4_sft_149m` | Instruction SFT (SmolTalk subsets, assistant-token loss) | Chat formatting, loss masks, measuring base-model drift during SFT |
| M5 | `m5_reasoning_149m` | Reasoning SFT with a mandatory think span (`<\|think\|>`) and `#### answer` | Traces as supervision, format learning, what a 150M model can and cannot reason about |
| M6 | `m6_rl_arith_149m` (+ stages B/C, `m6_rl_gsm_tools_149m`) | GRPO with programmatic verifiers, no critic, no reward model; later with a sandboxed Python tool (`slm/tools`) | Group-relative advantages, clipped ratios, KL to a reference, reward hacking, held-out generalization |
| M7 | `m7_ctx16k_149m`, `m7_ctx8k_retrieval*_149m` | YaRN RoPE scaling; a gated retrieval curriculum (templated needle documents, gate = worst-depth retrieval ≥ 80%) | Configured vs effective context, needle-in-a-haystack, why the window edge fails, short-context regression checks |
| M8 | `m8_base_stable_336m` → `m8_base_4k_336m` | The second base: 336M, 10B tokens, 2K then 4K rows, chat and tool-call conversations mixed into the decay phase | Scaling the model instead of the microbatch, size/throughput trade-offs, pretraining the chat format |

Everything is measured. Each run writes a self-contained `runs/<run>/report.html`, a JSONL metrics log,
bf16 snapshots every 100M tokens, and the command center renders all of it live.

## 2. The model

Two bases share one architecture family (Llama-3-style decoder-only Transformer). The first, `base_149m.yaml`, was
used for M1–M7 and for every ablation; the second, `base_336m.yaml`, is the model the project now builds on. The
size was chosen from a throughput/VRAM sweep (docs/results.md §1): 336M keeps the 149M model's utilization and
leaves headroom for 4K rows, while ~565M is the memory ceiling of a 16 GB card with fp32 Adam state.

| | base_149m (first) | base_336m (second) |
|---|---|---|
| Layers × width | 18 × 768 | 24 × 1024 |
| Attention heads | 12 query / 4 KV | 16 query / 8 KV |
| d_ff (SwiGLU) | 2304 | 3072 |
| RoPE base | 100 000 | 500 000 |
| Params (non-embedding) | 149.1M (123.9M) | 335.6M (302.0M) |
| Native context | 2K, extended to 8K | 2K for 75% of tokens, then 4K |
| Throughput | 62K tok/s at 2K | 28.6K tok/s at 2K, 25.4K at 4K |

Common to both:

| Component | Choice | Notes |
|---|---|---|
| Attention | GQA, head_dim 64, QK-norm | GQA handed to fused SDPA (`enable_gqa=True`); cuDNN backend on Windows (flash unavailable) |
| Positions | RoPE, table for 8192 positions | Scaling variants: linear, NTK, YaRN; `Transformer.set_rope` rebuilds tables for extension |
| MLP | SwiGLU (d_ff = 3·d_model), fused gate/up projection, no biases | |
| Norms | Pre-norm RMSNorm (eps 1e-6) + final norm | Norm gains excluded from weight decay |
| Output | Tied to the input embedding | An untied head would add 25M (149M) / 34M (336M) params |
| Init | N(0, 0.02), residual-writing projections scaled by 1/√(2L) | GPT-2 convention |
| Loss | Chunked cross-entropy returning `(loss_sum, n_valid)` | Never materializes [B·T, V] fp32 logits when chunked |
| Memory options | Gradient checkpointing (for 16K+), `loss_chunk_size` | |

Parameter arithmetic per layer of the 149M model: attention 768·(768 + 2·256) + 768·768 ≈ 1.6M; MLP 3·768·2304 ≈ 5.3M;
about 6.9M per block, 18 blocks ≈ 124M, plus 25M embedding. For the 336M model: attention 1024·(1024 + 2·512) + 1024² ≈
3.1M, MLP 3·1024·3072 ≈ 9.4M, 12.6M per block, 24 blocks ≈ 302M, plus 34M embedding. The Architecture page of the command
center derives this graph, the FLOPs per token and the memory budget from the real module tree.

A 26M sibling (`sanity_26m.yaml`: 8 × 384, 6q/2kv) is used for M1 and for cheap ablations, and
`tiny.yaml` for unit tests.

## 3. Tokenizer and chat format

Our own byte-level BPE (`slm/data/tokenizer.py`): 32,768 ids = 32,704 BPE merges + 64 reserved
specials at the top of the id range. GPT-4-style pre-tokenization regex except that **digits are always
single tokens** (better arithmetic for a small model). Trained on a character sample matching the
pretraining mix. Special tokens are not part of the HF tokenizer: raw text can never produce them, only
the chat formatter inserts them. The tokenizer is frozen and every checkpoint records its sha256.

Named specials: `<|bos|> <|eos|> <|pad|> <|system|> <|user|> <|assistant|> <|end|> <|think|> <|/think|>
<|python_call|> <|/python_call|> <|python_result|> <|/python_result|>` plus 51 `<|reserved_N|>`.

Chat format (one example, reasoning stage):

    <|bos|><|user|>What is 17 + 26?<|end|><|assistant|><|think|>17 + 26 = 43<|/think|>#### 43<|end|><|eos|>

Rules: `<|end|>` is a loss target and the generation stop token; `<|eos|>` is masked (the model cannot
know whether another turn follows); after reasoning SFT the think span is mandatory; verifiable
answers use the plain-text `#### <answer>` convention (last such line wins). Tool tokens are reserved
now and unused.

## 4. Data

All sources are Hugging Face parquet files, downloaded to `C:\slm-data\raw\<source>` and tokenized into
uint16 shards with document boundaries (`C:\slm-data\tokenized\v1\<source>\{train,val}`). Exact
volumes are on the Data page of the command center and in [docs/results.md](docs/results.md).

| Source | Kind | Role |
|---|---|---|
| fineweb-edu (sample-10BT): `fineweb-edu-10bt` = all 14 files (10.07B tokens, the second base); `fineweb-edu` / `fineweb-edu-b` = the first 2 / 7 files; `fineweb-edu-long` = docs ≥ 4096 tokens | educational web prose | 66–80% of the pretraining mixture; long-doc variant for the context phases |
| cosmopedia v2 | synthetic textbooks | ~10% |
| finemath 4+ | math web text with LaTeX | ~5% |
| python-edu (file contents fetched from Software Heritage S3) | Python | ~3% |
| stack-edu Shell (Software Heritage S3) | bash/sh | ~2% |
| tinystories | children's stories | M1 only |
| SmolTalk subsets (magpie-ultra, openhermes, systemchats, constraints, everyday) | chat | instruction SFT; also mixed into the second base's decay phase as `smoltalk-chat` (4.5%) |
| GSM8K / MetaMathQA / templated traces with Python tool calls (`gsm8k-tools`, `synthetic-reasoning-tools`) | tool-use reasoning | tool SFT; also mixed into the decay phase as `tool-chat` (1.5%) |
| `synth-retrieval*` (templated: facts inserted into real text + questions; key-value ledgers) | retrieval | context curriculum and 3.5% of the second base's mixture |
| GSM8K, MetaMathQA (converted to think spans + `#### X`), synthetic traces from our task generators | reasoning | reasoning SFT |

Language filtering: `language == "en"` where the source provides it plus an ASCII-letter-ratio test at
prepare time; validation splits are hash-partitioned by document text so they can never leak into
training. SFT examples are packed like pretraining rows with a parallel loss mask.

## 5. Training stack (the rules the code follows)

- **Config-driven.** Every numeric hyperparameter lives in dataclasses (`slm/config.py`,
  `slm/train/config.py`) filled from YAML plus `key.sub=value` command-line overrides.
- **Token-indexed schedules** (cosine, warmup-stable-decay, constant): progress is measured in tokens,
  so batch or sequence changes do not silently change the schedule.
- **Accumulation-invariant loss.** Loss functions return `(loss_sum, n_valid)`; the trainer divides by
  the global token count of the update, so gradient accumulation gives the same result as a bigger
  batch (tested).
- **fp32 master weights, bf16 autocast, fused AdamW**, weight decay only on ≥2-D tensors, grad clip 1.0.
- **Packing.** Rows are `seq_len + 1` consecutive tokens of one source's stream (may start
  mid-document; `<|bos|>`/`<|eos|>` mark boundaries); the source of each row is sampled from mixture
  weights. Loader state is captured after each consumed batch, so a resume replays exactly.
- **Checkpoints.** `latest.pt` (full state, atomic write, previous copy kept) every 15 minutes and on
  Ctrl-C or a `STOP` file; bf16 `snap_<tokens>.pt` every 100M tokens; `best.pt`, `final.pt`;
  `checkpoints/index.json` with exact token counts. Rerunning the same command resumes.
- **Phase changes** use `init_from` (weights only by default; optimizer state optional) and a fresh
  schedule, which is how M2 → M3a → M3b → M4 → M5 → M6 → M7 chain together.
- **Observability.** `metrics.jsonl` (one record per event, keyed by tokens and time), console lines
  with ETA and GPU temperature/power, `report.html` refreshed every 10 minutes and at milestones,
  fixed-prompt samples every 100M tokens, an up-front wall-clock estimate written into `run.json`.
- **Windows specifics** matter: the flash SDPA kernel is unavailable (cuDNN is the fast path), the
  allocator spills into host memory above 16 GiB instead of raising OOM (60× slower, so peaks are kept
  near 11 GiB), open files cannot be replaced, and KV-cache decoding must not use cuDNN attention.

Throughput and memory for every (sequence length, microbatch) pair are measured once with
`scripts/bench_throughput.py` and recorded in [docs/results.md](docs/results.md); training configs take
their microbatch from that table.

## 6. Post-training

**Instruction SFT (M4)** reuses the pretraining loop with `data.kind: sft`: examples are packed exactly
like pretraining, a parallel mask marks assistant tokens and their `<|end|>`, schedules are in epochs,
and a second validation set on the pretraining mixture tracks base-model drift.

**Reasoning SFT (M5)** trains on `<|think|>` traces: GSM8K, MetaMathQA (numeric answers only) and
synthetic traces for arithmetic, multi-step expressions, linear equations and word problems. Prompts
end with `<|think|>` so the model must reason before answering.

**RL (M6)** is GRPO-style (`slm/train/rl.py`, `slm/rl/`): for each prompt sample a group of G
completions, score them with a programmatic verifier (last `#### <number>`, fraction-aware), compute
group-relative advantages (mean-centered, optionally std-normalized), and optimize the clipped
importance-ratio objective plus a k3 KL penalty to the frozen bf16 starting policy. Old and reference
log-probs are recomputed teacher-forced after sampling so the ratio is honest. Every rollout is saved to
`rollouts/step_<n>.jsonl`. Train and held-out prompts are split by a hash of the problem text, so
generalization is measured on problems the policy never saw. Curriculum: A arithmetic → B multi-step
and algebra → C code with unit tests and logic puzzles.

## 7. Context extension

Context length changes no parameters. It changes the RoPE table, the attention cost, activation memory
and the KV cache. The plan: bulk pretraining at 2K, the final phase at 8K on a long-document mixture
(M3b), then branch from the finished 8K base to 16K with YaRN (factor 2) and a short continuation at
16K with gradient checkpointing (M7), optionally 32K after that. The base checkpoint is never modified,
and each extension is checked against the needle-in-a-haystack eval and the short-context validation
loss so a regression would be visible.

## 8. Evaluation

- Validation loss/perplexity on held-out shards of the training mixture, at a fixed token cadence.
- `slm.eval.lm_eval_wrapper`: an lm-evaluation-harness adapter (HellaSwag, ARC-Easy, PIQA, MMLU subsets)
  with a loglikelihood implementation that respects BPE merges.
- `slm.eval.reasoning`: greedy accuracy on held-out programmatic tasks and GSM8K test, with malformed
  rate and lengths.
- `slm.eval.long_context`: single- and multi-needle retrieval across lengths and depths.
- `slm.eval.diagnostics`: per-layer residual norms, dead SwiGLU units, attention entropy and BOS-sink
  mass, head/layer ablations, weight spectra (writes `diagnostics.html` next to the checkpoint).
- Fixed-prompt samples (greedy and sampled) during every run.

## 9. Command center

`python -m slm.portal` serves a local web UI at http://127.0.0.1:8765 (FastAPI, torch-free server;
vendored Preact + uPlot, no build step). Pages: **Overview** (pipeline table, live runs, GPU state, data
readiness), run detail (live charts including GPU temperature/power, milestones, samples timeline,
checkpoints, events, config), **Data** (sources, mixture, a Documents browser over tokenized shards and
raw parquet with text/tokens/ids views and SFT loss masks), **Tokenizer** playground, **Inference**
(checkpoint slots A/B in a torch worker subprocess with a GPU guard, streaming generation with per-token
log-probs, scoring), and **Architecture** (interactive module graph, FLOPs/memory, LR/RoPE
illustrations computed by the real functions). It is a supplement to working in the repo, not the
primary interface. Details: [docs/command_center.md](docs/command_center.md).

## 10. Running it

Setup (Windows, native; see [docs/runbook.md](docs/runbook.md) for the full operations guide):

```bash
pip install uv
uv venv .venv --python 3.13
uv pip install --python .venv/Scripts/python.exe "torch==2.14.0+cu130" --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/Scripts/python.exe "triton-windows==3.8.0.post28"
uv pip install --python .venv/Scripts/python.exe -e ".[portal,dev]"
.venv/Scripts/python.exe -m pytest
```

The pipeline, in order (all commands use `.venv/Scripts/python.exe`):

| Step | Command |
|---|---|
| download raw corpora | `python -m slm.data.download fineweb-edu --n-files 2` (`--list` shows sources) |
| fetch code contents | `python -m slm.data.swh python-edu --max-files 200000` |
| train tokenizer | `python scripts/train_tokenizer.py --out C:/slm-data/tokenizer/v1 --chars 1.5e9` |
| tokenize to shards | `python -m slm.data.prepare fineweb-edu --tokenizer C:/slm-data/tokenizer/v1` (`--min-doc-tokens 4096 --name fineweb-edu-long` for the long-doc source) |
| benchmark | `python scripts/bench_throughput.py --config configs/model/base_336m.yaml --seq 2048 4096` |
| pretrain / continue | `python -m slm.train.pretrain --config configs/train/m8_base_stable_336m.yaml` (`scripts/pipeline_m8.sh` chains both phases) |
| SFT data | `python -m slm.data.sft smoltalk-openhermes-100k --tokenizer C:/slm-data/tokenizer/v1` (`--think-required` for reasoning sets); `python -m slm.rl.synth --n 40000` for synthetic traces |
| instruct / reasoning SFT | `python -m slm.train.pretrain --config configs/train/m4_sft_149m.yaml` (same loop, `data.kind: sft`) |
| GRPO RL | `python -m slm.train.rl --config configs/train/m6_rl_arith_149m.yaml` |
| evaluate | `python -m slm.eval.lm_eval_wrapper --checkpoint … --tasks hellaswag,arc_easy,piqa`; `python -m slm.eval.reasoning --checkpoint … --gsm8k 200`; `python -m slm.eval.long_context --checkpoint … --lengths 1024 4096 8192`; `python scripts/diagnose.py <ckpt> --root C:/slm-data/tokenized/v1 --source fineweb-edu` |
| command center | `python -m slm.portal` |

Long runs go in the background with unbuffered output: `python -u -m slm.train.pretrain --config … > runs/<run>/train.log 2>&1`.
Stop with Ctrl-C or `runs/<run>/STOP`; rerun the same command to resume.

## 11. Repository map

```
slm/
  config.py            ModelConfig, RopeScaling, YAML/override plumbing
  model/               attention.py (GQA, QK-norm, KVCache), rope.py (none/linear/ntk/yarn), mlp.py (SwiGLU),
                       transformer.py (blocks, generate), loss.py (chunked CE), introspect.py (module graph)
  data/                sources.py (registry), download.py, swh.py (code contents), tokenizer.py, prepare.py (shards),
                       loader.py (packing/mixture/resume), chat.py (format + loss mask), sft.py (SFT shards/loaders)
  train/               config.py, schedule.py, pretrain.py (Trainer: pretrain + SFT), rl.py (RlTrainer: GRPO)
  rl/                  tasks.py (generators, disjoint splits), rewards.py (verifier), advantages.py, objectives.py,
                       rollout.py (groups, greedy eval), synth.py (correct-by-construction traces)
  eval/                generation.py, sampling.py, diagnostics.py, reasoning.py, long_context.py, lm_eval_wrapper.py
  utils/               checkpoint.py, logging.py, metrics.py, report.py, sdpa.py, profiling.py, gpu.py
  portal/              app.py, settings.py, api/ (runs, data, tokenizer, model, arch, system), services/
                       (runs, datasets, tokenizer, hparams, worker, harness), static/ (Preact SPA)
configs/model/         tiny (tests), sanity_26m (M1), base_149m (M2–M7), base_336m (M8+), base_250m/360m/500m/620m (size sweep candidates)
configs/train/         one YAML per run, named after the run
scripts/               bench_throughput, param_count, train/eval_tokenizer, diagnose, backfill_ckpt_index,
                       pipeline_after_m2.sh, portal_smoke
tests/                 model math, data/loader resume, trainer resume equivalence, SFT, RL, GPU telemetry,
                       portal API + Playwright e2e
runs/<run>/            metrics.jsonl, run.json, report.html, train.log, samples/, checkpoints/ (+ rollouts/ for RL); untracked
artifacts/bench/       throughput benchmark JSON/logs
docs/                  see below
```

## 12. Conventions

- Run names: `m<milestone>_<role>_<params>` (`m2_base_149m`, `m3_base_stable_149m`, `m6_rl_arith_149m`).
  Roles say what a run produces; token counts never go in names (they live in `checkpoints/index.json`).
  Config files are named after the run they define.
- Tests: `.venv/Scripts/python.exe -m pytest` (CUDA tests skip without a GPU; `tests/e2e` needs
  Playwright Chromium). Any change to portal JS must pass the module-parse test.
- Never train on the SDPA math backend; never pipe a long job through `grep`/`tail`; verify the GPU is
  free (`nvidia-smi`) before launching; keep peak VRAM under ~14.5 GiB.
- `CLAUDE.md` holds the short version of these rules for coding agents.

## 13. Documentation

| File | Contents |
|---|---|
| [docs/design.md](docs/design.md) | Implementation internals: config system, model math, tokenizer, data formats, trainer loop, checkpoint and metrics schemas, RL objective, context extension, evaluation |
| [docs/runbook.md](docs/runbook.md) | Operations: environment, data preparation, launching/monitoring/stopping runs, phase changes, pre-launch checklist, Windows gotchas, troubleshooting |
| [docs/results.md](docs/results.md) | Measured numbers: benchmarks, data volumes, every run's outcome, evaluations, diagnostics, sample quality |
| [docs/command_center.md](docs/command_center.md) | The web UI: architecture, pages, API, worker, tests, backlog |
| [docs/roadmap.md](docs/roadmap.md) | Remaining work with time estimates and open decisions |
| [docs/log.md](docs/log.md) | Dated project log: what happened, incidents, decisions |

## 14. For agents joining the project

1. Read this file, then `CLAUDE.md`, then [docs/runbook.md](docs/runbook.md). Skim
   [docs/design.md](docs/design.md) for the subsystem you will touch.
2. Check what is running before using the GPU: `nvidia-smi` and the Overview page. A training run owns
   the GPU; evaluations and the Inference page fall back to CPU or wait.
3. Find state in files, not in memory: `runs/<run>/metrics.jsonl` and `run.json` are the source of
   truth for any run; `checkpoints/index.json` for exact token counts; `docs/results.md` and
   `docs/log.md` for the narrative.
4. Add a config, not a code path, for a new experiment. Add a test for any behavior you rely on
   (resume equivalence and accumulation invariance are the two that matter most).
5. Keep the documentation current: results go in `docs/results.md`, incidents and decisions in
   `docs/log.md`, plan changes in `docs/roadmap.md`. This README should only change when the approach
   or the structure changes.

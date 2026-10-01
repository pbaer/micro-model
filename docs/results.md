# Results and measurements

Numbers that change as runs finish. Update this file when a run completes or an evaluation is run;
the command center shows the live version of the same data. Last updated 2026-09-16 13:30.

## 1. Throughput benchmark (149M, RTX 4080 SUPER, cuDNN attention)

`scripts/bench_throughput.py --config configs/model/base_149m.yaml`, artifacts in `artifacts/bench/`.
Rows marked SPILL ran at 1% MFU because the WDDM allocator spilled into host memory (not an OOM).

| seq | microbatch | compile | loss chunk | tok/s | step ms | peak VRAM | MFU (PaLM) |
|---|---|---|---|---|---|---|---|
| 2048 | 4 | yes | 0 | 56.2K | 146 | 6.6 GiB | 66% |
| 2048 | 8 | yes | 0 | 58.2K | 281 | 11.1 GiB | 69% |
| 2048 | 8 | yes | 4096 | 56.4K | 291 | 10.5 GiB | 67% |
| 2048 | 16 | yes | 0 | 0.96K | 34,297 | 20.2 GiB SPILL | 1% |
| 4096 | 4 | yes | 0 | 43.4K | 377 | 11.1 GiB | 65% |
| 4096 | 8 | yes | 0 | 0.77K | 42,486 | 20.2 GiB SPILL | 1% |
| 8192 | 4 | yes | 0 | 0.60K | 54,532 | 20.2 GiB SPILL | 1% |
| 8192 | 8 | yes | 4096 | 0.34K | 189,969 | 38.0 GiB SPILL | 1% |
| 2048 | 8 | no (eager) | 0 | 9.5K | 1,723 | 18.5 GiB | 11% |
| 8192 | 1 | yes | 4096 | 40.5K | 202 | 6.6 GiB | 87% |
| 8192 | 2 | yes | 4096 | 41.9K | 391 | 10.9 GiB | 90% |

Size candidates for the second base (2026-09-16, 2K context, compiled, cuDNN, largest non-spilling microbatch):

| model (config) | params | non-embed | mb | tok/s | MFU | peak VRAM | 10B tokens |
|---|---|---|---|---|---|---|---|
| base_149m (18×768) | 149.1M | 123.9M | 8 | 62K | 69% | 12.1 GiB | 45 h |
| base_250m (20×896) | 210.6M | 181.2M | 4 | 44.6K | 73% | 8.7 GiB | 62 h |
| base_360m (24×1024) | 323.0M | 289.5M | 4 | 30.2K | 74% | 12.3 GiB | 3.8 d |
| base_500m (26×1152) | 440.4M | 402.6M | 2 | 20.3K | 66% | 10.8 GiB | 5.7 d |
| base_620m (28×1280) | 565.0M | 523.1M | 2 | 16.9K | 69% | 13.2 GiB | 6.8 d |

Chosen: **base_336m** (24×1024, 16q/8kv, d_ff 3072, RoPE 500K): 2K × mb4 = 28.6K tok/s (72% MFU), 4K × mb2 = 25.4K tok/s
(78%), 12.5 GiB peak in the benchmark, 13.8 GiB in the live run. The next microbatch up spills in every case (14.6–20 GiB). fp32 weights + grads + Adam cost 16 B/param (8.4 GiB at 565M), so
~565M is the VRAM ceiling and ~440M the last size with a usable microbatch; 323M keeps the 149M model's MFU with headroom.

Conclusions baked into the configs: 2K × mb 8 (M2/M3a/M4), 8K × mb 2 with `loss_chunk_size 4096`
(M3b: 41.9K tok/s, 10.9 GiB; the PaLM-style MFU overstates efficiency at 8K because it counts full attention), 16K × mb 1 with gradient checkpointing (M7). In
production the M2/M3a runs sustain 61–63K tok/s, slightly above the benchmark.

## 2. Data volumes (tokenizer v1)

| Source | Train tokens | Val tokens | Documents | Notes |
|---|---|---|---|---|
| fineweb-edu | 3.04B | 15.2M | 2.90M | sample-10BT files 0–1 |
| fineweb-edu-b | 5.32B | 27.1M | 5.10M | files 0–6, min 16 tokens (M3 mixture) |
| fineweb-edu-long | 730M | 3.6M | 88K | docs ≥ 4096 tokens |
| cosmopedia | 1.62B | 8.1M | 2.25M | files 0–5 (was 538M / files 0–1 until 2026-09-16 20:45; old shards `cosmopedia-v1`) |
| finemath | 1.36B | 6.6M | 926K | files 0–8 (was 455M / files 0–2; old shards `finemath-v1`) |
| python-edu | 589M | 2.9M | 1.23M | 1.3M files fetched from Software Heritage (was 140M / 248K files until 2026-09-16; the old shards are `python-edu-v1`) |
| stack-edu-shell | 505M | 2.5M | 560K | 560K scripts from Software Heritage (was 51M / 60K; old shards `stack-edu-shell-v1`) |
| tinystories | 463M | 4.6M | 2.10M | M1 only |
| fineweb-edu-10bt | 10.07B | 51.5M | 9.66M | all 14 files of sample-10BT, min 16 tokens (second base) |
| smoltalk-chat | 225M | 31.3M | 285K | the 5 SmolTalk SFT sets re-laid as a pretraining source (chat format, no mask) |
| tool-chat | 7.5M | 0.5M | 67K | gsm8k-tools + synthetic-reasoning-tools + metamathqa-tools + synthetic-multiturn-tools as a pretraining source |
| tool-chat-v2 | 36.1M | 2.9M | 225K | the same four plus `synthetic-python-tools`; phase 2's tool source (1.5% of 2.5B ≈ 0.96 epochs) |
| synth-retrieval | 400M | 2.0M | 79K | regenerated 2026-09-16 with seed 7 (was 150M; old shards `synth-retrieval-v1`);  templated retrieval docs (needle facts in real text 70%, key-value ledgers 30%), 512–16K tokens, log-uniform; context curriculum only |

SFT shards (`C:\slm-data\sft\v1`): smol-magpie-ultra 121K examples / 162M tokens (88% targets,
16K dropped for length), openhermes-100k 94K / 36M, systemchats-30k 34K / 20M, smol-constraints 34K /
7M, everyday-conversations 2.3K / 0.4M, metamathqa-reasoning 44K / 10M, gsm8k-reasoning 7.4K / 1.4M,
synthetic-reasoning 40K / 3.2M, synthetic-multiturn-tools 19K conversations / 2.8M (2–4 turns, Python calls reusing the session variable),
synthetic-python-tools 158K train conversations / 28.4M tokens + 11.8K val / 2.4M (grammar-generated, 259K tool
calls; feature coverage of the call spans: loops 77%, list ops 61%, string methods 20%, declared-function calls 11.0%,
`def` 11.0%, `math.` 6.6%; hold-out families `pipeline.dict` and `declared.distance` are val-only),
and 4096-token rebuilds of the five SmolTalk sets (`*-4k`: magpie-ultra keeps 134K conversations vs 121K at 2048). numina-cot-100k (105K / 54M) is prepared but unused. Raw downloads
total 20 GB, tokenized shards 21 GB, SFT shards 1 GB.

Tokenizer: 32,768 ids, sha256 `c2a7b5dbd660944b79fd5934b919dec4d22cb170cff9e5b68d902ff03433ac7e`.

## 3. Runs

| Run | Stage | Init | Tokens | Wall | Final train / val loss | Notes |
|---|---|---|---|---|---|---|
| m1_tinystories_26m | pretrain, 26M | random | 600M | 52 min (192K tok/s) | 1.336 / 1.391 (ppl 4.0) | Coherent stories; diagnostics clean (0% dead units, all layers useful) |
| m2_base_149m | pretrain, 149M, 2K, WSD | random | 1.00B | 4.6 h (60.7K tok/s) | 3.060 / 3.051 | Pre-decay snapshot `snap_800M.pt` (val 3.20) seeds M3a |
| m3_base_stable_149m | pretrain, constant LR (stable phase) | m2 snap_800M | 3.40B | 15.2 h (62.2K tok/s) | 2.768 / 2.878 | Finished 09-14 02:23; val 3.168 → 2.878 with the LR still flat (decay happens in M3b); weights have seen 4.2B tokens |
| m3_base_8k_149m | pretrain, 8K context, long-doc mixture, WSD decay (last 60%) | m3a final | 800M | 5.4 h (41.3K tok/s) | 2.683 / 2.693 | **The base checkpoint** (weights have seen 5.0B tokens). Val is on the 8K mixture, so not comparable to the 2K numbers; decay took it 2.82 → 2.69. GPU peak 72 °C, no throttle warnings |
| m7_ctx8k_retrieval_149m | context curriculum stage 1: 8K, 15% retrieval docs | m3b final (base) | 600M | 4.1 h (41K tok/s) | 2.53 / 2.586 (mixture) | needle effective 1K → 6K; 8K depth 0–0.1 still fails (see needle tables); pretraining val 2.729 → 2.709 |
| m4_sft_149m | instruct SFT | m3b final (base) | 450M (2 epochs) | 2.1 h (62.8K tok/s) | 1.434 / 1.660 | Pretraining-mixture val 2.81 → 2.89 (drift +0.08 nats) |
| m5_reasoning_149m | reasoning SFT | m4 final | 45M (3 epochs) | 12 min | 0.500 / 0.525 | Pretraining val 2.96 → 2.99 |
| m6_rl_arith_149m | GRPO stage A (arith1/arith2) | m5 final | 200 steps | 11 min | held-out acc 0.53 → 0.58 | KL 0.005, length 32, no malformed; resumed once at step 125 |
| m6_rl_multi_149m | GRPO stage B (arith2/arith2mul/arith_multi/algebra/word) | m6 A final | 300 steps | 21 min | held-out acc 0.37 → 0.39 | KL 0.012, 38% of groups without signal (arith_multi is all-zero) |
| m7_ctx16k_149m | context extension 8K → 16K (YaRN ×2), long-doc mixture | m3b final (base) | 200M | 2.3 h (24.5K tok/s, mb 1 + grad checkpointing, 4.3 GiB) | 2.70 / 2.712 | Val on the 16K mixture. Short-context check: 2K loss on fineweb-edu-b 3.069 vs base 3.076; lm-eval unchanged (see below) |
| m5_reasoning_tools_149m | reasoning SFT with the Python tool | m4 final | 14M (3 epochs) | 4 min | 0.50 / 0.675 | Tool use 93% of GSM8K answers, 68% of final numbers from a call, tool errors 1%; templated tasks 98–100% except multi-step (34%); GSM8K 0.8% (plans wrong, 21% of attempts loop to the length cap); without the tool it is lost (87% malformed: it expects the tool) |
| m6_rl_gsm_tools_try1_149m | GRPO with the tool on GSM8K-train (attempt 1) | m5 tools final | 190 of 400 steps (reboot) | 2.8 h | held-out 9% → 12% at step 175 | Collapsed from ~step 110: entropy 0.8 → 5, KL 0 → 0.28, garbage tokens; kl_coef 0.01 too weak. Archived; attempt 2 uses kl 0.05, lr 1e-6, collapse guards |
| m6_rl_gsm_tools_149m | GRPO with the tool on GSM8K-train (attempt 2: kl 0.05, lr 1e-6, guards) | m5 tools final | 222 steps (KL guard fired at 0.159) | 2.6 h | held-out 9% → 11.5% (best step 175) | No collapse (entropy 0.4–1.1); GSM8K with tool 1.3% (from 0.8%), malformed 18% (from 30%), tool use 94%, answers-from-tool 78%; all templated tasks 100% |
| m7_ctx8k_retrieval2_149m | context curriculum stage 1b: 8K, 20% early-depth retrieval docs | stage 1 final | 300M | 2.0 h (41K tok/s) | 2.53 / 2.609 (mixture) | 8K depth 0 / 0.1: 0% → 12%, 6% → 50%; everything else 100%; effective still 6000 (gate not passed); pretraining val 2.709 → 2.711 |
| m7_ctx8k_retrieval3_149m | context curriculum stage 1c: 8K, 20% of a 6K–8K early-fact source | stage 1b final | 300M | 2.0 h | 2.55 / 2.643 (mixture) | 8K depth 0 / 0.1: 12% → 50%, 50% → 94%; all else 100%; effective still 6000 by the strict gate; 2K loss 3.064 (base 3.076) |
| m7_ctx8k_retrieval4_149m | context curriculum stage 1d: same recipe as 1c, +300M | stage 1c final | 300M | 2.1 h | 2.55 / 2.647 (mixture) | 8K depth 0 = 44% (plateau), depth 0.1 = 94%, all else 100%; effective 6000 by the strict gate. Depth probe: 8K needles fail only inside the first ~2% (≈160 tokens): 50/31/69/69% at depths 0/0.005/0.01/0.02, 94% at 0.05, 100% from 0.1 |
| m8_base_stable_336m | second base, phase 1: 336M, 2K, constant LR | random | 7.5B (finished 09-19 12:48) | 2d 23h 28m | best val 2.6081 @ 7.4B (final 2.6086, ppl 13.6); needle 100% @1K, 95% @2K (n=16); judged overall 2.80 ex-bash | 30K tok/s, 13.8 GiB, 70–78 °C |
| m8_base_4k_336m | second base, phase 2: 4K, WSD decay, chat+tool+retrieval mixed in | m8 phase 1 final | 2.5B (finished 09-20 18:18) | 1d 3h 41m | val 2.3869 (drift vs the phase-1 mixture 2.4666, from 2.6086); needle effective **4096** (min-depth 82.8% at 4K, n=64); HellaSwag 32.8 / ARC-Easy 57.9 / PIQA 66.4 (full); facts 69.1%; judged overall 3.06 (peak 3.46 @1.5B) | 26.4K tok/s, 13.8 GiB |
| m4_sft_rehearsal_149m | instruct SFT (rehearsal on the 1B base) | m2 final | 450M (2 epochs) | 2.1 h | 1.629 / 1.878 | Pretraining-mixture val drifted 3.097 → 3.208 |
| m5_reasoning_rehearsal_149m | reasoning SFT (rehearsal) | m4 rehearsal final | 45M (3 epochs) | 13 min | 0.554 / 0.589 | Pretraining val 3.27 → 3.31 |
| m6_rl_arith_rehearsal_149m | GRPO stage A (rehearsal) | m5 rehearsal final | 200 steps, 259K completion tokens | 11 min | held-out acc 0.33 → 0.48 | KL ≈ 0.02, no malformed completions, no length blow-up |

Validation loss trajectory of the 149M base (2K context, nats per token):

| Tokens | 0.1B | 0.2B | 0.5B | 0.8B | 1.0B (M2 end, after decay) | 1.8B | 2.4B | 2.8B | 3.2B |
|---|---|---|---|---|---|---|---|---|---|
| M2 (WSD) | 4.593 | 3.812 | 3.350 | 3.201 | 3.051 | | | | |
| M3a (constant LR from 0.8B) | | | | 3.168 (at +0.1B) | 3.010 (+1.0B) | 2.946 (+1.8B) | 2.914 (+2.4B) | 2.898 (+2.8B) | 2.882 (+3.2B); 2.878 at +3.4B (end) |

The 336M base at the same token counts (constant LR, 524K-token updates, 100M warmup):

| Tokens | 100M | 200M | 300M | 400M | 500M |
|---|---|---|---|---|---|
| 149M (M2, WSD) | 4.593 | 3.812 | 3.562 | 3.435 | 3.350 |
| 336M (M8 phase 1) | 5.345 | 4.298 | 3.740 | 3.501 | 3.358 |

(M3a counts tokens from its own start; add 0.8B for tokens seen by the weights. The final decay
happens in M3b, so M3a's loss is a stable-phase loss and will drop further at decay.)

## 4. Evaluations

lm-evaluation-harness, accuracy (acc_norm in parentheses):

| Checkpoint | HellaSwag | ARC-Easy | PIQA | Limit |
|---|---|---|---|---|
| m2_base_149m final (1B tokens) | 27.6 (29.0) | 47.1 (41.5) | 60.3 (58.7) | full |
| m3_base_8k_149m final (5.0B tokens, the base) | 29.3 (32.7) | 51.6 (45.5) | 64.0 (62.5) | full |
| m3_base_8k_149m final, same 2000-sample limit | 32.6 (39.1) | 51.9 (45.6) | 64.0 (62.5) | 2000 |
| m4_sft_149m final | 33.1 (39.6) | 50.0 (46.7) | 63.5 (61.5) | 2000 |
| m7_ctx16k_149m final (16K extension of the base) | 32.6 (39.6) | 52.0 (46.8) | 63.8 (62.6) | 2000 |
| m4_sft_rehearsal_149m final | 31.7 (36.5) | 43.7 (41.0) | 61.8 (59.4) | 2000 |
| **m8_base_4k_336m final (10.0B tokens, the second base)** | **32.8 (38.3)** | **57.9 (52.5)** | **66.4 (65.7)** | full |

The first 2000 HellaSwag samples are easier than the full set, so compare rows with the same limit only.
The 16K extension costs nothing on short-context tasks (all three within ±1 point of the base at the same limit).

Reference points: random is 25% / 25% / 50%; GPT-2 small (124M, ~10B tokens) scores about 29–31 on
HellaSwag; SmolLM-135M (600B tokens) about 42.

Reasoning benchmark (`slm.eval.reasoning`, greedy, held-out prompts; n = 100 per task, arith1 n = 10,
GSM8K test n = 200):

| Checkpoint | arith1 | arith2 | arith2mul | arith_multi | algebra | word | GSM8K | malformed (GSM8K) |
|---|---|---|---|---|---|---|---|---|
| m5_reasoning_149m (real base) | 40% | 50% | 42% | 3% | 34% | 68% | 2.0% | 15% |
| m6_rl_arith_149m (stage A, 200 steps) | 70% | 61% | 49% | 2% | 37% | 67% | 1.5% | 8% |
| m5_reasoning_tools_149m (Python tool, greedy; arith1 n=10) | 100% | 100% | 100% | 34% | 100% | 98% | 0.8% | 30% |
| m5_reasoning_tools_149m, tool disabled | 10% | 1% | 2% | 0% | 2% | 0% | 0.5% | 87% |
| m6_rl_gsm_tools_149m best.pt (step 175), Python tool | 100% | 100% | 100% | 100% | 100% | 100% | 1.3% | 18% |
| m6_rl_multi_149m (stage B, +300 steps) | 50% | 61% | 48% | 3% | 32% | 67% | 2.0% | 16% |
| m5_reasoning_rehearsal_149m | 60% | 25% | 22% | 0% | 31% | 58% | 2.5% | 9.5% |
| m6_rl_arith_rehearsal_149m (200 GRPO steps on arith1/arith2) | 80% | 44% | 36% | 0% | 29% | 55% | 2.5% | 8.5% |

On the real base, stage A took arith2 from 50% to 61% and arith2mul from 42% to 49%; stage B (multi-step,
algebra, word) moved little because arith_multi yields almost no correct samples to learn from (38% of
groups carry no signal) — a curriculum gap to fix with easier intermediate tasks or partial credit.
GSM8K stays at ~2%: beyond this model at this stage. Rehearsal (1B base): arith2 25% → 44% after stage A.

Long-context needle retrieval (`slm.eval.long_context`, v2: haystack = continuous *validation* text with
document boundaries removed, 7 depths, n = 16 per cell, greedy, 6-digit secret). The first version used a
loop of 8 filler sentences and n = 4; it overstated retrieval by ~20 points at 2K (99% vs 79% for the base) and
its numbers are superseded by the tables below.

**2K-trained model (m3_base_stable_149m, RoPE table 8192):**

| Length | d=0.0 | d=0.1 | d=0.25 | d=0.5 | d=0.75 | d=0.9 | d=1.0 | mean | min |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 100% | 94% | 100% | 100% | 100% | 94% | 100% | 98% | 94% |
| 1024 | 100% | 94% | 94% | 81% | 100% | 94% | 100% | 95% | 81% |
| 2048 | 56% | 38% | 56% | 62% | 88% | 94% | 100% | 71% | 38% |
| 3072 | 0% | 0% | 0% | 19% | 81% | 81% | 94% | 39% | 0% |
| 4096 | 0% | 0% | 0% | 0% | 6% | 69% | 94% | 24% | 0% |
| 8000 | 0% | 0% | 0% | 0% | 0% | 0% | 94% | 13% | 0% |

Effective context (min over depths ≥ 80%): **1024**.

**8K base (m3_base_8k_149m):**

| Length | d=0.0 | d=0.1 | d=0.25 | d=0.5 | d=0.75 | d=0.9 | d=1.0 | mean | min |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 94% | 100% | 88% | 94% | 94% | 94% | 100% | 95% | 88% |
| 1024 | 100% | 94% | 94% | 81% | 94% | 100% | 100% | 95% | 81% |
| 2048 | 94% | 56% | 62% | 69% | 75% | 94% | 100% | 79% | 56% |
| 4096 | 6% | 12% | 38% | 50% | 62% | 94% | 100% | 52% | 6% |
| 6000 | 0% | 0% | 0% | 44% | 88% | 81% | 100% | 45% | 0% |
| 8000 | 0% | 0% | 0% | 6% | 75% | 69% | 94% | 35% | 0% |

Effective context (min over depths ≥ 80%): **1024**.

**16K extension, first attempt (m7_ctx16k_149m, YaRN ×2 + 200M tokens):**

| Length | d=0.0 | d=0.1 | d=0.25 | d=0.5 | d=0.75 | d=0.9 | d=1.0 | mean | min |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | 81% | 94% | 88% | 81% | 81% | 94% | 100% | 88% | 81% |
| 4096 | 62% | 69% | 44% | 62% | 69% | 94% | 94% | 71% | 44% |
| 8000 | 6% | 12% | 6% | 38% | 81% | 100% | 100% | 49% | 6% |
| 12000 | 0% | 0% | 0% | 31% | 62% | 50% | 100% | 35% | 0% |
| 16000 | 0% | 0% | 0% | 0% | 38% | 81% | 100% | 31% | 0% |

Effective context (min over depths ≥ 80%): **2048**.

**Context curriculum stage 1 (m7_ctx8k_retrieval_149m: base + 600M tokens at 8K with 15% templated retrieval docs):**

| Length | d=0.0 | d=0.1 | d=0.25 | d=0.5 | d=0.75 | d=0.9 | d=1.0 | mean | min |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 2048 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 4096 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 6000 | 88% | 100% | 100% | 100% | 100% | 100% | 100% | 98% | 88% |
| 8000 | 0% | 6% | 94% | 100% | 100% | 100% | 100% | 71% | 0% |

Effective context **6000** (from 1K). The remaining failures are needles in the first ~10% of a full 8K context, and
they are near-misses (556423 for 556413): the model finds the needle but copies the last digits wrong at maximum
distance. Short context unchanged (pretraining-mixture val 2.7286 → 2.7085 over the run). Stage 1b targets those
cells with 3K–8K retrieval documents whose facts sit in the first 15% half of the time.

**Stage 1d (m7_ctx8k_retrieval4_149m, +300M, same recipe as 1c):** 8K: 44% / 94% / 100% / 100% / 100% / 100% / 100%, mean 91%.
The depth-0 cell no longer moves (50% → 44%). A finer probe at 8K (n = 16, depths 0, 0.005, 0.01, 0.02, 0.05, 0.1) gives
50% / 31% / 69% / 69% / 94% / 100%. A second probe across lengths (depths 0, 0.005, 0.02, 0.05) shows the same shallow
needles at 100% for 4K and 6K, 88–100% at 7K, 62–94% at 7.5K and 38–88% at 8K: the failure is the far edge of the
trained window (relative distance ≥ ~7,500 tokens), not a property of the first tokens. Full-window distances are rare in
packed 8K rows (a long document only spans the whole row when it starts at a row boundary), which is why targeted data
plateaued. Strict effective context: **7K**; 8K for anything outside the first ~2% of the window. Extending the window
(16K rows) makes 7–8K distances interior and common, which is how stage 1 fixed the base's 4K edge.

**Stage 1c (m7_ctx8k_retrieval3_149m, +300M tokens, 6K–8K documents with 80% early facts):** 100% at every depth for
1K–6K; 8K: 50% / 94% / 100% / 100% / 100% / 100% / 100%, mean 92%, min 50%. Only the depth-0 cell (needle in the first ~30
tokens after `<|bos|>`) is left; its trajectory over stages 1 → 1b → 1c is 0% → 12% → 50%, so stage 1d repeats the recipe.

**Stage 1b (m7_ctx8k_retrieval2_149m, +300M tokens with early-depth retrieval docs):** 100% at every depth for 1K–6K;
8K: 12% / 50% / 100% / 100% / 100% / 100% / 100% (depths 0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0), mean 80%, min 12%. Same near-miss
signature at depth 0–0.1 (248967 for 248955). Stage 1c (6K–8K documents, 80% early facts) is the last targeted push.

Reading: none of the checkpoints retrieves reliably beyond ~1K tokens of real text, including at their own
training length; failures are the model ignoring the needle (it answers with a number from the text or
`10000000000`), not misreading it. Long-document training helps (the 16K run beats the base at every shared
length), so the fix is targeted training at each length with a gate before extending — see the context
curriculum in `roadmap.md`. Filler-haystack control on the base: 99% at 2K, 57% at 8K.

Factual-recall probe (`slm.eval.facts`, 194 items, greedy, whole-word match):

| Checkpoint | mode | total | capitals | science | history | culture | geography | units | language |
|---|---|---|---|---|---|---|---|---|---|
| m3_base_8k_149m (the 149M base) | completion | 43.3% | 64% | 40% | 57% | 25% | 40% | 10% | 40% |
| m4_sft_149m | chat | 45.9% | 70% | 42% | 60% | 15% | 40% | 25% | 35% |

## 5. Diagnostics

`scripts/diagnose.py` on the final checkpoints (validation batches from fineweb-edu / tinystories):

- m1_tinystories_26m: base loss 1.441; 0% dead SwiGLU units in every layer; skipping any single
  block raises loss (layer 0 by +3.70, others +0.21 to +0.54).
- m3_base_8k_149m (the base, on fineweb-edu-b val at 2K): base loss 3.076; 0% dead units; layer-skip deltas
  +5.29 (layer 0), +0.69 (layer 1), then +0.03 to +0.28 with the last layer at +0.84.
- m7_ctx16k_149m at 2K context: base loss 3.069 (no ablation run), 0% dead units — no short-context regression.
- m2_base_149m: base loss 3.354; 0% dead units; layer-skip deltas +5.29 (layer 0), then +0.01 to +0.22
  through the stack with the last layer at +0.67, so every layer contributes and depth is not wasted.

## 6. Sample quality (qualitative)

- M1 (26M, TinyStories): fluent multi-sentence stories with consistent characters.
- M2/M3a (149M): fluent encyclopedic prose; factual prompts are hit or miss ("The capital of France
  is" continues with "located in the heart of the country" under greedy decoding and "Paris" when
  sampled); Python prompts produce syntactically valid but wrong functions; greedy decoding loops.
- M5/M6: well-formed think spans and `#### answer` lines nearly always; arithmetic within the trained
  digit range is reliable after RL, multi-step expressions are not.

## 7. History of numbers that changed decisions

- mb 16 at 2K "worked" in the first benchmark at 1K tok/s: that was the host-memory spill, which is why
  every config keeps peaks near 11 GiB.
- Eager mode at 9.5K tok/s vs compiled 58K: `torch.compile` via `triton-windows` is mandatory.
- cuDNN attention during generation: samples stalled a training run; decoding now uses the efficient
  backend.

## 7a. Needle sweep cost (`slm.eval.needle_sweep`, 336M, n=64, 5 depths, 1K+2K = 640 prompts per snapshot)

| device | per snapshot | notes |
|---|---|---|
| CPU, 24 threads, fp32, below-normal priority | 1,769 s | beside the live run; training throughput unchanged (29.9K tok/s) |
| GPU, bf16, 32K-token batches | ~35 s (estimate from in-run eval timings) | GPU must be free |

## 8. Judged quality (LLM-scored prompt suite; `docs/quality_eval.md`)

35 prompts (facts, prose, Python, bash, arithmetic, pattern continuation, definitions, narrative, why-questions),
greedy outputs from every milestone snapshot generated on CPU beside the live run, scored blind by Claude Sonnet
on correctness / coherence / task (1-5 each; overall = mean). Suite v1, rubric v1. Numbers are means over 35
items, so a single prompt moves the overall by about 0.03 and a category (2-8 prompts) by 0.1-0.5: read the
trend, not the last digit.

### Second base, phase 1 (`m8_base_stable_336m`, constant LR, 2K rows) — complete, 30 snapshots to 7.5B (overall excludes bash from 2026-09-19; the bash column is still scored)

| tokens | overall | correctness | coherence | task | facts | pattern | python | prose | arithmetic | bash |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25B | 1.12 | 1.00 | 1.19 | 1.19 | 1.17 | 1.25 | 1.17 | 1.17 | 1.00 | 1.00 |
| 0.50B | 1.40 | 1.12 | 1.59 | 1.47 | 1.38 | 1.42 | 1.39 | 1.42 | 1.33 | 1.33 |
| 0.75B | 1.74 | 1.38 | 1.94 | 1.91 | 1.75 | 1.58 | 2.17 | 1.50 | 1.42 | 1.22 |
| 1.00B | 1.83 | 1.31 | 1.97 | 2.22 | 2.04 | 1.67 | 1.83 | 1.67 | 1.67 | 1.33 |
| 1.25B | 1.80 | 1.41 | 2.00 | 2.00 | 1.88 | 1.58 | 1.78 | 2.00 | 1.58 | 1.44 |
| 1.50B | 2.13 | 1.84 | 2.03 | 2.53 | 2.67 | 2.25 | 2.00 | 2.00 | 1.42 | 1.33 |
| 1.75B | 2.16 | 1.84 | 2.19 | 2.44 | 2.00 | 2.42 | 2.67 | 2.25 | 1.33 | 1.56 |
| 2.00B | 2.12 | 1.81 | 2.19 | 2.34 | 2.04 | 2.92 | 2.17 | 2.00 | 1.42 | 1.33 |
| 2.25B | 2.06 | 1.88 | 2.06 | 2.25 | 1.92 | 2.50 | 2.89 | 1.83 | 1.58 | 1.22 |
| 2.50B | 2.07 | 1.94 | 2.00 | 2.28 | 2.67 | 1.75 | 2.78 | 1.75 | 1.42 | 1.44 |
| 2.75B | 2.26 | 2.00 | 2.25 | 2.53 | 2.08 | 2.17 | 2.94 | 2.25 | 1.58 | 1.33 |
| 3.00B | 2.31 | 2.19 | 2.19 | 2.56 | 2.62 | 2.17 | 2.50 | 2.08 | 1.33 | 1.22 |
| 3.25B | 2.43 | 2.22 | 2.34 | 2.72 | 2.58 | 3.33 | 2.94 | 1.83 | 1.67 | 1.33 |
| 3.50B | 2.40 | 2.12 | 2.41 | 2.66 | 2.62 | 3.00 | 2.67 | 2.75 | 1.67 | 1.33 |
| 3.75B | 2.30 | 2.25 | 2.09 | 2.56 | 2.79 | 2.42 | 2.61 | 2.17 | 1.25 | 1.22 |
| 4.00B | 2.29 | 2.09 | 2.31 | 2.47 | 2.46 | 3.08 | 2.67 | 2.33 | 1.25 | 1.22 |
| 4.25B | 2.70 | 2.47 | 2.50 | 3.12 | 2.88 | 3.08 | 3.50 | 2.50 | 1.67 | 1.11 |
| 4.50B | 2.79 | 2.69 | 2.50 | 3.19 | 3.46 | 2.92 | 3.78 | 2.50 | 1.42 | 1.22 |
| 4.75B | 2.50 | 2.22 | 2.34 | 2.94 | 2.83 | 3.25 | 2.17 | 2.42 | 1.67 | 1.33 |
| 5.00B | 2.46 | 2.12 | 2.28 | 2.97 | 2.83 | 2.67 | 2.56 | 2.25 | 1.33 | 1.22 |
| 5.25B | 2.34 | 2.47 | 1.97 | 2.59 | 2.46 | 3.33 | 2.17 | 2.25 | 1.17 | 1.11 |
| 5.50B | 2.90 | 2.91 | 2.75 | 3.03 | 3.00 | 2.50 | 3.50 | 2.50 | 1.67 | 1.33 |
| 5.75B | 2.38 | 2.44 | 2.22 | 2.50 | 3.00 | 3.17 | 2.00 | 2.00 | 1.50 | 1.44 |
| 6.00B | 2.83 | 2.75 | 2.75 | 3.00 | 3.17 | 3.00 | 4.00 | 1.83 | 1.17 | 1.22 |
| 6.25B | 2.43 | 2.41 | 2.19 | 2.69 | 3.00 | 2.33 | 1.94 | 2.17 | 1.83 | 1.33 |
| 6.50B | 2.78 | 2.88 | 2.44 | 3.03 | 2.71 | 3.50 | 3.39 | 2.33 | 1.42 | 1.44 |
| 6.75B | 2.83 | 2.78 | 2.66 | 3.06 | 3.33 | 2.67 | 3.17 | 2.50 | 1.75 | 1.22 |
| 7.00B | 2.82 | 2.75 | 2.59 | 3.12 | 3.25 | 3.33 | 2.94 | 2.75 | 1.83 | 1.22 |
| 7.25B | 3.05 | 3.03 | 2.50 | 3.62 | 3.29 | 3.25 | 3.89 | 2.75 | 1.50 | 1.22 |
| 7.50B | 3.07 | 3.00 | 2.53 | 3.69 | 2.79 | 3.92 | 4.00 | 3.17 | 1.33 | 1.11 |

What the judge sees at 3.5B (all 14 checkpoints judged, 490 items): the right first sentence arrives long before
the model learns to stop. `The capital of France is` → "Paris, the capital of France is Paris. The capital of
France is Paris…" scores 5 / 2 / 4. `def fibonacci(n):` is still syntactically broken (1 / 2 / 1). The days of
the week come out in order and then repeat ("…Sunday, and Sunday", 3 / 2 / 3). `arithmetic` and `bash` have not
moved off the floor in 3.5B tokens: a base model at this size does not answer "What is 17 + 25?" from a Q/A
prompt (it restates the equation), and bash prompts drift into prose or number lists. Those two are the headroom
the SFT / tool stages are for, and this table is the baseline they will be compared against.

Reading the curve: overall (ex-bash) rises 1.12 → 3.07 at 7.5B, holding above 2.78 on every snapshot from 6.5B and jumping at 7.25B (task 3.6+, correctness 3.0); earlier there was checkpoint-to-checkpoint wobble of ±0.25 under the constant LR (3.75-4.0B dipped to 2.20, 4.5B jumped to 2.66, 5.0B sits at 2.35), far larger than the ±0.03 judge noise: at a constant learning rate the greedy behaviours (where a loop starts) move between snapshots even as validation loss improves monotonically, so read the trend over several points, and expect the decay phase to settle it;
`correctness` is the slowest rubric (1.0 → 2.0) and `task` the fastest (1.2 → 2.5), i.e. the model learns what
kind of text to produce before it learns to be right. Cost: 53 s of CPU per checkpoint, ~75K judge tokens per
40-item packet, zero measurable training-throughput impact.

### Second base, phase 1: needle sweep at n=64 (`slm.eval.needle_sweep`, 2026-09-19)

Every one of the 30 milestone snapshots re-measured with one fixed haystack draw, 64 samples per cell, five depths,
real-text haystack (48 s per snapshot on the free GPU). Selected rows (mean / worst depth):

| tokens | 1024 | 2048 | 4096 (untrained) | effective |
|---|---|---|---|---|
| 0.50B | 98% / 91% | 79% / 14% | 25% / 0% | 1024 |
| 1.00B | 99% / 98% | 96% / 81% | 44% / 0% | 2048 |
| 2.00B | 99% / 98% | 98% / 92% | 24% / 0% | 2048 |
| 2.25B | 99% / 97% | 83% / 28% | 20% / 0% | 1024 |
| 3.00B | 100% / 98% | 98% / 95% | 22% / 0% | 2048 |
| 4.00B | 100% / 98% | 91% / 66% | 20% / 0% | 1024 |
| 4.50B | 100% / 100% | 100% / 100% | 28% / 0% | 2048 |
| 6.00B | 100% / 98% | 98% / 91% | 21% / 0% | 2048 |
| 6.75B | 97% / 94% | 76% / 38% | 19% / 0% | 1024 |
| 7.00B | 100% / 100% | 95% / 86% | 27% / 0% | 2048 |
| 7.25B | 100% / 98% | 89% / 58% | 21% / 0% | 1024 |
| 7.50B | 99% / 98% | 96% / 83% | 26% / 0% | 2048 |

1K is solid from 500M on. 2K, the trained window edge, reaches 96% at 1B and then fluctuates from snapshot to
snapshot: 83% / 28% at 2.25B, 76% / 38% at 6.75B, 89% / 58% at 7.25B, 96% / 83% at the final. With 64 samples per
cell those dips are not draw noise (the in-run n=16 tracker's dips at 3.3B were: the sweep reads 92% / 86% there);
retrieval at the last trained position moves with the constant-LR weight noise the way the judged scores do. 4K,
never trained, sits at 20-44% mean with a 0% worst depth throughout (RoPE extrapolation retrieves shallow needles
only); this is the baseline phase 2's 4K training starts from.

### The 149M chain, stage by stage (retroactive, 2026-09-18; overall excludes bash from 2026-09-19)

Every stage's snapshots were generated on CPU and judged the same way (M3a every third snapshot). "last" is the
stage's final checkpoint, which is what the next stage started from.

| stage (run) | ckpts | first → last | best | correct | coherent | task | arithmetic | python | pattern | narrative |
|---|---|---|---|---|---|---|---|---|---|---|
| base 2K, cosine (`m2_base_149m`) | 10 | 1.20 → 1.86 | 1.86 @ 1.0B | 1.44 | 2.00 | 2.16 | 1.42 | 2.39 | 1.50 | 2.00 |
| base 2K stable (`m3_base_stable_149m`) | 12 | 1.61 → 2.48 | 2.48 @ 3.4B | 2.16 | 2.34 | 2.94 | 1.17 | 2.44 | 2.75 | 3.67 |
| base 8K + decay (`m3_base_8k_149m`) | 5 | 2.55 → 2.40 | 2.56 @ 700M | 2.25 | 2.22 | 2.72 | 1.50 | 2.61 | 3.17 | 2.67 |
| chat SFT (`m4_sft_149m`) | 2 | 3.17 → 2.70 | 3.17 @ 225M | 2.12 | 2.91 | 3.06 | 2.25 | 2.83 | 2.00 | 2.83 |
| reasoning SFT (`m5_reasoning_149m`) | 3 | 2.50 → 2.40 | 2.50 @ 15M | 1.97 | 2.75 | 2.47 | 2.92 | 1.94 | 2.33 | 1.83 |
| tool reasoning SFT (`m5_reasoning_tools_149m`) | 3 | 2.88 → 2.59 | 2.88 @ 5M | 2.34 | 2.69 | 2.75 | 3.25 | 1.78 | 1.33 | 2.17 |
| 8K retrieval curriculum (`m7_ctx8k_retrieval_149m`) | 6 | 2.37 → 2.49 | 2.49 @ 600M | 2.38 | 2.38 | 2.72 | 1.42 | 2.44 | 2.50 | 3.17 |
| GRPO on GSM8K with tools (`m6_rl_gsm_tools_149m`) | 9 | 2.59 → 2.67 | 2.70 @ step 200 | 2.38 | 2.78 | 2.84 | 4.08 | 2.06 | 1.33 | 1.83 |

What the chain says, and what it changes for the 336M post-training:

- **Chat SFT is the big step** (2.28 → 3.03 at its midpoint): `task` and `coherence` jump because the model
  learns to answer and stop; `correctness` barely moves (2.14 → 2.06), which is the base model's knowledge showing
  through. **Its second half is worse than its first** (3.03 → 2.61; `pattern` 2.92 → 2.00, `evens` continued
  correctly at 225M and merely restated at 450M). Validation loss picked `best.pt` at 380M. Pick SFT checkpoints
  by judged quality, and evaluate mid-run, not only at the end.
- **Reasoning SFT is a regression on everything but arithmetic** (2.61 → 2.29): narrow math data, and the
  marker-by-default defect (the checkpoint answers "What is the capital of France?" with `#### 24`). The tool
  variant regresses less and lifts `arithmetic` further (3.25) because the calls give real numbers. Both were
  trained before the 50/50 answer-style fix; the 336M chain uses the corrected sets, and this suite will show
  whether that closes the gap.
- **RL moves exactly one category.** From the tool-SFT final (2.49, arithmetic 3.25) GRPO takes `arithmetic`
  to 4.08 and leaves everything else within noise (215 of 315 RL items were byte-identical to another step's
  output, so most of the suite never changed under RL). Overall 2.49 → 2.54: the reward did what it rewarded.
- **Context work was cheap on this axis**: the 8K phase ends 0.09 below the 2K stable end, the retrieval
  curriculum ends 0.09 above where it started. Consistent with the ≤5% rule.
- **Per token, the 336M base is ahead from ~1B on**: 2.31 at 3.5B vs the 149M chain's 2.2 at the same
  cumulative count and 2.37 at 4.2B.

## 9. Packing diagnostic: does a row that starts mid-document hurt? (`slm.eval.row_positions`, 2026-09-19)

Phase-1 final checkpoint, 256 validation rows of 2048 tokens from `fineweb-edu-10bt`, packed exactly as training
packs them (every row started mid-document; 49% of tokens sit before the row's first `<|bos|>`), plus a control
pass of 256 rows each aligned to start at a document boundary. Per-token loss in nats.

| position in row (tail) / distance into document (in-doc) | tail: truncated prefix | in-doc: full prefix | doc after other text (packed) | doc at row start (aligned control) |
|---|---|---|---|---|
| 0 | 6.26 | | | |
| 1-3 | 4.95 | 5.02 | 5.02 | 5.05 |
| 4-15 | 3.74 | 3.45 | 3.45 | 3.51 |
| 16-63 | 3.17 | 2.97 | 2.97 | 3.03 |
| 64-255 | 2.91 | 2.86 | 2.86 | 2.90 |
| 256-1023 | 2.81 | 2.82 | 2.82 | 2.86 |

- **Truncated prefix**: costs ~0.3 nats in the first 16 positions of a row, ~0.2 through 64, ~0.05 through 256, nothing
  beyond; the curves are smooth and converge. Tail tokens average 2.82 vs 2.84 for all tokens (mid-document text is
  easier than document openings). The penalty lives in ~3% of tokens and is the short-context task itself.
- **Cross-document attention**: a document that starts after other text in the row scores 0.03-0.06 nats *lower*
  at every distance to 1023 than one that starts at the top of a row with nothing before it. Preceding text from an
  unrelated document does not hurt at this scale; the small positive delta is within what two 256-row samples can
  resolve. No case for boundary-aligned rows; document masking remains a next-base benchmark, not a fix for a
  measured problem.

### Second base, final measurements (`m8_base_4k_336m` final.pt, 10.0B cumulative tokens, 2026-09-20)

| measure | 336M base (10.0B) | 149M base (5.0B) |
|---|---|---|
| HellaSwag acc (acc_norm), full | 32.8 (38.3) | 29.3 (32.7) |
| ARC-Easy | 57.9 (52.5) | 51.6 (45.5) |
| PIQA | 66.4 (65.7) | 64.0 (62.5) |
| facts probe (194 items, completion) | 69.1% | 43% |
| needle effective context (n=64, min over 5 depths >= 80%) | **4096** | 8000 at 2K-6K, 7K strict |
| val loss on the phase-1 pretraining mixture | 2.4666 | 2.8778 |

Needle by length at n=64 (mean / worst depth): 1024 100/100, 2048 99.6/96.9, 3072 98.7/96.9, 4096 94.0/82.8.
Facts by category: capitals 88.6%, history 83.3%, science 67.5%, culture 65%, language 60%, geography 55%, units 35%.

The 336M base beats the 149M base on every benchmark and by 26 points on the facts probe, and it is a genuine
4K model by the strict gate. Reference points: random is 25/25/50; GPT-2 small (124M, ~10B tokens) scores ~29-31
on HellaSwag; SmolLM-135M (600B tokens) ~42.

## 10. M9 post-training on the second base (`m9_sft_336m`, `m9_tool_336m`, 2026-09-21)

### Stage A — chat SFT (200M tokens, 5 SmolTalk sets at 4K, mandatory but empty think span)

| measure | base (`m8_base_4k_336m` final) | stage A final |
|---|---|---|
| judged overall | 3.06 | **3.47** |
| judged correctness / coherence / task | 3.41 / 2.72 / 3.06 | 3.03 / **3.56** / **3.81** |
| tool misfire rate (28 eligible prompts) | 0.00 | 0.00 |
| HellaSwag (acc_norm) / ARC-Easy / PIQA, limit 2000 | — | 35.1 (42.6) / 58.4 / 66.6 |
| facts probe (194 items) | 69.1% | 70.1% |
| needle effective context (n=64, strict gate) | 4096 | **4096** |

Stage A did what chat SFT is for: coherence +0.84 and task-following +0.75, retrieval and knowledge intact.
Correctness fell 0.38, which is the base's decay trade being partly given back. Judged quality wobbles
+-0.15 across its 8 checkpoints with no trend after 50M, so 200M tokens was more than this stage needed.

### Stage B v1 — reasoning and Python tools (120M tokens, 34% chat rehearsal) — **FAILED its gate**

| measure | stage A | stage B v1 final | best stage B ckpt (105M) |
|---|---|---|---|
| judged overall | **3.47** | 3.00 | 3.16 |
| correctness / coherence / task | 3.03 / 3.56 / 3.81 | 2.38 / 3.41 / 3.22 | 2.53 / 3.50 / 3.44 |
| **tool misfire rate** | **0.00** | **0.429** | 0.429 |
| ARC-Easy / PIQA / facts probe | 58.4 / 66.6 / 70.1% | 56.9 / 66.8 / 69.6% | — |
| needle effective context | 4096 | **3072** (70.3% worst depth at 4K) | — |

Judged categories, stage A -> stage B v1 final:

| arithmetic | python | prose | narrative | definition | facts | qa | pattern |
|---|---|---|---|---|---|---|---|
| 2.83 -> **4.50** | 4.22 -> 3.94 | 3.25 -> 3.17 | 2.67 -> **3.17** | 4.17 -> 3.17 | 3.08 -> 2.46 | 3.50 -> **1.50** | 4.00 -> **1.58** |

The targeted skill was learned and everything else paid for it. The diagnosis is one number: **every category
that emitted a `<|python_call|>` regressed, and every category that did not held or improved.** The misfire
rate was 0.500 at the first 15M-token checkpoint and never fell below 0.429 in 120M tokens — flat, so no amount
of further training was going to fix it. `What is the capital of France?` produced an invented
`city_population(...)` call (NameError) and the answer "That would be Aldershaw."

The tool mechanism itself is sound where the prompt resembles the training grammars
(`slm.eval.reasoning --tools`, stage B v1 final): arith1 1.00 acc at 0.90 tool use, arith2 0.94 at 0.81,
arith2mul 0.82 at 0.70, arith_multi 0.99 at 1.00, word 0.89 at 0.41, all with a 0.00 tool-error rate.
GSM8K is the opposite failure — 4.5% accuracy at a 0.17 tool-use rate and 12% malformed — because real word
problems are out of distribution for those grammars. Both failures are the same thing: **the model routes on
surface form, not on need.**

Cause: v1's mixture contained "empty think, then answer" (chat rehearsal) and "think with a tool call, then
answer" (tools) and no third case. Once the model moved off empty think spans, the only non-empty span it knew
how to write contained code — and short, data-free questions look exactly like the tool grammars' prompts.
`slm/data/direct_think.py` and `configs/train/m9_tool2_336m.yaml` are the fix; `tool_misfire` is now reported
per checkpoint by `slm.eval.quality` so routing is visible during a run instead of after it.

### Stage B v3 — conversation-balanced mixture (`m9_tool3_336m`, 120M tokens) — best of the three, still short

| measure | stage A | v1 | **v3** |
|---|---|---|---|
| judged overall | **3.47** | 3.00 | 3.24 |
| correctness / coherence / task | 3.03 / 3.56 / 3.81 | 2.38 / 3.41 / 3.22 | **3.06** / 3.22 / 3.44 |
| tool misfire (mean over checkpoints) | 0.000 | 0.459 | **0.259** |
| needle effective context (n=64, 7 depths) | **4096** (84% worst) | 3072 (70%) | 3072 (75%) |
| HellaSwag (norm) / ARC-Easy / PIQA | 35.1 (42.6) / 58.4 / 66.6 | 35.1 (42.4) / 56.9 / 66.8 | 34.8 (**42.9**) / 57.6 / 66.8 |
| facts probe | 70.1% | 69.6% | **71.1%** |

Judged categories, stage A -> v1 -> v3:

| arithmetic | definition | qa | facts | narrative | prose | python | pattern |
|---|---|---|---|---|---|---|---|
| 2.83 -> 4.50 -> **4.58** | 4.17 -> 3.17 -> **4.33** | 3.50 -> 1.50 -> **4.00** | 3.08 -> 2.46 -> 2.92 | 2.67 -> 3.17 -> 3.17 | 3.25 -> 3.17 -> 2.92 | 4.22 -> 3.94 -> **3.44** | 4.00 -> 1.58 -> **1.67** |

`slm.eval.reasoning --tools`, accuracy / tool-use rate:

| | arith2 | arith2mul | word | algebra | gsm8k |
|---|---|---|---|---|---|
| v1 | 0.94 / 0.81 | 0.82 / 0.70 | 0.89 / 0.41 | **0.44** / 0.00 | 0.045 / 0.17 |
| v3 | **0.99 / 0.94** | **0.92 / 0.91** | **0.93 / 0.64** | **0.22** / 0.02 | 0.060 / **0.28** |

The `math_tool : math_no_tool` rebalance (0.8 : 1 -> 2.0 : 1) did what it was for — every tool-use rate rose and
the accuracies with it — but cutting the prose-reasoning math sets halved algebra, which uses no tool at all
(0.02 rate). Mean accuracy is unchanged (0.730 vs 0.732) because the two cancel.

Two measurement corrections came out of this run, both of which changed conclusions:

- **The misfire metric undercounted.** It counted only spans the tool loop executed, and the loop honours
  `<<...>>` markup solely inside the think span, so an answer that reaches for the sandbox — or runs out of
  `max_new_tokens` mid-program — scored zero calls. That hid 6 of v3's 8 misfires at 15M (0.179 reported
  against 0.286 real). `_ran_code` now checks the markup opener too.
- **The in-run needle eval is optimistic.** It uses 5 depths; the full eval uses 7. v3's 30M snapshot read
  "effective 4096" in-run and 3072 under `slm.eval.long_context`. Do not read a gate from the in-run number.

The 30M snapshot scored judged 3.58, above stage A, but measured identically to the final on every hard
number (needle 3072, ARC-E 57.5, facts 70.6%, algebra 0.19, gsm8k 0.03). With judged scores wobbling +-0.25
across eight checkpoints, that peak is selection noise, not a better model.

Where the remaining gap sits (n = 32 scored items, bash excluded): recovering `pattern` is worth +0.29 of
overall and `python` +0.15 — 84% of the 0.23 shortfall. Recovering all four below-stage-A categories would
give 3.76.

### Stage B v4 — the stage B output (`m9_tool4_336m`, 120M tokens, 2026-09-21)

Judged, final checkpoint of each run:

| | stage A | v1 | v3 | **v4** |
|---|---|---|---|---|
| overall | 3.47 | 3.00 | 3.24 | **3.58** |
| correctness / coherence / task | 3.03 / 3.56 / 3.81 | 2.38 / 3.41 / 3.22 | 3.06 / 3.22 / 3.44 | **3.22 / 3.84** / 3.69 |
| tool misfire (mean over 8 checkpoints) | 0.000 | 0.459 | 0.259 | 0.255 |
| arithmetic | 2.83 | 4.50 | 4.58 | 4.42 |
| python | 4.22 | 3.94 | 3.44 | **4.22** |
| pattern | 4.00 | 1.58 | 1.67 | 2.50 |
| qa / narrative / prose | 3.50 / 2.67 / 3.25 | 1.50 / 3.17 / 3.17 | 4.00 / 3.17 / 2.92 | **4.50 / 4.00 / 3.33** |
| facts | 3.08 | 2.46 | 2.92 | 2.88 |
| needle effective (4K worst depth) | **4096** (84%) | 3072 (70%) | 3072 (75%) | 3072 (73%) |
| ARC-Easy / PIQA / HellaSwag(norm) | 58.4 / 66.6 / 42.6 | 56.9 / 66.8 / 42.4 | 57.6 / 66.8 / 42.9 | 57.8 / **67.2** / 42.8 |
| facts probe | 70.1% | 69.6% | 71.1% | **72.7%** |

**Read the overall figure carefully.** 3.58 is v4's last checkpoint and also its maximum; its per-checkpoint
curve is 3.38 / 3.30 / 3.47 / 3.40 / 3.37 / 3.51 / 3.50 / 3.58, mean **3.44**, against stage A's mean of 3.49.
The honest claim is that v4 is *level with stage A on general quality while adding arithmetic 2.83 -> 4.42 and
working tool use* — the capability came free, not that v4 is a better chat model.

What each v4 change bought, against the prediction made before the run:

| change | predicted | measured |
|---|---|---|
| `humanize()` list answers (reprs 6.7% -> 0.51%) | pattern recovers | 1.67 -> 2.50, partial |
| restore `smoltalk-smol-constraints` (0.06) | python recovers to ~4.22 | 3.44 -> **4.22**, exact |
| restore `synthetic-reasoning` in full | algebra 0.22 -> ~0.44 | 0.22 -> 0.35, partial |

`slm.eval.reasoning --tools`, accuracy / tool-use rate:

| | arith2 | arith2mul | word | algebra | gsm8k |
|---|---|---|---|---|---|
| v1 (math_tool:math_no_tool 0.8:1) | 0.94 / 0.81 | 0.82 / 0.70 | 0.89 / 0.41 | **0.44** / 0.00 | 0.045 / 0.17 |
| v3 (2.0:1) | 0.99 / **0.94** | 0.92 / **0.91** | 0.93 / 0.64 | 0.22 / 0.02 | 0.060 / 0.28 |
| v4 (1.3:1) | 0.91 / 0.48 | 0.75 / 0.47 | 0.89 / 0.54 | 0.35 / 0.00 | 0.040 / **0.30** |

The `math_tool : math_no_tool` conversation ratio controls tool use on math almost proportionally, and it
trades against algebra, which uses no tool at all (rate 0.00-0.02). v4 deliberately sits between v1 and v3.

**Retrieval: the post-SFT model is a 3K model.** Three mixtures with completely different balances eroded 4K
needle identically (worst depth 70% / 75% / 73% against the base's 84%). Per the project rule that long context
is claimed only where needle holds, `m9_tool4_336m` is documented as **effective context 3072**, not 4096.
Treat this as a property of tool SFT at this scale rather than a mixture defect; benchmark performance was
given priority over context length (Peter, 2026-09-22).

## 11. M9 stage C — GRPO (`m9_rl2_336m`, best.pt at step 141, 2026-09-22)

Two attempts. Try 1 (`m9_rl_336m`) stopped at step 69 on a false-positive collapse guard and is kept only as
the evidence for the two fixes it produced; try 2 is the run.

| | pre-RL (v4) | step 50 | step 100 | step 141 (best.pt) |
|---|---|---|---|---|
| held-out accuracy (96 prompts) | 0.250 | 0.323 | 0.375 | **0.385** |
| malformed | 0.15 | 0.11 | 0.09 | **0.07** |
| mean completion length | 138 | 117 | 103 | 92 |

Stopped by the KL guard on a sustained KL of 0.174 against its 0.15 budget, held-out still rising.

`slm.eval.reasoning --tools`, accuracy / tool-use rate — the clearest win of the whole milestone:

| | arith1 | arith2 | arith2mul | arith_multi | algebra | word | gsm8k | mean |
|---|---|---|---|---|---|---|---|---|
| v4 | 1.00/0.90 | 0.91/0.48 | 0.75/0.47 | 0.96/1.00 | 0.35/0.00 | 0.89/0.54 | 0.04/0.30 | 0.701 |
| RL | 0.80/1.00 | 0.93/0.98 | 0.75/0.98 | 0.95/1.00 | **0.99/0.96** | **0.98/0.97** | 0.06/**0.78** | **0.779** |

RL taught the model to actually reach for the sandbox — rates went from 0.00-0.54 to 0.96-1.00 — and algebra
followed from 0.35 to 0.99 because linear equations are now solved in the tool instead of attempted mentally.
**GSM8K barely moved (0.04 -> 0.06) despite tool use nearly tripling**: the bottleneck there is reading the
problem and setting it up, not the arithmetic, and RL fixed only the routing half.

Core benchmarks are unchanged, so the drift cost nothing there:

| | needle (4K worst) | HellaSwag (norm) | ARC-Easy | PIQA | facts probe |
|---|---|---|---|---|---|
| stage A | **4096** (84%) | 35.1 (42.6) | **58.4** | 66.6 | 70.1% |
| v4 | 3072 (73%) | 34.8 (42.8) | 57.8 | **67.2** | **72.7%** |
| RL best.pt | 3072 (75%) | 34.5 (42.8) | 58.0 | 66.6 | 71.1% |

Judged, and this is where RL costs something:

| | overall | corr | coh | task | misfire | arith | defin | facts | narra | patte | prose | pytho | qa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stage A | 3.47 | 3.03 | 3.56 | **3.81** | **0.000** | 2.83 | 4.17 | 3.08 | 2.67 | **4.00** | 3.25 | 4.22 | 3.50 |
| v4 | **3.58** | **3.22** | 3.84 | 3.69 | 0.250 | 4.42 | 4.17 | 2.88 | **4.00** | 2.50 | 3.33 | 4.22 | 4.50 |
| RL | 3.49 | 3.06 | **3.94** | 3.47 | 0.321 | **4.67** | 3.50 | **2.00** | 3.50 | 3.25 | **3.58** | **4.28** | **5.00** |

Overall is flat within checkpoint noise (3.58 -> 3.49), with arithmetic, pattern, prose, python and qa all up
and coherence the best of any checkpoint in the project. Two regressions are real and have the same cause:

- **judged facts 2.88 -> 2.00**, far more than the facts *probe* moved (72.7% -> 71.1%). In chat mode the RL
  policy answers knowledge questions with terse verifier-shaped templates: "Who wrote Hamlet?" -> "So the
  answer is 2.", "What is the chemical symbol for gold?" -> "So the answer is True.", "Name the seven
  continents." -> "That would be China."
- **tool misfire 0.250 -> 0.321.** `reward_scheme: tool` pays 1.0 for correct-with-a-call against 0.5 for
  correct-without, so the policy was rewarded for reaching for the sandbox unconditionally.

Both follow from RL having seen only math, tool and constraint families: with no chat or knowledge task in the
mixture and nothing but a KL term anchoring it, the whole policy moved toward tool-and-terse. The fix for a
future run is a rehearsal family of ordinary chat prompts scored on something other than a verifier, or a
tighter KL budget; the per-family `reward_schemes` mechanism already exists to carry it.

**Which checkpoint is the M9 output.** `m9_rl2_336m/checkpoints/best.pt`. It ties v4 on every core benchmark,
is far ahead on reasoning and tool use (mean 0.779 vs 0.701, algebra 0.99 vs 0.35), and its judged overall
difference (-0.09) is inside the +-0.25 checkpoint wobble. `m9_tool4_336m/checkpoints/final.pt` remains the
better choice for pure factual chat, and both are kept.

## 12. MMLU: at chance, and not worth tracking (2026-09-22)

First MMLU run on any checkpoint in the project (`lm_eval` `mmlu` group, 40 questions per subject, 61 subjects).

| checkpoint | MMLU |
|---|---|
| random baseline (4 choices) | **25.0%** |
| base, `m8_base_4k_336m` final | 24.3% |
| stage B, `m9_tool4_336m` final | 22.9% |
| stage C, `m9_rl2_336m` best.pt | 23.6% |

All three are at or below chance and the differences between them are noise. The per-subject spread (8-42%)
is sampling variance at 40 questions each, not signal: no subject result here should be quoted.

This is the expected outcome for 336M parameters at 10B tokens — MMLU needs far more of both before it rises
above chance — but it had been an open "known gap" and is now closed. **Conclusion: MMLU stays out of the
regular suite.** It has no resolution at this scale, so it cannot track progress between stages. The facts
probe (`slm.eval.facts`, 194 items, completion-style) is the knowledge metric that does: it reads 69-73%
across these same checkpoints and moves measurably between stages.

## 13. Benchmark revision: what has resolution at 336M (2026-09-26)

The criterion (Peter, 2026-09-26): benchmarks we will not easily saturate and are not at chance on, covering
basic world knowledge and problem solving. Every lm-eval candidate available in the harness was run on the
base, the stage B output and the stage C output at `--limit 1000` (300 for the generation tasks). What
matters is not where the base sits but whether a task *moves between stages* and has room on both sides.

| task | chance | base | stage B (v4) | stage C (RL) | verdict |
|---|---|---|---|---|---|
| LAMBADA (openai) | ~0 | 32.1 | 33.1 | 33.0 | **keep** — solid resolution; a base-quality signal, and it should barely move under post-training, which it does not |
| OpenBookQA (acc_norm) | 25 | 32.4 | 31.6 | 31.4 | **keep** — resolution on the normalised metric; raw `acc` is at chance |
| SciQ | 25 | 81.5 | 82.8 | 83.5 | **keep** — easy but not saturated (frontier ~97), and it moves with post-training |
| HellaSwag / ARC-Easy / PIQA | 25 / 25 / 50 | (unchanged, §10-11) | | | keep — the existing three |
| BoolQ | 50 | 61.2 | 58.0 | 58.5 | **drop** — BoolQ's majority class is 62.2% of the validation set; every checkpoint is at or below it, so the model is not reading the passage |
| WinoGrande | 50 | 53.0 | 52.9 | 51.9 | **drop** — chance |
| ARC-Challenge | 25 | 24.2 / 26.8n | 25.3 / 26.1n | 24.9 / 26.8n | **drop** — chance on both metrics, like MMLU |
| TriviaQA (exact match) | ~0 | 1.7 | 2.7 | 3.0 | **drop** — at the floor; the facts probe covers recall with resolution |
| ASDiv | ~0 | 0.3 | 0.0 | 0.0 | **drop from lm-eval** — a harness mismatch, not capability: it scores free generation against a fixed answer format, and our models answer with a think span and `#### n`. The stage C model scores 0.98 on our word problems. SVAMP through our own reasoning eval is the honest version of this benchmark |

Standard set from here: HellaSwag, ARC-Easy, PIQA, LAMBADA, OpenBookQA, SciQ (`scripts/measure_stage.sh`),
plus SVAMP and GSM8K through `slm.eval.reasoning --tools`, plus the homebrew evals (facts probe, judged suite,
needle, multi-turn). Not tracked: MMLU, ARC-Challenge, WinoGrande, BoolQ, TriviaQA, ASDiv.

Two of the standard tasks are only usable on a normalised metric (OpenBookQA) or sit near where small models
plateau (SciQ), so the homebrew evals carry more of the weight than the standard ones — which is the intended
balance: standard benchmarks for comparability, homebrew ones for resolution on what this project is for.

### 13a. SVAMP, and what the base already knew (2026-09-26)

`slm.eval.reasoning --tools`, accuracy / tool-use rate. SVAMP is 300 one-step word problems; the synthetic
families are n=50 here, GSM8K n=200. **This is the first time the base was measured with `--tools`.**

| | arith2 | algebra | word | GSM8K | SVAMP |
|---|---|---|---|---|---|
| base (`m8_base_4k_336m`) | **1.00 / 1.00** | **1.00 / 1.00** | **1.00 / 1.00** | 0.01 / 0.90 | **0.10** / 0.96 |
| stage B (`m9_tool4_336m`) | 0.90 / 0.54 | 0.30 / 0.00 | 0.92 / 0.56 | 0.04 / 0.30 | 0.05 / 0.48 |
| stage C (`m9_rl2_336m` best.pt) | 0.92 / 1.00 | 1.00 / 1.00 | 0.96 / 0.94 | 0.06 / 0.78 | 0.09 / 0.86 |

Two things this table settles.

**The base already had the tool.** Its decay phase carried `tool-chat` at 1.5% of tokens — the same grammar-
generated Python-tool conversations the SFT sets are built from, folded in as pretraining rows — and that was
enough for the base to solve every synthetic family perfectly, calling the sandbox 90-100% of the time. Stage B
SFT then *suppressed* it on math (algebra 1.00 -> 0.30, tool use 1.00 -> 0.00): the prose-reasoning math sets
crowded out the tool traces, the conversation-count imbalance described in §10. Stage C recovered it. So the
headline "RL lifted algebra 0.35 -> 0.99" is true but misdescribed: RL restored a pretraining capability that
SFT had damaged, rather than teaching one. The most likely reading is that a stage B whose mixture had not
suppressed the tool would have made stage C's largest gain unnecessary.

**SVAMP has resolution, barely, and it is the right band.** 0.05-0.10 against a floor near zero, one notch
above GSM8K, with tool use at 0.5-0.96: the model reaches for the sandbox and sets up the wrong computation.
That is the same bottleneck GSM8K has — reading the problem, not the arithmetic — one difficulty step lower,
which makes SVAMP the place where progress on comprehension will show first. It is tracked from here through
`scripts/measure_stage.sh`.

### 13b. Multi-turn chat, measured for the first time (`slm.eval.multiturn`, n=64, greedy, 2026-09-26)

Scripted three-turn conversations: the user states a fact, asks an unrelated question, then asks something
that needs the fact. Scored deterministically, no judge.

| | recall | format | misfire | templated | tokens/turn |
|---|---|---|---|---|---|
| stage A (`m9_sft_336m`) | 0.578 | 0.703 | 0.000 | 0.000 | 54 |
| stage B (`m9_tool4_336m`) | 0.531 | **0.875** | 0.000 | 0.000 | 49 |
| stage C (`m9_rl2_336m` best.pt) | 0.562 | **0.875** | 0.000 | 0.000 | **22** |

The model remembers a fact from two turns earlier a little over half the time, at every stage — post-training
neither helped nor hurt recall. Format (every turn reaching `<|end|>`) improved from 0.70 to 0.875 at stage B
and held. Zero tool misfires and zero templated answers even on the RL checkpoint, which reads 0.32 misfire on
the judged suite: these turns are conversational ("My cat is called Biscuit. Please remember that.") rather
than short factual queries, and the RL drift is specific to the latter surface form. RL's answers are less
than half the length of the SFT models'. Recall at ~0.55 is the number to move; it is the first homebrew eval
with resolution on multi-turn behaviour and is run per stage from here.

### 13c. Chat-family reward signal on the stage B output (`scripts/rl_signal_probe.py`, 8 prompts x 6)

| family | pass | groups with spread |
|---|---|---|
| constraints | 0.19 | 0.75 |
| **chat** (`plain` reward) | 0.90 | **0.50** |
| gsm8k | 0.02 | 0.12 |

The `plain` reward -- present, not a `####` line, not a verifier template, no tool call -- yields spread in
half its groups on the stage B model, so it will carry gradient in RL run 3 rather than sit at all-ones. That
is the condition for it to work as an anchor.

## 14. M9 stage C, run 3: GRPO with a chat anchor (`m9_rl3_336m`, 2026-09-26)

Run 2's regression -- judged facts 2.88 -> 2.00, tool misfire 0.25 -> 0.32, "Who wrote Hamlet?" -> "So the
answer is 2." -- came from RL seeing only verifiable math/tool/constraint families. Run 3 adds a `chat` family
(short, non-computational SmolTalk first turns) with a `plain` reward: 1 only if the answer is present, is not
a `####` line, is not a verifier template, and made no tool call. Verifiable, no judge. 3 of 13 draws.
Everything else identical to run 2 (lr 4e-6, 8 prompts x 6 rollouts, guard on a 10-step window).

Stopped by the KL guard at step 150 (sustained 0.156 vs 0.15), held-out still rising: 0.438 -> 0.490 -> 0.542
-> **0.562**, malformed 0.25 -> 0.08. (The held-out set now contains chat prompts, so its level is not
comparable to run 2's 0.250; its gain, +0.124, is.) The chat family sat at reward 0.95-0.96 with **zero** tool
calls for the whole run -- the anchor held, and the spread it showed in a third of its groups is what pushed
back.

### Judged (final checkpoint of each run; run 3 = best.pt, step 150)

| | overall | corr | coh | task | **misfire** | arith | defin | **facts** | narra | patte | prose | pytho | qa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stage A | 3.47 | 3.03 | 3.56 | 3.81 | 0.000 | 2.83 | 4.17 | 3.08 | 2.67 | 4.00 | 3.25 | 4.22 | 3.50 |
| stage B (v4) | 3.58 | 3.22 | 3.84 | 3.69 | 0.250 | 4.42 | 4.17 | 2.88 | 4.00 | 2.50 | 3.33 | 4.22 | 4.50 |
| RL run 2 | 3.49 | 3.06 | 3.94 | 3.47 | 0.321 | 4.67 | 3.50 | 2.00 | 3.50 | 3.25 | 3.58 | 4.28 | 5.00 |
| **RL run 3** | **3.65** | **3.41** | 3.88 | 3.66 | **0.143** | 4.58 | **5.00** | 2.79 | 3.67 | 3.42 | 3.33 | 4.11 | 3.50 |

Facts recovered from 2.00 to 2.79 (v4: 2.88), misfire more than halved, correctness the highest of any
checkpoint in the project. Per checkpoint within run 3:

| checkpoint | overall | facts | misfire | RL held-out |
|---|---|---|---|---|
| step 50 | 3.81 | 3.71 | 0.18 | 0.490 |
| **step 100** | **3.96** | **3.96** | **0.07** | 0.542 |
| step 150 (best.pt) | 3.65 | 2.79 | 0.14 | **0.562** |

Every run 3 checkpoint scores above every other model's *maximum* (3.58), so "run 3 is the best chat model
we have" does not depend on which one is picked. Step 100 leads on exactly what the anchor targets, which is
consistent with less KL drift; whether it also holds the hard measures is checked below before choosing.

### Hard measures (best.pt)

| | needle | HellaSwag(n) | ARC-E | PIQA | LAMBADA | OBQA(n) | SciQ | facts probe | mt recall | mt format | tok/turn |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stage B (v4) | 3072 (73%) | 42.8 | 57.8 | 67.2 | -- | -- | -- | 72.7% | 0.531 | 0.875 | 49 |
| RL run 2 | 3072 (75%) | 42.8 | 58.0 | 66.6 | -- | -- | -- | 71.1% | 0.562 | 0.875 | 22 |
| RL run 3 | 3072 (72%) | 43.0 | 58.0 | 66.6 | 32.9 | 31.6 | 83.3 | **72.7%** | **0.578** | **0.906** | 35 |

Core benchmarks identical; facts probe back to v4's level; the best multi-turn recall and format of any
checkpoint; answers no longer run 2's terse 22 tokens.

| reasoning (acc / tool use) | arith2 | arith2mul | algebra | word | GSM8K | SVAMP |
|---|---|---|---|---|---|---|
| RL run 2 | 0.93 / 0.98 | 0.75 / 0.98 | **0.99 / 0.96** | 0.98 / 0.97 | 0.06 / **0.78** | 0.09 / 0.86 |
| RL run 3 | 0.95 / 0.90 | **0.86 / 0.83** | 0.74 / 0.71 | 0.95 / 0.79 | 0.04 / 0.54 | 0.05 / 0.68 |

The cost: tool use on math came down across the board (algebra 0.96 -> 0.71, GSM8K 0.78 -> 0.54), and
algebra accuracy with it. A reward that penalises *any* tool call on chat prompts generalised "call the tool
less" somewhat beyond chat. A softer anchor (penalise only calls that fail, or only on prompts with no
numbers) is the obvious next dial; this run establishes that the mechanism works.

### 14a. pass@k on word problems: selection, not capability (`scripts/pass_at_k.py`, run 3 best.pt, k=32, n=30 each)

| | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | **pass@32** | majority vote | distinct answers / 32 |
|---|---|---|---|---|---|---|---|---|
| GSM8K | 0.021 | 0.033 | 0.100 | 0.133 | 0.233 | **0.333** | 0.033 | 22.4 |
| SVAMP | 0.035 | 0.033 | 0.100 | 0.167 | 0.367 | **0.533** | 0.167 | 24.0 |

The question this was run to answer: with GSM8K at 0.04-0.06 greedy, is the right answer *in* the model at
all? It is -- in a third of GSM8K problems and half of SVAMP ones, somewhere among 32 samples, a 15-16x
lift over pass@1. So goal 5's swarm has something to select from. But the errors are not systematic: 22-24
distinct answers per problem means the samples almost never agree, the mode is usually wrong, and **majority
voting recovers almost nothing** (0.033 / 0.167). The design consequence is precise: parallel sampling needs a
*verifier* to pick the winner, not a vote. That is the sandbox's job -- re-derive, check the setup against
the problem, execute. n=30 per set, so pass@32 carries about +-0.09; the direction is robust, the exact
numbers are not.

### 14b. Which run 3 checkpoint: best.pt, and why not the judged argmax

Step 100 leads the judged suite (3.96 vs 3.65; facts 3.96 vs 2.79; misfire 0.07 vs 0.14). Measured on the
full hard suite before choosing:

| | reasoning mean | arith2 | algebra | GSM8K | mt recall | mt format | benchmarks / needle / facts probe |
|---|---|---|---|---|---|---|---|
| **best.pt (step 150)** | **0.686** | **0.95 / 0.90** | **0.74 / 0.71** | **0.04 / 0.54** | **0.578** | 0.906 | identical |
| step 100 | 0.568 | 0.75 / 0.82 | 0.65 / 0.50 | 0.02 / 0.47 | 0.516 | 0.922 | identical |

Step 100's judged advantage is real, not noise -- but it is bought with 0.12 of reasoning accuracy and lower
tool use across every family, plus four fewer multi-turn recalls in 64. That is the wrong trade for a model
whose tool use is meant to be a core capability, and the judged gap is concentrated in `facts`, eight items.
**`runs/m9_rl3_336m/checkpoints/best.pt` is the M9 output.** Second time the rule has paid for itself: check
the hard suite before the judged argmax (run 2's 30M snapshot was the first, where the check showed noise;
here it showed a trade).

## 15. M9 stage C, run 4: `tool_strict` does not recover math tool use (2026-09-26)

One change from run 3: math and tool families scored under `tool_strict` (correct-without-a-call earns 0.25
instead of 0.5), the chat anchor unchanged. Hypothesis: with the mental route worth half a tool answer, the
anchor's "call less" pressure had tipped the balance on math, and widening the gap would tip it back.
Stopped by the KL guard at step 150 like run 3; held-out 0.344 -> 0.427 (+8 of 96), malformed 0.22 -> 0.05.

`slm.eval.reasoning --tools`, accuracy / tool-use rate:

| | arith2 | arith2mul | algebra | word | GSM8K | SVAMP | mean |
|---|---|---|---|---|---|---|---|
| run 2 (no anchor) | 0.93 / 0.98 | 0.75 / 0.98 | **0.99 / 0.96** | 0.98 / 0.97 | 0.06 / **0.78** | 0.09 / 0.86 | **0.779** |
| run 3 (anchor, `tool`) | 0.95 / 0.90 | 0.86 / 0.83 | 0.74 / 0.71 | 0.95 / 0.79 | 0.04 / 0.54 | 0.05 / 0.68 | 0.686 |
| run 4 (anchor, `tool_strict`) | 0.87 / 0.70 | 0.75 / 0.66 | 0.57 / 0.45 | 0.93 / 0.64 | 0.06 / 0.54 | 0.07 / 0.55 | 0.646 |

**The hypothesis was wrong.** Tool use fell further in every family, opposite to the prediction across the
board -- more than one-seed variance explains. The reason is visible after the fact: `tool_strict` only acts
on the families RL trains on, where tool use was already 0.9-1.0 (§14, rollout stats), so it had almost
nothing to push. The families that dropped -- algebra, arith2, word -- are not in the RL mix. Their tool use
is generalisation, and the chat anchor's "short question -> do not call" generalises to short questions that
contain numbers. No reward change inside the mix reaches that.

What would: train the contested surface form directly. `algebra`, `arith2` and `word` are generators already
in `GENERATORS`, cheap to add to the mix under a tool scheme, so that "short question with numbers -> call"
is learned rather than inferred. Whether that is the next run, or whether the swarm discussion (§14a) changes
what RL should optimise for, is Peter's call. Until then run 3's `best.pt` remains the M9 output: run 4's
judged and hard-suite numbers follow below, but a 0.04 drop in reasoning mean is not what any judged gain
could buy back given the goals.

### 15a. Run 4 hard suite: the decision does not need the judges

| | needle | HellaSwag(n) | ARC-E | PIQA | LAMBADA | OBQA(n) | SciQ | facts probe | mt recall | mt format | mt misfire | tok/turn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RL run 3 | 3072 (72%) | 43.0 | 58.0 | 66.6 | 32.9 | 31.6 | 83.3 | 72.7% | **0.578** | 0.906 | 0.047 | 35 |
| RL run 4 | 3072 (72%) | 43.4 | 58.1 | 67.1 | 32.9 | 31.2 | 83.2 | 72.2% | 0.422 | **0.938** | **0.000** | 31 |

Benchmarks, needle and the facts probe are identical within noise. Multi-turn recall fell 0.578 -> 0.422 (ten
fewer of 64) while format and misfire improved slightly -- the same direction as the reasoning table: run 4
answers more cleanly and knows less of what to do with the question. **Run 3's `best.pt` remains the M9
output**; run 4 is kept as the ablation that rules out the reward-gap explanation. Judged scores and pass@k
for run 4 are appended below when scored, for the record rather than for the decision.

### 15b. Run 4 judged: the trade is real, and 15a overstated the decision

| | overall | corr | coh | task | misfire | arith | defin | facts | narra | patte | prose | pytho | qa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RL run 3 best | 3.65 | 3.41 | 3.88 | 3.66 | 0.143 | 4.58 | 5.00 | 2.79 | 3.67 | 3.42 | 3.33 | 4.11 | 3.50 |
| **RL run 4 best** | **3.97** | **3.44** | **4.22** | **4.25** | **0.000** | 4.67 | 4.67 | **3.75** | **4.50** | **3.75** | 3.42 | 3.89 | 4.00 |

Run 4 per checkpoint: step 50 4.00 / facts 4.08 / misfire 0.07; step 100 3.66 / 3.50 / 0.00; step 150 3.97 /
3.75 / 0.00 -- every checkpoint above run 3's best, misfire at zero on two of three. This is the best judged
chat model of the project by a wide margin, and facts at 3.75 is higher than any checkpoint including stage A.

**15a said the decision did not need the judges. That was wrong.** The hard suite showed run 4 giving up
reasoning (0.686 -> 0.646) and multi-turn recall (0.578 -> 0.422); the judges show what it bought with them.
That is not a checkpoint-selection artefact of the kind the "hard suite before judged argmax" rule guards
against -- it is a genuine trade between goal 3 (a coherent chatbot: run 4, clearly) and goal 4 (tool use as a
core capability: run 3, clearly), with goal 2 (benchmarks) a tie and multi-turn memory, part of goal 3, on
run 3's side. One seed each, so the size of the judged gap is uncertain; its existence (+0.32, misfire 0.14 ->
0.00, facts +0.96) is not within the suite's +-0.25 wobble.

What both runs agree on: the chat anchor's gains on facts and misfire survive across seeds and schemes
(run 3 2.79 / 0.14, run 4 3.75 / 0.00 against run 2's 2.00 / 0.32). The mechanism is settled. What is not
settled is tool use on families outside the RL mix, and the roadmap's next run -- train `algebra`, `arith2`
and `word` in the mix -- is the candidate to keep run 4's chat quality without its reasoning cost. The
choice of M9 output between run 3 and run 4 is Peter's.

Run 4 pass@k (k=32, n=30 each), for the record: GSM8K pass@1 0.020 -> pass@32 **0.367**, majority 0.033,
22.5 distinct answers per problem; SVAMP 0.035 -> **0.533**, majority 0.067, 23.5 distinct. Within noise of
run 3 (0.333 / 0.533). The pass@k ceiling and the failure of voting are properties of the model, not of the RL
variant -- a verifier-based swarm has the same raw material whichever run it is built on.

## 16. M9 stage C, run 5: the anchor plus the out-of-mix families trained in the mix (2026-09-26)

Run 3's config (chat anchor, `tool` scheme) with `algebra`, `arith2` and `word` added to the RL mix under `tool`
-- the families whose tool use runs 3 and 4 had eroded by generalisation -- so "short question with numbers ->
call the sandbox" is trained rather than inferred. 17 draws: chat 3, constraints 4, gsm8k 3, pytool_numbers 2,
pytool_declared 1, algebra 2, arith2 1, word 1. Seed 12.

**The only stage C run to complete all 250 steps.** KL stayed at 0.01-0.07 for the whole run, never near the
0.15 guard that stopped runs 2-4 at 141-150. It also survived a crash: at step 60 the policy wrote a slice with a
float index and the sandbox let Python's `TypeError` through to the trainer (fixed at the slice site and at the
`run_tool` boundary, §log 2026-09-26); the run resumed from the step-51 checkpoint with full state.

| held-out | pre-RL | step 50 | 100 | 150 | 200 | 250 |
|---|---|---|---|---|---|---|
| accuracy | 0.438 | 0.604 | 0.615 | **0.625** | 0.625 | 0.615 |
| malformed | 0.11 | 0.08 | 0.10 | 0.05 | 0.07 | 0.04 |

`best.pt` = step 150. (The held-out set contains the new families, so its level is not comparable to earlier
runs; the +0.19 gain is the largest of any run.)

| window | KL | entropy | reward | no-signal groups |
|---|---|---|---|---|
| 1-50 | 0.014 | 0.94 | 0.593 | 0.66 |
| 51-100 | 0.050 | 0.84 | 0.649 | 0.79 |
| 101-150 | 0.032 | 0.95 | 0.648 | 0.76 |
| 151-200 | 0.063 | 0.93 | 0.682 | 0.77 |
| 201-250 | 0.072 | 1.03 | 0.667 | 0.76 |

Per family, first 30 steps -> last 30 (reward / tool-call rate): **algebra** 0.95/0.94 -> 0.97/**1.00**,
**arith2** 0.86/0.81 -> 1.00/**1.00**, **word** 0.86/0.80 -> 1.00/**1.00**, chat 0.80/0.01 -> 0.95/**0.00**,
gsm8k 0.02/0.45 -> 0.06/**0.75**, pytool_numbers 0.77/0.90 -> 0.95/1.00, constraints 0.40 -> 0.45. The two
pressures coexist in one mix: the trained math families saturate at 100% tool use while the chat anchor holds
at zero. Whether that transfers to the post-run evals -- the reasoning suite, judged facts and misfire,
multi-turn -- is the next table.

### 16a. Run 5 reasoning: training the families in the mix restores their tool use

`slm.eval.reasoning --tools`, accuracy / tool-use rate, best.pt of each run:

| | arith2 | arith2mul | algebra | word | GSM8K | SVAMP | mean |
|---|---|---|---|---|---|---|---|
| run 2 (no anchor) | 0.93 / 0.98 | 0.75 / 0.98 | 0.99 / 0.96 | 0.98 / 0.97 | 0.06 / **0.78** | -- | 0.779* |
| run 3 (anchor) | 0.95 / 0.90 | 0.86 / 0.83 | 0.74 / 0.71 | 0.95 / 0.79 | 0.04 / 0.54 | 0.05 / 0.68 | 0.686 |
| run 4 (anchor, `tool_strict`) | 0.87 / 0.70 | 0.75 / 0.66 | 0.57 / 0.45 | 0.93 / 0.64 | 0.06 / 0.54 | 0.07 / 0.55 | 0.646 |
| **run 5 (anchor, families in mix)** | **0.99 / 1.00** | **1.00 / 1.00** | **1.00 / 1.00** | **1.00 / 1.00** | 0.04 / 0.69 | **0.09 / 0.84** | 0.755 |

*run 2's mean is over six tasks (no SVAMP); on the shared tasks run 5 matches or beats it everywhere except
GSM8K tool use.

The diagnosis from §15 was right: the erosion was generalisation onto families outside the mix, and putting
them inside fixes it completely -- algebra tool use 0.45 -> 1.00, arith2mul 0.66 -> 1.00 -- while the chat
anchor held at zero calls throughout training. SVAMP is the best of any run (0.09 at 0.84 tool use). GSM8K is
unchanged at 0.04: comprehension, as before. Judged chat quality, misfire and multi-turn follow in 16b.

### 16b. Run 5 hard suite

| | needle | HellaSwag(n) | ARC-E | PIQA | LAMBADA | OBQA(n) | SciQ | facts probe | mt recall | mt format | mt misfire | tok/turn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RL run 3 | 3072 (72%) | 43.0 | 58.0 | 66.6 | 32.9 | 31.6 | 83.3 | 72.7% | **0.578** | 0.906 | 0.047 | 35 |
| RL run 4 | 3072 (72%) | 43.4 | 58.1 | 67.1 | 32.9 | 31.2 | 83.2 | 72.2% | 0.422 | 0.938 | 0.000 | 31 |
| **RL run 5** | 3072 (**75%**) | 43.0 | 57.6 | 67.0 | 33.0 | 30.6 | 83.1 | 71.6% | 0.484 | **0.969** | **0.000** | 40 |

Benchmarks, needle and the facts probe are within noise of each other across the three runs (the standard
tasks move by less than a point; the facts probe by ~1). Multi-turn format is the best of any checkpoint and
misfire on conversational turns is zero; recall sits between run 4's and run 3's. So on the hard suite run 5
gives up nothing against run 3 except six multi-turn recalls in 64, and adds the reasoning result of 16a.
Judged chat quality follows in 16c.

### 16c. Run 5 judged, and the M9 output

| | overall | corr | coh | task | misfire | arith | defin | facts | narra | patte | prose | pytho | qa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stage A | 3.47 | 3.03 | 3.56 | 3.81 | 0.000 | 2.83 | 4.17 | 3.08 | 2.67 | 4.00 | 3.25 | 4.22 | 3.50 |
| stage B (v4) | 3.58 | 3.22 | 3.84 | 3.69 | 0.250 | 4.42 | 4.17 | 2.88 | 4.00 | 2.50 | 3.33 | 4.22 | 4.50 |
| RL run 3 | 3.65 | 3.41 | 3.88 | 3.66 | 0.143 | 4.58 | 5.00 | 2.79 | 3.67 | 3.42 | 3.33 | 4.11 | 3.50 |
| RL run 4 | **3.97** | 3.44 | **4.22** | **4.25** | **0.000** | 4.67 | 4.67 | **3.75** | **4.50** | 3.75 | 3.42 | 3.89 | 4.00 |
| **RL run 5** (best.pt, s150) | 3.84 | **3.56** | 3.97 | 4.00 | 0.036 | 4.50 | 4.50 | 3.38 | 2.67 | **4.67** | 2.83 | **4.39** | 3.67 |

Run 5 per checkpoint: s50 3.68 / facts 4.00 / misfire 0.18; s100 3.95 / 3.88 / 0.04; **s150 3.84 / 3.38 / 0.04**;
s200 4.07 / 3.88 / 0.00; s250 3.88 / 3.42 / 0.07. Five-checkpoint mean **3.88 -- identical to run 4's** (4.00,
3.66, 3.97). Correctness 3.56 and pattern 4.67 are the highest of any checkpoint in the project; python 4.39
the highest of any post-training checkpoint. Where run 4 still leads -- narrative (n=2), prose (n=4),
facts (n=8) -- are the smallest categories.

**Decision: `runs/m9_rl5_336m/checkpoints/best.pt` (step 150) is the M9 output.** This is not the trade of
§15b any more. Run 5 has run 4's chat quality (judged means level, misfire ~0, correctness better), run 2's
tool use (100% on the trained families, the best SVAMP), multi-turn format and recall better than run 4's,
and benchmarks and needle tied. Against run 3 it gives up six multi-turn recalls in 64 and nothing else. Every
one of the five goals is served by the same checkpoint, which is what goal 1 asks for.

Step 200 scored 4.07 judged with misfire 0.00 -- within the suite's wobble of best.pt. Per the rule, it is
measured on the hard suite (§16e) before anything is read into it; if it holds reasoning and multi-turn, it
is a free upgrade, and if it does not, best.pt stands.

**What the four RL runs settled**, in order: (2) RL restores and extends tool use but drifts the whole policy
toward tool-and-terse when it sees only verifiable families; (3) a verifiable `plain` chat reward anchors it,
recovering facts and misfire, at a cost in tool use on families outside the mix; (4) that cost is not a reward
-gap problem -- widening the gap made it worse; (5) it is a coverage problem -- training the families in the
mix restores their tool use completely with the anchor intact. The mechanism is: verifiable rewards for every
behaviour you want, including the negative ones, and every contested surface form inside the mix.

### 16d. Run 5 pass@k: a better sampler raises the swarm's ceiling

| best.pt, k=32, n=30 | pass@1 | pass@8 | pass@16 | **pass@32** | majority vote | distinct answers / 32 |
|---|---|---|---|---|---|---|
| GSM8K, run 3 / run 4 | 0.021 / 0.020 | 0.133 / 0.167 | 0.233 / 0.267 | 0.333 / 0.367 | 0.033 / 0.033 | 22.4 / 22.5 |
| **GSM8K, run 5** | 0.025 | 0.067 | 0.233 | **0.467** | 0.067 | 23.8 |
| SVAMP, run 3 / run 4 | 0.035 / 0.035 | 0.167 / 0.200 | 0.367 / 0.467 | 0.533 / 0.533 | 0.167 / 0.067 | 24.0 / 23.5 |
| **SVAMP, run 5** | 0.049 | 0.200 | 0.533 | **0.733** | 0.133 | 22.7 |

§14a called the pass@k ceiling a property of the model rather than of the RL variant. Runs 3 and 4 supported
that; run 5 refutes it. Training the math families in the mix lifted pass@32 by about +0.10-0.13 on GSM8K and
+0.20 on SVAMP (n=30, so about +-0.09 each; the SVAMP move is well outside it) while pass@1 barely moved. So a
policy that calls the tool reliably produces a *more diverse-and-correct* pool even when its greedy answer is
no better -- and majority vote still recovers almost none of it (0.07 / 0.13), with ~23 distinct answers per
32 samples. For goal 5 that means two things: selection is still the whole problem, and the ceiling a
selector is aiming at is now 0.47 / 0.73 rather than 0.33 / 0.53.

### 16e. Step 200 on the hard suite: a free upgrade, and the M9 output

| | reasoning mean | algebra | GSM8K | SVAMP | needle | HellaSwag(n) | ARC-E | PIQA | facts | mt recall | mt format | mt misfire | judged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| best.pt (step 150) | 0.755 | 1.00 / 1.00 | 0.04 / 0.69 | 0.09 / 0.84 | 3072 (75%) | 43.0 | 57.6 | 67.0 | 71.6% | 0.484 | **0.969** | 0.000 | 3.84 |
| **step 200** | 0.757 | 1.00 / 1.00 | 0.03 / 0.74 | 0.10 / 0.84 | 3072 (73%) | 42.5 | 57.6 | 66.8 | 71.6% | **0.562** | 0.938 | 0.000 | **4.07** |

Everything held within noise, and step 200 is better where the two differ: judged +0.23, multi-turn recall
+5 of 64 (back near run 3's 0.578), misfire zero. The only ticks down -- multi-turn format by two
conversations, HellaSwag by 0.5 -- are inside their noise. Third application of the "hard suite before the
judged argmax" rule: the first found noise (run 2's 30M snapshot), the second a trade (run 3's step 100),
this one an upgrade.

**`runs/m9_rl5_336m/checkpoints/step_00200.pt` is the M9 output.** best.pt (step 150) is kept beside it.

## 17. Goal 5, the swarm: baseline on the M9 output (`scripts/swarm_eval.py`, step 200, k=16, n=50 each, 2026-09-26)

The pipeline (`slm/swarm.py`, `docs/design.md` "Swarm inference"): sample k, collapse by parsed answer with sandbox
evidence, then one greedy pass of the *same* model over a selection prompt. Measured before any selection training,
so `selector` here is the model reading a prompt format it has never seen. Ceilings from the same 16 samples:
`oracle` = a correct sample exists (pass@16), `oracle_verified` = a correct *sandbox-verified* sample exists,
`in_prompt` = the correct group survived into the selector prompt.

| set | greedy | majority | verified maj. | **selector** | oracle | oracle verified | in prompt | groups / verified per 16 |
|---|---|---|---|---|---|---|---|---|
| GSM8K | 0.00 | 0.06 | 0.08 | **0.06** | 0.30 | 0.20 | 0.30 | 12.0 / 9.3 |
| SVAMP | 0.10 | 0.14 | 0.12 | **0.10** | 0.58 | 0.50 | 0.54 | 11.7 / 11.8 |

What it says:
- The pool has the answer far more often than anything picks it: 0.30 / 0.58 at k=16 against 0.00 / 0.10 greedy. The
  budget throws away almost nothing (in_prompt = oracle on GSM8K, 0.54 vs 0.58 on SVAMP), so the selection prompt
  is not the bottleneck.
- Verification is weaker than hoped as a filter: 9-12 of 16 candidates are "verified" (the answer came out of a
  call that ran), because a wrong setup computed correctly is still verified. `verified_majority` gains +0.02 on
  GSM8K and loses 0.02 on SVAMP -- noise. Provenance says "the arithmetic was done", not "the setup was right".
- The untrained selector is at majority level and picked the right group in 3 of 15 (GSM8K) and 5 of 27 (SVAMP)
  of the prompts that contained it, never calling the tool (0.00). Its disagreements with majority cancel
  (1:1 and 0:2). So `in_prompt - selector` = 0.24 / 0.44 is the whole opportunity, and it is a training question:
  the `select` SFT set and RL family (§18 when run 6 lands).
- Cost: ~12 s per problem for 16 samples + selection (605 s / 431 s per 50), against ~1 s greedy.

Crash note: the first attempt died at GSM8K problem 50 when a runaway sample produced a 400-digit number and
`float()` overflowed in `answer_key` (9b038ce guards it). The eval's numbers above are from the rerun.

## 18. Selection SFT + RL run 6: the single-prompt selector learns nothing, run 6 is the new M9 output (2026-09-26/27)

The plan after §17: teach the swarm's single-prompt selector (`selector_messages`: every distinct answer with support
and provenance, pick one) with a format SFT on the model's own pools (`slm.rl.synth_select`), then an RL `select`
family. The SFT took three tries (log, 2026-09-26 evening): the first two learned the set's conditional tokens and
never moved the first think token; v3 (40M tokens, lr 1.5e-5, the set at 0.15, on half the correct pools, the other
half the RL family's) installed the format. `m9_rl6_336m` = run 5's mix + `select` x3/20, binary reward, from that
SFT (`runs/m9_select_336m`, kept as v3; v1/v2 deleted).

### Swarm evals, k=16, n=50 each (`scripts/swarm_eval.py`)

| checkpoint | set | greedy | majority | verified maj. | selector | oracle | oracle verified | in prompt |
|---|---|---|---|---|---|---|---|---|
| rl5 step 200 (§17, untrained selector) | GSM8K | 0.00 | 0.06 | 0.08 | 0.06 | 0.30 | 0.20 | 0.30 |
| | SVAMP | 0.10 | 0.14 | 0.12 | 0.10 | 0.58 | 0.50 | 0.54 |
| select SFT v3 (= rl6 best.pt, step 0) | GSM8K | 0.06 | 0.06 | 0.06 | 0.04 | 0.26 | 0.18 | 0.22 |
| | SVAMP | 0.08 | 0.12 | 0.12 | 0.10 | 0.50 | 0.42 | 0.48 |
| rl6 step 100 | GSM8K | 0.02 | 0.08 | 0.10 | 0.06 | 0.30 | 0.28 | 0.30 |
| | SVAMP | 0.10 | 0.16 | 0.16 | 0.14 | 0.46 | 0.44 | 0.46 |
| **rl6 final (step 250)** | GSM8K | **0.10** | 0.10 | 0.10 | 0.04 | 0.36 | 0.32 | 0.30 |
| | SVAMP | 0.08 | 0.14 | 0.14 | 0.08 | 0.48 | 0.48 | 0.48 |

The selector never rises above majority at any checkpoint (right pick in 2 of 11 / 4 of 24 in-prompt cases for the
SFT, versus 3/15 and 5/27 untrained), never calls the tool, and RL's `select` reward stayed at the family's ceiling
(0.24 / 0.18 / 0.28 / 0.19 / 0.11 by 50-step window; only 25% of its pool holds a correct group). Verdict: a
template that *announces* a pick teaches the template, not the choice, and "one of twelve" is not a decision a 336M
model learns from 185 examples. Peter's redesign, a pairwise tournament, is §19.

### The hard suite: run 6 is the new M9 output

| | reasoning mean | GSM8K | SVAMP | algebra | word | mt recall | mt format | misfire | facts | HellaSwag(n) | ARC-E | needle | judged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rl5 step 200 (M9 output until now) | 0.757 | 0.03 / 0.74 | 0.10 / 0.84 | 1.00 / 1.00 | 1.00 / 1.00 | 0.562 | 0.938 | 0.000 | 71.6% | 42.5 | 57.6 | 3072 | 4.07 |
| select SFT v3 (rl6 best.pt) | 0.755 | 0.05 / 0.66 | 0.06 / 0.69 | 1.00 / 1.00 | 0.97 / 0.81 | 0.531 | -- | 0.016 | 74.2% | 43.1 | 57.5 | 4096 | 3.23 |
| **rl6 final.pt** | **0.769** | **0.085 / 0.90** | 0.10 / **0.96** | 1.00 / 1.00 | 0.99 / 1.00 | **0.609** | -- | 0.000 | 70.6% | 42.9 | 57.0 | 3072 | **4.11** |

(accuracy / tool-use rate; judged = claude-sonnet-5 on the blind v1 suite.) Two things in the same run: the format SFT
alone cost judged 4.07 -> 3.23 (facts 3.88 -> 1.88, definition 4.00 -> 2.50), and run 6's rewards -- the chat anchor
above all -- brought it back to **4.11** (facts 4.46, task 4.45, the project's best overall; pattern 4.33 -> 3.50 is
the one category down). Its weakest categories are the writing ones: prose 3.42, pattern 3.50, narrative 3.67,
definition 4.00 -- against arithmetic 4.67, qa 4.50, facts 4.46 (the base never saw fiction; see the M10 plan). Greedy GSM8K nearly tripled (0.03 -> 0.085) with tool use 0.74 -> 0.90, and SVAMP tool use
0.84 -> 0.96: the `gsm8k` reward with the select prompts' word problems in the mix did what §16a did for algebra.
Multi-turn recall 0.609 is the best measured. Benchmarks and needle tied; facts probe 70.6% vs 71.6% is inside noise.

**Decision: `runs/m9_rl6_336m/checkpoints/final.pt` is the M9 output** (fifth application of the hard-suite-then-judge
check: an upgrade). The selection SFT contributed nothing the selector could use, but the RL run it was built for
did; and the SFT's chat damage being fully repaired by RL says the format-then-anchor order is safe to reuse (§19).

## 19. The tournament: the pairwise decision is at chance, trained or not (2026-09-27)

Peter's redesign of the swarm's selection: a single-elimination bracket of pairwise comparisons ("Answer A ...
Answer B ... which is correct? `#### A`/`#### B`"), rounds in parallel, so the model never faces more than a binary
choice. `slm.swarm.tournament` (seeded first-vs-last, every other pair presented swapped, evidence fallback when no
pick parses), `scripts/pair_eval.py` (the atomic decision on its own: accuracy on gold-vs-wrong pairs from the
model's own pools, balanced 50/50 so chance is 0.5 and the A-share reads the position bias), and the same training
ladder as §18 on run 6's final.pt: pairwise SFT `m9_pair_336m` (322 pairs, v3 recipe), then RL `m9_rl7_336m` with a
`pair` family (337 disjoint pairs, binary reward) in run 6's mix.

| checkpoint | pair accuracy (n=300) | parsed | A-share | tournament GSM8K / SVAMP | majority | greedy |
|---|---|---|---|---|---|---|
| rl6 final (untrained) | 0.12 (≈0.49 of the parsed) | 0.24 | 0.11 | -- | | |
| rl5 step 200 (untrained, the §17 base) | -- | | | 0.06 / 0.08 | 0.06 / 0.14 | 0.00 / 0.10 |
| pair SFT (m9_pair final) | **0.507** | 0.997 | 0.475 | 0.08 / 0.08 | 0.08 / 0.10 | 0.08 / 0.10 |
| rl7 step 100 | **0.487** | 1.000 | 0.043 | 0.08 / 0.14 | 0.08 / 0.12 | 0.06 / 0.12 |

- The SFT installed the format completely (99.7% parsable picks, position bias gone at 0.475) and the decision not
  at all: 0.507, chance, on every source (GSM8K 0.53, SVAMP 0.48).
- RL run 7 had the ideal GRPO signal -- a balanced binary reward at chance means every group has spread -- and its
  `pair` reward went 0.43 / 0.55 / 0.48 / 0.58 by 20-step window while the A-share fell to 0.04: the policy drifted
  to the constant answer, which is what a policy does when no feature in the prompt predicts the label. Stopped at
  step 100 (checkpoint kept); the remaining GPU went to the comparison models (§20).
- The bracket itself is correct (unit-tested against a scripted judge, and the portal's streamed bracket matches
  `swarm_answer` round for round); it inherits the atomic decision, so tournament = majority ± noise.

**Verdict.** At 336M, judging which of two worked rationales is right is not learned from 322 imitation pairs plus
100 RL steps, and the pool's evidence (support, sandbox provenance) does not separate right from wrong either (§17:
9-12 of 16 candidates "verified" because a wrong setup computes correctly). The swarm's ceiling is real (oracle
0.30-0.36 GSM8K, 0.48-0.66 SVAMP at k=16) and nothing at this scale reaches it by judgment. What is left to try,
in order of cost: (1) *cross-method agreement* as a mechanical selector -- an answer reached both by a code path and a
prose path, or by two differently-worded re-askings of the problem, is right more often than support alone says
(needs a k=16 sweep with candidates stored; cheap); (2) a much larger pairwise set -- every GSM8K-train pool at k=8
(~1.2K correct pools, ~5K pairs) plus host-generated "spot the wrong step" pairs from the synthetic families, to
test whether the judge is data-starved rather than incapable; (3) accept selection as out of reach at this size and
use parallel sampling only where an external verifier exists (unit tests, constraints), which the RL families already
do. The training ladder is reusable as is: a format SFT then the chat-anchored RL (§18) costs nothing measurable.

## 20. Where we stand against local open-weight models of the same size (2026-09-27)

Peter's ask: similarly sized open-source models, run *locally in our own harness* on the same suites, as rows in the
Evals tab. Seven models (`slm/eval/external.py`; weights under `C:\slm-data\models`, offline at run time): SmolLM2-135M
and 360M (base + Instruct; trained on the same data family we sampled, 2T / 4T tokens), Qwen2.5-0.5B (base + Instruct;
up to 18T tokens), gpt2-medium (355M, 2019, a floor). Ours: `m9_rl6_336m` final.pt, 336M, ~10B pretraining tokens.
Same lm-eval tasks and limits, same items and verifiers for the homebrew evals, each model's own tokenizer and chat
template as shipped, our decoding settings; anything that depends on our tool protocol is n/a for them.

| model | params / tokens | HellaSwag(n) | ARC-E | PIQA | LAMBADA | OBQA(n) | SciQ | facts | GSM8K strict / lenient | SVAMP lenient | mt recall | needle | judged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt2-medium (base) | 355M / ~? | 43.4 | 49.6 | 67.3 | 41.3 | 29.6 | 75.7 | 46.4 | -- | -- | -- | 1024 (max) | -- |
| SmolLM2-135M (base) | 135M / 2T | 44.2 | 64.2 | 68.3 | 42.5 | 33.0 | 83.8 | 70.6 | -- | -- | -- | 2048 | -- |
| SmolLM2-135M-Instruct | 135M / 2T | 43.5 | 53.8 | 66.6 | 38.6 | 33.2 | 81.6 | 71.1 | 0.000 / 0.015 | 0.040 | 0.516 | 4096 | 3.94 |
| **ours, m9_rl6 final** | **336M / 10B** | 42.9 | 57.0 | 67.1 | 33.7 | 31.6 | 82.7 | 70.6 | **0.085** / -- | 0.100 (strict) | 0.609 | 3072 | 4.11 |
| SmolLM2-360M (base) | 362M / 4T | 51.8 | 70.3 | 71.8 | 52.8 | 37.8 | 91.1 | 87.1 | -- | -- | -- | 4096 | -- |
| SmolLM2-360M-Instruct | 362M / 4T | 52.2 | 56.2 | 71.0 | 49.1 | 36.8 | 72.7 | 86.1 | 0.005 / 0.075 | 0.173 | 0.734 | 4096 | 4.46 |
| Qwen2.5-0.5B (base) | 494M / 18T | 48.9 | 64.4 | 70.6 | 51.9 | 35.2 | 92.9 | 80.9 | -- | -- | -- | 4096 | -- |
| Qwen2.5-0.5B-Instruct | 494M / 18T | 49.9 | 65.3 | 70.5 | 50.3 | 34.2 | 91.8 | 70.1 | 0.000 / 0.330 | 0.560 | 0.750 | 4096 | 4.31 |

(lm-eval zero-shot, limit 2000, acc_norm for HellaSwag/OBQA, acc otherwise; facts = the 194-item completion probe;
GSM8K/SVAMP = our reasoning eval, greedy, 256 tokens, no tools for the external models -- *strict* requires our
`#### <answer>` line, *lenient* counts the last number; mt recall = the scripted 3-turn recall eval; needle =
effective context at min-over-depths >= 0.8 in each model's own tokens, real haystack; judged = claude-sonnet-5 on
the blind v1 suite, chat form. Base models get only the evals that have a completion form.)

What it says, honestly:
- **On the public benchmarks we are a 135M-class model, not a 360M-class one.** Against SmolLM2-360M (base) we trail
  by 9 points on HellaSwag, 13 on ARC-E, 19 on LAMBADA, 8 on SciQ; against SmolLM2-135M (base, 2T tokens) we are
  within noise on HellaSwag, PIQA, OBQA and SciQ and behind on ARC-E and LAMBADA. gpt2-medium is below us on
  everything except HellaSwag(n). That ordering is the training-token ordering: 10B versus 2T versus 4T versus 18T.
  Parameter count is not what these tables measure.
- **Where the post-training recipe shows.** Strict-format GSM8K: ours 0.085, every external model 0.000-0.005 --
  they were never taught our marker. Counting the last number instead, Qwen-Instruct solves 0.33 of GSM8K and 0.56
  of SVAMP to our 0.085 / 0.10: a model with 50x the pretraining and a real math SFT is a different class of
  reasoner, and our tool-use gains (§16a, §18) sit on a weak comprehension base. SmolLM2-360M-Instruct, closest in
  size, is at 0.075 / 0.17 lenient -- near us.
- **Chat quality.** Judged: SmolLM2-360M-Instruct 4.46 > Qwen-Instruct 4.31 > ours 4.11 > SmolLM2-135M-Instruct 3.94.
  We are between the two SmolLM2 sizes, on a suite written for our model, judged blind. Multi-turn recall follows
  the same order (0.734 / 0.750 / 0.609 / 0.516).
- **Facts.** The 360M base's 87% shows what the probe can read; we are at the 135M level (70.6%).
- **Context.** Every external model except gpt2 holds the needle at 4096 in its own tokens; ours holds 3072 (the M8
  4K stage's known erosion). The chat models needed the haystack as a user turn of their template (log,
  2026-09-27): on a raw completion prompt SmolLM2-360M-Instruct ends its turn instead of answering.
- **Not comparable, deliberately.** No external model has our tool protocol, so tool-use rate, sandbox verification,
  the selector and the tournament are n/a for them; the swarm eval reports only greedy / majority / oracle
  (Qwen-Instruct oracle 0.10 / 0.24 at k=16 without tools, ours 0.36 / 0.48 with them -- sampling with a sandbox
  gives us a wider pool than a stronger model without one).

The picture for the roadmap: the recipe (goal 1) is doing its job -- a 10B-token model behaves like a 2T-token
model of the same size on chat and beats it on format-following and tool use -- but nothing in post-training
substitutes for pretraining tokens on knowledge and comprehension, which is where every gap in this table is.
Rows and hover details are in the Evals tab (`runs/ext_<name>/`); re-run any of them with
`scripts/measure_external.sh <name>`.

## 21. Shape ablation before the third base: depth vs width at 149M, and z-loss (2026-09-27/28)

Four runs, same data (the M8 mixture without shell), same 600M tokens, cosine to the end, seed 0, 2K context;
val loss is plain cross-entropy on the 4M-token val mixture; benchmarks lm-eval zero-shot at limit 2000 on final.pt.

| run | shape | non-embed params | z-loss | val loss @600M | log Z (mean) | HellaSwag(n) | ARC-E | PIQA | LAMBADA | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| `m10_abl_wide_149m` | 18 x 768 (base_149m) | 123.9M | 1e-4 | **3.261** | 11.9 | 34.9 | 43.1 | 57.5 | 17.2 | 62.4k |
| `m10_abl_deep_149m` | 26 x 640 | 127.8M | 1e-4 | 3.271 | 12.1 | 35.2 | 42.8 | 58.0 | 15.3 | (14.1k, spilled) |
| `m10_abl_deeper_149m` | 32 x 576 | 123.9M | 1e-4 | 3.303 | 12.2 | 34.2 | 43.6 | 58.3 | 16.9 | 52.7k (mb4) |
| `m10_abl_wide_noz_149m` | 18 x 768 | 123.9M | 0 | 3.259 | 13.1 | 34.9 | 41.9 | 58.9 | 17.2 | 62.5k |

- **Depth does not win at this budget.** Val loss orders wide < deep < deeper (3.261 / 3.271 / 3.303), and the
  benchmark differences are inside the +-1-point noise of limit 2000. The deep run's gap to wide shrank through
  training (0.136 nats at 100M, 0.015 at 450M, 0.010 at 600M), so a longer run might close it, but there is no
  evidence it crosses, and the deeper shapes are 15-18% slower per token -- at a fixed wall-clock budget that is
  15-18% fewer tokens, which §20 says is the quantity that matters. **The third base keeps 24 x 1024.**
- **z-loss is harmless and does what it says.** With it, log Z sits at 11.9 instead of 13.1 (the term pulls the
  logit scale down by 1.2 nats) at a val-loss cost of 0.002 -- noise. It stays on at 1e-4 for the long run.
- Engineering note: 26 x 640 at microbatch 8 reserved 15.6 GiB and spilled into host memory (14k tok/s for ten
  hours); the run's numbers are unaffected, only its clock. Deeper shapes carry more activations per token at the
  same parameter count -- the microbatch has to come down with depth (32 x 576 ran at mb4, 52.7k tok/s).

## 22. Chat diagnostics before M10: the multi-turn behaviours we are about to train (2026-09-30)

`slm.eval.multiturn` now runs four scripted kinds on held-out tables (tables and rules disjoint from the training
families in `slm.rl.synth_chat`), greedy, no judge; the scorers are the RL verifiers themselves (`verify_recall`,
`verify_constraints`), so a gain here is a gain the reward can see. M9 output (`m9_rl6_336m` final.pt), n=32 per
kind, CPU:

| kind | what is scored | M9 output |
|---|---|---|
| recall | the fact stated two turns ago is in the reply | 0.50 |
| recall_absent | asked about something never stated: names nothing from the table *and* says so | **0.03** |
| revise | rewrite the previous answer under 1-3 constraints: share satisfied (all satisfied) | 0.42 (0.19) |
| sysrule | a system-prompt rule kept in a new reply: share (all) | 0.67 (0.59) |
| format / misfire | every turn terminated / any tool call in a chat turn | 0.95 / 0.00 |

The absent case is the finding: the model treats "what is my partner's hobby?" as a cue to produce *a* name from
the history, 31 times out of 32. Nothing in its training ever showed a recall question whose answer is "you did not
tell me" -- the same shape as the tool-misfire lesson (teach the negative case). Stage C of M10 adds the `recall`
family with one absent case in four, plus `revise` and `sysrule`; these five numbers are its gate, alongside the
judged suite and the hard suite.

## 23. M10: the prose-sprinkled base and the M9 recipe replayed on it (2026-09-30 → )

Plan (log, 2026-09-30): continue the M8 base with narrative prose rather than try to teach prose at SFT, then
replay the M9 post-training recipe stage for stage and compare each stage to its M9 counterpart.

### 23a. Phase 1: `m10_base_prose_336m` — the floor gate passes, prose moves at the base stage

Continued from `m8_base_4k` latest.pt (weights, AdamW moments, data cursors), 2B tokens at 4K, LR re-warmed to 1.3e-4
and decayed over the last 40%, z-loss 1e-4. Mixture: 75% replay of the phase-2 sources (fineweb-edu 45, cosmopedia 8,
finemath 5, python-edu 5, synth-retrieval 2, multi-turn SmolTalk 8, tool chats 2), 25% new prose: filtered Gutenberg
PG-19 12%, the curated adult-canon supplement 3% (about one pass), long fineweb-edu documents 10%. 21 h at 26.5k tok/s.

| | M8 base (`m8_base_4k` final) | **M10 base (`m10_base_prose` final)** |
|---|---|---|
| val, old distribution (phase-1 mixture) | 2.467 | **2.452** |
| HellaSwag(n) / ARC-E / PIQA | 41.7 / 57.5 / 66.4 | 41.8 / 57.1 / 66.6 |
| LAMBADA / OBQA(n) / SciQ | -- (M9 output: 33.7 / 31.6 / 82.7) | 36.5 / 33.0 / 82.5 |
| facts probe | 69.1% | 68.6% |
| needle (real haystack, min over depths >= 0.8) | 4096 | **4096** (4096: mean 0.97, min 0.875) |
| judged, completion form, mean over milestones | 3.03 (prose 2.62, narrative 3.40) | **3.45 (prose 3.26, narrative 3.63)**; the 2B final: 3.62 (prose 3.67, narrative 3.17) |

The gate: nothing on the old distribution moved outside noise -- the replayed loss is in fact 0.015 lower after the
decay, benchmarks and facts are within a point -- and the needle holds 4096 at every milestone. What did move is the
thing the run was for: judged prose +0.6 and narrative +0.2 in completion form, averaged over eight milestones (single
milestones swing by up to 0.3; the final checkpoint reads 3.62 overall, its prose 3.67 the highest of any base milestone). log Z went 12.96 -> 6.5 under z-loss
with no cost. Stage A (`m10_sft_336m`, the m9_sft recipe) started at 06:45.

### 23b. Stage A replayed: the same chat SFT on the new base judges 3.96 against 3.49

`m10_sft_336m` is `m9_sft_336m`'s config unchanged (the SmolTalk `-4k-think` sets, 200M tokens) on the M10 base.
Judged in chat form at all eight milestones, blind:

| | `m9_sft` (M9 stage A) | **`m10_sft`** |
|---|---|---|
| judged overall, mean over 8 milestones (range) | 3.49 (3.43-3.60) | **3.96 (3.76-4.08)** |
| prose / narrative | 3.08 / 3.10 | **3.72** / 3.02 |
| facts / definition / qa (at 200M) | 3.08 / 4.17 / 3.50 | **4.62 / 4.67 / 4.00** |
| python / arithmetic (at 200M) | 4.22 / 2.83 | 4.17 / 2.92 |
| SFT val / old-distribution val | 1.317 / 2.478 | **1.270 / 2.460** |
| HellaSwag(n) / ARC-E / LAMBADA | 42.7 / 58.4 / -- | 42.5 / 56.0 / 38.1 |
| facts probe / needle | 70.1% / 4096 | 71.6% / 4096 |

The distributions do not overlap: M10's worst milestone beats M9's best. The gain is wider than prose -- definition,
qa and the judged facts category (the coherence of factual answers; the facts *probe* is flat) all move, and the
same SFT data fits better (val 1.270 vs 1.317): a base that has read more kinds of English reads chat data better.
Narrative is flat and noisy at three items. ARC-E 56.0 vs 58.4 is the one number outside the usual +-1 and is
carried forward as a watch item, not a verdict (stage C of M9 read 57.0-57.6).

### 23c. Stage B replayed: judged 3.92 against 3.44, misfire halved, judged python down

`m10_tool_336m` is `m9_tool4_336m`'s config unchanged (the v4 mixture, 120M tokens) on `m10_sft`. Eight milestones each:

| | `m9_tool4` (M9 stage B) | **`m10_tool`** |
|---|---|---|
| judged overall, mean over 8 milestones | 3.44 | **3.92** |
| prose / narrative | 3.34 / 3.94 | **3.82** / 3.60 |
| definition / facts (judged) / pattern | 4.29 / 2.74 / 2.02 | **4.73 / 4.05 / 3.03** |
| python (judged) | 3.90 | **3.54** (3.17-4.11 vs 3.44-4.22) |
| tool misfire at 120M | 0.25 | **0.11** |
| algebra / word (accuracy / tool use) | 0.35 / 0.00, 0.89 / 0.54 | 0.43 / 0.00, 0.92 / 0.51 |
| GSM8K / SVAMP (accuracy / tool use) | 0.045 / 0.30, -- | 0.045 / 0.28, 0.067 / 0.44 |
| facts probe / needle | 72.7% / 3072 | 73.2% / 3072 |
| HellaSwag(n) / ARC-E | 42.8 / 57.8 | 42.5 / 55.4 |

Two new signals beside the repeat of 23b: the tool-misfire rate at the stage that produces it is less than half of M9's,
and judged python is lower at every milestone (the judges name `find_max` using the forbidden `max()` and
`distance_to` ignoring its argument). ARC-E reads 2.4 points under M9 for the second stage running. Both go to stage
C as watch items; M9's stage C recovered python to 4.2.

# Results and measurements

Numbers that change as runs finish. Update this file when a run completes or an evaluation is run;
the command center shows the live version of the same data. Last updated 2026-09-14 08:05.

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

Conclusions baked into the configs: 2K × mb 8 (M2/M3a/M4), 8K × mb 2 with `loss_chunk_size 4096`
(M3b: 41.9K tok/s, 10.9 GiB; the PaLM-style MFU overstates efficiency at 8K because it counts full attention), 16K × mb 1 with gradient checkpointing (M7). In
production the M2/M3a runs sustain 61–63K tok/s, slightly above the benchmark.

## 2. Data volumes (tokenizer v1)

| Source | Train tokens | Val tokens | Documents | Notes |
|---|---|---|---|---|
| fineweb-edu | 3.04B | 15.2M | 2.90M | sample-10BT files 0–1 |
| fineweb-edu-b | 5.32B | 27.1M | 5.10M | files 0–6, min 16 tokens (M3 mixture) |
| fineweb-edu-long | 730M | 3.6M | 88K | docs ≥ 4096 tokens |
| cosmopedia | 538M | 2.7M | 749K | |
| finemath | 455M | 2.4M | 309K | |
| python-edu | 140M | 0.7M | 248K | |
| stack-edu-shell | 51M | 0.2M | 59K | |
| tinystories | 463M | 4.6M | 2.10M | M1 only |

SFT shards (`C:\slm-data\sft\v1`): smol-magpie-ultra 121K examples / 162M tokens (88% targets,
16K dropped for length), openhermes-100k 94K / 36M, systemchats-30k 34K / 20M, smol-constraints 34K /
7M, everyday-conversations 2.3K / 0.4M, metamathqa-reasoning 44K / 10M, gsm8k-reasoning 7.4K / 1.4M,
synthetic-reasoning 40K / 3.2M. numina-cot-100k (105K / 54M) is prepared but unused. Raw downloads
total 20 GB, tokenized shards 21 GB, SFT shards 1 GB.

Tokenizer: 32,768 ids, sha256 `c2a7b5dbd660944b79fd5934b919dec4d22cb170cff9e5b68d902ff03433ac7e`.

## 3. Runs

| Run | Stage | Init | Tokens | Wall | Final train / val loss | Notes |
|---|---|---|---|---|---|---|
| m1_tinystories_26m | pretrain, 26M | random | 600M | 52 min (192K tok/s) | 1.336 / 1.391 (ppl 4.0) | Coherent stories; diagnostics clean (0% dead units, all layers useful) |
| m2_base_149m | pretrain, 149M, 2K, WSD | random | 1.00B | 4.6 h (60.7K tok/s) | 3.060 / 3.051 | Pre-decay snapshot `snap_800M.pt` (val 3.20) seeds M3a |
| m3_base_stable_149m | pretrain, constant LR (stable phase) | m2 snap_800M | 3.40B | 15.2 h (62.2K tok/s) | 2.768 / 2.878 | Finished 09-14 02:23; val 3.168 → 2.878 with the LR still flat (decay happens in M3b); weights have seen 4.2B tokens |
| m3_base_8k_149m | pretrain, 8K context, long-doc mixture, WSD decay (last 60%) | m3a final | 800M | 5.4 h (41.3K tok/s) | 2.683 / 2.693 | **The base checkpoint** (weights have seen 5.0B tokens). Val is on the 8K mixture, so not comparable to the 2K numbers; decay took it 2.82 → 2.69. GPU peak 72 °C, no throttle warnings |
| m4_sft_rehearsal_149m | instruct SFT (rehearsal on the 1B base) | m2 final | 450M (2 epochs) | 2.1 h | 1.629 / 1.878 | Pretraining-mixture val drifted 3.097 → 3.208 |
| m5_reasoning_rehearsal_149m | reasoning SFT (rehearsal) | m4 rehearsal final | 45M (3 epochs) | 13 min | 0.554 / 0.589 | Pretraining val 3.27 → 3.31 |
| m6_rl_arith_rehearsal_149m | GRPO stage A (rehearsal) | m5 rehearsal final | 200 steps, 259K completion tokens | 11 min | held-out acc 0.33 → 0.48 | KL ≈ 0.02, no malformed completions, no length blow-up |

Validation loss trajectory of the 149M base (2K context, nats per token):

| Tokens | 0.1B | 0.2B | 0.5B | 0.8B | 1.0B (M2 end, after decay) | 1.8B | 2.4B | 2.8B | 3.2B |
|---|---|---|---|---|---|---|---|---|---|
| M2 (WSD) | 4.593 | 3.812 | 3.350 | 3.201 | 3.051 | | | | |
| M3a (constant LR from 0.8B) | | | | 3.168 (at +0.1B) | 3.010 (+1.0B) | 2.946 (+1.8B) | 2.914 (+2.4B) | 2.898 (+2.8B) | 2.882 (+3.2B); 2.878 at +3.4B (end) |

(M3a counts tokens from its own start; add 0.8B for tokens seen by the weights. The final decay
happens in M3b, so M3a's loss is a stable-phase loss and will drop further at decay.)

## 4. Evaluations

lm-evaluation-harness, accuracy (acc_norm in parentheses):

| Checkpoint | HellaSwag | ARC-Easy | PIQA | Limit |
|---|---|---|---|---|
| m2_base_149m final (1B tokens) | 27.6 (29.0) | 47.1 (41.5) | 60.3 (58.7) | full |
| m4_sft_rehearsal_149m final | 31.7 (36.5) | 43.7 (41.0) | 61.8 (59.4) | 2000 |

Reference points: random is 25% / 25% / 50%; GPT-2 small (124M, ~10B tokens) scores about 29–31 on
HellaSwag; SmolLM-135M (600B tokens) about 42.

Reasoning benchmark (`slm.eval.reasoning`, greedy, held-out prompts; n = 100 per task, arith1 n = 10,
GSM8K test n = 200):

| Checkpoint | arith1 | arith2 | arith2mul | arith_multi | algebra | word | GSM8K | malformed (GSM8K) |
|---|---|---|---|---|---|---|---|---|
| m5_reasoning_rehearsal_149m | 60% | 25% | 22% | 0% | 31% | 58% | 2.5% | 9.5% |
| m6_rl_arith_rehearsal_149m (200 GRPO steps on arith1/arith2) | 80% | 44% | 36% | 0% | 29% | 55% | 2.5% | 8.5% |

RL moved the trained tasks (arith2 +19 points, and arith2mul +14 without being trained on it) and left
the others within noise; GSM8K is beyond this model at this stage.

Long-context needle eval: the M2 attempt failed on a RoPE-table assertion at 8192 (the eval now reserves
answer room); no needle numbers yet. Runs for the 8K base and the 16K extension are planned.

## 5. Diagnostics

`scripts/diagnose.py` on the final checkpoints (validation batches from fineweb-edu / tinystories):

- m1_tinystories_26m: base loss 1.441; 0% dead SwiGLU units in every layer; skipping any single
  block raises loss (layer 0 by +3.70, others +0.21 to +0.54).
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

# Project log

Dated entries, newest last. Incidents, decisions and their reasons. Numbers live in `results.md`.

## 2026-09-12 — brief, planning, scaffold

- Brief adopted: from-scratch SLM stack on one RTX 4080 SUPER, milestones M0–M7, teaching orientation.
- Decisions from Peter: code and checkpoints in the repo, training data on `C:\slm-data`; English +
  Python + bash only; own tokenizer with 64 reserved specials (chat, think, tool tokens reserved now);
  text `#### answer` convention; think span mandatory after reasoning SFT; no teacher models or
  model-generated data; robust logging, checkpointing, hourly self-contained HTML reports, interruptible
  runs; first serious model kept small (149M) to allow analysis and ablations.
- M0: environment (torch 2.14+cu130, triton-windows, compile works; flash SDPA unavailable, cuDNN is the
  fast backend), model, config system, tests, data registry, download and Software Heritage fetch,
  tokenizer, shards, loader, trainer, checkpointing, HTML report.
- Command center plan drafted by a planning agent and accepted (FastAPI + vendored Preact, torch worker
  subprocess with GPU guard). Playwright Chromium download approved. The web app is a supplement; the
  chat session stays the primary interface.

## 2026-09-13 (night) — M1, benchmark, M2 launch

- M1 (26M, TinyStories, 600M tokens) finished in 52 minutes, val 1.391, coherent stories, diagnostics
  clean.
- Incident: the first 149M benchmark was piped through `grep`, silently kept running and shared the GPU
  with M1 for 45 minutes (M1 at a quarter speed, a sampling step stalled). Killed it, resumed M1 from
  its checkpoint. Rules: unbuffered logs to files, verify GPU jobs are gone.
- Incident: cuDNN attention re-plans per KV length; generation crawled. All decode paths now use the
  efficient backend (`sdpa_context("decode")`).
- Benchmark: 2K × mb8 = 58K tok/s at 69% MFU; mb16 "worked" at 1K tok/s because WDDM spills to host
  memory instead of OOM. Peaks are kept near 11 GiB; the benchmark flags spills.
- Incident: re-tokenization failed with EINVAL while the portal held a shard memmap; portal no longer
  caches memmaps, checkpoint writes retry on lock errors. RNG restore needed CPU tensors.
- M2 (149M × 1B tokens, 2K, WSD) launched at 03:45. Decisions taken without asking: WSD so the pre-decay
  800M snapshot can be continued; bulk at 2K with the 8K phase last; Numina CoT kept but unused
  (too hard for 150M); reasoning SFT from GSM8K + MetaMathQA (numeric answers) + synthetic traces.
- Built while M2 ran: SFT pipeline and configs, reasoning-SFT data, RL stack (tasks, verifier,
  advantages, objectives, rollouts, GRPO trainer), long-context eval, lm-eval adapter, portal pages
  (runs, data browser, tokenizer, model harness, architecture explorer).

## 2026-09-13 (day) — M2 done, rehearsal pipeline, M3a, portal review

- M2 finished at 08:20: val 3.051, HellaSwag 27.6/29.0, ARC-Easy 47.1, PIQA 60.3; all 18 layers
  useful, no dead units.
- Rehearsal of the whole post-training pipeline on the 1B base: M4 SFT (val 1.878, pretraining val
  drift 3.05 → 3.21), M5 reasoning SFT (val 0.589), RL stage A 200 steps (held-out accuracy 33% → 48%;
  benchmark arith2 25% → 44%). RL launch bugs fixed on the way: train/held-out split leaked across
  overlapping generators (now hashed by prompt text); the parser marked everything malformed because
  the prompt already opened the think span (parser now keys on the closing tag); duplicate
  `grad_norm` kwarg. Added an end-to-end GRPO smoke test.
- M3a launched at 11:10 with microbatch 16 by mistake → spill; killed and relaunched with mb 8
  (61.8K tok/s, 12.1 GiB). ETA ~15 h. The run dir is `runs/m3_base_149m_stable`; a background job renames
  it to `m3_base_stable_149m` on exit (Windows locks open files).
- Command center reviewed against real data at Peter's request. Fixed: finished runs showing 99%
  (max tokens over all records), page hang from `<|bos|>` inside htm templates, chart crash on nulls,
  derived/SFT sources invisible, checkpoint metadata from filenames → `index.json`, elapsed from trainer
  counters, RL runs at 0% progress (now by step) and a misleading "train loss 0" (policy objective is
  zero-mean by construction; RL tiles show reward/held-out/KL/length instead), chat datasets not
  browsable and the tokens view hanging (transcripts + chat-formatter tokenization with loss mask).
- UI decisions from Peter: Home and Runs merged into one Overview; tokenized and raw modes merged into
  one Documents browser with a text/tokens/ids toggle; Model tab renamed Inference with a raw-text vs
  tokens view that keeps reserved tokens visible; consistent run naming
  `m<milestone>_<role>_<params>` applied retroactively; all datasets browsable, pay-for-play.
- Question answered in chat: context extension changes no parameters (only RoPE tables/scaling,
  attention cost, activation memory, KV cache); weak factual recall and arithmetic come from scale and
  token count, not from the 2K context; short-context loss is tracked through every extension phase.

## 2026-09-13 (evening) — telemetry, layout, documentation

- GPU temperature/power/utilization sampled from `nvidia-smi` on a daemon thread, attached to every
  train record (pretraining, SFT and RL trainers), charted next to VRAM on run pages and in the HTML
  report, with a console warning and a `warn` event above 80 °C (rate-limited). M3a keeps running
  without it; M3b onward has it.
- Portal boot diagnostics: a failed module import or a page crash now shows an error panel instead of a
  blank page (a tab loaded during a server restart had gone blank). Documents browser fills the
  viewport; all pages fit the available width (table cells wrap, grids shrink, Overview pipeline table
  reduced from 14 to 10 columns, Inference selector shrinks, GQA caption moved out of the SVG).
- Documentation restructured: README as a stable high-level description; `docs/design.md`,
  `docs/runbook.md`, `docs/results.md`, `docs/command_center.md`, `docs/roadmap.md`, this log. The
  morning status report and the command-center plan were folded into these files.
- Config fix: `m7_ctx16k_149m.yaml` pointed at a stale base path; now `runs/m3_base_8k_149m/checkpoints/final.pt`.
- M3a at 2.46B of 3.4B tokens (val 2.914), ETA ~02:15 on 09-14.

## 2026-09-14 — M3a finished, M3b

- M3a finished at 02:23: 3.40B tokens in 15h 13m at 62.2K tok/s, val 2.878 (train 2.768) with the LR
  still flat. The background job renamed the run dir to `runs/m3_base_stable_149m`.
- 8K benchmark: mb 1 = 40.5K tok/s / 6.6 GiB, mb 2 = 41.9K tok/s / 10.9 GiB (no spill).
- M3b (`m3_base_8k_149m`: 8K context, long-doc mixture, WSD decay over the last 60% of 800M tokens)
  launched at 02:27 from `m3_base_stable_149m/checkpoints/final.pt`; expected ~5.5 h.
- M3b finished at 07:51: 800M tokens in 5h 24m at 41.3K tok/s, val 2.693 on the 8K mixture (2.82 before
  the decay). `runs/m3_base_8k_149m/checkpoints/final.pt` is the base checkpoint (5.0B tokens seen).
- The rehearsal runs on the 1B base were renamed to free the real names: `m4_sft_rehearsal_149m`,
  `m5_reasoning_rehearsal_149m`, `m6_rl_arith_rehearsal_149m` (their `run.json` lineage updated).
- `scripts/pipeline_after_m3.sh` launched at 08:05: base evals (lm-eval, diagnostics, needle 1K–8K) →
  M4 → M5 → M6 stage A → new M6 stage B (`m6_rl_multi_149m`: arith2/arith2mul/arith_multi/algebra/word,
  300 steps) → M7 16K → needle/lm-eval on M7. Each training run has a spill guard (STOP if VRAM ≥ 14 GiB
  or tok/s < 3K after warmup). Expected ~9 h total.
- Base evals (07:53–07:58): HellaSwag 29.3/32.7, ARC-Easy 51.6, PIQA 64.0 (vs 27.6 / 47.1 / 60.3 for the
  1B-token M2); needle retrieval 100% up to 4K, and at 8K only the depth-0.1 cell fails; diagnostics clean.
- M4 (real base) finished 10:05: val 1.660, pretraining val drift 2.81 → 2.89; lm-eval HellaSwag 33.1/39.6.
  M5 finished 10:18: val 0.525; reasoning benchmark arith2 50%, word 68%, GSM8K 2%.
- Incident: the pipeline's spill guard read tok/s = 0 for the RL run (RL logs steps, not tokens/sec)
  and stopped M6 stage A at step 125 (held-out acc already 0.57). Guard fixed (throughput test only when
  reported), the script got a resume-from-stage argument, chain relaunched from M6 at 10:29; the RL
  trainer resumed from `latest.pt` at step 125.
- M6 stage A finished 10:33 (held-out 0.58; arith2 61%, arith2mul 49%). Stage B (`m6_rl_multi_149m`,
  300 steps) finished 10:59 with little movement: arith_multi produces no correct samples, so 38% of
  groups have zero advantage. Benchmark after B: arith2 61%, algebra 32%, word 67%.
  Lesson for stage C: the curriculum needs tasks the policy sometimes solves, or partial credit.
- M7 (16K YaRN extension, 200M tokens, mb 1 with gradient checkpointing) started 11:01.
- M7 finished 13:21: 200M tokens at 16K in 2h 19m (24.5K tok/s, 4.3 GiB with gradient checkpointing;
  microbatch 2 without checkpointing would likely be faster next time). Val 2.712 on the 16K mixture.
  Needle: depth-0.9 retrieval 100% at every length to 16K; early-depth cells at 12–16K fail. Short context
  unchanged: 2K loss 3.069 vs base 3.076, lm-eval within ±1 point at the same 2000-sample limit.
- Gap noticed: the M7 config had no `extra_val_mixture`, so the short-context loss was not tracked during
  the run (checked afterwards with the diagnostics loss instead). Added to the config for future runs.
- Pipeline finished 13:22. The full M0–M7 stack has now run end to end on the real base.

## 2026-09-14 (evening) — needle retrieval is the problem; context curriculum

- Peter's direction: long context only counts if needle retrieval holds; work up from 2K and stop wherever
  retrieval breaks, 32K is optional.
- Needle eval v2: real validation text as haystack (boundaries stripped), 7 depths, n = 16, failure samples,
  effective-context summary; the trainer can run it at every eval (`eval.needle_lengths`) and the portal /
  report chart `needle_<L>`. Measured (real text): 2K model effective 1K (71% at 2K); 8K base effective 1K
  (79% at 2K, 35% at 8K); first 16K run effective 2K (88% at 2K, 31% at 16K). The filler haystack had
  overstated retrieval by ~20 points.
- New source `synth-retrieval` (`slm.data.synth_retrieval`, templated, no model): 150M tokens of real
  training text with inserted facts + questions at the end (70%) and key-value ledgers (30%), 512–16K tokens.
  Bug caught by inspection before use: two facts from the same template in one document made a question
  ambiguous; templates are now distinct per document (tested).
- Stage 1 `m7_ctx8k_retrieval_149m` launched 21:07 from the base: 600M tokens at 8K, 15% retrieval docs,
  needle tracked at 1K/2K/4K/8K every 50M tokens; gate to 16K is min-depth ≥ 80% at all four lengths.
- Tool-use project queued (Peter): GSM8K at 2% is arithmetic compounding per step; a tool removes it. Peter asked
  for Python rather than a bare calculator, sandboxed simply but safely, erring on the side of safety. Built
  `slm/tools`: sandboxed Python-subset interpreter (no exec/eval, no imports, no attributes, budgets), protocol
  on the reserved tool tokens, batched tool loop, SFT/RL masks, evals, GSM8K-train RL tasks, tool SFT data,
  configs `m5_reasoning_tools_149m` / `m6_rl_gsm_tools_149m`, `scripts/pipeline_tools.sh` (starts after the
  context pipeline). 56 new tests (sandbox refusals and limits, protocol round trip, loop bookkeeping).

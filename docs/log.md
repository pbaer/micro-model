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
- Tool protocol revised at Peter's request: the four specials are now `<|python_call|>` / `<|/python_call|>` /
  `<|python_result|>` / `<|/python_result|>` (same ids, renamed in the tokenizer meta; the sha covers only the
  BPE so old checkpoints are unaffected) and the call body is plain code (no "python:" prefix). Future tools will
  be Python functions in the session namespace. `PySession` gives REPL semantics: variables persist across calls
  and across turns of one conversation (one session per conversation in SFT conversion, per row in rollouts).
  Incentive to use the tool: RL reward scheme `tool` pays 1.0 only when the final number came out of a
  non-trivial call (print(42) does not count), 0.5 for a correct answer computed in the head; evals report
  tool_use_rate and answer_from_tool_rate. Sandbox errors now carry line number, source line and a hint.

## 2026-09-15 — tool protocol details, command-center chat

- Tool calls are bare expressions (REPL echo; no print()); the echoed number after a result is dropped from
  training data; calls are only allowed inside the think span (SFT conversion, generation loop, parser and
  reward all enforce it). Tool SFT data regenerated (GSM8K example: 122 -> 107 tokens).
- Inference page: multi-turn chat with the Python tool. Well-formed assistant turns are appended to the
  conversation with their exact token ids (no re-execution of calls) and an empty user turn; one sandbox
  session per conversation on the worker; inserted result tokens shown distinctly; python-calls panel;
  `new conversation` resets the session. Harness tested in-process with a scripted sampler.
- Stage 1 finished 01:14 (600M tokens, 4h 06m). Gate measurement: effective context 6000 — perfect through 4K,
  88% worst-depth at 6K, but 0–6% for needles at depth 0–0.1 of 8K, as near-miss copies (last digits wrong).
  Diagnosis: the retrieval data was log-uniform over 512–16K tokens, so full-length recall from the start of an
  8K document was rare. Stage 2 correctly did not launch. Built `synth-retrieval-8k` (80M tokens, 3K–8K docs,
  `--early-frac 0.5`) and `m7_ctx8k_retrieval2_149m` (300M tokens from the stage-1 checkpoint); the gated
  pipeline now has a `stage1b` entry that waits for the tool track, then runs 1b → gate → stage 2.
- The tool track was released by hand at 01:27 (its wait condition only fires when the gate passes) and is
  running: M5 tools SFT → evals → M6 tools RL → full GSM8K with tools.
- Overnight: the machine rebooted at ~04:50. Tool SFT + evals had finished (GSM8K with tool 0.8%, tool use 93%,
  answers-from-tool 68%; templated tasks 98–100% except multi-step 34%). The tool RL was
  killed at step 190/400 — and had been collapsing since ~step 110 (entropy 0.8 → 5, KL → 0.28, garbage
  rollouts) with kl_coef 0.01; held-out accuracy still crept 9% → 12%. Archived as `m6_rl_gsm_tools_try1_149m`.
- RL trainer hardening: `best.pt` by held-out accuracy (index carries heldout_acc; read back on resume) and
  collapse guards `entropy_stop` / `kl_stop` that stop the run with a warn event. Attempt 2 config: kl 0.05,
  lr 1e-6, temperature 0.8, entropy_stop 2.5, kl_stop 0.15, 300 steps; evals use best.pt.
- Stage 1b started 07:40 (GPU was idle). Re-sequenced the day with `scripts/pipeline_day.sh`: stage-1b gate →
  tool RL attempt 2 + full GSM8K eval → stage 2 (16K) only if 1b passed → 16K gate. The earlier gated pipeline
  shell was stopped (not the training) to make room for the RL before stage 2.
- 09:48 stage 1b measured: 8K depth 0/0.1 = 12%/50% (from 0%/6%), all other cells 100%; effective context still
  6000, gate not passed. Correction to the earlier log line: the tool-SFT templated numbers were misquoted; the
  real ones are 98–100% on every templated task except multi-step arithmetic (34%).
- 12:26 tool RL attempt 2 ended at step 222 when the KL guard fired (0.159); no collapse (entropy stayed 0.4–1.1).
  best.pt = step 175, held-out 11.5% (from 9%). Full GSM8K with the tool: 1.3% (from 0.8%), malformed 18% (from
  30%), tool use 94%, answers-from-tool 78%; every templated task now 100% (multi-step 34% → 100%).
- 12:45 stage 1c launched (`m7_ctx8k_retrieval3_149m`: +300M tokens, 20% of a 6K–8K-only source with 80% early
  facts). Decision pending if it still misses the 8K gate: report effective 6K, or extend anyway with the caveat.
- The Claude session crashed around 13:00; stage 1c finished on its own at 14:47 and the GPU sat idle until
  19:05. Stage 1c measured: 8K depth 0 = 50%, depth 0.1 = 94%, everything else 100% (effective 6000 by the
  strict gate). Stage 1d (`m7_ctx8k_retrieval4_149m`, same recipe, +300M) launched 19:15; the gated pipeline
  (`STAGE1=m7_ctx8k_retrieval4_149m bash scripts/pipeline_ctx.sh`) measures it and starts stage 2 if it passes.
- 21:21 stage 1d gate: 8K depth 0 = 44% (no longer improving), depth 0.1 = 94%, all else 100%; effective 6000 by
  the strict gate; stage 2 not launched. Depth probe at 8K: failures are confined to the first ~160 tokens after
  `<|bos|>` (31–69%), 94% at 400 tokens, 100% from 800 tokens — a position artifact, not a distance limit.
  Decision on how to proceed (redefine the gate to skip the sink zone vs stop at 8K vs investigate) left to Peter.
- Inference page: stage badges and a base-checkpoint warning in chat mode; think switch and greedy default follow
  the loaded checkpoint; RL training snapshots labelled. Peter's chat test had used the collapsed try1 snapshot.
- 2026-09-16 00:15 — corrected diagnosis: shallow needles pass 100% at 4K–6K and degrade from 7K (7.5K: 62–94%, 8K:
  38–88%), so the blind spot is the edge of the trained window (max relative distance), not the first tokens.
  Strict effective context 7K. Recommendation to Peter: extend to 16K, gate on 8K-all-depths + 16K-from-5%.
- 01:00 Peter's redirect: strength per parameter first, context second (≤ 5% cost), use the VRAM for a bigger model.
  Benchmarked 211M/323M/440M/565M candidates (table in results.md): 323M keeps 74% MFU at 30K tok/s, 440M drops to
  66% at 20K, 565M is the memory ceiling. Recommended 323M on ≥ 10B tokens; started downloading FineWeb-Edu files
  7–13 for the larger token budget.
- 09:00–13:00 Peter chose 323M-class / 10B tokens and asked for an architecture and data-mix sanity check. Adopted:
  8 KV heads (16q/8kv, +12.6M params → 336M), RoPE base 500K, a 2K→4K sequence schedule inside the run (phase 2 at
  4K carries the decay), chat and tool conversations mixed into the decay phase (SmolTalk subsets and the tool SFT
  sets converted to pretraining sources; Peter: pretraining need not be human-text only), peak LR 4e-4, 524K-token
  updates. Data: all 14 FineWeb-Edu 10BT files tokenized (10.07B tokens); a second Software Heritage fetch for
  ~1M Python files started (python-edu would otherwise repeat 3.6× over 10B tokens).
- 13:07 M8 phase 1 (`m8_base_stable_336m`) launched via `scripts/pipeline_m8.sh` (phase 2 and measurements chained):
  30K tok/s, 13.8 GiB, ETA ~3 d 7 h. README/design/roadmap/results/CLAUDE.md updated for the second base.
- Built a factual-recall probe (`slm.eval.facts`, 194 items over 7 categories, completion or chat form) so
  "reasonable factual knowledge" is measured: the 149M base scores 43% (capitals 64%, units 10%), the instruct
  checkpoint 46% in chat form. This is the reference the 336M base must beat.
- Post-training data for the 4K base prepared while phase 1 trains: SmolTalk SFT sets rebuilt at 4096 tokens
  (`*-4k`), a multi-turn tool-conversation generator (`slm.rl.synth_multiturn`, 19K conversations, follow-ups
  reuse the REPL variable; tested for consistency), and `tool-chat` rebuilt to include it (67K docs, 7.5M tokens)
  before phase 2 samples it.
- 13:43 Python swap: the Software Heritage fetch delivered 1.3M files → `python-edu` now 589M tokens (old shards
  kept as `python-edu-v1`). Phase 1 was stopped gracefully at 63M tokens, the directory swapped under the same
  source name (the loader's stream state keys by name), and the pipeline relaunched; it resumed at update 121 at
  29.6K tok/s. Cost: ~4 minutes.
- 15:30 first sanity check (250M tokens): train loss 336M 4.35 vs 149M 3.84 at 200M tokens — above the reference,
  but the comparison is not like-for-like this early: the 336M run uses 524K-token updates (half the optimizer steps
  per token) and a 100M-token warmup vs 262K / 20M for the 149M run. Per update it is ahead (4.35 at update 381 vs
  4.61 for the 149M at its update 381) and its slope is steeper (−0.41 vs −0.26 per 50M). Kept running; the
  decisive check is val at 500M (must be ≤ 3.35) and 750M (≤ 3.22), the 149M values.
- 17:52 500M check: val 3.358 vs the 149M's 3.350 at the same token count — the gap closed from 0.75 (100M) to
  0.008 with the 336M curve still falling at 0.14 per 100M vs 0.085; it crosses below by ~600M. Needle retrieval
  switched on at 500M (92% at 1K, 75% at 2K). Continuing. Next check at 1B: the constant-LR peer is M3a at 3.17
  (M2's 3.05 includes its decay).
- 20:46 second data swap (Peter: expose the model to more data even if the gain is minor): cosmopedia 538M → 1.62B
  tokens (4 more files), finemath 455M → 1.36B (6 more files), shell 51M → 505M (560K scripts via Software
  Heritage), synth-retrieval 150M → 400M (new seed). Graceful stop at 808M tokens, directories swapped under the
  same names (old shards kept as `*-v1`), resumed at update 1541. Every source in the 10B plan is now ≤ 1.2 epochs.
- 21:40 Peter caught a data defect: GSM8K/MetaMathQA SFT rows answered with `#### N` although the user turn never
  asked for it, so reasoning SFT taught the marker as the default answer style (visible in the chat UI). Added
  `slm/data/answers.py` (50/50 instruction+marker vs bare question+sentence, consistent per conversation), applied
  it in SFT prep, both templated generators and the multi-turn set; regenerated all seven sets and `tool-chat`
  (phase 2 samples the corrected version; phase 1 uses none of this data). Lenient verifier for bare answers.

## 2026-09-17
- 00:09 in-run needle sample size raised from 4 to 16 over five depths (0/0.25/0.5/0.75/1.0) for both M8
  phases, after a single 92% cell at 2048 turned out to be one miss out of twelve. Paid for by batching:
  every prompt of a given length is exactly that length, so `answer_all` decodes a whole cell at once
  (`eval.needle_batch_tokens`, 16384 prompt tokens per batch, VRAM taken from the headroom that exists
  while training activations are freed). 80 samples per length now cost roughly what 12 cost before.
  Applying it needed a restart: graceful stop at 1.15B tokens, resumed at update 2189. GPU idle 10.5 min,
  most of it my own mistaken wait loop (it polled for *any* python.exe and the portal is one).
- 08:04 the batched needle eval cost 8% of training throughput, permanently. Throughput held at 30,030
  tok/s for ten log lines after the 00:09 restart and fell to 27,550 at the exact update of the first
  eval, then stayed there: 19,030 ms/update instead of 17,500, 282 W instead of 303, 100% util, no
  throttle flags, `nvidia-smi` showing 15.9 GiB of 16.4 GiB in use. Cause is reserved memory, not peak
  allocated (unchanged at 13.8 GiB): the generation KV cache (8 rows x 2060 tokens x 49 KB = ~800 MB)
  needs contiguous segments the training blocks cannot supply, so the allocator took new ones from the
  driver and kept them, and WDDM started paging. Fix: `torch.cuda.empty_cache()` after the needle run,
  `needle_batch_tokens` 16384 -> 8192, and `vram_reserved_gib` now logged next to the peak (console shows
  `peak/reserved`). Restarted at 1.92B tokens, 22 s downtime, back to 30,035 tok/s at 13.9 GiB reserved.
  Verified at the 2.00B eval: 30,035 tok/s before, 51 s eval, 30,009 tok/s after, reserved 13.9 -> 14.0 GiB
  (the residue is 0.1 GiB, not the ~1 GiB that was being kept). Needle 99% at both 1024 and 2048.
- 22:10 needle worst-depth at 2048 dipped to 37.5% at 3.30B tokens (mean 87.5%), recovered to 87.5%/95% at
  3.40B while val loss improved monotonically through it (2.7536 -> 2.7477 -> 2.7389). Not a model problem:
  2048 is the trained window edge (1024 has been a flat 100% throughout) and each eval drew a *different*
  random haystack, so cell-to-cell variance rode on top of the length effect. Two changes, which land at the
  phase 1 -> phase 2 handover with no restart: `eval.needle_seed` (default 0) fixes the haystacks so the curve
  tracks the model rather than the draw, and each eval record now carries `retrieval_by_depth` plus
  `retrieval_worst` with failure examples. Both keys deliberately avoid the `needle_` prefix, which is what
  selects the chart series.

## 2026-09-18
- 00:30 judged-quality eval (Peter: score our periodic sample prompts with a cheaper judge model on a few rubrics,
  broaden the prompt set to what a model this size should eventually do well, run it retroactively on every
  stage with CPU inference, chart it). `slm/eval/quality_suite.py` (35 prompts, 9 categories, completion + chat
  forms, expectations, rubric text) and `slm/eval/quality.py` (generate / pack / ingest / summary / status).
  CPU generation beside the live run: 34 tok/s for 336M, 53 s per checkpoint, training throughput unchanged
  (29.9K tok/s). Judge = a Claude subagent on Sonnet, blind to checkpoint identity (shuffled packets, opaque
  ids), scores validated on ingest. Run page: two charts + a `quality` tab. Phase 2 config generates the suite
  on the GPU at every milestone (`eval.quality_suite: true`). Protocol: docs/quality_eval.md.
- 23:45 needle sweep (`slm.eval.needle_sweep`): every snapshot re-measured at n=64 with one fixed haystack draw,
  overlaid on the run page's needle chart. Measured cost: 35 s per 336M snapshot on the GPU (estimate), 1769 s
  (29.5 min) on CPU with 24 threads at 1K+2K, with training throughput unaffected (29.9K tok/s). Phase 1's 30
  snapshots are backfilling on the idle CPU (~14 h); phase 2 gets swept on the GPU at 1K/2K/4K once the pipeline
  finishes. The in-run tracker moved from n=4 to n=16 mid-run, which is why one consistent curve is wanted.

## 2026-09-19
- 00:05 CPU needle backfill stopped after two snapshots (Peter: too slow at 29.5 min each). Both M8 runs get swept
  on the GPU at 1K/2K/4K, n=64, when the pipeline reports done (~25 min per run); a watcher is armed for it.
- 12:10 Data page refactored around the mixture as the spine (docs/proposals/data_tab_refactor.md, approved with all
  four open questions answered yes). Two Opus 5 agents, ~40 min of wall-clock: recipes (config = plan, run = what ran,
  `plan_differs` flag), truthful mixtures for all three stages (the old page showed 0 available for every SFT set and
  for smoltalk-chat / tool-chat, and 500'd on RL configs), the raw / prepared / training-row inspector with the packed
  SFT window and mask, compare (default = init_from parent), chain (actual per-source exposure from the new
  `sources` checkpoint field where present), RL prompt sample + rollouts viewer, a catalog page replacing the
  sources / mixture / documents tabs, run-page data link, portal main process now torch-free at import (asserted).
  Training side: `PretrainLoader.consumed()` logged as `sources` in checkpoint/finish records; `prepare.py` writes a
  `shard_NNNNN.src.npy` raw-row sidecar for new shards.
- 11:30 phase 2 fenced off with a pre-placed STOP file (the pipeline chains it automatically; Peter wants to inspect
  the mid-training data first). The needle sweep of phase 1's snapshots runs on the GPU in that gap.
- 13:10 Peter: the model only needs to write Python; bash is out of scope from here and may fade. `stack-edu-shell`
  removed from the phase-2 mixture and from its drift-tracking `extra_val_mixture` (its 1.5% moved to python-edu,
  which is now 6.5%); scope lines in CLAUDE.md / README updated. The judged-quality `overall` now excludes the
  bash category for every run (the 3 bash prompts stay in the suite and are still scored, `overall_all` keeps the
  old definition); every summary.json regenerated. Phase 1 of the second base keeps its 1.5% shell to the end.
- 13:00 Peter: the tool datasets teach `<|python_call|>` as a calculator (measured: 0 of 87,593 spans contain a
  loop, def, list, string method, `if` or `math.`). Decision: before phase 2, build a grammar-generated,
  sandbox-verified Python-tool set covering the whole supported subset (plus a sandbox-filtered public source),
  and add declared functions (`<|python_def|>sig<|python_comment|>text<|/python_def|>`, masked, registered into the
  session) to the vocabulary and the data. Two Opus agents in sequence: plumbing, then the generator.
- 12:48 M8 phase 1 FINISHED: 7.50B tokens in 2d 23h 28m, best val 2.6081 at 7.4B (final 2.6086), 149M chain's
  equivalent 2.88; the power-law fit from 3B predicted 2.612 at 7.5B. The pre-placed STOP file ended phase 2 after
  one update (55 s); that stub run dir is set aside as `runs/m8_base_4k_336m_stub` so the real phase 2 starts
  fresh from phase 1's final.pt with the shell-free mixture (a resumed loader state would still carry the shell
  stream). The n=64 needle sweep of phase 1's 30 snapshots is running on the freed GPU.
- Declared functions landed (plumbing half of the 13:00 decision): three reserved slots named
  `<|python_def|>` / `<|python_comment|>` / `<|/python_def|>` (ids 32717-32719; naming a reserved slot changes no id
  and not the tokenizer sha256, and `SlmTokenizer` re-applies NAMED_SPECIALS by position at load time, so the frozen
  v1 tokenizer gains them without being retrained and every checkpoint's `tokenizer_sha256` still matches). Registry
  `slm/tools/functions.py` (`FunctionDecl`, `render_defs`/`parse_defs`, `functions_env`), `PySession(functions=...)` /
  `register()`, `format_chat(functions=...)` (masked blocks after `<|bos|>`, segment label `python_def`, impls
  registered so `<<code=result>>` markup calling them resolves at conversion time), `sample_with_tools(functions=...)`,
  and the portal's `POST /api/model/generate` `functions` field with the def blocks highlighted in the token view.
  The data generator builds on this API next.
- 15:40 Grammar-generated Python-tool set landed (the data half of the 13:00 decision): `slm/rl/synth_python.py`
  writes `synthetic-python-tools` (170K conversations, 28.4M train + 2.4M val tokens, 259K calls) from small
  grammars rather than templates — data pipelines (source -> 1-3 transforms -> aggregate, several idioms per
  step), strings, number theory and sequences, loop-with-state simulations, multi-turn REPL (a helper `def` or a
  variable defined once and reused across 2-4 turns), "run this code and tell me what it prints", error-and-recover
  (the first call really trips a sandbox hint, taken from a live `PySession`, never hand-written), declared-function
  services, and the old arithmetic word problems at 6%. Real sentences and words are drawn from the `fineweb-edu-b`
  val shards so inputs look like text. Everything is correct by construction: the gold is cross-checked against an
  independently computed host value and `format_chat(tools=True)` runs every program again at conversion time.
  Measured feature coverage of the call spans (the point of the exercise): loops 77%, list ops 61%, string methods
  20%, declared-function calls 11.0%, `def` 11.0%, `math.` 6.6% — against 0% for all of them in the old sets.
  Two whole families, `pipeline.dict` and `declared.distance`, are written to val only, so validation measures
  generalisation to unseen program shapes. `tool-chat-v2` rebuilt from the four old tool sets plus this one:
  36.1M train / 2.9M val tokens, 225K docs (the existing `tool-chat` dir is untouched; the swap is a rename at
  launch time). At 1.5% of phase 2's 2.5B that is ~0.96 epochs.
- 14:05 `tool-chat` swapped: the 7.6M-token calculator-only set is now `tool-chat-v1`; `tool-chat` is the rebuilt
  36.1M-token source (the four old sets + `synthetic-python-tools`, 30.8M: 170K grammar-generated conversations,
  9 families, 77% of calls with loops, 11% with `def`, 11% calling declared functions, 13 real error hints;
  hold-outs `pipeline.dict` and `declared.distance` in val only). At 1.5% of phase 2's 2.5B that is 0.96 epochs.
  Phase 2 is ready to launch on Peter's go (STOP file removed, stub run set aside, config shell-free).
- 13:55 Peter: "That makes eraser." reads wrong. `natural_answer` now picks numeric templates (that makes / that
  gives / comes to / the result is) only for numeric answers and text templates (the answer is / it is / that would
  be / so it's) otherwise. `synthetic-python-tools` regenerated (same seed) and `tool-chat` rebuilt (36.0M tokens;
  the previous build kept as `tool-chat-v2`).
- 14:20 Peter spotted a GSM8K trace in tool-chat stating "72 ounces ... because 12 x 6 = <call>": the answer before
  its own computation, which teaches that the tool call is decorative. Measured: 8.3% of gsm8k-tools calls, 2.8% of
  metamathqa-tools, ~1% of the synthetic sets. `hoist_calls` (slm/tools/protocol.py, applied by the SFT prep
  `--tools` path) moves the computation in front of such a sentence ("2 x 16 = <<2*16=32>>. He eats 32 pieces."),
  so a number only ever appears after the tool result. gsm8k-tools and metamathqa-tools regenerated, tool-chat
  rebuilt (previous build kept as `tool-chat-v3`).
- 14:35 Peter: are stored `<|python_result|>` spans exactly what the sandbox produces? Verified by replay
  (`scripts/verify_tool_results.py`: one fresh session per conversation, declared functions registered from the
  generator's service registry, calls replayed in order): 360,516 calls across the five tool sets, 10,937 of them
  stored error hints, 0 mismatches. A sampled version now runs in the test suite so a sandbox change that alters
  a hint or a number format is caught against the data.
- 14:32 M8 phase 2 (`m8_base_4k_336m`) launched on Peter's go: fresh start from phase 1's final.pt, 4K rows x mb 2,
  2.5B tokens, WSD decay over the last 80%, mixture fineweb 66 / cosmopedia 11 / finemath 7 / python-edu 6.5 /
  synth-retrieval 3.5 / smoltalk-chat 4.5 / tool-chat 1.5 (no shell; tool-chat = the rebuilt 36.0M-token Python-tool
  source at 1.04 epochs), needle n=64 at 1K/2K/4K with 32K-token batches and a fixed haystack seed, quality suite
  generated at every milestone, per-source stream state in every checkpoint record. Final measurements run at
  n=64. Expected ~27 h.
- 17:05 phase 2's first two evals took 421 s and 622 s (phase 1: 47 s). Training resumed at full speed after each
  (reserved back at 13.91 GiB thanks to empty_cache), so the paging was confined to the eval itself: with the
  allocator already at 13.95 GiB reserved, the 4K needle's 32K-token batches needed an fp32 KV cache of 3.2 GB
  (the cache took the model's master dtype) in contiguous segments the free pool could not supply, and WDDM paged
  for the duration. Fix: bf16 KV cache whenever generation runs under autocast (halves it), needle_batch_tokens
  8192 for phase 2 (2 rows at 4K). Restarted phase 2 to apply (~1 min).
- 18:15 Peter caught a real waste: phase 2 started its data streams at token 0, so it re-read what phase 1 had
  already trained on. Verified from the checkpoints: phase 1 ended at fineweb shard 54, phase 2 was at shard 2.
  Of its first 311M tokens, 280M (94%) were a second epoch; only the two new sources (smoltalk-chat, tool-chat)
  were fresh. Fix: `init_loader_from` (config) + `PretrainLoader.adopt_stream_positions`, pointed at the PARENT's
  latest.pt (final.pt is weights-only and carries no loader state); shared sources continue, new ones start at 0,
  dropped ones are ignored. The live run's latest.pt was patched the same way and resumed at 311M rather than
  restarted: per data-constrained scaling a 2nd epoch is worth ~95% of fresh data, so the 280M cost ~0.6% of the
  phase's value, against 3.2 h (12% of wall clock) to redo it.

## 2026-09-20
- 01:50 4K retrieval trigger fired at the 1B milestone. Depth 0 (a needle at the very start of a 4096-token
  window) peaked at 75% around 700M and then fell for four consecutive evals during the decay: 75 / 55 / 39 / 23%.
  Depth 0.25 held at ~80-97%, 0.5 and beyond at 100%, and 1K/2K are 100%. Same window-edge failure as the 149M
  model: full-window dependencies are rare in packed rows of ~1K-token documents, and as the LR decays the model
  specialises toward the dominant web text. Acted on the pre-announced trigger: swapped `synth-retrieval` (400M,
  512-16K docs, facts anywhere) for the contingency set built yesterday (150M, 3072-4096-token docs, 80% of facts
  in the first 15%), same source name and 3.5% weight, so every full-width row carries a start-to-end dependency.
  Old corpus kept as `synth-retrieval-16k`; that stream reset to shard 0 (new corpus, and the old cursor at shard
  5 is outside the new set's 4 shards). Val loss for 3.5% of val tokens changes content; the drift line does not.
  Resumed at 1.01B.
- 15:10 RL beyond arithmetic. Two new task families for M9's GRPO stage: `pytool_*` (six families plus an
  umbrella `pytool`) drawn from the *same* grammars as the tool SFT set — `synth_python.sample_question` now
  hands `slm.rl.pytool` the `(prompt, gold, declarations)` of a generated conversation, so there is still one
  copy of every grammar — and `constraints`, writing prompts with 1-3 machine-checkable instructions (18 types,
  checked from the spec, never from the English). Declared-function tasks carry their `FunctionDecl`s through
  `Task.meta["functions"]` into the prompt blocks and into each rollout's `PySession`, so RL can train a
  capability the model has to call rather than guess. Gold answers are often a list, a word or a boolean now,
  hence `verify_exact` (tolerant of packaging, strict about content) next to `verify_numeric`, dispatched by
  `verify_answer`. `reward_scheme` became resolvable per family (`reward_schemes`, exact name then group
  prefix) with a new `fraction` scheme: constraint tasks have no tool to use, so under `tool` they could never
  have scored above 0.5. Also fixed a quiet bias in `make_tasks`: the gsm8k pool is pre-filtered by split and
  always lands, while a generated prompt lands in held-out 1 time in 10, so held-out sets were ~75% gsm8k;
  the drawn family is now retried instead of redrawn. `configs/train/m9_rl_336m.yaml` written (30% gsm8k /
  40% pytool / 30% constraints, group 6 x 4 prompts at 4K), not launched.
- 18:18 M8 phase 2 FINISHED: 2.50B tokens in 1d 3h 41m, val 2.3869, drift vs the phase-1 mixture 2.4666 (from
  2.6086 — the decay bought 0.142, more than the whole 149M->336M parameter step was worth). Needle effective
  context **4096** by the strict gate (n=64: 1K 100/100, 2K 99.6/96.9, 3K 98.7/96.9, 4K 94.0/82.8). HellaSwag
  32.8, ARC-Easy 57.9, PIQA 66.4 (full), facts 69.1% (149M base: 29.3 / 51.6 / 64.0 / 43%).
- 18:30 base checkpoint chosen: `final.pt` (2.5B). 1.5B scored higher on the judged suite (3.46 vs 3.06) but
  fails the 4K gate (worst depth 60.9%) and is 0.053 worse on drift loss; the 2.0B-2.5B judged gap (0.10) is
  inside the measured checkpoint wobble (+-0.25) while the loss trend is monotone. The decay traded coherence
  (3.31 -> 2.72) for correctness (3.25 -> 3.41), which is the right trade for a base: SFT reliably adds
  coherence (149M m4: 2.11 -> 2.83) and cannot add knowledge.
- 18:33 M9 stage A (`m9_sft_336m`) launched: chat SFT, 200M tokens, 5 SmolTalk sets at 4K with a mandatory
  (empty for plain chat) think span so the format never changes under the model again. scripts/pipeline_m9.sh
  chains stage B (reasoning+tools, 34% chat rehearsal) and the per-stage measurements; RL is launched by hand
  after the gates are read.

## 2026-09-21 — M9 stage A passed, stage B failed on routing, stage B v2 launched

- Stage A (`m9_sft_336m`, chat SFT, 200M tokens) passed its gate: judged overall 3.06 -> **3.47**, coherence
  +0.84, task +0.75, needle still **4096** by the strict gate, facts 70.1%, ARC-Easy 58.4. Correctness fell
  0.38 (the base's decay trade partly given back). Quality wobbles +-0.15 across its 8 checkpoints with no
  trend after 50M: 200M tokens was more than this stage needed.
- Incident: stage A crashed at its first eval with an inductor bounds check, `index out of bounds:
  0 <= tmp4 < 32768`. Cause: chunked cross-entropy (`loss_chunk_size`, first used in SFT) under
  `torch.compile` lowers to a kernel that gathers at the target index *before* applying the ignore mask, so
  `IGNORE_INDEX = -100` trips the check. Pretraining never hit it because every token is a target.
  `_chunk_loss` now masks the per-token losses instead (arithmetically identical, index-safe) and there is a
  regression test.
- Stage B v1 (`m9_tool_336m`, reasoning + Python tools, 120M tokens, 34% chat rehearsal) **failed its gate**.
  It learned the targeted skill — judged arithmetic 2.83 -> 4.50 — and gave back more than it gained:
  overall 3.47 -> 3.00, pattern -2.42, qa -2.00, definition -1.00, facts -0.62, and needle effective context
  fell 4096 -> 3072.
- The diagnosis is one correlation: **every judged category that emitted a `<|python_call|>` regressed, and
  every category that did not held or improved** (narrative +0.50, prose -0.08). "What is the capital of
  France?" produced an invented `city_population(...)` call and the answer "That would be Aldershaw."
- Two things I had wrong until the numbers were normalised. I first read the raw call counts (17/35 -> 13/35)
  as "more training is slowly fixing it" and planned a *shorter* v2; per-eligible-prompt the misfire rate is
  0.464 at 15M and 0.393 at 120M — **flat**, so duration and LR were never the lever. And my first cut of the
  metric counted a call on a `python` prompt as correct use, when those prompts ask the model to *write* a
  function, not run one. No prompt in the suite requires a tool, so the suite can only measure misfires;
  correct use is `slm.eval.reasoning --tools`, where the mechanism is in fact sound (arith 0.82-1.00 accuracy
  at 0.70-1.00 tool use, 0.00 tool-error rate). GSM8K is the opposite failure: 4.5% at a 0.17 tool-use rate,
  because real word problems are out of distribution for the synthetic grammars. Both are one thing — **the
  model routes on surface form, not on need** — and short data-free questions share the grammars' surface form.
- Cause: a missing case, not a missing regulariser. v1's mixture had "empty think, then answer" (rehearsal)
  and "think with a tool call, then answer" (tools) and nothing else, so once the model moved off empty think
  spans the only non-empty span it knew how to write contained code. Rehearsal at 34% could not teach a
  behaviour that was absent from the data.
- Fix, in three parts. `slm/data/direct_think.py` supplies short prose think spans that reach for no tool, and
  `slm.data.sft --direct-think 0.45` applies them to the chat sets (`-4k-direct`); a deliberately
  over-inclusive `is_computational()` guard keeps "no code is needed here" off any turn that might want the
  tool — the first probe run attached it to a probability question, which would have taught the opposite
  lesson. `slm.eval.quality` now reports `tool_misfire` per checkpoint (stage A 0.00, stage B v1 0.39-0.46) so
  routing is visible in the run page during a run. (These rates are the corrected ones: the first cut of the
  metric counted only spans the tool loop executed, and the loop honours markup solely inside the think span,
  so an answer that reaches for the sandbox and runs out of tokens part-way scored zero calls. v1 reads
  0.500-0.429, not 0.464-0.393.) `configs/train/m9_tool2_336m.yaml` rebalances rehearsal
  34% -> 52% and tool-bearing data 50% -> 33.5%, keeping 40M absolute tool tokens — v1 had the skill saturated
  after 7.5M of them.
- 10:02 stage B v2 (`m9_tool2_336m`) launched, 120M tokens. Gate: misfire <= 0.10 first, then judged overall
  >= 3.40 with no category more than ~0.3 below stage A except arithmetic, needle >= 80% at 4K, and generated
  tool families no worse than v1. GSM8K is explicitly not gated here; closing it is the GRPO stage's job.

## 2026-09-21 (later) — stage B v2 stopped at 27M; the ratio that actually governs routing

- v2 (`m9_tool2_336m`, direct-think rehearsal, tool tokens 50% -> 33.5%) reported misfire **0.429** at its
  first 15M checkpoint against v1's 0.500. Barely moved — but the outputs showed the fix working where it had
  data: prose, python and qa were clean, with the templated think lines visible ("Let me think about what the
  user is asking for"), sandbox errors fell 16 -> 2, and the misfiring categories narrowed from five to three.
  Only short factual prompts still reached for code.
- My first explanation — that short prompts were not being treated — was wrong, and measuring it said so:
  short prompts are 15-19% of the rehearsal rows and are treated at the *highest* rate (37-40%).
- The real cause: **mixture weights are token shares, but reaching for a tool is decided once per
  conversation**, and a synthetic tool conversation is 70-220 tokens against a chat conversation's 400-1470.
  Worse, only ~16% of a chat set's conversations have a short first user turn and only ~38% of those drew a
  think line, so a `-direct` set teaches the contested decision about a fourteenth as often as its token share
  suggests. v2's real ratio on short question-shaped prompts was **33.7 : 1** in favour of calling the tool.
  Cutting the token share from 50% to 33.5% could never have fixed that, which is exactly why it didn't.
- `scripts/mixture_decisions.py` now prints this for any candidate config: conversations per set, the share of
  them that actually demonstrate the contested decision, epochs, and the ratio per decision class. It found a
  second misrouting immediately, one I had written up as distribution shift: v1 taught
  math-word-problem-without-tool 113k times against math-with-tool 89k — **0.8 : 1 against** the behaviour we
  want, which is the whole of the 0.17 GSM8K tool-use rate. The prose-reasoning math sets were crowding out the
  tool-math ones.
- v2 stopped at 27M rather than run its remaining hour: both of its failure modes were already explained, and
  v3 contains all of v2's data plus the fix, so v3's own checkpoint curve answers the same questions.
- `slm.data.sft --short-only N` extracts conversations whose first user turn is at most N tokens, dropping the
  ones a tool might legitimately serve, and pairs with `--direct-think 1.0`. magpie is deliberately excluded
  from the resulting `-short` sets: its conversations are multi-turn and still average 1384 tokens, so the
  filter buys almost nothing there. The useful ones are systemchats (609 tok/conv), openhermes (269) and
  everyday-conversations (225).
- 10:25 v3 (`m9_tool3_336m`) launched, 120M tokens. Measured ratios: short_tool : plain_no_tool **1.2 : 1**
  (v2: 33.7 : 1), math_tool : math_no_tool **2.0 : 1** (v1: 0.8 : 1). Same gate, misfire read first.

- Correction to the misfire metric, found on v3's first checkpoint. It counted only spans the tool loop
  actually executed, and the loop honours `<<...>>` markup solely inside the think span (tool calls are part of
  thinking, never of the answer). A model that reaches for the sandbox in its ANSWER span, or runs out of
  `max_new_tokens` part-way through the program, therefore scored zero calls. That hid 6 of v3's 8 misfires.
  `_ran_code` now also checks for the markup opener — the opener rather than the full pattern, because a
  truncated call rarely carries its closing `>>>`. Corrected: stage A 0.000 throughout, v1 0.500 -> 0.429
  (still flat), v2 0.429 at 15M, **v3 0.286 at 15M**.
- v3 at 15M: needle effective **4096** (v2 read 2048 at the same point), misfiring categories down to facts and
  pattern only, and the wanted behaviours are visible in the outputs — "List the days of the week" now draws
  "I do not need to run anything here. Let me answer from knowledge." followed by the correct list, and
  "17 + 25" draws a real `<<17+25=42>>` calculator call.

## 2026-09-22 — M9 stage B settled on v4, stage C (GRPO) run, milestone complete

- Stage B v4 (`m9_tool4_336m`) is the stage B output: judged 3.58 at its last checkpoint against stage A's
  3.47, python fully recovered to 4.22 by restoring `smoltalk-smol-constraints` (exactly the predicted effect),
  qa/narrative/prose all above stage A. Read as level-with-stage-A rather than better — v4's per-checkpoint
  mean is 3.44 against stage A's 3.49, and 3.58 is both its last point and its maximum. The capability came
  free: arithmetic 2.83 -> 4.42 and working tool use at no net cost.
- The `humanize()` fix worked partially: pattern 1.67 -> 2.50. Python reprs in generated answers fell 6.7% ->
  0.51% and the model stopped answering "List the days of the week" with a list repr, but the broader habit of
  answering short prompts with a terse template survived. I had attributed most of the pattern collapse to the
  reprs; that was only part of it.
- Retrieval: three mixtures with very different balances all eroded 4K needle the same way (worst depth 70% /
  75% / 73% against the base's 84%), so the post-SFT model is documented as a **3072** model. Peter's call
  (2026-09-22): benchmarks over context length, cap it if needed.
- Stage C try 1 (`m9_rl_336m`) stopped at step 69 with "collapse guard: entropy 2.54". **It was not a
  collapse.** Over 69 steps entropy ran 0.00-2.54, stdev 0.63, corr with step -0.13 — stationary noise from a
  24-rollout step — while KL to the reference sat at 0.0003. A policy that has not moved cannot have collapsed.
  The guard compared a single step against its limit; m6 try 1, the failure it was written for, went 0.8 -> 5
  and stayed. It now tests the mean over `guard_window` (10) steps and cannot fire before the window fills.
- The same log showed the real problem, and it is the more useful lesson: **KL averaged 0.00028 against a 0.15
  budget — 0.2% of the movement the run was allowed.** That is why reward was flat over 50 steps and held-out
  moved 0.250 -> 0.271 (two prompts of 96). The KL budget, not the learning rate, is the safety mechanism, and
  it was going unused. lr 8e-7 -> 4e-6 and prompts_per_step 4 -> 8.
- Try 2 (`m9_rl2_336m`) then learned cleanly: held-out 0.250 -> 0.323 -> 0.375 -> **0.385**, malformed
  0.15 -> 0.07, train reward rising monotonically from step 20 while try 1 had wandered. Stopped at step 141 by
  the KL guard (0.174 vs 0.15), held-out still climbing. KL had accelerated — I extrapolated it linearly to
  ~0.12 by step 250 and it reached 0.174 by 141.
- Before launching, `scripts/rl_signal_probe.py` measured what GRPO actually depends on: the share of groups
  whose rollouts do not all score the same (a group with no spread has zero advantage). It found the config's
  40% `pytool` share was the worst-spent part of the budget — those families sit at 0.56-0.94 pass with
  0.85-1.00 tool use — while `constraints` yields spread in 62% of groups at only 0.12 pass, because the
  `fraction` scheme gives partial credit. Reweighted to 40/30/20/10 constraints/gsm8k/pytool_numbers/
  pytool_declared, raising the signal-weighted share of useful updates from 0.36 to 0.45. The trainer already
  logs `no-signal groups` per step, so the probe's value was in measuring it *per family before committing*.
- Stage C result: reasoning mean 0.701 -> 0.779, algebra **0.35 -> 0.99**, tool-use rates 0.00-0.54 -> 0.96-1.00,
  core benchmarks flat. GSM8K barely moved (0.04 -> 0.06) despite tool use tripling to 0.78 — the bottleneck is
  comprehension and setup, not arithmetic.
- The cost, and it has one cause: judged facts 2.88 -> 2.00 and tool misfire 0.250 -> 0.321. RL saw only math,
  tool and constraint families, so with nothing but a KL term anchoring it the whole policy moved toward
  tool-and-terse; in chat mode it now answers "Who wrote Hamlet?" with "So the answer is 2." A future RL run
  needs a chat rehearsal family or a tighter KL budget. `reward_schemes` already supports per-family schemes.
- M9 output: `runs/m9_rl2_336m/checkpoints/best.pt`. `m9_tool4_336m/checkpoints/final.pt` is kept as the
  better pure-chat model.

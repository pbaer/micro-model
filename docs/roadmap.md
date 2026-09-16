# Roadmap

Remaining work, with GPU-time estimates from measured throughput. Updated 2026-09-14 afternoon: the full M0–M7 chain has run on the real base (see `results.md`); items 1–9 below are done except RL stage C, which is not built.

## GPU critical path (~35 GPU-hours)

| # | Work | Estimate | Depends on |
|---|---|---|---|
| 1 | M3a stable continuation to 3.4B tokens (running) | ~4 h left | — |
| 2 | Rename `runs/m3_base_149m_stable` → `m3_base_stable_149m` (background job on exit) | minutes | 1 |
| 3 | Measure 8K × microbatch 2 throughput (`bench_throughput --seq 8192 --max-mb 2`) | 10 min | 2 |
| 4 | M3b: 8K phase with WSD decay, 800M tokens (`m3_base_8k_149m`) → the base checkpoint | ~7 h (at ~30K tok/s) | 3 |
| 5 | Base evals: lm-eval (HellaSwag, ARC, PIQA, MMLU subsets), diagnostics, needle 1K–8K | ~1 h | 4 |
| 6 | M4 instruct SFT on the real base | ~2 h | 4 |
| 7 | M5 reasoning SFT | ~15 min | 6 |
| 8 | M6 RL stage A (arithmetic), B (multi-step, algebra), C (code with unit tests, logic puzzles) | ~1 h each | 7, and C needs its tasks built |
| 9 | M7 16K extension (YaRN ×2, 200M tokens, grad checkpointing) + needle eval + short-context check | ~4 h | 4 |
| 10 | Optional 32K branch (YaRN ×4 from the 16K model) | ~6 h | 9, decision below |

## Engineering (CPU-side, can overlap with training)

- RL stage C: code tasks with sandboxed unit tests, logic puzzles with deterministic checkers;
  synthetic traces for them; reward robustness review.
- Ablations on the 26M model (cheap, ~1 h each): QK-norm on/off, tied vs untied embeddings, d_ff ratio,
  a Muon-style optimizer comparison. Report as a table in `results.md`.
- Portal P1/P2 items (see `command_center.md` backlog): batch replay, run comparison, diagnostics
  viewer, animations, stage lenses, run scheduler.
- Milestone write-ups in `results.md` after M3b, M6 and M7 (what the numbers say, what was learned).
- Keep `docs/log.md` current; add a `docs/results.md` entry whenever a run finishes.

## Decisions pending (Peter)

1. **Extend M3a beyond 3.4B tokens?** The stable phase can simply continue (the data supports ~5B
   tokens before repeating) and would be the cheapest way to a stronger base; costs ~4.5 h per extra
   billion tokens and delays M3b.
2. **Do the 32K branch?** Decide after the 16K needle curves; costs ~6 h and gradient checkpointing at
   microbatch 1.
3. **Attention alternatives** (sliding-window or hybrid layers) as an ablation track, or stay with full
   attention throughout.
4. **A larger model** (e.g. 300–400M) as a second base once the 149M pipeline is fully validated, at the
   cost of iteration speed.

## Context curriculum (current focus, 2026-09-14 evening)

Long context is only claimed where needle retrieval holds. Gate at every stage: `needle_min_<L>` ≥ 0.8
(worst depth, real-text haystack) at every length up to the stage length, and no short-context regression
(pretraining-mixture val, lm-eval at the same limit). Each stage branches from the previous stage's final
checkpoint; the base is never modified. The old `m7_ctx16k_149m` stays as the "before" record.

| Stage | Run | Length | Data | Gate to next |
|---|---|---|---|---|
| 1 (done, gate failed) | `m7_ctx8k_retrieval_149m` | 8K rows | 15% synth-retrieval + 35% long docs + pretraining mix, 600M tokens | effective 6000: 100% to 4K, 88% min at 6K, 0–6% at 8K depth 0–0.1 |
| 1b (queued after the tool track) | `m7_ctx8k_retrieval2_149m` | 8K rows | 20% `synth-retrieval-8k` (3K–8K docs, half the facts in the first 15%), 300M tokens from stage 1 | needle min ≥ 80% at 1K, 2K, 4K, 6K, 8K |
| 2 | `m7_ctx16k_retrieval_149m` | 16K, YaRN ×2 | same recipe with 16K retrieval docs, 600M tokens, from stage 1b | needle min ≥ 80% at 8K and 16K |
| 3 (optional) | `m7_ctx32k_retrieval_149m` | 32K | only if stage 2 passes | needle min ≥ 80% at 16K and 32K |

If a stage stalls below the gate after its token budget, the honest result is "effective context = previous
stage" and the report says so.

**Corrected 2026-09-16:** the failing cells are the far edge of the trained window (shallow needles are 100% at 4K–6K,
degrade from 7K on), not an attention-sink zone. Strict effective context 7K. Recommendation: proceed to 16K; gate 16K on
all depths ≥ 80% at 8K (the old edge, now interior) and depths ≥ 0.05 at 16K, reporting the new edge cells as the known
weak zone.

**Status 2026-09-15 evening (decision pending):** stages 1–1d reached 100% at every depth up to 6K and 94–100% at 8K for
every depth from 0.05 on; the only failing cells are needles in the first ~160 tokens of an 8K context (31–69%), which no
longer improve with data. Options: (a) proceed to 16K with the gate redefined to exclude the first 2% of the context
(documented as a known blind spot), (b) stop the curriculum at 8K, (c) investigate the blind spot first (attention-sink
mitigation such as a few pad/sink tokens after `<|bos|>` in the eval, or training rows that put facts in that zone).

## Tool use track (queued behind the context curriculum, 2026-09-14 evening)

Goal: push GSM8K by letting the model offload arithmetic. Built and tested: sandboxed Python subset interpreter
(`slm/tools/pysandbox.py`), calculator, tool protocol on the reserved tokens, batched tool-aware generation, SFT
masks, RL masks, evals with `--tools`, GSM8K-train prompts as RL tasks, tool SFT data (7.3K GSM8K, 682 MetaMathQA,
40K synthetic). `scripts/pipeline_tools.sh` waits for `CTX_PIPELINE_DONE`, then runs `m5_reasoning_tools_149m`
(SFT from the real M4) → reasoning eval with/without tools on the full GSM8K test → `m6_rl_gsm_tools_149m`
(GRPO on GSM8K-train prompts) → full GSM8K test with tools. Targets: 5–8% without tools is the size-class
stretch; with tools 15–25% is plausible. Later: redo the post-training chain (M4→M5 tools→M6 tools) from the
context-extended base once the curriculum settles.

## Next candidates (after the decisions below)

- RL curriculum fix: stage B learned nothing from arith_multi (no correct samples). Add intermediate tasks
  (two-step expressions, small numbers), partial credit, or a reward for a correct intermediate line; then
  stage C (code with unit tests, logic puzzles).
- M7 speed: try microbatch 2 without gradient checkpointing (4.3 GiB peak leaves room).

## Known gaps

- MMLU subsets not run yet on any checkpoint.
- `runs/` is untracked; there is no off-machine backup of checkpoints.

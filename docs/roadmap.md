# Roadmap

Remaining work, with GPU-time estimates from measured throughput. Updated 2026-09-13 evening.

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

## Known gaps

- No needle numbers yet (the M2 attempt hit the RoPE-table assertion; fixed since).
- MMLU subsets not run yet on any checkpoint.
- The M4–M6 numbers in `results.md` are rehearsal results on the 1B-token base and will be replaced.
- `runs/` is untracked; there is no off-machine backup of checkpoints.

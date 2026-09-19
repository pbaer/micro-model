# Roadmap

## Direction set 2026-09-16 (Peter)

Priority 1: the strongest model the parameter footprint allows — multi-turn chat, tool calling, basic problem solving,
reasonable factual knowledge. Priority 2: extend context only as far as it costs ≤ 5% on the short-context evals; a
great 2K/4K model beats a mediocre 8K one. The machine is dedicated; use the VRAM by growing the model, not the
microbatch. Pretraining may include chat/tool data that would otherwise only appear in SFT.

## The second base: M8 (running since 2026-09-16 13:07)

`base_336m` (24 × 1024, 16q/8kv, d_ff 3072, RoPE base 500K, 336M params), chosen from the size sweep in `results.md` §1.
`scripts/pipeline_m8.sh` chains the phases and the measurements; each phase resumes from `latest.pt` if relaunched.

| Phase | Run | Tokens | Rows | Mixture | Time |
|---|---|---|---|---|---|
| 1 stable | `m8_base_stable_336m` | 7.5B | 2K, mb 4 | fineweb-edu-10bt 72 / cosmopedia 11 / finemath 7 / python-edu 5 / shell 1.5 / synth-retrieval 3.5 | ~73 h at 28.6K tok/s |
| 2 decay | `m8_base_4k_336m` | 2.5B | 4K, mb 2 | fineweb-edu-10bt 66 / cosmopedia 11 / finemath 7 / python-edu 6.5 / synth-retrieval 3.5 / smoltalk-chat 4.5 / tool-chat 1.5 (no shell from 2026-09-19; the grammar-generated Python-tool set is ready as `tool-chat-v2`, 36.1M tokens ≈ 0.96 epochs — rename it over `tool-chat` at launch); LR decay over the last 80%; needle n=64 at 1K/2K/4K | ~27 h at 25.4K tok/s |
| measure | needle 1K–4K (n = 16), lm-eval full + 2000-limit, diagnostics | | | | ~1 h |

Sanity rule for phase 1: loss below the 149M curve at matching token counts from ~200M on, or stop. Python data: `python-edu` was
enlarged to 589M tokens (1.3M files) and swapped in at 63M tokens of phase 1, so the 5% share is 0.85 epochs, not 3.6.

## After the base (in order)

1. Post-training on the 336M base: instruct SFT (multi-turn kept), tool-use reasoning SFT, GRPO with the Python tool
   (`tool` reward scheme, collapse guards, best.pt), all with the multi-turn/REPL evals from the command center.
2. Context: 8K only through the gated retrieval curriculum, and only if short-context evals stay within 5%. The 149M
   curriculum reached 7K effective (window-edge failures beyond ~7.5K); 16K/32K are off the table unless 8K is clean.
3. RL stage C (code with unit tests, logic puzzles) and a factual-recall probe so knowledge is measured.
4. Ablations on the 26M model, portal P1/P2, writeups (unchanged).

## First base (149M): status at hand-off

M0–M7 all ran on it (results in `results.md`): base val 2.693 at 8K; HellaSwag 29.3, ARC-Easy 51.6, PIQA 64.0; tool
SFT + RL reach 100% on the templated tasks and 1.3% on GSM8K; retrieval curriculum effective context 7K. It stays as
the ablation/reference model.

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

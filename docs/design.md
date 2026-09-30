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

**Registry** (`sources.py`). `Source(name, repo, pattern, text_col, kind, license, content_via_swh, custom_prepare)`;
`custom_prepare` names a module that replaces the generic preparer (PG-19, below).
Kinds: prose, math, code, chat, math_cot, math_qa. Code datasets ship only Software Heritage blob ids;
`swh.py` fetches contents from `s3://softwareheritage/content/<id>` (anonymous, gzip) with a thread
pool and writes text parquet next to the ids.

**Prepare** (`prepare.py`). Streams parquet row groups, applies `keep_doc` (≥ 64 chars; `language ==
"en"` when the column exists and the source is not code; ASCII-letter ratio ≥ 0.95 for prose, ASCII
ratio ≥ 0.9 for code), drops docs shorter than `--min-doc-tokens` (default 16) or longer than 65,536
tokens, and writes:

    tokenized/<tag>/<name>/train/shard_00000.bin      uint16 tokens, 100M per shard, docs = <|bos|> … <|eos|>
    tokenized/<tag>/<name>/train/shard_00000.idx.npy  int64 document start offsets within the shard
    tokenized/<tag>/<name>/train/shard_00000.src.npy  int32 [n_docs, 3] raw-row sidecar, aligned with .idx.npy
    tokenized/<tag>/<name>/val/…                       10 ‰ of documents by sha1(text[:2048]); never trained on
    tokenized/<tag>/<name>/manifest.json               counts, tokenizer sha256, min_doc_tokens

The sidecar names the raw row every kept document came from — `(file_index, row_group, row)`, where
`file_index` indexes `manifest["files"]` — so a shard offset can be traced back to its source text;
`manifest["sidecar"] == "src"` marks the shards that have it (older shards do not, and no reader requires it).

`--name` creates derived sources from the same raw files (e.g. `fineweb-edu-long` with
`--min-doc-tokens 4096`).

**Narrative prose: PG-19** (`gutenberg.py`, 2026-09-30). Project Gutenberg books from the PG-19 release,
selected by rules rather than sampled. Per book, on the text with Gutenberg framing stripped (START/END markers
when present; PG-19's surviving "End of the Project Gutenberg EBook" line and what follows; production-credit
paragraphs at the top), in order, the first rule failed is the book's verdict:

| rule | default | catches |
|---|---|---|
| `min_year` (before download) | 1850 | older register |
| `min_words` | 2,000 | fragments, pamphlets |
| `min_stopword_share` | 0.30 | non-English books (share of words in a small function-word list; English ~0.5) |
| duplicate | exact, whitespace-normalised | re-issues; a val book wins over a train copy |
| `max_caps_share` | 0.06 | plays (speaker names), indexes, tables: ALL-CAPS, table-like or mostly non-letter lines |
| `max_verse_share` | 0.15 | verse: lines inside runs of >= `verse_run` (4) lines shorter than `verse_line_chars` (55) |
| `max_archaic_per_1k` | 1.5 | thee/thou/thy/thine/hath/doth/dost/hast/shalt per 1K words |
| `min_dialogue` | 0.10 | books without conversation (share of lines with `"`, curly doubles, or `'` opening a capitalised word) |

Surviving train books are ranked by dialogue density and taken until `target_tokens` (5e8); the density of the
last one taken is the cutoff. The val split is PG-19's own validation + test books through the same rules and
cutoff (topped up in density order to 2M tokens if the cutoff leaves less), so train and val never share a book.
Stats are measured on the hard-wrapped text; what is tokenized is `normalize()`d: `[Illustration]` tags and
`_italic_` underscores removed and the ~70-character wraps joined inside prose paragraphs (a paragraph whose lines
are mostly short keeps its breaks). A book is tokenized once and cut into documents of at most `segment_tokens`
(32,768) at the nearest paragraph start within 5% of an even split, each `<|bos|> … <|eos|>`; a book's segments
stay together and books are written in a seeded shuffle, never in density order. The manifest adds `books` (counts,
mean tokens per book) and `filter` (thresholds, per-rule removed / fail-at-all counts with example titles, the
funnel); `books.jsonl` beside it has every book's statistics and verdict, which the Data tab's raw trace reads.
Stats and token counts are cached in `raw/gutenberg-pg19/book_stats.json` (keyed by the settings they depend on).
Known noise: PG-19's `publication_date` is often the edition's, not the first publication's (a 1907 Boccaccio
translation passes); quotation-mark density also rewards biographies that quote letters and glossaries that quote
citations (Hobson-Jobson, 1.6M tokens, made the v1 cut); transcriber's notes and "Project Gutenberg also has an HTML
version" notes are not stripped (present in about a third of kept books, ~0.03% of characters).

**Curated supplement: `gutenberg-canon`** (`gutenberg.py canon`, 2026-09-30). The dialogue-density ranking skews
gutenberg-pg19 to chatty, sanitized popular fiction and drops most of the adult literary canon (Flaubert, Zola,
Chopin, Stoker, Melville, George Eliot, most of Conrad and Hardy). `CANON` is a hand-written list of author
entries (a regex on the part of the PG-19 title after the last " by ") with ordered title patterns and a per-entry
cap (`CanonConfig.per_author_cap`, 6 unless the entry sets one); an entry without titles takes the author's books in
PG-19 id order. All hits of one title pattern are one work: one edition is kept (the longest; a "Complete" /
"Vols. 1-4" edition, or an unmarked one >= 1.5x the largest volume, over its volumes; otherwise the longest per
volume number). Nothing gutenberg-pg19 holds is repeated: same id, same text hash, same title, same work (author +
title without volume markers, unless both are differently numbered volumes), or a collection titled after a work it
holds. The hygiene rules are unchanged except: no dialogue rule, `max_archaic_per_1k` 3.0 instead of 1.5
(translations, Hardy, Adam Bede), and a play whose speaker-name lines are >= 4% of lines, with recurring names
covering >= 80% of them, is judged on its caps share without those lines. Val: PG-19's validation/test books, then
seeded whole-book draws to 2% of books. Tokenized exactly like gutenberg-pg19 (one tokenization per book, <= 32K
documents at paragraph starts, seeded shuffle). The manifest's `canon` block lists every book with the rule relaxed
for it (`dialogue-density cutoff` = passed every rule, ranked below pg19's cutoff) and a status line for every title
pattern (included, already in gutenberg-pg19, fails a rule, not downloaded because PG-19 dates it before 1850, not in
PG-19). Seen while building it: gutenberg-pg19 itself holds the Casanova memoirs twice (the Complete edition and
Vols. I-VI) and Rhoda Fleming Complete beside its volumes, because its duplicate rule is exact-text only.

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

**What a continuation inherits.** Three independent switches, because a continuation is not always a continuation
of everything: `init_from` (weights, from any checkpoint incl. a bf16 `snap_*.pt`/`final.pt`), `init_optimizer`
(AdamW moments; needs a full `latest.pt`), and `init_loader_from` (per-source data-stream cursors, also a full
`latest.pt`). The last one is the one that is easy to forget and invisible when wrong: without it every
`TokenStream` starts at shard 0 offset 0 and the phase re-reads whatever the parent already consumed. M8 phase 2
did exactly that for its first 311M tokens (94% of them a second epoch, caught only because Peter asked). The
trainer now warns at startup when a pretraining run sets `init_from` without `init_loader_from`, and
`PretrainLoader.adopt_stream_positions` moves only the sources present in both mixtures, so a changed mixture
(new sources start at 0, dropped ones ignored) is handled. Per-source cursors are logged as `sources` in every
`checkpoint` record, so the Data page's chain view can show actual rather than expected exposure.

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
| `start`, `resume`, `stop`, `finish` | `msg`; `finish`/`stop` carry `elapsed_s` and `sources` |
| `checkpoint` | `msg` (file, seconds), `sources` |
| `train` | `update, loss, lr, grad_norm, tok_s, tok_s_ema, step_ms, fwd_ms, bwd_ms, opt_ms, data_ms, vram_gib, gpu_temp_c, gpu_power_w, gpu_util, elapsed_s, eta_s` (+ RL fields below) |
| `eval` | `val_loss, val_ppl, best, eval_s, val_pt_loss` (extra validation), RL: `heldout_acc, train_acc, heldout_malformed, heldout_len` |
| `milestone` | `tokens, segment_s, elapsed_s, tok_s, loss, val_loss` |
| `warn` | `msg` (GPU temperature) |

`sources` is `PretrainLoader.consumed()`: per mixture source `{tokens, epoch, shard, offset}`, the tokens
taken from that source's stream since the run started (`epoch` fractional); `null` for loaders without it.

RL `train` records add `reward_mean, success_rate, group_std_mean, groups_no_signal, adv_abs_mean,
len_mean, len_correct, len_wrong, malformed_rate, length_term_rate, kl, entropy, clip_frac, ratio_mean`.
`slm/utils/metrics.py` (torch-free) reads this incrementally for both the HTML report and the portal;
`summary()` derives status, progress (tokens, or steps for RL), ETA, cumulative tokens along the
`init_from` chain, and the latest GPU state.

### z-loss (`optim.z_loss`, 2026-09-27)

`slm.model.loss.chunked_cross_entropy(..., z_loss)` adds PaLM's `z_loss * (log Z)^2` per valid token inside each
chunk, `Z` the softmax normaliser. Cross-entropy is invariant to a shared logit offset, so over a long bf16 run the
logits can drift up together until `exp` overflows; the term pulls `log Z` toward 0 and changes nothing the model
predicts. Off by default (0 = byte-identical to every run before it; `tests/test_zloss.py` checks the formula on
both the chunked and the single-chunk path and that one gradient step shrinks `log Z`). The trainer passes it only
into the *training* forward, so `val_loss` stays plain cross-entropy and comparable across runs, and every eval
record carries `val_logz_mean` / `val_logz_max` from the first val batch whether or not the term is on. Set at
1e-4 (PaLM's value) for the third base and its ablations.

## 6. RL trainer (`slm/train/rl.py`, `slm/rl/`)

**Tasks** (`tasks.py`): generators yield `{id, prompt, answer, task, meta}`. `make_tasks` splits
train/held-out by disjoint seeds and a hash of the canonical prompt text, so overlapping generators
cannot leak; a drawn family is retried until it lands in the requested split, so the held-out set keeps
the training mix instead of filling up with the one pool (gsm8k) that is pre-filtered by split.
Families:

| family | source | gold |
|---|---|---|
| `arith1` `arith2` `arith2mul` `arith_multi` `algebra` `word` | templates in `tasks.py` | integer |
| `gsm8k` | GSM8K *train* split (test is reserved for the benchmark) | number after `####` |
| `pytool_pipeline` `pytool_strings` `pytool_numbers` `pytool_sim` `pytool_runcode` `pytool_declared`, umbrella `pytool` | `pytool.py`, over the grammars of `synth_python.py` | whatever the sandbox printed: a number, a word, a list, a boolean |
| `constraints` | `constraints.py` | the JSON constraint spec |

`pytool.py` holds no grammar of its own: `synth_python.generate_sample` produces the conversation and
`synth_python.sample_question` returns its `(prompt, gold, declarations)`, so the SFT set and the RL
tasks are generated by one copy of each grammar. The gold is produced by the sandbox while the task is
generated (anything it cannot reproduce is dropped), so tasks are correct by construction;
`meta["codes"]` keeps the reference program for re-verification. Declared-function tasks carry their
`FunctionDecl`s (with impls) in `meta["functions"]`: `rollout.py` renders the masked declaration blocks
into the prompt (`format_chat(functions=…)`) and passes them to `sample_with_tools(functions=…)`, which
registers the impls in every rollout's `PySession` — so the model can actually call them.

`constraints.py` emits a writing prompt plus 1-3 machine-checkable instructions (18 types: sentence /
word / paragraph / bullet counts, required or forbidden words and phrases, all-lowercase, no commas, a
`<<title>>`, a first word, a last phrase, a `P.S.` line, a number in a range, a per-sentence length cap).
The spec is the truth: the English rule is composed from the spec and the checker evaluates the spec,
never the prompt. One constraint per group, conflicting groups (a `P.S.` in an all-lowercase answer)
excluded, and a feasibility check (60 words in at most 2 short sentences) — every task is satisfiable.
These prompts do not ask for `####` at all (`meta["answer_style"] = "free"`); the answer *is* the writing.

**Rollouts** (`rollout.py`): `rollout_group` samples G completions for one prompt (same prompt length,
no padding), stops on `<|end|>`, parses think/final spans, computes the reward, and recomputes old and
reference log-probs teacher-forced. `greedy_accuracy` batches prompts by length *and* by declared
function set (one session env per batch) for evaluation.

**Rewards** (`rewards.py`): `parse_final_answer` takes the LAST `#### …` line and the first number-like
token after it (commas and `$` stripped, fractions allowed); `binary` reward = exact numeric match;
a malformed completion (no think span when required, no answer line) scores 0 and is counted.

**Verifiers**: `verify_answer(answer_text, gold, kind="auto")` is the single entry point.
`auto` dispatches on the gold's shape (`slm.data.answers.is_numeric_answer`): a numeric gold takes the
numeric path (`verify_numeric`, unchanged), anything else `verify_exact`, and `kind="constraints"` takes
`verify_constraints`. `verify_exact` compares the whole `#### …` span (`parse_final_span`, not the first
number in it) and is tolerant of packaging — case, surrounding punctuation, quotes, list brackets and
spacing, `5` vs `5.0` — and strict about content: a wrong item, a missing item, an extra item or an extra
word fails. `verify_constraints` returns a `Verdict` whose `correct` is "all satisfied" and whose new
`fraction` field carries the share. `answer_from_tool` now also credits a non-numeric answer that is
exactly what a (non-trivial) call printed.

**Reward schemes per family**: `reward_from_verdict(v, malformed, scheme, from_tool)` gained
`fraction` (the verdict's partial credit, else 1/0 for a binary verdict; malformed is still 0 under every
scheme). `RlConfig.reward_scheme` remains the run default and `RlConfig.reward_schemes` overrides it per
family: `resolve_scheme(task, schemes, default)` takes an exact family name first, then the group prefix
(`pytool_declared` → `pytool`), then the default. A mixed run needs this — constraint tasks have no tool
to use, so under `tool` they could never score above 0.5.

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

## 6a. Final-answer style (`slm/data/answers.py`)

`#### <answer>` exists for verification, so the model must produce it only when asked. Every set with verifiable
answers (GSM8K, MetaMathQA, the templated traces, the multi-turn tool conversations) is written as a 50/50 mix:
either the user turn carries the instruction ("…give the final answer on its own line as '#### <number>'") and the
answer is `#### X`, or the question is bare and the answer is a sentence ("So the answer is 624."). One coin flip
per conversation. RL rollouts and the reasoning benchmark always prompt with the instruction and verify strictly;
`verify_numeric(strict=False)` accepts the last number for bare-question answers. Before 2026-09-16 the GSM8K and
MetaMathQA sets had the marker without the instruction, which taught `#### N` as the default reply to any math
question.

## 6b. Chat and tool data inside pretraining (`scripts/sft_to_pretrain.py`)

SFT shards (`tokens_*.bin` + `idx_*.npy`, chat-formatted examples `<|bos|>…<|eos|>`) are re-laid as an ordinary
pretraining source (`shard_*.bin` + `shard_*.idx.npy`) with the loss mask dropped, so a pretraining run can sample
whole conversations, including the `<|user|>`/`<|assistant|>`/`<|think|>`/`<|python_call|>` tokens and the inserted
tool results, as plain next-token targets. The second base mixes `smoltalk-chat` (225M tokens) at 4.5% and a tool set
at 1.5% into its 4K decay phase, following the SmolLM2 recipe of putting instruction data in the decay. Decision
(Peter, 2026-09-16): pretraining is not limited to human text.

The tool source was rebuilt for phase 2 as `tool-chat-v2` (36.1M train / 2.9M val tokens, 225K docs): the four
calculator-style sets (`gsm8k-tools`, `synthetic-reasoning-tools`, `metamathqa-tools`, `synthetic-multiturn-tools`,
7.6M tokens together, the old `tool-chat`) plus `synthetic-python-tools` (28.4M), the grammar-generated set below.
At 1.5% of a 2.5B-token phase it is sampled for ~0.96 epochs, which is why the size was chosen. The old `tool-chat`
directory is kept as it is; swapping the data is a rename at launch time.

## 7. Context extension

`RopeScaling` in the training config's `model:` block rebuilds the RoPE tables when the model is
constructed; `init_from` then loads the 8K weights unchanged. The training row length comes from
`data.seq_len`; memory is handled with `grad_checkpointing: true` and `loss_chunk_size`. The
long-context eval (`slm/eval/long_context.py`) inserts a needle sentence at given depths in filler
text of a given total length, reserves 16 tokens for the answer within the RoPE table, and reports
retrieval accuracy per (length, depth); multi-needle mode asks for several facts at once. Extension
runs also keep `extra_val_mixture` pointed at the ordinary 2K validation mixture so short-context loss
is tracked next to the long-context metric.

Every prompt the eval builds for one length is exactly that length (the haystack budget absorbs the
needle and question tokens), so all cells of a length decode in one batch: `answer_all` groups prompts
by length and fills batches up to `max_batch_tokens` prompt tokens (`eval.needle_batch_tokens`, 16384).
Decoding 16 rows costs about what one row costs, which is what makes a statistically useful `n`
affordable inside a training run. A batch that runs out of memory is retried row by row, and rows that
still fail are dropped from that cell rather than counted as misses. The batched path is asserted to
return exactly the row-at-a-time answers (`tests/test_long_context.py`).

`slm/eval/needle_sweep.py` re-measures every milestone snapshot of a finished run at one sample count and one
fixed haystack draw (`runs/<run>/needle_sweep.json`, resumable, ~35 s per 336M snapshot on the GPU at n=64), and
the run page overlays it on the needle chart as heavy lines. Use it when a run's in-run settings changed
mid-way, or when the in-run n was too small for the gate to be trustworthy: at n=16 a true 90% cell reads below
the 80% gate 21% of the time, at n=64 1.4%.

In-run sample size matters more than it looks: at `needle_n: 4` over three depths a cell is 4 samples
and the logged worst-depth line is the minimum of three 4-sample estimates, which swings by 25 points
for one miss. The M8 runs use `needle_n: 16` over five depths (80 samples per length), matching the
`n >= 16` the gate uses, so the in-run curve and the end-of-phase gate measure the same thing.

## 8. Evaluation tools (`slm/eval/`)

- `generation.py` / `sampling.py`: fixed-seed greedy and sampled completions; one `sample_next`
  (temperature, top-p, top-k) shared by the trainer, RL rollouts, the evals and the portal.
- `diagnostics.py`: residual-stream RMS per layer and relative block update size; dead/rare SwiGLU
  units; per-head attention entropy, BOS-sink mass, mean distance, local mass; head and layer ablation
  deltas on validation loss; effective rank and spectral norms; embedding usage. `scripts/diagnose.py`
  writes `diagnostics.json` and `diagnostics.html`.
- `reasoning.py`: greedy accuracy per task on held-out prompts and on GSM8K test with the chat
  format and a mandatory think span.
- `multiturn.py`: scripted chat conversations, greedy, scored without a judge, `--n` of each `--kinds`:
  `recall` (a fact, a distractor, a question that needs it), `recall_absent` (the last question asks about something
  never stated; correct = names none of the stated fact's table and says so, by `slm.rl.rewards.verify_recall`, the
  RL reward's rule), `revise` (a question with a *given* answer, then "rewrite it so that ..." with 1-3
  `slm.rl.constraints` types; share kept and all kept by `verify_constraints`) and `sysrule` (a system-prompt rule,
  sometimes two-part, kept by the given first answer and checked on the reply to a new question). `format` and
  `misfire` count every generated turn of every kind. Each kind draws from its own seeded RNG, so `recall` keeps the
  conversations it always had; the three newer kinds use material held out from `slm.rl.synth_chat` (their own
  fact tables, 30 hand-written paragraphs instead of SmolTalk, their own rule and instruction wording, phrases and
  banned words; `tests/test_multiturn_eval.py` asserts the disjointness on the constants). External chat models
  run the same conversations through their template, the `sysrule` prompt through its system role (which replaces
  the template's default system prompt). The result keeps a per-kind breakdown and every conversation for the
  Evals detail page.
- `lm_eval_wrapper.py`: an `lm_eval` `LM` subclass implementing `loglikelihood` (context/continuation
  split that respects BPE merges), `loglikelihood_rolling` and greedy `generate_until`; run from the
  CLI with `--tasks` and `--limit`.
- `quality_suite.py` / `quality.py`: judged quality over training. A versioned 35-prompt suite (facts, prose,
  Python, bash, arithmetic, pattern continuation, definitions, narrative, why-questions), each prompt in a
  completion form for base checkpoints and a chat form for SFT/RL ones, run greedily on every snapshot (CPU
  beside a training job, ~1 min per 336M checkpoint) and scored blind by an LLM judge on correctness,
  coherence and task (1-5). `pack` writes shuffled packets with opaque item ids and the rubric text; `ingest`
  validates the judge's JSON and rebuilds `quality/summary.json`, which the run page charts. The three prompts
  the trainer has always sampled are averaged separately (`legacy3`). Protocol and rubric: `docs/quality_eval.md`.
- `external.py`: **external comparison models** — seven similarly sized open-weight LMs (SmolLM2-135M/360M base and
  Instruct, Qwen2.5-0.5B base and Instruct, GPT-2 medium; Apache-2.0 / MIT) scored by our own evals as comparison
  rows. `EXTERNAL_MODELS` is the registry (hf id, params, license, chat or base, max positions, published training
  tokens); weights live in `<data root>/models/<name>/` (`python -m slm.eval.external download`, the only networked
  step, run in a child process). Importing the module forces `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` /
  `HF_DATASETS_OFFLINE`, so no eval or inference touches the network. `HfChatModel` wraps the HF forward pass in our
  own decode loop (`sample_next`, left-padded batches, stop tokens from the model's own config), so decoding settings
  mean what they mean for our checkpoints and no `generation_config.json` default leaks in; each model uses its own
  tokenizer and chat template, and a base model is never given a template. Every eval takes `--external <name>`
  (lm-eval through its own `hf` backend; facts completion form for every model; reasoning, multi-turn, judged quality
  and the swarm's sampling half for chat models only; needle for all, with lengths counted in the model's own tokens
  on the same corpus windows and secrets as ours, and lengths beyond its position table n/a) and writes the same
  file shape to `runs/ext_<name>/` with `"checkpoint": "external:<name>"` and a `"model"` block. What depends on our
  tool protocol or think span (tool-use rates, sandbox verification, the selector, multi-turn misfire) is null, not
  approximated. `scripts/measure_external.sh <name>` runs them all; the Evals tab shows the rows as a separate
  "external models" group after ours. The portal's Inference tab loads them into a slot (`external:<name>`) next to
  our checkpoints; `HfChatModel.stream_ids` is its token-streaming path (same sampler, seeding and stop rule as
  `generate_ids`), see `docs/command_center.md`.

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

    ... 120 - 36 = <|python_call|>120-36<|/python_call|><|python_result|>84<|/python_result|>84 pages left ...

Tool calls are part of thinking: they are only converted inside the think span in SFT, a call emitted after
`<|/think|>` is refused by the generation loop (the row terminates as malformed), and the parser flags tool tokens
in the answer as malformed, so the `tool` reward scheme pays nothing for them. The dataset's echoed number after a
result ("<<12*52=624>>624 pages") is dropped in conversion: the result span carries it, and the model is not trained to
repeat tool output (only the final `#### N` is repeated, for the verifier).
The model generates through `<|/python_call|>`; the harness runs the code and appends the result span; generation
resumes. Result tokens are environment-written: loss mask 0 in SFT (`format_chat(tools=True)`), `gen_mask` 0 in RL
so they are excluded from the policy gradient and the KL term. One `PySession` per conversation keeps variables and
functions across calls and across turns (REPL semantics); further tools are Python functions exposed in that
namespace (see declared functions below), not new token types. Text form for datasets and display: GSM8K's own `<<expr=result>>` for one expression
and `<<<code>>>` for a short program; `split_markup` runs the code while converting so the recorded result is exactly
what inference would insert, and `render_tools` turns generated ids back into the same markup.

Declared functions (`slm/tools/functions.py`): a conversation can hand the model capabilities it could not write
itself — a price list, a lookup, a measurement. Each is declared once, right after `<|bos|>` and before the first
turn, as one block per function:

    <|python_def|>def unit_price(item: str) -> float<|python_comment|>Catalogue price of an item in dollars. Use it instead of guessing.<|/python_def|>

The signature is a real Python `def` line without a body; the comment is natural language (what it does, when to
use it). Every token of every block is loss-masked, like a tool result: the environment wrote it, the model only
reads it. `FunctionDecl(name, signature, comment, impl)` is the registry entry, `render_defs` / `parse_defs` are the
id<->declaration pair (the portal parses a prompt back). `format_chat(functions=[...])` emits the blocks (segment
label `python_def`) and, with `tools=True`, registers the impls in the conversation's `PySession`, so `<<code=result>>`
markup that calls them resolves at conversion time exactly as inference would; `sample_with_tools(functions=[...])`
does the same for every row's session. A conversation stored as data can carry its declarations instead as a leading
`{"role": "functions", "decls": [...]}` message, so one serialized conversation stays self-contained; the parameter
wins if both are given. The functions are ordinary Python callables we provide, registered in the session namespace
under a name that may shadow nothing: a call goes through the normal call path (same result rendering, same error
hints, reachable from a `def` written in the sandbox), an exception it raises comes back as `error: ...` like any
other failure, and a name that was never declared still gets the plain NameError hint. A declaration without an impl
(the portal sends `{name, signature, comment}` only) is still registered, so calling it says it has no implementation
here instead of looking like a name the model invented. Nothing else changes: no new call syntax, no new tokens
beyond the three, and a conversation without declarations is byte-identical to before.

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
`slm.rl.synth --tools` writes templated traces whose every step is a call (programs for multi-step tasks);
`slm.rl.synth_multiturn` writes 2–4-turn conversations whose follow-ups ("now add 5", "double it") reuse the session
variable set in the first turn, so multi-turn REPL behaviour is trained directly.

Those three teach the tags as a *calculator*: measured over their 87,593 call spans, not one contains a loop, a
`def`, a list, a string method, an `if` or `math.`, although the sandbox supports all of them. `slm/rl/synth_python.py`
(`synthetic-python-tools`, 2026-09-19) fixes that by generating programs from small grammars instead of templates —
each step of a recipe has several idioms (loop with an accumulator, comprehension, builtin) and the prose question is
composed from the same recipe, so the set has tens of thousands of distinct program shapes rather than dozens of
templates. Families: **pipeline** (produce a list / string / range / dict, apply 1–3 transforms, aggregate),
**strings**, **numbers** (gcd, lcm, primes, digit sums, factorials, powers, fibonacci-like recurrences, collatz,
base conversion, hypotenuse, doublings), **simulation** (loops with state: growth, interest, inventories, scoring),
**multiturn** (a helper `def` or a variable defined in one turn and reused in 2–4 later ones, never recomputed),
**runcode** ("what does this print?": the user's own program, executed verbatim), **error** (a first call that really
trips a sandbox hint, the hint read in the think text, then a corrected call), **declared** (1–3 `FunctionDecl`
capabilities — a price list, a population register, distances, a unit conversion, postage, a sensor service, a
warehouse lookup; a few conversations declare a function that is *not* needed, and a few call a name that was never
declared and recover), and the old arithmetic word problems at a minority share. Real sentences and words are drawn
from the `fineweb-edu-b` validation shards (`load_corpus`, with a built-in fallback so tests need no data root) so the
inputs read like text. Correctness is doubly enforced: the gold answer is computed independently on the host and
cross-checked against a real `PySession` run while generating (mismatches are dropped), and `format_chat(tools=True)`
runs every program again at conversion time, so the stored result span is exactly what inference would insert. The
one exception to the markup path is the error family: `split_markup` deliberately drops a *failing* call back to plain
text, so those turns are encoded as explicit call spans (`_turn_ids`) and `encode_sample` then re-asserts the
invariant that no `<|python_result|>` token is a loss target. Two whole families — `pipeline.dict` and
`declared.distance` — are written only to the validation split, so val measures generalisation to program shapes that
were never trained on. `tests/test_synth_python.py` checks per-family verification, the real hints, declared-function
resolution, the masks, feature coverage and skeleton diversity floors, and the hold-out.

## Swarm inference (`slm/swarm.py`, `slm/rl/synth_select.py`, `scripts/swarm_eval.py`)

Goal 5: use the batch parallelism one GPU gives (dozens of completions per prompt at once) as a *system* around one
model, not as a way to run one model faster. The pipeline is sample → collapse → verify → select, and every stage is
a plain function that returns its intermediate state so the eval can report the ceiling at each step and the
portal's inference tab can show what the swarm saw and chose.

- **Sample.** `sample_candidates` runs `sample_with_tools` on `k` copies of the chat prompt (temperature 0.8,
  top-p 0.95, the tool available, `max_calls` 6). Each completion becomes a `Candidate`: think span, answer span,
  the parsed `#### <answer>`, its tool calls, and `from_tool` (the answer came out of a real call,
  `slm.rl.rewards.answer_from_tool`). `verified` = `from_tool and n_errors == 0` — sandbox-backed evidence that
  the model cannot fake in its head and that majority voting ignores.
- **Collapse.** `answer_key` canonicalises answers (numbers by value, so 42 / 42.0 / 42.00 agree; text
  lowercased) and `collapse` groups candidates into `Group`s carrying `support`, `verified` (how many members were
  sandbox-backed) and a representative rationale (a verified member's think, shortest first). Groups are ordered
  verified-support first, then support: with 22–24 distinct answers per 32 samples (`docs/results.md` §14a),
  raw support is nearly noise and provenance is the signal. `majority` and `verified_majority` are the two
  mechanical baselines.
- **Select.** `selector_messages` renders the task and the groups as one user turn (`SELECT_INTRO` /
  `SELECT_ASK` are module constants so SFT, RL and inference all see one wording) inside a token budget
  (2400 by default, measured with the real tokenizer; rationales are shortened before groups are dropped, never
  below three). `select` is one greedy pass of the *same* model with the tool available, so it can re-check.
  `final` is the selector's parsed answer, else `verified_majority`.
- **Teaching the selector.** Selection is a prompt format the model has never seen, so it is trained like every
  other behaviour. `slm.rl.synth_select` samples k=8 candidates from a checkpoint on GSM8K-train, SVAMP-train and
  the synthetic word families, collapses them, and writes (a) an SFT set `select-sft` — the selection prompt as
  the user turn, a one-line templated think stating what the pool showed, `#### <gold>` — for pools that contain
  a correct group, and (b) `select_pool.jsonl` for the RL `select` family (`slm.rl.tasks.select_pool`, in
  `POOLED`), which keeps *every* pool including those with no correct candidate: there the only rewarded move is
  to solve the problem afresh, which is the override behaviour a selector needs and that imitation cannot teach.
  No teacher is involved: the candidates are the model's own, the evidence is mechanical, the target is the
  dataset's gold. `reward_schemes: {select: binary}` — the evidence is already in the prompt, so no tool credit.
- **Ceilings.** `scripts/swarm_eval.py` reports, from the same k samples, greedy / majority / verified_majority /
  selector, and the ceilings oracle (pass@k), oracle_verified (a correct verified candidate exists) and in_prompt
  (the correct group survived into the selector prompt). The gaps locate the work: oracle − oracle_verified is
  what verification throws away, oracle_verified − in_prompt what the budget throws away, in_prompt − selector
  what the selector still gets wrong.

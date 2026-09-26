# Command center (web UI)

`python -m slm.portal` → http://127.0.0.1:8765. A local supplement for looking at runs, data, the
tokenizer, checkpoints and the architecture. The chat/CLI workflow remains the primary interface; the
portal never launches training.

## Architecture

- **Server** (`slm/portal/app.py`, FastAPI + uvicorn) is torch-free: it reads `runs/`, the data root and
  the tokenizer directly. Routers under `slm/portal/api/`, logic under `slm/portal/services/`. Static
  assets are served with `Cache-Control: no-cache`, so a browser reload picks up JS edits; Python edits
  need a restart. `--runs-root`, `--port`, `--no-browser`, `--reload` flags; settings in `settings.py`.
- **Frontend** (`slm/portal/static/`): no build step; vendored ESM Preact + htm and uPlot; hash router
  in `app.js`; pages in `pages/`, shared pieces in `components/` (`chart.js` with an optional second
  y-axis, `tokens.js` token chips with loss-mask colouring, `util.js` formatters and `api()`, `info.js` +
  `cards.js` the explanatory info cards, see below).
  `index.html` carries boot diagnostics: a failed module import, a boot exception or a 10-second stall
  shows an error panel; an `ErrorBoundary` in `app.js` contains page crashes.
- **Worker** (`services/worker.py`): all torch work runs in a lazily spawned subprocess. GPU guard:
  loads default to CPU while a training run is live (unless forced); idle auto-stop; explicit
  `POST /api/model/worker/stop`. A crash cannot take the server down. `services/harness.py` runs inside
  the worker: checkpoint slots, streaming generation with per-token log-probs and top-k alternatives,
  teacher-forced scoring, diagnostics.
- **Data access** (`services/datasets.py`): row-group addressed parquet reads and per-request memmaps of
  token shards (never cached, because Windows cannot replace a mapped file). Chat/math rows are
  rendered as transcripts and normalized through the same `rows_to_messages` the SFT pipeline uses.
- **Metrics** come from `slm/utils/metrics.py` (shared with the HTML report): incremental JSONL tailing,
  status detection, series thinning, checkpoint listing from `index.json`, cumulative tokens along the
  `init_from` chain.

## Pages

| Page | What it shows |
|---|---|
| Overview | Header with GPU memory/utilization/temperature/power/throttle state; live-run cards (progress, loss, ETA); the pipeline table (stage → runs with status, tokens this run and cumulative, progress, loss, result, tok/s, elapsed + ETA, started + git, init chain); data readiness. Click a run to open it. |
| Run detail (`#/runs/<run>`) | Tiles (progress, ETA, elapsed vs initial estimate, tok/s, losses or RL reward/held-out/KL/length, LR, grad norm, VRAM, GPU °C/W with run max, step timing); charts with x in tokens/updates/time and log-y (loss, validation incl. pretraining-mixture drift, tok/s, LR, grad norm, step time, VRAM, GPU temperature/power, RL charts); milestones table; samples timeline (per 100M tokens, greedy vs sampled); needle chart overlays `needle_sweep.json` (every snapshot re-measured at one n) as heavy lines when present; judged quality (chart of overall / per-rubric / legacy-3 scores and a per-category chart when `quality/summary.json` exists; the `quality` tab lists every prompt's output for a checkpoint with its three scores and the judge's note, plus a category table); checkpoints with exact tokens; events (start/resume/stop/finish/checkpoint/warn); config. Live tail via SSE. |
| Data (`#/data`) | **Recipes** (default): one row per training config and per run, by stage, with a marker where the yaml no longer matches what the run started with. A recipe (`#/data/recipes/<id>`, id = `run:<name>` or `config:<path>`) is the mixture table with weight, planned and available tokens, epochs (red above 1.5), the loader's own per-source consumption when the checkpoint records carry it, the `extra_val_mixture` drift set, and for RL a prompt/reward panel with the deterministic prompt list (rendered as the exact generation prompt) and, for a run, a rollouts viewer (step selector, prompt + completion chips, reward, parsed answer, malformed flag). Clicking a mixture row opens the inspector: **raw** (a parquet row of the source it was prepared from, with a *trace* that re-runs the preparation filters and reports kept/dropped and the split), **prepared** (the stored document, or the stored SFT example with green loss-mask chips), **row** (a training row of exactly `seq_len + 1` tokens with red document — or SFT example — boundaries and the target count). **Compare** (`#/data/compare/<a>/<b>`, client-side from two recipe payloads): header diff and the union of sources with weight A / B, the delta in percentage points, planned tokens and epochs; a run's default pairing is its `init_from` parent. **Chain** (`#/data/chain/run:<name>`): the `init_from` walk back to the root, one row per stage with tokens used and per-source tokens (the loader's own counters where the run logged them, `tokens x weight` otherwise — the row says which), a totals row and a stacked bar. **Catalog** (`#/data/catalog`) lists every raw, tokenized and chat-formatted set, with unused and `*-v1` sets behind a "show unused" toggle; `#/data/source/<name>` carries raw files, prepared artifacts per tag, provenance both ways, the recipes that use it with their weights, and a **browse** panel with the old documents browser (tokenizer tag / source / split / shard or parquet file / row group; text, tokens or ids; `window`; `stats` with length percentiles and long-doc counts). The list and preview fill the viewport. |
| Tokenizer | Playground: encode text in document or chat mode, coloured chips with offsets and ids, vocabulary lookup. |
| Inference | Two checkpoint slots (A/B) loaded in the worker (device auto/cuda/cpu, force flag), completion, chat and swarm modes (swarm: see below), a think toggle, streaming tokens with log-probs and top-k alternatives, raw-text vs tokens view (reserved tokens stay visible), prompt scoring, cancel, release GPU. |
| Architecture | Any model config: interactive expandable module graph with symbolic and numeric shapes (B and T sliders), per-node params and FLOPs, GQA diagram, parameters by family, KV-cache size, memory budget vs measured benchmark, LR schedule / RoPE / batch / cadence illustrations computed by the real training functions. |

## Info cards (`components/info.js`, `components/cards.js`)

The portal is a learning tool, so every chart, tile, column header and section whose meaning is not obvious carries a
small "?" that opens a card explaining the concept: what it measures, how to read it (what good and bad look like),
and how it connects to the rest of the pipeline. Hover opens it, a click or tap pins it, keyboard focus opens it,
Escape / blur / scroll closes it. The card is `position: fixed`, clamped to the viewport (it flips above the icon when
there is no room below) and rendered only while open, so it never shifts the layout; it uses the theme tokens.

- **Content** lives in one registry, `cards.js`: ``CARDS[key] = { t: "Title", b: html`<p>…</p>` }``, one card per
  concept, reused wherever the concept appears (e.g. `val_loss` on the tile and the chart). Keep to the pattern: a
  paragraph on what it is, then how to read it, then an optional muted `<p class="see">` on the connection; explain
  concepts, not current numbers (those belong on the page and in `docs/results.md`).
- **Placement**: `<${Info} k="val_loss" />` right after the title text (inside an `h2`/`h3`/`th`/tile `.k`);
  `<${Chart} … info="lr" />` for a chart (Chart renders its own title so the icon sits inline); run-page tiles take
  `info=`. The run page shows one icon after the tab buttons that explains whichever tab is open (`tab_<name>`).
  Never put one inside a `<label>` or `<summary>` (a click there toggles the control). A missing key logs a
  `console.warn` and renders nothing.

## API (all under `/api`)

`GET /meta`, `GET /system/gpu` · runs: `GET /runs`, `/runs/{run}`, `/runs/{run}/series`, `/events`,
`/checkpoints`, `/samples`, `/samples/{tokens}`, `/quality`, `/quality/{tokens}`, `/rollouts?step=`, `/report`, `/live` (SSE) · data: `/data/sources`,
`/data/configs`, `/data/recipes`, `/data/recipe?id=`, `/data/chain?run=`, `/data/source/{name}`,
`/data/rl/prompts?id=&split=`, `/data/mixture` (legacy shape, kept for one release),
`/data/raw/{source}/files|docs|doc|sample`, `POST /data/raw/{source}/trace`,
`/data/tokenized/{tag}/{source}/{split}/shards|docs|doc|window|stats`,
`/data/sft/{tag}/{set}/{split}/shards|examples|example|window|stats` · tokenizer: `GET /tokenizers`,
`POST /tokenizers/{tag}/encode`, `GET /tokenizers/{tag}/vocab`, `/token/{i}` · model:
`GET /model/status`, `POST /model/worker/stop`, `GET /model/checkpoints`, `POST /model/slots/{slot}/load|unload`,
`POST /model/score`, `POST /model/generate` (SSE), `POST /model/swarm` (SSE), `POST /model/streams/{sid}/cancel`, `POST /model/diagnostics` ·
arch: `/arch/configs`, `/arch/graph`, `/arch/hparams`, `/arch/benchmark`.

## Tests and checks

- `tests/test_portal_runs.py`: API on synthetic runs; every JS module must parse (`node --check`).
- `tests/test_portal_live.py`: SSE tail against a real uvicorn server.
- `tests/test_portal_model.py`: worker load/generate/score/cancel on CPU.
- `tests/test_portal_swarm.py`: `/api/model/swarm` validation and SSE shape against a stub worker, `Harness.swarm` stages
  with scripted sampling/selection (incl. cancel), and one real pass on a tiny CPU model.
- `tests/e2e/test_portal_ui.py`: Playwright + Chromium on hermetic data; every page opened, every
  button clicked, every select cycled; no JS errors, no hangs, no raw template text.
- `scripts/portal_smoke.py`: the same click-through against a live portal with real data.
- Layout rules verified at a 1000 px viewport: no page scrolls horizontally; the Overview table is the
  one element that may still need its own scroll below ~1100 px.

## Backlog (from the original plan)

P1: batch replay inspector (exact rows of a given update), run comparison overlays, sampled document
search, A/B side-by-side with divergence marker, prompt-scoring view. P2: diagnostics viewer in the
portal, animations on the architecture page, stage "lenses", STOP control for runs, a run scheduler.

## Inference page: chat with the Python tool (2026-09-15)

Chat mode is a multi-turn conversation. When a generated assistant turn is well-formed (ends with `<|end|>`,
think span closed, no tool call outside the think span) it is appended to the message list verbatim — the
exact generated token ids travel with the message so the next prompt reuses them without re-running any tool
call — followed by an empty user turn to fill in. Editing a generated turn drops its ids and re-encodes it.

The `python tool` checkbox lets the model call the sandbox: generation pauses at `<|/python_call|>`, the code
runs in the conversation's `PySession` (variables persist across calls and turns), a *python calls* panel lists
code and results, and the inserted result tokens are shown in blue with no log-prob (they were not sampled).
`new conversation` resets the session on the worker and clears the messages. The worker keeps up to 32
sessions keyed by conversation id; the API is `POST /api/model/generate` with `tools`, `session_id`,
`max_tool_calls`, `functions`, and `POST /api/model/sessions/{id}/reset`.

*declared functions* (a JSON list of `{name, signature, comment}`, collapsed under the message list) prepends one
masked `<|python_def|>signature<|python_comment|>comment<|/python_def|>` block per function to the prompt; the
`prompt` event carries the chat segments, so those tokens are highlighted in amber in both views. The portal holds
no implementations, so a call to a declared function reports that it is declared but not available here — the point
of the control is to see the blocks the model reads and how it reacts to them.

## Inference page: swarm mode (2026-09-26)

The third mode button runs swarm inference (`slm/swarm.py`) on one slot: the task is sampled k times in one batch
with the Python tool available, the attempts are collapsed by parsed final answer (support = attempts that agree,
verified = attempts whose answer came out of a Python call that ran without error), and one greedy pass of the same
model over the selection prompt (`selector_messages`, built under a token budget) picks the final answer. It is the
same pipeline `scripts/swarm_eval.py` measures, stage for stage and with the same generator seeding.

- **Controls**: slot, k (1-64, default 16), temperature (default 0.8: a swarm needs diverse attempts), top-p, max new
  tokens (512), max tool calls (6), seed (blank = the worker draws one and reports it; "reuse seed" pins it), *append
  answer instruction* (default on: appends `slm.data.answers.SUFFIX`, the `#### <number>` line, to the sampling prompt
  only), selector budget (2400 tokens) and max groups (12). An optional *expected answer* turns on the ceilings table.
- **While it runs**: a stage strip (sample k → collapse by answer → select → done) with elapsed time; after the
  collapse the groups table appears before the selector finishes. Cancel takes effect at the next stage boundary (the
  k samples are one batch); cancelled after sampling, the selector is skipped and the final answer is the verified majority.
- **Result**: the final answer (and where it came from: the selector, or the verified-majority fallback when the
  selector gave no `####` line), majority and verified majority for comparison; the groups table (answer, support,
  verified, whether it survived into the selector prompt, representative rationale; click a row for its member
  samples with think span, tool calls and results, answer, verified flag and token count); the samples with no
  parsable answer; the selector prompt (collapsible), think and answer. With an expected answer, a ceilings table
  shows pass@k, verified pass@k, in-prompt and selector for this one task (client-side match, close to but not the
  eval's verifier).
- **API**: `POST /api/model/swarm` with `{slot, text, k, temperature, top_p, max_new_tokens, max_calls, seed,
  budget_tokens, max_groups, answer_suffix}` (validated: k <= 64, max_calls <= 16, non-empty text). SSE events:
  `start` (stream id), `stage` (`sampling`; `collapsed` with the groups, majority and verified majority; `selecting`
  with the selector prompt's token count), `done` with the whole `SwarmResult.to_dict()` plus a per-candidate
  `verified` flag and `meta` (seed, settings, `cancelled`, `selector_parsed`, `groups_in_prompt`, `prompt_tokens`).
  The worker method is `Harness.swarm` (a streaming method, run under `sdpa_context("decode")` and `no_grad`, with
  `torch.cuda.empty_cache()` at the end when the slot is on the GPU).
- **Cost**: the KV cache grows with k × (prompt + max new tokens). On CPU a 149M model does k=8 × 160 tokens in
  seconds; on a GPU shared with a training run, keep k and max new modest.
- Info cards: `swarm`, `swarm_k`, `swarm_suffix`, `swarm_support`, `swarm_majority`, `swarm_selector`, `swarm_budget`,
  `swarm_oracle`.

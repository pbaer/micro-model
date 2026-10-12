# Command center (web UI)

`python -m slm.portal` → http://127.0.0.1:8930 (this project's ports are 8930-8939; `slm.portal.settings.PORT_RANGE`). A local supplement for looking at runs, data, the
tokenizer, checkpoints and the architecture. The chat/CLI workflow remains the primary interface; the
portal never launches training.

## Architecture

- **Server** (`slm/portal/app.py`, FastAPI + uvicorn) is torch-free: it reads `runs/`, the data root and
  the tokenizer directly. Routers under `slm/portal/api/`, logic under `slm/portal/services/`. Static
  assets are served with `Cache-Control: no-cache`, so a browser reload picks up JS edits; Python edits
  need a restart. `--host`, `--port`, `--runs-root`, `--data-root`, `--gpu-policy auto|cuda|cpu`, `--default-chat`,
  `--no-browser`, `--reload` flags; settings in `settings.py`. A taken port is refused with the holder's name and a free
  spare of the range (the portal never moves on its own: chats are stored per browser origin). FastAPI's own API docs
  are at `/api/docs`.
- **Frontend** (`slm/portal/static/`): no build step; vendored ESM Preact + htm and uPlot; hash router
  in `app.js`; pages in `pages/`, shared pieces in `components/` (`chart.js`, every chart on the dashboard: an optional
  second y-axis; hover values in a card beside the cursor, highest value on top, for the chart under the pointer only (a
  sparse series shows its nearest point within 24 px, unlabelled; values in one right-aligned column); a static legend below (name + a line in the
  series colour, never a box: hover highlights the series, click hides it); only the vertical crosshair line, no
  horizontal one, and only in the hovered chart (no cursor sync between charts); Peter, 2026-10-10; `tokens.js` token chips with loss-mask colouring and the shared text view (`TextWithSpecials`, see "Text view
  and reserved tokens" below), `specials.js` its pure splitting helper, `util.js` formatters and `api()`, `info.js` +
  `cards.js` the explanatory info cards, see below; `context.js` the context-window meter, `swarm.js` the swarm panel and
  bracket, `arena_views.js` the Arena's episode model and views).
  The nav carries a GPU tile on every page (memory, utilization, power, temperature, live training runs; polled every
  5 s). `index.html` carries boot diagnostics: a failed module import, a boot exception or a 10-second stall
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
| Overview (`#/`, also `#/runs`) | Header with GPU memory/utilization/temperature/power/throttle state; live-run cards (progress, loss, ETA); the **run lineage graph** (`pages/home.js`, since 2026-10-10): a node per run, an arrow where a run started from another run's checkpoint (`init_from`), laid out left to right by generation, one tree per base family, newest family on top, siblings oldest first. A node shows the run name, a status dot, its stage (coloured edge), parameter size, tokens or steps, and its headline number (judged quality, else held-out accuracy, else best val loss); a label on an arrow names the starting checkpoint when it was not the parent's final.pt; a dashed node is a parent missing from the runs folder. Hover shows a details card (status and ETA, stage and params, parent checkpoint, tokens this run and along the lineage, loss / val / held-out / judged quality, needle length, time and tok/s, start and git, data sources, RL tasks, children) and lights up the run's ancestors and descendants; a click opens the run page. Zoom: fit (width, never below 75%) / − / +; the graph scrolls inside its panel. Data readiness below. |
| Run detail (`#/runs/<run>`) | Header with report.html and data-recipe links (and a note when the yaml no longer matches what the run started with); tiles (progress, ETA, elapsed vs initial estimate, tok/s, losses or RL reward/held-out/KL/length/policy objective, needle effective context when measured, LR, grad norm, VRAM, GPU °C/W with run max, step timing); charts with x in tokens/updates/time and log-y (loss, validation incl. pretraining-mixture drift, tok/s, LR, grad norm, step time, VRAM peak and reserved, GPU temperature/power, RL charts, tool misfire per checkpoint when the quality outputs carry it); milestones table; samples timeline (per 100M tokens, greedy vs sampled); needle chart overlays `needle_sweep.json` (every snapshot re-measured at one n) as heavy lines when present; judged quality (chart of overall / per-rubric / legacy-3 scores and a per-category chart when `quality/summary.json` exists; the `quality` tab lists every prompt's output for a checkpoint with its three scores and the judge's note, plus a category table); checkpoints with exact tokens; events (start/resume/stop/finish/checkpoint/warn); config. Live tail via SSE. |
| Evals (`#/evals`) | One table: rows = every checkpoint with at least one result file, columns = the public benchmarks (lm-eval) and the homebrew evals (facts, reasoning families, multi-turn, needle, judged, pass@k, swarm) under group headers; cells coloured per column from red (worst) to green (best), grey n/a. Details in "Eval tab" below. |
| Data (`#/data`) | **Recipes** (default): one row per training config and per run, by stage, with a marker where the yaml no longer matches what the run started with. A recipe (`#/data/recipes/<id>`, id = `run:<name>` or `config:<path>`) is the mixture table with weight, planned and available tokens, epochs (red above 1.5), the loader's own per-source consumption when the checkpoint records carry it, the `extra_val_mixture` drift set, and for RL a prompt/reward panel with the deterministic prompt list (rendered as the exact generation prompt) and, for a run, a rollouts viewer (step selector, prompt + completion chips, reward, parsed answer, malformed flag). Clicking a mixture row opens the inspector (on the shard's first document or example, in the text view): **raw** (a parquet row of the source it was prepared from, with a *trace* that re-runs the preparation filters and reports kept/dropped and the split), **prepared** (the stored document, or the stored SFT example with green loss-mask chips), **row** (a training row of exactly `seq_len + 1` tokens with red document — or SFT example — boundaries and the target count). **Compare** (`#/data/compare/<a>/<b>`, client-side from two recipe payloads): header diff and the union of sources with weight A / B, the delta in percentage points, planned tokens and epochs; a run's default pairing is its `init_from` parent. **Chain** (`#/data/chain/run:<name>`): the `init_from` walk back to the root, one row per stage with tokens used and per-source tokens (the loader's own counters where the run logged them, `tokens x weight` otherwise — the row says which), a totals row and a stacked bar. **Catalog** (`#/data/catalog`) lists every raw, tokenized and chat-formatted set, with unused and `*-v1` sets behind a "show unused" toggle; `#/data/source/<name>` carries raw files, prepared artifacts per tag, provenance both ways, the recipes that use it with their weights, for a source prepared by a selecting preparer (`gutenberg-pg19`) a **selection filter** panel (books in, kept, dialogue cutoff, mean tokens per book, raw and tokenized size, and one row per rule with its threshold, books removed, tokens removed and example titles; the catalog marks such sources "filtered", and the raw trace reports the per-book verdict from `books.jsonl` instead of re-running `keep_doc`), for a curated set (`gutenberg-canon`) a **curated list** panel (books, tokens, how often each rule was relaxed, every book by list entry with year, tokens, split and the rule relaxed for it, and a toggle for the seed titles not in the set with the reason; the catalog marks it "curated"), and a **browse** panel with the old documents browser (tokenizer tag / source / split / shard or parquet file / row group; selecting a source, shard, file or row group opens its first document right away; text (default), tokens or ids; a raw chat row's text view is the SFT chat format with its loss mask; `window`; `stats` with length percentiles and long-doc counts). The list and preview fill the viewport. |
| Tokenizer | Playground: encode text in raw, document or chat mode; text view (default: decoded, reserved tokens as chips, loss mask in chat mode) or coloured token chips with offsets and ids; vocabulary lookup. |
| Inference (`#/inference`) | **Chat mode** (default): a plain chat with the best all-round chat checkpoint, loaded on arrival, streamed replies with the think span folded under "reasoning", several conversations kept in the browser, the few settings that matter behind a gear (see "Inference tab: chat mode" below). **Advanced mode** (`#/inference/advanced`): two checkpoint slots (A/B) loaded in the worker (device auto/cuda/cpu, force flag), each holding one of our checkpoints or a local open-weight comparison model (external slot: see below), completion, chat and swarm modes (swarm: see below), a think toggle, streaming tokens with log-probs and top-k alternatives, A/B side by side with the first diverging token marked, text (default) vs tokens view (reserved tokens are chips in both), prompt scoring, cancel, release GPU. `#/model` is an old alias of the advanced mode. |
| Arena (`#/arena`) | The robot grid of `slm/arena/world.py` on a loaded slot, or an episode file in viewer mode: controls, a live SVG grid (robots, doors, comm range, path trails, messages as labelled arcs), a turn scrubber, a task panel, an episode summary strip, an interaction timeline (swimlanes), a message log, a compact table of every robot's turn and one selected robot's transcript; JSON episode download. Details in "Arena tab" below. |
| Architecture | Any model config: interactive expandable module graph with symbolic and numeric shapes (B and T sliders), per-node params and FLOPs, GQA diagram, parameters by family, KV-cache size, memory budget vs the training-throughput benchmarks measured on the selected config's model file (`/arch/configs` lists them as `training_benchmarks`, leaving out other kinds such as `bench_inference.py`'s `infer_*.json`; `/arch/graph` reports the `model_file` a model / train / run selection rests on; none measured says so), LR schedule / RoPE / batch / cadence illustrations computed by the real training functions. |

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
`/checkpoints`, `/samples`, `/samples/{tokens}`, `/quality`, `/quality/{tokens}`, `/rollouts?step=`, `/report`, `/live` (SSE) · `GET /evals`, `/evals/detail?run=&checkpoint=&key=&file=&filter=&q=&offset=&limit=` · data: `/data/sources`,
`/data/configs`, `/data/recipes`, `/data/recipe?id=`, `/data/chain?run=`, `/data/source/{name}`,
`/data/rl/prompts?id=&split=`, `/data/mixture` (legacy shape, kept for one release),
`/data/raw/{source}/files|docs|doc|sample`, `POST /data/raw/{source}/trace`,
`/data/tokenized/{tag}/{source}/{split}/shards|docs|doc|window|stats`,
`/data/sft/{tag}/{set}/{split}/shards|examples|example|window|stats` (doc / example / window / trace / RL prompts and
rollouts also carry `runs`, see below) · tokenizer: `GET /tokenizers` (with each tag's `specials`),
`POST /tokenizers/{tag}/encode`, `GET /tokenizers/{tag}/vocab`, `/tokenizers/{tag}/token/{i}` · model:
`GET /model/status`, `POST /model/worker/stop`, `GET /model/checkpoints` (ours, then the external models; every entry carries `chat_ok`, one carries `default_chat`), `POST /model/slots/{slot}/load|unload`,
`POST /model/score`, `POST /model/generate` (SSE), `POST /model/sessions/{sid}/reset`, `POST /model/swarm` (SSE), `GET /model/arena/tasks`,
`/model/arena/episodes`, `/model/arena/episodes/{name}`, `/model/arena/world`, `POST /model/arena` (SSE), `POST /model/streams/{sid}/cancel`,
`POST /model/diagnostics` (no page uses it yet) ·
arch: `/arch/configs`, `/arch/graph`, `/arch/hparams`, `/arch/benchmark`.

## Tests and checks

- `tests/test_portal_runs.py`: API on synthetic runs; every JS module must parse (`node --check`); the server process
  never imports torch; data endpoints (replaced shards noticed, manifest normalizer and recipes incl. an RL yaml, SFT
  window masks, a filtered source and a book's recorded trace verdict, the `init_from` chain, rollouts, RL prompts);
  the Architecture page's training benchmarks.
- `tests/test_portal_evals.py`: the eval table on a synthetic runs tree (attribution, file priority, quality rows by token
  count, colour positions, mtime cache), `/api/evals`, an info card for every column, `node --check` of the page; the
  detail pages on synthetic result files of every kind (judged outputs joined with scores and aliases, a reasoning dump
  split by block size, facts / needle / multi-turn / pass@k / swarm rows, lm-eval aggregate-only with two sources and a
  log tail, verdict filter, search, paging, 404s, the file-size cap, mtime cache) and `/api/evals/detail`; external
  models as their own group after ours (registry fields, n/a where our tool protocol is needed, the shared colour
  scale); the multi-turn kinds' columns and detail rows.
- `tests/test_portal_runs.py` also covers the text view: `text_runs` on a real tokenizer (exact specials, a literal
  "<|user|>" stays text, loss flags, multi-byte characters whole), the `runs` field of the doc / window / SFT example /
  RL prompt endpoints and `specials` on `/api/tokenizers`, `splitSpecials` under node against the tokenizer's actual
  special list, and a source check that every text/tokens toggle defaults to text and both data browsers auto-open the
  first document.
- `tests/test_portal_live.py`: SSE tail against a real uvicorn server.
- `tests/test_portal_model.py`: worker load/generate/score/cancel on CPU; external slots with a stubbed `HfChatModel`
  behind an in-process worker (`tests/_ext_stub.py`: the listing, loading, streamed completion and chat, the refusals),
  and a real CPU smoke test (SmolLM2-135M-Instruct through the worker subprocess, skipped when the weights are absent).
- `tests/test_portal_swarm.py`: `/api/model/swarm` validation and SSE shape against a stub worker, `Harness.swarm` stages
  with scripted sampling/selection (incl. cancel), the tournament path with a scripted `compare_batch` (round events,
  bracket equal to `slm.swarm.tournament`'s, both modes, the entrant cap, cancel at a round boundary), one real pass
  on a tiny CPU model, and an external slot's swarm (stubbed model: own template, verification n/a, majority fallback).
- `tests/test_portal_arena.py`: `Harness.arena` with a scripted `arena_generate` (event shape per turn, the messages list, the declared tools read back from the prompts, stop at done, the cancel path, refusals incl. an external slot), `arena_messages` against `deliver()`, one real turn of a tiny CPU model, `/api/model/arena` validation and SSE shape against a stub worker, `/api/model/arena/tasks`, the nav entry, an info card for every `arena_*` key, `node --check` of the changed JS; the viewer endpoints (episode list and read with path safety, `/api/model/arena/world`) and the grid's messages toggle.
- `tests/test_portal_tools.py`: the Python tool through the in-process `Harness` with a scripted sampler (tool event,
  inserted-token flags, REPL state across calls with one session id, the well-formed assistant summary), generated ids
  reused verbatim by `format_chat`, declared functions in chat mode.
- `tests/test_portal_charts.py`: the chart rules as source checks on `components/chart.js` (no uPlot live legend, no
  horizontal crosshair, no cursor sync, the hover card sorted highest first with no "at <x>" qualifiers, line swatches
  rather than boxes) and that no page builds its own uPlot.
- `tests/test_portal_worker.py`: `WorkerClient` against stub workers (`tests/_worker_stub.py`) that die at spawn or mid-request:
  the call and the stream must fail with the exit code within seconds, the client resets so the next request starts a
  fresh worker, and the idle stop never fires while a request holds the lock; `describe_exit` messages; the port range
  (8930 default, clear of 8765 and common dev ports) and a taken port refused with its holder and a free spare.
- `tests/test_portal_home.py`: the lineage layout under node (`layoutLineage`, `parentOf`, `relatives` on a realistic
  forest: every path form of `init_from`, a ghost parent, column per generation, parents centred on children, newest family
  first, no overlaps) and the page wiring.
- `tests/test_portal_chat.py`: `mark_default_chat` (the configured checkpoint, the stand-in order, `chat_ok`), the chat page's
  wiring (routing, API use, localStorage with only changed settings stored, the 1536-token default in both modes, a
  `ContextMeter` in both modes, no native dialogs, a seed per reply, an info card for every `k`) and `node --check` of
  the page.
- `tests/e2e/test_portal_ui.py`: Playwright + Chromium on hermetic data; every page opened, every
  button clicked, every select cycled; no JS errors, no hangs, no raw template text. Also the lineage graph (nodes,
  the edge label, the hover card and dimming), the chart behaviour in a real browser (static legends, one hover card,
  the vertical cursor line only in the hovered chart, no horizontal line), and chat mode as the Inference default.
- `scripts/portal_smoke.py`: the same click-through against a live portal with real data (`--generate` also loads the
  newest checkpoint into slot A on CPU and generates).
- Layout rule: no page scrolls horizontally at a 1000 px viewport; wide content (the lineage graph, the eval table, the
  bracket, the arena timeline) scrolls inside its own panel. Checked by hand, not by a test.

## Text view and reserved tokens (2026-10-01)

Every place that shows tokenized content opens in the **text** view; the tokens (and ids) view is a toggle. Defaults:
the data inspector's *show as* (was tokens), the RL prompt sample (was tokens), the Tokenizer playground (new text /
tokens toggle; it showed only chips), the source browser and rollouts (already text), the Inference output and scoring
(already text). Eval detail pages, the swarm panel and the run page's samples / judged outputs have no token view; their
text goes through the same component.

- **Reserved tokens are always chips.** In the text view `<|bos|>`, `<|user|>`, `<|assistant|>`, `<|think|>`,
  `<|python_call|>` and every other special of the tokenizer's reserved block render as the same dark `chip special` the
  tokens view uses, inline in the running text, never as plain text and never dropped.
- **One component**, `TextWithSpecials` in `components/tokens.js` (with `SpecialChip`, the chip itself, and
  `useSpecials`, the registered list loaded once from `/api/tokenizers`). It takes either
  - `runs`, the API's exact form where the ids are known: `[{text} | {special, i}]`, each with `loss` when there is a
    mask. `services/tokenizer.py:text_runs` builds it from the ids (identity from the id, never from the text, so a
    literal "<|user|>" in a document stays text; ordinary ids are decoded per run, so a character split across
    byte-level ids comes out whole). Returned as `runs` by `/data/tokenized/.../doc|window`, `/data/sft/.../example|window`
    (with the loss mask), `/data/raw/{source}/trace`, `/data/rl/prompts`, `/runs/{run}/rollouts` (loss = policy target) and
    `POST /tokenizers/{tag}/encode`; or
  - `text`, a plain string (eval rows, swarm samples, run samples, think spans), split by `splitSpecials`
    (`components/specials.js`, pure, no imports) on the **registered** special strings only, longest first; any other
    `<|...|>` stays text.
  With `lossMask` the text runs are green-backed (target) or grey (masked), as in the tokens view; `boundaries` puts the
  red document-boundary outline on the specials at those token indices (the window / training-row views).
- **Inference** keeps its per-token text view (each word hovers its log-prob); the specials in it are `SpecialChip`s.
  The worker's `prompt` event carries a per-id `special` flag list (ours: id >= base vocab; an external model: its own
  special ids) and scored tokens a `special` flag, so the page no longer guesses from the piece's shape.
- **First document**: the source browser and the recipe inspector open the first document (example) of the selected
  source / shard / file / row group immediately instead of an empty "pick a document" panel; the list and *random*
  work as before.

## Backlog (from the original plan)

P1: batch replay inspector (exact rows of a given update), run comparison overlays, sampled document
search. P2: diagnostics viewer in the portal (`POST /api/model/diagnostics` exists, no page calls it), animations on
the architecture page, stage "lenses", STOP control for runs, a run scheduler. Done since the plan: A/B side by side
with a divergence marker and the prompt-scoring view (both in the Inference advanced mode).

## Inference tab: chat mode (2026-10-05)

`pages/chat.js`. The Inference tab opens in a chat a regular user can drive; everything the worker can do beyond that is
the **advanced mode** (`pages/model.js`, `#/inference/advanced`, linked at the bottom of the conversation list and back
from the page's title).

- **Model.** `GET /api/model/checkpoints` marks every chat-capable entry `chat_ok` (our SFT / reasoning / RL checkpoints,
  external chat models with weights on disk; never a base checkpoint) and one entry `default_chat`:
  `PortalSettings.default_chat_checkpoint` (`m10_rl_336m/final.pt`; `python -m slm.portal --default-chat <run>/<file>`
  overrides) when it is listed, else the stand-in `mark_default_chat` picks (RL before reasoning before SFT, a later
  milestone before an earlier, `final.pt` before `best.pt`, newest last). The page uses slot A: an empty slot loads the
  default on arrival (device auto, so CPU while a training run is live); a slot already holding a chat-capable
  checkpoint is used as is (the advanced mode's choice is respected); the gear's model picker loads what it names.
  The header pill shows the model and device, "loading …" while the worker loads, "training live" when a run is.
- **Sampling.** Defaults temperature 0.8, top-p 0.95, top-k off, 1536 new tokens (Peter, 2026-10-09; the advanced mode's
  default is 1536 too), Python tool on, measured on the model
  (`scripts/chat_sampling_defaults.py`, results §27); a fresh random seed per reply, shown under the reply (the
  advanced mode's fixed seed stays there). `think_required` follows slot A's stage as in the advanced mode; an
  external chat model takes neither the think span nor the tool. Settings persist in `localStorage`
  (`slm.chat.settings.v2`, only the values that differ from the defaults, so a changed default reaches every browser;
  v1 stored a full snapshot and is removed on load); "reset to defaults" restores the measured ones.
- **Conversations.** `localStorage` `slm.chat.v1`: id, title (the first user line unless renamed), timestamps, one
  Python `session` id each (variables persist across turns; deleting a conversation resets its session), and the
  messages. An assistant message keeps the generated turn's `ids` when it closed properly, so re-encoding never
  re-runs its tool calls; one that ran out of tokens is kept without ids, marked "stopped early", and still feeds the
  next turn. New / rename (✎ or a double-click: the title becomes an input; Enter or leaving it saves, Escape cancels,
  an emptied title returns to the automatic one) / delete (✕: an empty conversation goes at once, one with messages asks
  "Delete this chat?" inside its row) in the list -- never `window.prompt`/`confirm`, which embedded browsers such as
  the Claude desktop pane block; "regenerate" drops the last reply and re-rolls with a new seed; Enter
  sends, Shift+Enter breaks a line; the stop button cancels the stream (`POST /streams/{id}/cancel`) and keeps what
  arrived.
- **Streaming and the think span.** The SSE tokens are routed by the reserved tokens they carry: `<|think|>` /
  `<|/think|>` switch between the folded reasoning and the answer, `<|python_call|>` … diverts the code into a pending
  call, `tool` events supply the result, inserted (result) tokens are skipped. A finished turn's think text carries
  the calls as `<<code=result>>` markup (`slm/tools/protocol.py`), split for display into text and code/result
  blocks. The reasoning row is folded by default (word count, number of calls; "thinking…" while the model is still in
  the span) and expands on click. Answers go through a small markdown renderer (fenced code, headings, bullet and
  numbered lists, inline code / bold / italic) that builds nodes, never HTML strings.
- **Context window.** `components/context.js` `ContextMeter`: a slim bar and "42% of 4096" (green, amber past 75%, red
  past 90%), the exact count and what is left on hover. The worker reports the window with every loaded slot
  (`context`: our `max_seq_len`, an external model's max positions) and on each generation (`prompt.context`;
  `done.n_prompt`, `done.used` = prompt + everything generated incl. tool results = the tokens in the KV cache,
  `done.context`). Chat mode shows it above the composer: live during a reply (prompt + tokens so far), otherwise
  what the last reply left (stored on the message as `ctx`), with a note when the next reply would be cut short.
  The advanced mode shows one under each slot's output (prompt + generated), and the slot card names the context.
- **Layout.** A two-column panel the height of the viewport: the conversation list (collapses below 900 px) and the
  chat column (header with the model pill and gear, optional settings strip, the scrolling transcript with a welcome
  card and four suggestions when empty, the composer). Info cards `chat_mode`, `chat_model`, `chat_settings`,
  `chat_context`.

## Inference page: chat with the Python tool (2026-09-15)

This and the next two sections describe what is now the **advanced mode** (`pages/model.js`, `#/inference/advanced`).
Its chat mode is a multi-turn conversation. When a generated assistant turn is well-formed (ends with `<|end|>`,
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
  budget_tokens, max_groups, answer_suffix, mode, pair_budget_tokens, max_entrants}` (validated: k <= 64, max_calls <= 16,
  non-empty text, mode one of select / tournament / both, max_entrants 2-64; the tournament fields are described below). SSE events:
  `start` (stream id), `stage` (`sampling`; `collapsed` with the groups, majority and verified majority; `selecting`
  with the selector prompt's token count), `done` with the whole `SwarmResult.to_dict()` plus a per-candidate
  `verified` flag and `meta` (seed, settings, `cancelled`, `selector_parsed`, `groups_in_prompt`, `prompt_tokens`).
  The worker method is `Harness.swarm` (a streaming method, run under `sdpa_context("decode")` and `no_grad`, with
  `torch.cuda.empty_cache()` at the end when the slot is on the GPU).
- **Cost**: the KV cache grows with k × (prompt + max new tokens). On CPU a 149M model does k=8 × 160 tokens in
  seconds; on a GPU shared with a training run, keep k and max new modest.
- Info cards: `swarm`, `swarm_k`, `swarm_suffix`, `swarm_support`, `swarm_majority`, `swarm_selector`, `swarm_budget`,
  `swarm_oracle`, and for the bracket `swarm_tournament`, `swarm_seeding`, `swarm_swap`, `swarm_fallback`, `swarm_bracket`.

### Tournament selection (2026-09-27)

`slm.swarm.tournament` picks among the distinct answers with a **pairwise single-elimination bracket** instead of (or
next to) the one-prompt selector: each match is a two-answer prompt (`pair_messages`, answer A / answer B with support,
provenance and a short rationale, reply `#### A` or `#### B`), one greedy batch per round. A binary comparison is the
easiest decision a small model can be asked for; picking one of twelve is not.

- **Mode control** ("pick by"): *select* (the selector only), *tournament* (the bracket only), *both* (default: both run,
  the final answer follows the bracket as in `swarm_answer(mode="both")`, the selector's pick is shown next to it). The
  selector budget / max groups inputs show when the selector runs, pair budget (1200 tokens) and max entrants (16,
  2-64) when the bracket does. The stage strip becomes sample k → collapse → select → tournament r/n → done.
- **Bracket rules** (the library's, replicated round by round in `Harness._tournament_rounds` so each round streams):
  the entrants are the first `max_entrants` groups in evidence order (verified support, then support); each round pairs
  first against last (`seed_pairs`), the middle entrant of an odd count gets a bye; the 2nd, 4th, ... pair of every
  round is presented swapped (the better seed as B) so position bias cancels; a comparison with no parsable pick falls
  back to the evidence order (verified, then support, ties to A); the next round is the winners then the bye. A
  seeded run in the portal and `swarm_answer` produce the same bracket (a test compares the two with a scripted
  `compare`).
- **Bracket diagram** (`Bracket` in `components/swarm.js`): columns = rounds, then the champion; each match box shows
  its id (R1 M2), the two rows in the order the model saw them (A on top) with seed, answer, support (×n) and verified
  count (✓n), the ⇄ swapped marker, "picked A/B" or "no pick → evidence"; the winner row is green, the loser struck
  through; byes are dashed boxes; elbow lines connect each entrant to the box it came from, the champion's path in blue.
  The seeding is drawn as soon as the groups arrive (the client replays `seed_pairs` from the groups and max entrants
  and checks it against the server's round logs), undecided rounds show "winner of R1 M2" placeholders, and each round
  event fills its column in (a short slide-in animation; the column being decided pulses). Tags mark the majority,
  verified majority, the selector's pick (both mode) and the expected answer wherever they appear, so it is visible
  where each was eliminated. The box scrolls horizontally on narrow screens.
- **Comparison**: the final-answer panel lists tournament, selector (both mode), majority and verified majority (each
  with yes/no against the expected answer); in both mode a banner says whether the selector and the tournament agree,
  and if not, in which match the selector's pick lost and to what. The groups table gains a *bracket* column (champion,
  "lost R2 to X", "(evidence)" when the fallback decided it, "not entered" beyond max entrants); the ceilings table gains
  *in bracket* and *tournament* rows.
- **API**: `SwarmRequest` adds `mode` (`select` | `tournament` | `both`, default `both`), `pair_budget_tokens` (1200,
  200-8192) and `max_entrants` (16, 2-64). Extra SSE `stage` events with `stage: "tournament"`: `round: 0` once the
  bracket is seeded (`entrants` in seed order, `matches: []`, `n_rounds_expected`), then `round: 1..n` per decided round
  with `matches` (exactly the library's `{a, b, swapped, pick, winner}` log, `pick` 0 = A, 1 = B, null = fallback),
  `entrants` going in, `byes` and `seconds`. `done`'s result carries `tournament` (the champion or null), `rounds`, and
  `meta` gains `mode`, `pair_budget_tokens`, `max_entrants`, `n_entrants`, `n_rounds_expected`, `tournament_complete`
  and `selector_final` (the selector's parsed pick, null when it gave none or did not run). Cancel is honoured before
  the selector and at every round boundary; the decided rounds are kept and, with no champion, the final answer is the
  verified majority (the library's rule).

## Inference page: external models in a slot (2026-09-27)

Either slot can hold one of the local open-weight comparison models (`slm.eval.external`: SmolLM2-135M/360M base and
Instruct, Qwen2.5-0.5B base and Instruct, GPT-2 medium) instead of one of our checkpoints, so the same prompt can be run
through both side by side. Purely local: weights load from `<data root>/models/<name>/` with the offline switches on;
a model whose weights are missing is listed but cannot be loaded (`python -m slm.eval.external download <name>`).

- **Loading**: `GET /api/model/checkpoints` returns our checkpoints, then one entry per registered model with
  `external: true`, `group: "external models"`, `path: "external:<name>"`, `hf_id`, `params`, `license`, `is_chat`,
  `context` (position table size), `train_tokens`, `notes`, `available` (weights on disk). The dropdown shows them as
  a second `<optgroup>` (missing ones disabled). `POST /api/model/slots/{slot}/load` with `checkpoint: "external:<name>"`
  (404 for an unknown name or missing weights) goes through the same GPU guard as ours; `Harness.load` calls
  `load_external` on the chosen device, bf16 on cuda (or the requested dtype), fp32 on cpu. The slot info carries
  `external: true`, `stage: "external"` and the registry fields; the slot card shows the orange *external* badge, the
  hf id, params, license, chat/base, context and device/dtype, and one line on what the slot supports. Unload drops the
  model, runs `gc.collect()` and `torch.cuda.empty_cache()`.
- **Generation** (`Harness._generate_external`, the same `prompt` / `token` / `done` events): completion mode for every
  model (the text encoded with the model's own tokenizer, no BOS); chat mode through the model's own chat template for
  chat models, refused for a base model with an error (a template is never invented). Tokens stream through
  `HfChatModel.stream_ids` (our sampler and seeding, identical ids to `generate_ids`), pieces are decoded incrementally
  (a character split across byte-level BPE ids shows up once complete, the ids before it as empty chips), special
  tokens such as `<|im_end|>` stay visible, and log-prob / rank / entropy / top-k are the model's own over its own
  vocabulary. A chat reply that ends on a stop token is appended to the conversation as text (no think span, no ids).
  Our protocol does not apply: `tools`, `functions` and `think_required` are refused with an error naming the option;
  in the UI the python tool and force-think checkboxes are greyed out and the declared-functions box disabled while an
  external slot is part of the run (A, or A and B side by side), and the three are sent off. `session_id` is ignored.
  Teacher-forced scoring and diagnostics are n/a (an error; the score button is disabled for an external slot A).
- **Swarm** (`Harness.swarm`, same stages and events): chat models only. The k replies are one `batch_generate_chat`
  batch through the model's own template (task + answer instruction, the same seed), `#### <answer>` is parsed with
  `parse_final_span` and grouped with `slm.swarm.collapse`; a reply has no think span, so each group's rationale is its
  shortest member's reply up to the `####` line. The selector (`selector_messages`) and each tournament round
  (`pair_messages`, one batch per round) go through the same template as the user message, greedy; the prompt budgets
  are measured with the model's tokenizer. There is no sandbox, so **verification is n/a**: events carry
  `verification: "n/a"` and `external: true`; `n_verified`, `verified_majority` and each candidate's `verified` are
  null; `meta.verification` is `"n/a"` (`"sandbox"` for ours) and `meta.max_calls` 0; every fallback that uses the
  verified majority for ours uses the plain majority. The panel shows a notice with the badge, greys out max tool calls,
  and shows n/a for verified counts, the verified majority, the verified oracle and the candidates' badge. A base model
  is refused.
- Info card `external_slot`: what an external slot is and where it is not apples-to-apples (tokenizer and token-level
  numbers, chat template and its default system prompt, no tool protocol, no verification, scoring n/a).

## Arena tab (2026-10-02)

`#/arena` (`pages/arena.js`, the episode model and the at-a-glance views in `components/arena_views.js`) runs the
arena of `slm/arena/world.py` on one slot: an N×N grid of robots, each one a conversation with the model, one batched
generation per turn (`Runner.step`), streamed turn by turn. Viewer mode drives the same views from an episode file.

- **Controls**: the two slot cards (the Inference page's `SlotCard`, so a checkpoint can be loaded here, CPU or cuda),
  slot, task (`key_door`, `relay`, `triangulate`, and the competitive `team_relay`, `tag`, `gold_rush`, each with its
  description from `GET /api/model/arena/tasks`), grid 4-16, robots 1-32 (each task's own rule, `Task.check`: key_door an
  even count, team_relay an even count of at least 4 on a grid at least robots - 1 cells across), seed, max turns (empty = no limit, the default since 2026-10-11: the episode runs until the game ends or you press stop;
  else 1-1000), max new tokens (512), temperature (0.3, top-p 0.95; 0 = greedy as the evals run; higher costs calls, results §28c; every run
  sends a random `sample_seed`, recorded in `meta` with the temperature, so the same world seed plays a new game), max tool calls (4),
  history (1, the World default: earlier exchanges kept per robot), stop when done; run / stop; download episode log.
- **Grid** (SVG): cells with coordinates, doors (amber, green ✓ when opened), the triangulate target hidden until dug (a
  checkbox reveals it for analysis; a failed dig leaves a grey ×), robots as circles coloured by their unique tool and
  labelled R1..Rn (several on a cell share it on a small ring). Hover or select a robot: its comm range and sight are shaded;
  the robot's tooltip lists its tools, cells moved, its action this turn (calls and results) and what it heard. Every robot
  also has an identity colour (`robotColor`, golden-angle hues, distinct up to 32) used for its trail, its messages, its
  timeline arrows and its name in the tables. Between turns the robots glide (CSS transform transition). **Trails**: each
  robot's full path from the start to the viewed turn, fading from faint (early) to solid, a small square at the start,
  slightly offset per robot so shared cells stay readable; toggles all / selected robot / none. Each message of the viewed
  turn draws as an arc from the speaker to every robot that hears it, in the speaker's colour, with the text (22 chars) at
  the arc's midpoint (every label when the turn has at most 6 messages, else only the focused robot's); a dashed ring when
  nobody is in range. Arcs touching the focused robot are bright, the others pale; a message picked in the log is the only
  bright one ("clear message highlight" undoes it). **Messages** (radio): arrows + text (default), arrows only, or hidden on
  the grid (the message log and the per-turn list below the grid are unaffected). **Comm range of every robot** (checkbox) outlines each robot's
  Manhattan diamond, clipped to the grid. Opened doors turn green with ✓, the dig target appears when dug up. The legend
  covers roles, shading, arcs, trails, ranges and doors. Below the grid the messages of the turn as text.
- **Scrubber**: start / previous / play / next / latest and a slider over 0..turns (0 = the start state). While a run
  streams, the view follows the newest turn unless you step back.
- **Task panel** (above the arena since 2026-10-11): the task's description and roles, grid / robots / comm range / sight / seed / checkpoint, the score at
  the viewed turn (the hidden target is masked), the events, and at the end the result (solved in k turns, or not;
  turns played, seconds, robot turns, tool calls, messages; "stopped by you" after a cancel).
- **Episode summary** (`SummaryStrip`): four small SVG charts over turns, click a turn to jump there: task progress
  (`state.score.progress`; for files without it, opened/of or hops_reached/hops), with a dashed mark at every turn after
  which an event shows up in the state; messages per turn; tool calls per turn; failed calls per turn (`error:`, `Traceback`,
  "Wrong code", "is not the code"). Then the verdict (solved in k turns / not solved / stopped), totals (messages, calls,
  calls with errors, calls that did nothing) and the events (click = jump to the turn the event appeared).
- **Interaction timeline** (`Timeline`): swimlanes, one row per robot (sticky name column with role dot and identity
  colour), one column per turn (horizontal scroll; row height 24 / 18 / 14 px for up to 8 / 16 / 32 robots, so 32 rows fit
  in ~470 px). A cell is the robot's action that turn (`classify`): the first tool its calls invoked as a glyph and tint
  (move = arrow of the direction it actually went, else the literal argument; » say; ◉ look; R read_key / read_code; ★
  open_door / submit / dig, green border when it worked; ◎ distance_to_target; {} code that called no tool; · no call; "+"
  when it called several tools), a red corner for a failed call, an amber dot for a call that ran but did nothing
  ("Blocked:", "Unknown direction", "Move there first", "Nothing here"). Messages are arrows (cubic, speaker's colour) from
  the speaker's cell to each hearer's cell in the next column, since delivery is next turn; a message of the last turn
  points into an extra "next" column; a dashed circle = said to nobody. Arrows touching the selected or hovered robot are
  bright. Hover a cell: tooltip with what it heard, the question (the user turn minus the answer instruction), every call
  with its result, what it said and to whom, and the answer. Click a cell: selects the robot and scrubs to that turn (grid,
  table and transcript follow); click a header number: scrub; ★ in the header: a turn after which an event appeared. The
  scrubber's turn is a highlighted column, the selected robot a highlighted row. A glyph legend sits below it.
- **Message log** (`MessageLog`): every `say` of the episode in order (turn, speaker → hearers or "nobody within range",
  the text, "✓ heard t+1" or "not delivered" when the episode ended first), the viewed turn's rows tinted; filter by robot
  (speaker or hearer) and by free text (substring of the text, or an exact robot name); click = scrub to the turn, select
  the speaker and light that message's arc alone on the grid. Renders the first 400 matches.
- **Robots**: one compact table for all robots at the viewed turn (position, cells moved so far, unique tool, the tool its
  prompt declared this turn, the calls (red when one failed), its inbox this turn, what it said and who heard it, its
  answer). The transcript of ONE selected robot (click a row, a robot or a timeline cell) sits to the right of the arena
  (sticky, scrolls on its own; below it on narrow screens): its declared
  functions, the briefing (`world.system_prompt`, collapsible), and per turn (newest first, the viewed turn open) its
  inbox ("heard: R1 said: '...'" lines), the situation line (the world's record, not shown to the model), the user turn
  exactly as the model saw it, the declared tools, the generated turn with its reserved tokens as chips
  (`TextWithSpecials`: think span, `<|python_call|>` / `<|python_result|>` spans, `<|end|>`), the tool calls with results,
  and the answer. With 32 robots nothing renders 32 transcripts.
- **API**: `POST /api/model/arena` with `{slot, task, n, n_agents, seed, turns, max_new_tokens, max_calls, max_history,
  stop_when_done}` (`ArenaRequest`, validated as above). SSE: `start` (stream id), then `start` from the worker with
  `state` (`world.state()`), `robots` (id, name, system prompt, tools as `{name, signature, comment}`) and `meta`
  (settings, checkpoint, device, comm range, sight); one `turn` per turn with `turn`, `seconds`, `records` (the Runner's,
  plus `agent_id`, `raw` = the generated turn decoded with its specials, `declared` = the tools that robot's prompt
  declared, parsed back from its `<|python_def|>` blocks, and `heard` = its inbox, the messages delivered at the start of
  that turn as `"R1 said: '...'"` strings, the same list the status line's "Heard:" part joins), `state`, and `messages`
  (`{speaker, speaker_id, pos, text,
  heard_by, heard_by_ids}`, computed from `world.pending` before the next `deliver()` with the same range rule); `done` with
  `result` (score, events, turns, seconds, cancelled, meta, messages per turn, the full transcript). `GET
  /api/model/arena/tasks`: key, title, description, roles, min agents, comm range, sight. Viewer endpoints (read-only,
  `services/arena_files.py`, torch-free): `GET /api/model/arena/episodes` lists `<runs_root>/arena/*.json` (top level only)
  newest first with name, size, mtime and a peek (`kind`: `cli` | `portal` | `summary` | `unreadable` | `too_large`, task,
  grid, robots, seed, turns, success; cached per mtime and size); `GET /api/model/arena/episodes/{name}` returns one file
  as written (a bare `[A-Za-z0-9][A-Za-z0-9._-]*.json` name without `..` that resolves to a direct child of that directory,
  else 400; 404 missing; 413 above 64 MiB); `GET /api/model/arena/world?task&n&n_agents&seed[&max_history]` rebuilds the
  start of an episode without a model (World is deterministic): `state`, `robots` (as in `start`), `meta` (comm range,
  sight); 422 for counts the task cannot be built for.
- **Viewer mode**: a row under the controls: a dropdown of the episode files (sweep summaries listed but disabled: no
  transcript), "open", and a file picker for any local file. Both shapes load: the tab's download (`{request, meta,
  robots, start, turns, result}`, exactly the streamed events) as is, and the CLI's `python -m slm.arena --out`
  (`{task, n, agents, seed, turns, score, events, seconds, transcript}`) through `normalizeEpisode`: the start state,
  robots and comm range from `/arena/world`; per-turn positions from the records (`pos`, end of turn); events filtered by
  turn (an event's "turn k" label is `World.turn` while step k+1 runs, so it belongs to the state after record turn k+1);
  scores derived per task as `Task.score` computes them (doors opened from the events; hops reached = highest robot whose
  inbox held a 4-digit message; digger distance and progress from the target in the file's final score), the final turn
  taking the file's own score; messages from the say calls (text from the result's "You said: '...'" or the literal
  argument, hearers by the delivery rule from end-of-turn positions, the same as `arena_messages`), plus any line of a
  robot's next inbox (parsed from its status line) that no parsed call explains; inboxes from the status lines. A note
  says the file was rebuilt, and flags robots whose turn-1 position contradicts the rebuilt start. A viewed episode can be
  re-downloaded in the tab's shape. Every view (grid, summary, timeline, log, table, transcript) runs from the file; no
  slot is needed.
- **Worker**: `Harness.arena` (a streaming method). The turns are generated by `harness.arena_generate`, the library's
  sampler and per-turn seed on the slot's own device (`Runner._generate` builds its generator on cuda, so a CPU slot needs
  this). Cancel is honoured between turns (a turn is one batch); `torch.cuda.empty_cache()` at the end on cuda.
- **Cost**: one batch of n_agents rows per turn; on the GPU decode is launch-bound, so up to ~32 rows cost about one row
  per step (docs/results.md §24). On CPU, keep to a few robots: the 336M model does 4 robots × 64 tokens in ~3-4 s a turn.
- **The 3D arena** (2026-10-10, default; the 2D map is a switch away, remembered per browser): `components/arena3d.js`, a
  hand-written WebGL renderer (no library: the portal vendors its dependencies and has no build step) -- toon-shaded,
  ink-outlined cartoon robots (side or role colour, the robot's own colour as a band, googly eyes, a wobbling antenna whose
  bulb glows when it speaks, red after an error, green on a goal) on a candy checkerboard that floats on a plinth; coins
  spin, doors are arches whose panel swings open, the triangulate target is an X. Robots hop between cells and face where
  they go; a speech bubble shows what each said and envelopes fly to every hearer; emotes from the turn's calls (💥 failed
  call, ❓ no call, 👀 look, 🧱 blocked / refused, 🤫 secret read, 🪙 coin, 🫵 tag, 🎉 goal); a tagged runner keels over
  (x_x); confetti on goals and a falling shower plus a banner on a verdict; smoke on errors; footprints for trails; the
  selected robot's comm-range diamond. Camera: drag orbit (drag right turns the board right, 2026-10-11), right-drag / Shift pan, wheel zoom, W A S D move the centre, Q/E
  turn, R reset, T top-down; keys act only while the pointer is over the view (never while typing). Click a robot to
  select and follow it (Esc, F, a pan or the button toggle following); its panel opens with **its context window**: the
  declared tool block, the earlier exchange the arena kept, the question, and what the model wrote, with a
  `ContextMeter` of prompt + reply tokens against the model's window (records carry `history`, `prompt_tokens`,
  `gen_tokens`, `context` since 2026-10-10). Without WebGL the view says so and the map remains. Info cards `arena_3d`,
  `arena_context`. One renderer per canvas is reused across episodes (losing the context on cleanup broke the next one).
- **Collisions and fair prompts** (2026-10-11): robots scoot between cells (a wind-up, a dash that overshoots with dust
  behind, a rocking stop) instead of hopping. A failed move (`record.bumped`, or "Bump!" in older files) makes the robot
  lunge towards the cell, bounce back with a squash and a spark, sway dizzily under four circling stars; a robot bumped
  into while standing still rocks. The comm range is drawn as the square of cells that hear the robot (clipped to the
  board; the map view too), and the board's plinth has no ink outline (the inverted hull is inflated before scaling, so on
  a board-sized box it showed as black slabs along the edge). The form has a *prompt* select (`GET /api/model/arena/prompts`,
  request field `prompt`; info card `arena_prompt`), and the robot panel shows the whole context window in the order the
  model read it: declared tools, the system prompt (or that there is none), worked examples, the earlier exchange, the
  question, the reply; "the exact text" folds open `record.prompt_text` with reserved tokens as chips; a ✅ / ❌ line
  compares the robot's call with the one the scaffold would have spelled out (`tool_ok`, `call_ok`, `expected`).
- **Model against model** (2026-10-10): for a competitive task the controls offer an *opponent (side B)*: the other
  slot's model, ours or an external one (any slot may now hold an external model in the arena: it answers in plain text
  and the call it writes is run, `slm.arena.world.text_action`). The form names what each side plays (from the tasks
  endpoint's `sides`). *Batch size* (1-32, empty = all) caps the robots per generation call; the line under the run
  button shows the last turn's peak / reserved VRAM and turns red above 14.5 GiB reserved. The task panel names each
  side's model, what it plays and the winner side. Request fields `opponent` (the other slot) and `batch_size`; the turn
  events carry `vram`, the start event `meta.players`, the result `vram_peak_gib`. Info card `arena_versus`.
- **Competitive tasks** (2026-10-10): robots are coloured by side when the state carries a `team` (team red / team blue,
  chaser red / runner green), else by their unique tool (gold rush: all prospectors). Coins are gold circles and vanish
  when grabbed; a robot's coin count sits beside it; a tagged runner fades with a dashed outline. The task panel's badge and
  the result line show the score's `verdict` ("team red wins", "the chasers win: every runner tagged", "R3 wins with 4
  coins"), or the game's state when there is none ("runners survive: 1 of 4 still free", "R2 leads with 1 of 12 coins")
  instead of "not solved". The viewer rebuilds these from the events for CLI files (tagged runners, coins taken, coin
  counts), and its event filter uses the events' 1-based turns (it read them as 0-based until 2026-10-10, so every event
  showed a turn late).
- Info cards: `arena`, `arena_key_door`, `arena_relay`, `arena_triangulate`, `arena_team_relay`, `arena_tag`, `arena_gold_rush`, `arena_range`, `arena_cost`, `arena_scaffold`,
  `arena_turn`, `arena_grid`, `arena_timeline`, `arena_messages`, `arena_summary`, `arena_viewer`.

## Eval tab (2026-09-26)

`#/evals` (`pages/evals.js`), `GET /api/evals` (`api/evals.py`), `services/evals.py` (`EvalIndex`). One table of every
evaluated checkpoint against every eval the project runs, so a stage's gains and costs read across one row and a
capability's history down one column.

- **Sources** (all under `runs/<run>/`): `lm_eval*.json` (HellaSwag and OpenBookQA as acc_norm, ARC-Easy, PIQA, LAMBADA
  and SciQ as acc, following `docs/results.md`; both metrics are in the hover), `bench_ll.json` (LAMBADA / OBQA / SciQ at
  limit 1000, used only where no lm-eval file has them), `facts*.json`, `reasoning_eval*.json` / `reasoning_s*.json` /
  `reasoning_svamp.json` (per-family accuracy, the file's mean, mean tool-use rate), `multiturn*.json` (recall, not stated
  = `recall_absent`, revise, sysrule, and format / misfire over every kind; a file written before the kinds holds
  recall only and leaves the three new columns n/a), `needle_v2.json` / `needle_s*.json` / `needle_step*.json` / `needle_snap*.json` (effective context = longest length whose every shorter length also has
  min-over-depths >= the threshold; real-text haystack only; v1 `needle.json`, the filler control and the depth probes
  are ignored), `quality/summary.json` (judged overall and tool misfire), `pass_at_k.json`, `swarm_eval*.json`.
  MMLU and the dropped benchmark-revision tasks (§13 of results.md) are not shown.
- **Attribution**: a file belongs to the run whose directory holds it (some `checkpoint` fields name a run's pre-rename
  directory; the hover says so) and to the checkpoint named by the file name in its `checkpoint` field (`_s200` style
  suffixes are the fallback). Checkpoints of one run with the same token count in `checkpoints/index.json` are the same
  weights and share a row (`best.pt = step_00150.pt`). Judged-quality points attach to those rows; a run with no other
  result file gets one row for its last judged checkpoint. When two files fill one cell, lm-eval prefers limit 2000 over
  the full set over limit 1000 and reasoning prefers the tool-enabled main file over the no-tool one over
  `reasoning_svamp.json`; the loser is listed in the hover ("also measured").
- **Rows** are ordered by model size (largest first), then stage (base, sft, reasoning / tool, rl), run name and tokens.
  Each shows the stage, the checkpoint and its aliases, and cumulative tokens along the `init_from` chain (this run's
  share in parentheses). The local open-weight comparison models (`runs/ext_<name>/` with a `model.json` registry entry,
  result files naming `"checkpoint": "external:<name>"`) follow ours as their own **external models** group (stage
  `external`, the registry's params and published training tokens, hf id / license / chat or base in the hover; no link to
  a run page); cells that need our tool protocol are n/a, and the colour scale spans every row.
- **Colour**: per column, t = (value - min) / (max - min), flipped for `higher_is_better: false` columns (misfire), mapped
  to a hue from red (0) to green (120); a column with one distinct value is amber, n/a is grey. The server computes `t`,
  the page only paints it.
- **Page**: sticky header block and first column inside one scroll box, group toggles, compact cells (the score
  only, 2-5 px padding) and a hover card per score (file, n / k / limit, the eval's details, "also measured"), a "?"
  card per column (`ev_*` in `cards.js`) plus `ev_table`, `ev_colour`, `ev_sort` and `ev_detail`.
- **Sorting** (client-side, `sortRows` in `pages/evals.js`): a column header cycles best first → worst first → default
  order, respecting `higher_is_better`; n/a rows go below their own separator; ties keep the chain order. Sorted, the
  size separators are dropped and each row shows its size instead; clicking the checkpoint header restores the default.
  Sort and hidden groups survive a visit to a detail page (module state).
- **Detail pages** (`#/evals/<run>/<checkpoint>/<column>`, `EvalDetailPage`; any alias of the row's checkpoint works):
  every score links to one. `GET /api/evals/detail` (`services/eval_detail.py`, `EvalDetail`) opens a file that
  measured the cell (the table records all of them per cell, winner first; `file=` picks another) and returns summary
  numbers, aggregate tables with the cell's row highlighted, the file's scalar fields, the run's `run.json` config, the
  last 40 lines of the eval's `.log` (progress bars removed), and per-item rows (`{ok, c, x}`: verdict for this column,
  short cells, expansion blocks) paged 100 at a time (max 500), filterable by verdict and text. Where the rows come from:
  judged = `quality/outputs/<tokens>.jsonl` joined with every `quality/scores.jsonl` record by item id (verdict:
  correctness 4-5 / 1-2; misfire column: ran code on an eligible prompt); reasoning = the `reasoning_dump_<tag>.jsonl`
  beside `reasoning_eval_<tag>.json`, split into tasks by the file's per-task n (arith2mul rows say "arith2"), which only
  the M5/M6 runs have (the others: aggregates only, and the page says `--dump` was not used); facts, multi-turn (every
  turn with its think span, given turns and the system prompt marked, the verifier's verdict and failed constraints;
  each kind's column lists only that kind's conversations, format and misfire list all, plus a per-kind table), needle (per cell, failures only), pass@k and swarm (flags and answers per problem, no
  sample text; GSM8K / SVAMP problem text looked up by id in the local test parquet) inside the result file; lm-eval
  and `bench_ll.json` never (run without `--log_samples`: per-task acc / acc_norm / stderr / n and the limit only).
  Parsed files are cached by mtime and size (32 entries); a file above 64 MiB is refused with a note.
- **Cache**: rebuilt only when a matching file's mtime or size changes (also `run.json` and `checkpoints/index.json`);
  a rebuild reads ~100 small JSON files in well under a second.

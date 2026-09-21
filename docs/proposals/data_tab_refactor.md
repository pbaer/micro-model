# Data page refactor: the mixture as the spine

Status: proposal, 2026-09-19. No code changed. Line numbers refer to the tree at commit `d268d5d`.

## (a) Problem

The Data page (`slm/portal/static/pages/data.js`) is three tabs that grew independently: *sources* is a flat table of
the registry plus a second table of SFT manifests, *mixture* is a per-config weight table, *documents* is a browser
over tokenized shards or raw parquet. The same numbers appear in two tabs (train/val tokens are in sources *and* in
mixture's "available" column; `/api/data/sources` is fetched by sources, documents and the Overview's "Data readiness"
panel), while other facts live in exactly one place with no path between them: epochs only in mixture, length
percentiles only in documents/stats, SFT manifests only in sources, shard lists only in documents. The owner's actual
question ("for this training config or run, what does the mixture look like, what raw data feeds it, and what does the
model literally see per row?") cannot be answered without re-selecting the same source in three tabs, and for two of
the three stages it cannot be answered at all: the mixture tab reports `available 0 / epochs null` for every SFT config
and every source converted with `scripts/sft_to_pretrain.py`, returns HTTP 500 for RL configs, and the SFT *shards*
(the packed `tokens_*.bin` + `mask_*.bin` the trainer reads) are not browsable anywhere; the "exactly what SFT trains
on" view (`data.js:153`) is a live re-encoding of the raw parquet row, not the prepared example.

## (b) Inventory of the current tabs, with redundancies and silos

**Sources tab** (`data.js:20-35`; payload from `DataCatalog.overview`, `services/datasets.py:280-305`)

- One row per registry source (`slm/data/sources.py:40-83`) with kind, license, raw files/bytes/rows, and per tokenizer
  tag: train tokens, val tokens, docs, dropped %. Rows for derived tokenized sets (`fineweb-edu-10bt`, `fineweb-edu-b`,
  `fineweb-edu-long`, `smoltalk-chat`, `tool-chat`, `synth-retrieval*`, the `*-v1` leftovers) are appended with
  `raw_files 0` (`datasets.py:292-304`). Provenance is derived from `manifest.source`, which is wrong for two writers:
  `scripts/sft_to_pretrain.py:42` writes `source` = the output name (so the live page shows "derived from smoltalk-chat"
  for smoltalk-chat) and `slm/data/synth_retrieval.py:132` writes `name` only (live: "derived from ?").
- A second table of SFT manifests from `C:\slm-data\sft\<tag>\*\manifest.json` (`data.js:30-33`,
  `datasets.py:268-278`): examples, tokens, target share, max_len, think, dropped/too long.
- The link between the two tables is a string heuristic (`data.js:27`: `n.startsWith(s.name)` with a `gsm8k` special
  case); `metamathqa-reasoning` (source `smoltalk-metamathqa-50k`) and `synthetic-*` (no raw source) do not match it.

**Mixture tab** (`data.js:37-51`; `api/data.py:18-30`; `DataCatalog.mixture`, `datasets.py:323-334`)

- Select one of `configs/train/*.yaml`; table of weight, planned tokens (`schedule.total_tokens x weight`), available
  train tokens, epochs, val tokens; red if epochs > 1.5.
- Bugs that make it wrong for two of three stages and for the current 336M phase 2:
  - `mixture()` reads only tokenized manifests (`datasets.py:324-325`); `data.kind == "sft"` and `sft_root` are ignored,
    so `m4_sft_149m`, `m5_*` show `available 0 / epochs "no data"` (live payload checked).
  - It reads `train_tokens`, but `sft_to_pretrain` manifests use `train: {tokens, docs, shards}`; live
    `m8_base_4k_336m` shows `smoltalk-chat` and `tool-chat` as `available 0`, though they are 225M and 7.6M tokens.
  - `schedule.epochs > 0` configs (m4, m5) show the dataclass default `total_tokens = 1B`; the trainer computes the real
    total at start (`slm/train/pretrain.py:73-75`).
  - RL yaml goes through `load_train_config` and `from_dict` raises on unknown keys (`slm/config.py:93-95`): HTTP 500.
  - `extra_val_mixture` (the drift set used by every SFT/context run) is not shown.
  - Only yaml files are selectable; the mixture a run *actually* used is in `runs/<run>/run.json` (`config.data.mixture`,
    written by `pretrain.py:114-125`) and can differ from the yaml (configs are edited between phases; m3a's comment
    documents a data swap mid-run).

**Documents tab** (`data.js:58-179`; `api/data.py:33-99`; `RawSource` and `TokenizedSplit`, `datasets.py:41-239`)

- Tokenized sources: tag/source/split/shard selects, document list (start, length), `doc` view (text/tokens/ids),
  `window` view (an exact contiguous slice with `<|bos|>`/`<|eos|>` boundaries in red, `datasets.py:205-214`), `stats`
  (percentiles, log-bin histograms, docs over 4K/8K, disk-cached, `datasets.py:216-239`).
- Raw-only sources (chat sets, gsm8k): parquet file/row-group browser; a row is rendered through `rows_to_messages` and,
  in tokens mode, re-encoded on demand via `POST /api/tokenizers/{tag}/encode` in chat mode with the loss mask in green
  (`data.js:97-101, 152-153`).
- Silos and gaps: length stats exist only here and only for tokenized sources; SFT shards (`sft.py:3-9`) are not
  browsable, so the *prepared* SFT example (after `apply_style` marker/natural coin flip `sft.py:97-98`, `--tools`
  conversion `sft.py:99-102`, `max_len` drop `sft.py:103-105`, test-file-to-val routing `sft.py:109-110`) is never
  shown; the legend "exactly what SFT trains on" is true only for plain chat sets at default flags. Nothing shows a
  *packed* SFT training row (several examples back to back with the mask, `SftStream.next_window_masked`,
  `sft.py:144-154`). Sources without raw parquet (`synth-retrieval*`, `smoltalk-chat`, `tool-chat`) have no "what did this
  come from" link even though the manifests know (`from_sft`, `backdrop`). Nothing at all for RL.

**Cross-tab redundancy, concretely**

| Fact | Sources | Mixture | Documents | Overview |
|---|---|---|---|---|
| train/val tokens per source | yes (`data.js:25-27`) | yes ("available", "val tokens", `data.js:46-48`) | per shard (`data.js:129`) | yes (readiness, `home.js:62-68`) |
| `/api/data/sources` fetched | yes | no | yes (`data.js:81`, only to list names) | yes (`home.js:35`) |
| tokenizer tag select | columns per tag | implicit (from config) | select (`data.js:123`) | per tag panels |
| SFT set list | manifests table | names only, with wrong numbers | as raw parquet, not as sets | count only |

## (c) Candidate information architectures

**A. Keep three tabs, fix the plumbing.** Fix `mixture()` for SFT/derived/RL, add run selection, link mixture rows to
the documents tab with the selection pre-filled, merge the SFT table into the sources table. Cost: ~1 day. It removes
the wrong numbers but keeps the user hopping between tabs and still has no home for "the training row of *this* run",
packed SFT rows, or RL prompts. Not recommended on its own, but everything in it is the first phase of B.

**B. Recipe spine (recommended).** The primary object is a *recipe*: the data section of one training config or one
run (pretrain, SFT, or RL). The page opens on a recipe list grouped by stage; a recipe view is the mixture table, and
every row expands into the same three-column inspector: raw record -> prepared record -> training row as the model
sees it. Sources become a secondary *catalog* (one page per source with provenance, prepared artifacts, stats, and
"used by" recipes), and the existing browser becomes the inspector's body rather than a tab. Compare and chain views
are recipe-level operations. This matches how the project thinks (configs and runs are the unit of work everywhere else
in the portal; the Overview's pipeline table already chains runs by `init_from`).

**C. Source spine.** The catalog is primary; each source page carries a "used by" matrix of configs and weights, and
the browser. Good for data curation (which sources are stale, which are unused, `numina-cot-100k`), poor for the
stated question: a mixture is a property of a config, and comparing two phases or summing exposure over a chain does
not fit a per-source page. B contains C's catalog as its second level, so C's value is not lost.

**Why B.** Every question in the brief is phrased from the recipe side ("for a chosen config or run", "compare phase 1
vs phase 2", "cumulative exposure over the init_from chain", "what the model sees during pretraining / SFT / RL"). The
one source-centric question ("the raw data that contributes") is one click down from a mixture row. B also gives the
missing stages a natural home: an SFT recipe's rows point at SFT shards, an RL recipe has no rows and shows its prompt
generator and reward scheme instead, in the same slot.

## (d) Recommended design

### Navigation model

Hash routes (the router in `app.js:58-70` already splits the hash into parts; `DataPage` needs to receive them the way
`RunDetail` does):

- `#/data` - recipe list (default) with the catalog as a second top-level tab.
- `#/data/recipe/run:<run_name>` and `#/data/recipe/config:<path>` - one recipe. Query-ish suffixes for the inspector
  state: `/<source>/<split>/<shard>/<doc>` so a view is linkable from a run page and from `docs/log.md`.
- `#/data/compare/<a>/<b>` - two recipes side by side.
- `#/data/chain/run:<run_name>` - exposure along the init_from chain ending at that run.
- `#/data/source/<name>` - catalog entry (provenance, prepared artifacts, stats, browser, used-by).

Config vs run: a recipe id is either `run:` (mixture from `runs/<run>/run.json`, the truth for what happened) or
`config:` (the yaml, the plan). The recipe list shows configs and runs in one table, one row per run with its config
path next to it, and a "plan differs from run" marker when the yaml's data block no longer equals `run.json`'s
(`json` compare of `data`, `schedule.total_tokens/epochs`, `init_from`). Configs without a run are listed as "not
started" so a mixture can be inspected before launch (the pre-launch checklist in `docs/runbook.md` section 3). Run
pages get a "data" link into `#/data/recipe/run:<name>`.

### Recipe list (`#/data`)

```
Data   [recipes] [catalog]                                  tokenizer: v1
+---------------------------------------------------------------------------------------------+
| stage     | recipe (run / config)            | status   | rows | seq | tokens        | init_from   |
| pretrain  | m8_base_stable_336m  (m8_..yaml) | running  | 6    | 2K  | 7.32B / 7.5B  | random      |
| pretrain  | m8_base_4k_336m      (m8_..yaml) | not run  | 8    | 4K  | 2.5B planned  | m8_stable   |
| sft       | m4_sft_149m                      | finished | 5+5v | 2K  | 450M (2 ep)   | m3_base_8k  |
| rl        | m6_rl_arith_149m                 | finished | tasks arith1,arith2 | 200 steps  | m5_reason   |
+---------------------------------------------------------------------------------------------+
   [compare selected]  [chain]
```

### Recipe view (`#/data/recipe/...`)

```
m8_base_4k_336m   plan (configs/train/m8_base_4k_336m.yaml)  ·  no run yet        [compare with…] [chain]
tokenizer v1 · seq_len 4096 · 2.50B planned tokens · wsd decay 80% · init_from m8_base_stable_336m/final.pt
+------------------------------------------------------------------------------------------------+
| source            | weight | planned | available | epochs | val   | prepared            | raw          |
| fineweb-edu-10bt  | 66.0%  | 1.65B   | 10.07B    | 0.16   | 51.5M | 101 shards, prose   | fineweb-edu 14 files |
| ...                                                                                            |
| smoltalk-chat     |  4.5%  | 112M    | 225M      | 0.50   | 31.3M | 5 shards, from 5 SFT sets | (SFT sets -> raw parquet) |
| tool-chat         |  1.5%  | 37.5M   | 7.6M      | 4.90 ! | 0.5M  | 1 shard, from 4 SFT sets  | ...          |
+------------------------------------------------------------------------------------------------+
extra_val_mixture (drift set): fineweb-edu-10bt 72 / cosmopedia 11 / ...  (val split only)

+--------------------- inspector for the selected row: tool-chat ------------------------------+
| RAW record            | PREPARED record                    | TRAINING ROW (what the model sees) |
| gsm8k train row 812   | tool-chat shard 0 doc 4413         | seq_len 4096 window at 1,234,567    |
| question / answer     | <|bos|><|user|>...<|end|><|assistant|> | crosses 3 docs (red bounds);       |
| meta cols             | <|think|>...<|python_call|>..      | every token is a target (pretrain)  |
|                       | tokens / ids / text toggle         | [prev] [next] [random]              |
+----------------------------------------------------------------------------------------------+
```

The list+preview pair keeps the `.two.fill` viewport sizing (`data.js:103-111`, `style.css:49-55`); the inspector is
that same pair with the list on the left (documents of the selected shard, or raw rows) and the three-stage preview on
the right as sub-tabs *raw / prepared / row*, so the existing window and loss-mask components are reused unchanged.

**What each column shows per stage**

*Pretraining recipe* (rows from `data.mixture`, root `data.tokenized_root`):
- Raw: the parquet row (`RawSource.doc`, `datasets.py:108-114`) when the source has raw files; for derived sets the
  raw column shows the provenance chain instead (`fineweb-edu-long` <- `fineweb-edu` files 0-1 with
  `min_doc_tokens 4096`; `smoltalk-chat` <- five SFT sets <- five SmolTalk parquet subsets; `synth-retrieval` <-
  templated over `fineweb-edu-b/train` backdrop) with a link to the parent's catalog page. A new *trace* action runs the
  raw row through `keep_doc` + `is_val` + `encode_doc` (`prepare.py:54-69`, `tokenizer.py:116`) and reports "kept ->
  train, 812 tokens" or "dropped: language != en" so the raw->prepared edge is explained even though there is no stored
  row->doc mapping (prepare.py records none; see risks).
- Prepared: the tokenized document exactly as today's `doc` view (bos/eos included).
- Training row: today's `window` view with `length` fixed to the recipe's `seq_len` (the loader cuts `seq_len + 1`
  consecutive tokens, `loader.py:38-48`; windows never straddle shards). Legend states the mixture probability of this
  source per row and that every token is a target.

*SFT recipe* (`data.kind == "sft"`, root `data.sft_root`, sets are directories with `tokens_*/mask_*/idx_*`):
- Raw: the parquet row rendered as messages (existing chat rendering), with the `rows_to_messages` drop reason when
  applicable (`datasets.py:75-90`).
- Prepared: the *stored* example from the SFT shard (new `SftSplit` reader): chips with the stored mask (green targets,
  grey masked, specials dark), and a derived segment strip (bos / role / think / python_call / python_result / end /
  eos) computed client-side from the special-token ids so the chat template is visible. Header shows the set's
  `max_len`, `think_required`, `tools`, `marker_mix` from its manifest.
- Training row: a packed window of `seq_len + 1` tokens from the SFT stream with the mask, example boundaries marked
  red like document boundaries, and the count of target tokens in the window (the loss denominator is the number of
  mask-1 positions, `sft.py:203`). This is the view that does not exist today.
- The `extra_val_mixture` block links to the tokenized val splits it reads.

*RL recipe* (no mixture; `RlConfig`, `slm/train/rl.py:41-95`): the mixture table is replaced by a prompt panel:
- Tasks with their share (task names, repeated names count as weight: `m6_rl_gsm_tools` lists `gsm8k` three times,
  `tasks.py:120-124`), `n_train_prompts` / `n_heldout_prompts`, the hash split rule (`tasks.py:29-31`), `group_size`,
  `prompts_per_step`, `max_new_tokens`, `think_required`, `tools` / `max_tool_calls`, `reward_scheme` with its rule
  text (`rewards.py:77-93`: binary / tool 1.0-0.5 / signed / shaped), `kl_coef`, collapse guards.
- A sample of the deterministic train prompt list (`make_tasks` is seeded and torch-free, `tasks.py:113-137`), each
  shown as the exact generation prompt `<|bos|><|user|>{prompt}{SUFFIX}<|end|><|assistant|><|think|>`
  (`format_chat(add_generation_prompt=True, think_required=True)`, `chat.py:97-100`; `SUFFIX` at `answers.py:14`) in
  chips, plus gold answer. For gsm8k tasks the raw column is the GSM8K train parquet row.
- For a *run*, the training-row column shows real rollouts from `runs/<run>/rollouts/step_<n>.jsonl`
  (`rollout.py:175-179`): prompt + completion chips, reward, parsed answer, malformed flag, with a step selector. This is
  "what the model saw" for RL: its own samples plus the reward.

### Compare view

Two recipes side by side, union of sources, columns weight A / weight B / delta (pp), planned tokens A / B, epochs A /
B, plus a header diff of `seq_len`, `total_tokens`, root, `extra_val_mixture`. Rows only in one recipe are
highlighted. Default pairing for a run is its `init_from` parent, so `m8_base_4k_336m` vs `m8_base_stable_336m` is one
click. Computed client-side from two recipe payloads; no new endpoint.

### Chain view

For a run, walk `init_from` back to the root the way `RunIndex._cumulative_tokens` does (`services/runs.py:137-158`,
using `checkpoints/index.json` for the exact tokens of the checkpoint the child loaded). Table: one row per run in the
chain with stage, tokens used from it, seq_len, and a column per source with expected tokens = tokens used x normalized
weight; bottom row = cumulative per source; a stacked bar per stage. The caption says these are expectations under the
mixture probabilities (`loader.py:133`), not counts from the loader's per-source cursors (those are inside `latest.pt`,
which the torch-free portal cannot read). SFT runs contribute their sets, RL runs contribute completion tokens only.

## (e) API

**Reused unchanged**: `/api/data/raw/{source}/files|docs|doc|sample`, `/api/data/tokenized/{tag}/{source}/{split}/
shards|docs|doc|window|stats`, `POST /api/tokenizers/{tag}/encode`, `/api/runs`, `/api/runs/{run}` (its `config` is the
run recipe), `/api/runs/{run}/checkpoints`.

**Fixed**: `GET /api/data/sources` keeps its shape, plus a normalized `prepared` block per source and correct
provenance for derived sets (`datasets.py:292-304`): `{"prepared": {"v1": {"kind": "tokenized"|"sft", "train_tokens",
"train_docs", "val_tokens", "shards", "from": ["fineweb-edu"] | ["gsm8k-tools", ...], "made_by": "prepare"|
"sft_to_pretrain"|"synth_retrieval"|"rl.synth"|"sft", "manifest": {...}}}}`. The normalizer lives in
`datasets.py` (one function that maps the four manifest dialects: `prepare.py:145-151`, `sft.py:120-124`,
`sft_to_pretrain.py:42`, `synth_retrieval.py:132-134` / `rl/synth.py:149`).

**New**

- `GET /api/data/recipes` -> `[{"id": "run:m8_base_4k_336m", "kind": "run", "stage": "pretrain"|"sft"|"rl",
  "run_name", "config_path", "status", "tokens", "total_tokens", "seq_len", "tag", "init_from", "sources": [...],
  "plan_differs": bool}]`. Runs from `RunIndex.summaries()`; configs from `configs/train/*.yaml` matched by `run_name`.
- `GET /api/data/recipe?id=` -> `{"id", "stage", "seq_len", "tag", "root", "total_tokens" (resolved for
  `schedule.epochs`), "init_from", "rows": [{"source", "weight", "planned_tokens", "available_tokens", "epochs",
  "val_tokens", "prepared": {...normalized...}, "raw": {"source": "gsm8k", "files": 2, "rows": 8792} | null,
  "provenance": [...]}], "extra_val": [{...same row shape...}], "rl": null | {"tasks", "task_weights",
  "n_train_prompts", "n_heldout_prompts", "group_size", "prompts_per_step", "max_new_tokens", "think_required",
  "tools", "max_tool_calls", "reward_scheme", "reward_rule": "…", "kl_coef", "entropy_stop", "kl_stop"}}`. Replaces
  `/api/data/mixture` (keep the old route as an alias for one release; the e2e test does not call it directly).
- `GET /api/data/rl/prompts?id=&split=train|heldout&offset=&limit=` -> `[{"prompt_id", "task", "prompt", "gold",
  "ids", "pieces"}]` built with `make_tasks` + `prompt_messages` + `format_chat(add_generation_prompt=True,
  think_required=cfg)`; all torch-free (`tasks.py`, `answers.py`, `chat.py` import no torch).
- `GET /api/runs/{run}/rollouts?step=&offset=&limit=` -> `{"steps": [1, 2, …], "step", "rollouts": [{"prompt_id",
  "task", "prompt", "gold", "prompt_ids", "completion_ids", "pieces", "reward", "correct", "parsed", "malformed",
  "termination", "n_tokens"}]}` reading `rollouts/step_%05d.jsonl`; `old_logprobs` and `ref_logprobs` dropped.
- `GET /api/data/sft/{tag}/{set}/{split}/shards|examples|example|window|stats` mirroring the tokenized routes.
  `example?shard=&ex=` -> `{"shard", "ex", "start", "length", "ids", "mask", "pieces": [{id, piece, special,
  loss}], "n_target"}`; `window?shard=&start=&length=` -> `{"ids", "mask", "pieces", "example_starts", "n_target",
  "shard_tokens"}`; `stats` adds target share and the `max_len` histogram. Implemented by a new `SftSplit` in
  `datasets.py` modelled on `TokenizedSplit` (`datasets.py:135-239`): memmaps opened per request (`datasets.py:169-172`),
  `idx_*.npy` cached with the mtime/size signature (`datasets.py:174-183`), `refresh()` on directory change. It must
  not reuse `SftStream` (`sft.py:131-154`): that module imports torch (`sft.py:24`) and caches memmaps.
- `POST /api/data/raw/{source}/trace` body `{"file", "rg", "row", "tag", "sft_set"?}` -> `{"kept", "reason",
  "split": "train"|"val", "ids", "pieces", "n_tokens"}` for pretraining sources (`keep_doc`, `is_val`, `encode_doc`,
  `MIN/MAX_DOC_TOKENS` from `prepare.py`, torch-free), or for SFT sets (`rows_to_messages`, `apply_style` both
  styles, `format_chat` with the set's `think_required`/`tools`, `max_len` check, `sft.py:93-110`). The `--tools`
  path runs `PySession` (`chat.py:61-70`); it is the sandbox interpreter, not torch, and is already used by the
  tokenizer page's chat mode.
- `GET /api/data/chain?run=` -> `{"runs": [{"run", "stage", "seq_len", "tokens_used", "checkpoint", "per_source":
  {name: tokens}}], "totals": {name: tokens}, "cumulative_tokens"}`. `RunIndex._cumulative_tokens` is refactored into
  a `chain(run)` that returns the list; the existing summary keeps calling it.
- `GET /api/data/source/{name}` -> raw files, prepared artifacts per tag (normalized), children (sets whose provenance
  names it), and `used_by: [{"recipe_id", "weight"}]` computed from `/api/data/recipes`.

**`services/datasets.py` changes**: manifest normalizer; `mixture()` becomes `recipe()` taking a plain dict (the yaml
via `slm.config.load_yaml`, or `run.json["config"]`) and dispatching on `data.kind` / presence of `tasks`, so RL yaml
never goes through `TrainConfig`; `SftSplit`; `sft_manifests()` reused for the SFT rows; `chain()` on `RunIndex`. RL
configs must not be parsed with `slm.train.rl.load_rl_config` in the main process (`rl.py:24` imports torch): either
read the yaml as a dict, or move `RlConfig` to a torch-free module (`slm/train/rl_config.py`) - the latter is cleaner
and also lets the Architecture page load RL configs.

## (f) Implementation plan

1. **Truthful mixtures (ships first, ~1 day).** Manifest normalizer, `recipe()` with SFT/derived/epochs/RL support,
   `/api/data/recipes` and `/api/data/recipe`, run-vs-config listing, `extra_val` rows, provenance fix in `overview()`.
   Frontend: recipe list + recipe table replacing the mixture tab; old sources/documents tabs stay for now. Unit tests
   in `tests/test_portal_runs.py` on synthetic manifests of all four dialects and one RL yaml.
2. **Inspector (~1.5 days).** Fold the documents browser into the recipe row inspector with raw / prepared / row
   sub-tabs; `SftSplit` and the `/api/data/sft/...` routes; SFT packed window with mask and example boundaries; `trace`
   endpoint. The `window`, `TokenChips` with `boundaries` / `lossMask`, and the `.two.fill` layout are reused verbatim.
3. **RL (~0.5 day).** Prompt panel, `/api/data/rl/prompts`, `/api/runs/{run}/rollouts`, rollouts viewer.
4. **Compare and chain (~1 day).** Client-side compare; `chain()` refactor and endpoint; stacked bar via `chart.js`.
5. **Catalog and cleanup (~1 day).** `#/data/source/<name>` page absorbing the sources table and SFT table; remove the
   old tabs; update `tests/e2e/test_portal_ui.py:202-236` (it clicks tabs named sources/mixture/documents and asserts
   the window legend text) and `scripts/portal_smoke.py:99-104`; update `docs/command_center.md` (Pages table and API
   list) and the README data section pointer.

Phase 1 alone fixes every wrong number on the page today; phases 2-3 deliver "what the model sees" for all three
stages; 4-5 are the comparison tooling and consolidation.

## (g) Risks and open questions

- **No raw-row -> tokenized-doc mapping exists.** `prepare.py` writes no row ids, and `is_val` hashes text. The
  inspector therefore pairs a raw row with its *re-derived* tokenization (exact by construction: same tokenizer, same
  filter) rather than with the stored doc, and pairs a stored doc with a raw row only via text search of the first
  ~200 chars in the parquet file listed in the manifest (row-group scan of one file, acceptable interactively for
  fineweb-sized files? to be measured; can be skipped if slow). Question: is a one-directional link (raw -> prepared)
  enough, or should `prepare.py` start writing a `docs_<shard>.parquet` sidecar `(file, rg, row)` for new shards?
- **SFT style coin flip is not reproducible per row.** `apply_style` draws from an RNG seeded per set and advanced per
  row (`sft.py:80, 97-98`), so the trace endpoint can only show both candidate styles, not the one that landed in the
  shard. The stored example in the SFT shard is the ground truth; the trace is explanatory.
- **Expected vs actual exposure in the chain view.** Per-source counts are `weight x tokens`; actual counts sit in the
  loader state inside `latest.pt`. Option: have the trainers log `stream_epochs` per source in the `checkpoint`
  metrics record (one line in `pretrain.py`), which would make the chain view exact for future runs.
- **Recipe identity when a run's yaml has been edited.** A run's recipe is `run.json`; the yaml may now describe a
  different plan under the same `run_name` (the M3a data swap; `m8_base_stable_336m` was launched with `needle_n 4`
  and the yaml now says 16). The list flags `plan_differs`; question: should the run page also show the diff?
- **Overview readiness panel** (`home.js:62-68`) duplicates the catalog. Proposal: keep it as a one-line link per tag.
- **Windows file locks.** All new readers open memmaps per request and the SFT `mask_*.bin` doubles the handle count;
  still nothing is cached. The rollouts reader opens jsonl files read-only per request. The rule in
  `docs/runbook.md` section 2 (stop viewers before re-tokenizing) stays.
- **Cost of `/api/data/sources`.** `overview()` stats every parquet file of every source per call (`datasets.py:47-56`,
  `285-291`); with 67 python-edu files it is fine, but the recipe list should call it once and the catalog page
  should not re-fetch on every navigation. Unchanged behaviour, noted because the new pages fetch more.
- **Open**: should `tinystories`, `numina-cot-100k` and the `*-v1` leftovers be hidden by default in the catalog
  ("unused" filter derived from `used_by`)? Should the recipe view for a *live* run show the loader's current shard
  and offset (available from nothing today; would need the same `checkpoint` record fields as the exposure fix)?

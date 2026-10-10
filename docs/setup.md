# Setup: from a fresh clone to a training run

This repo holds code, configs and documentation only. **No datasets, checkpoints or run outputs are
versioned** — the raw corpora are hundreds of gigabytes and the checkpoints are ~0.6 GB each. Everything
here is reproducible from public sources with the commands below.

[docs/runbook.md](runbook.md) is the day-to-day operations guide; this file is the one-time bootstrap.

---

## 1. What you need

| | |
|---|---|
| GPU | One CUDA GPU. Everything here was built on an RTX 4080 SUPER (16 GB). Smaller cards work for the 26M and 149M configs; `configs/model/base_336m.yaml` wants ~14 GB. |
| OS | Developed on native Windows 11. The code is not Windows-only, but the paths in configs are (`C:\slm-data`) and only Windows has been exercised — see [§7](#7-running-on-linux-or-macos). |
| Python | 3.13 |
| Disk | ~60 GB for a minimal setup, ~400 GB to reproduce the 336M base (see [§3](#3-download-the-raw-corpora)). Training data and the repo can live on different volumes. |
| Network | The corpora come from the Hugging Face Hub; `python-edu` additionally pulls file contents from Software Heritage's public S3 bucket. No account, token or paid API is needed. |

## 2. Environment

```bash
git clone https://github.com/pbaer/micro-model
cd micro-model
pip install uv
uv venv .venv --python 3.13
uv pip install --python .venv/Scripts/python.exe "torch==2.14.0+cu130" --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/Scripts/python.exe "triton-windows==3.8.0.post28"
uv pip install --python .venv/Scripts/python.exe -e ".[portal,dev]"
.venv/Scripts/python.exe -m pytest
```

Every command below uses `.venv/Scripts/python.exe`; it is written as `python` for brevity. Install torch
only from the cu130 index. The `triton-windows` wheel is what makes `torch.compile` work, which is a ~6x
speedup here and not optional. `nvidia-smi` must be on `PATH` for GPU telemetry.

**Where data goes.** Everything lands under `C:\slm-data`, overridable with the `SLM_DATA_ROOT` environment
variable. Configs name absolute paths (`tokenizer_dir`, `data.sft_root`, `data.tokenized_root`), so if you
change the root, change those too.

```
$SLM_DATA_ROOT/
  raw/<source>/           parquet as downloaded
  tokenizer/v1/           the trained BPE tokenizer
  tokenized/v1/<source>/  uint16 pretraining shards + document index
  sft/v1/<name>/          chat-formatted SFT examples + loss masks
  models/<short-name>/    external comparison models (optional, slm/eval/external.py)
```

## 3. Download the raw corpora

`python -m slm.data.download --list` prints the registry ([slm/data/sources.py](../slm/data/sources.py)).
All are public Hugging Face parquet datasets.

```bash
python -m slm.data.download fineweb-edu --n-files 14    # educational web prose; 14 files ~28.5 GB, ~10B tokens
python -m slm.data.download cosmopedia  --n-files 6     # synthetic textbooks, ~1.17 GB per file
python -m slm.data.download finemath    --n-files 9     # math web text with LaTeX, ~286 MB per file
python -m slm.data.download tinystories --all           # tiny, only needed for the M1 toy run
python -m slm.data.download python-edu  --all           # ids only -- see below
python -m slm.data.swh      python-edu  --max-files 200000   # fetch contents from Software Heritage S3
```

`python-edu` and `stack-edu-shell` ship *file identifiers*, not code, so the contents come from Software
Heritage in a second pass. That pass is slow (hours for 200K files) and resumable — rerun it if interrupted.
200K files is the first base's scale; the 336M base's `python-edu` holds ~1.3M files (raise `--max-files`, default
100000), so that it is not repeated several times over 10B tokens.
**`stack-edu-shell` is only needed to reproduce the first base and M8 phase 1**; shell was dropped from every
mixture on 2026-09-19 and no current config uses it.

**Narrative prose (PG-19).** `deepmind/pg19` on the Hub holds only a loading script and the split lists; the books
(and `metadata.csv` with title and publication date) are in the release's public GCS bucket
(`storage.googleapis.com/deepmind-gutenberg/`). The first command fetches the three split lists, the second the
metadata and every book with `publication_date >= 1850` (the date rule runs before download, so earlier books are
never fetched): 25,121 of 28,602 train books plus 132 of the 150 validation/test books, ~10 GB of text stored as
~4 GB of zstd parquet under `raw/gutenberg-pg19/{train,validation,test}/`. Resumable per 1,000-book chunk; ~8
minutes with `--workers 64`.

```bash
python -m slm.data.download gutenberg-pg19 --all          # data/{train,validation,test}_files.txt
python -m slm.data.gutenberg download --workers 64        # metadata.csv + books -> parquet
```

Chat, math and tool data (`slm.data.download` takes one source per call):

```bash
python -m slm.data.download smoltalk-smol-magpie-ultra --n-files 3   # the test file + the first two train files
for s in smoltalk-openhermes-100k smoltalk-systemchats-30k smoltalk-smol-constraints \
         smoltalk-everyday-conversations smoltalk-metamathqa-50k gsm8k svamp; do
    python -m slm.data.download $s --all
done
```

`svamp` is evaluation-only (`slm.eval.reasoning --svamp`).

**External comparison models** (optional). Seven open-weight models of similar size are scored by our own evals as
comparison rows (`slm/eval/external.py`, [design.md](design.md) §8). Their weights are downloaded once, like the
datasets, to `$SLM_DATA_ROOT/models/<short-name>/` (safetensors, configs and tokenizer files only; ~5.5 GB for all
seven); everything after this step runs with the Hugging Face offline switches forced on.

```bash
uv pip install --python .venv/Scripts/python.exe -e ".[external]"   # transformers + accelerate (+ lm-eval); torch untouched
python -m slm.eval.external download --all        # or name them: smollm2-360m-instruct qwen2.5-0.5b ...
python -m slm.eval.external list                  # registry, and what is on disk
bash scripts/measure_external.sh smollm2-360m-instruct   # every applicable eval -> runs/ext_<name>/
```

| short name | Hugging Face id | on disk |
|---|---|---|
| `smollm2-135m`, `smollm2-135m-instruct` | HuggingFaceTB/SmolLM2-135M, -Instruct | 0.27 GB each |
| `smollm2-360m`, `smollm2-360m-instruct` | HuggingFaceTB/SmolLM2-360M, -Instruct | 0.73 GB each |
| `qwen2.5-0.5b`, `qwen2.5-0.5b-instruct` | Qwen/Qwen2.5-0.5B, -Instruct | 1.00 GB each |
| `gpt2-medium` | openai-community/gpt2-medium | 1.52 GB (fp32) |

**Scale it down.** `--n-files 2` for fineweb-edu and 1 for cosmopedia is enough to exercise the whole
pipeline end to end on ~20 GB; you simply cannot train the 10B-token base on it. Every config's `mixture`
names sources, so a smaller run is a config edit, not a code change.

## 4. Train the tokenizer

```bash
python scripts/train_tokenizer.py --out C:/slm-data/tokenizer/v1 --chars 1.5e9
python scripts/eval_tokenizer.py C:/slm-data/tokenizer/v1
```

32K BPE with a reserved block of 64 special tokens (chat roles, think spans, tool calls, declared functions).
**Never change the tokenizer after a model has been trained with it** — a new tokenizer is a new
`tokenized/<tag>` tree and a new model.

## 5. Tokenize the pretraining sources

```bash
python -m slm.data.prepare tinystories cosmopedia finemath python-edu --tokenizer C:/slm-data/tokenizer/v1
python -m slm.data.prepare fineweb-edu --tokenizer C:/slm-data/tokenizer/v1 --name fineweb-edu-10bt
python -m slm.data.prepare fineweb-edu --tokenizer C:/slm-data/tokenizer/v1 --min-doc-tokens 4096 --name fineweb-edu-long
```

PG-19 has its own preparer, because selection is corpus-level (rank by dialogue density, take books until a token
target) and books are longer than the generic 65,536-token document cap (`slm.data.prepare` refuses the source and
points here). `--dry-run` writes only the selection report (`raw/gutenberg-pg19/selection_dryrun.json`); thresholds
are `GutenbergConfig` fields, overridable as `key=value`. Rules: [design.md](design.md) §4.

```bash
python -m slm.data.gutenberg prepare --tokenizer C:/slm-data/tokenizer/v1 --dry-run
python -m slm.data.gutenberg prepare --tokenizer C:/slm-data/tokenizer/v1          # e.g. target_tokens=4e8
```

The M10 continued pretraining also uses `gutenberg-canon`, a curated supplement of canon books the dialogue ranking
left out (books already in `gutenberg-pg19` are never repeated):

```bash
python -m slm.data.gutenberg canon --tokenizer C:/slm-data/tokenizer/v1          # --dry-run for the report only
```

Several sources in the base mixture are **generated here, not downloaded**. `slm.data.synth_retrieval` (its
`--backdrop`) and `slm.rl.synth_python` (its `--corpus`) read real text from `fineweb-edu-b` by default: the first 7
fineweb-edu files prepared under that name (`--name fineweb-edu-b` while only those files are downloaded), which is
what the shipped sets used. Point those flags at `fineweb-edu-10bt` instead if you skip it.

```bash
# retrieval: facts inserted into real text, plus key-value ledgers (context curriculum + 3.5% of the base).
# The set under this name since M8 phase 2 (3K-4K-token documents, 80% of facts in the first 15%):
python -m slm.data.synth_retrieval --tokenizer C:/slm-data/tokenizer/v1 --name synth-retrieval \
    --tokens 150000000 --min-len 3072 --max-len 4096 --early-frac 0.8 --seed 11
# M8 phase 1 trained on the earlier set (400M tokens, 512-16K-token documents), now kept as synth-retrieval-16k:
python -m slm.data.synth_retrieval --tokenizer C:/slm-data/tokenizer/v1 --name synth-retrieval-16k \
    --tokens 400000000 --min-len 512 --max-len 16384 --seed 7

# chat and tool-call conversations, folded into the decay phase as pretraining rows. These re-lay SFT sets, so run
# them after §6 (the tool sets) and after building the five plain SmolTalk sets:
python -m slm.data.sft smoltalk-smol-magpie-ultra smoltalk-openhermes-100k smoltalk-systemchats-30k \
    smoltalk-smol-constraints smoltalk-everyday-conversations --tokenizer C:/slm-data/tokenizer/v1
python scripts/sft_to_pretrain.py --out smoltalk-chat smoltalk-smol-magpie-ultra smoltalk-openhermes-100k \
    smoltalk-systemchats-30k smoltalk-smol-constraints smoltalk-everyday-conversations
python scripts/sft_to_pretrain.py --out tool-chat gsm8k-tools synthetic-reasoning-tools metamathqa-tools \
    synthetic-multiturn-tools synthetic-python-tools
```

`m10_base_prose_336m.yaml` names its tool source `tool-chat-v3`: per [log.md](log.md) (2026-09-19 14:20) that is the
build of the same five sets kept from just before the `hoist_calls` fix to the GSM8K and MetaMathQA tool traces; the
command above builds the current `tool-chat`.

## 6. Build the SFT sets

The M9 post-training configs name these exactly; the helper scripts build them in one go.

```bash
# chat SFT (stage A): mandatory but empty think span, so the format never changes under the model
bash scripts/prep_chat_4k_think.sh     # -> smoltalk-*-4k-think (5 sets)

# chat rehearsal with short prose think spans that reach for no tool (stage B v2 onward)
bash scripts/prep_chat_4k_direct.sh    # -> smoltalk-{smol-magpie-ultra,openhermes-100k}-4k-direct

# short, non-computational prompts only -- the "answer directly, no tool" signal (stage B v3 onward)
bash scripts/prep_chat_short.sh        # -> smoltalk-*-short (5 sets; magpie's is built but unused)

# grammar-generated tool, reasoning and multi-turn sets. Builds four, each from its module's default name:
#   synthetic-python-tools      slm.rl.synth_python     the whole permitted sandbox subset
#   synthetic-reasoning         slm.rl.synth            arithmetic/algebra/word problems, prose reasoning
#   synthetic-reasoning-tools   slm.rl.synth --tools    the same, every arithmetic step a tool call
#   synthetic-multiturn-tools   slm.rl.synth_multiturn  multi-turn REPL sessions
bash scripts/prep_synth_rebuild.sh

# math with think spans and #### answers, with and without tool calls
python -m slm.data.sft gsm8k --tokenizer C:/slm-data/tokenizer/v1 --think-required --name gsm8k-reasoning
python -m slm.data.sft gsm8k --tokenizer C:/slm-data/tokenizer/v1 --think-required --tools --name gsm8k-tools
python -m slm.data.sft smoltalk-metamathqa-50k --tokenizer C:/slm-data/tokenizer/v1 --think-required --name metamathqa-reasoning
python -m slm.data.sft smoltalk-metamathqa-50k --tokenizer C:/slm-data/tokenizer/v1 --think-required --tools --name metamathqa-tools
```

Why so many chat variants: they differ only in what the think span contains, and that difference decided
whether a post-training stage regressed. See [results.md](results.md) §10 and
[slm/data/direct_think.py](../slm/data/direct_think.py).

## 7. Running on Linux or macOS

Nothing in the training code is Windows-specific, but only Windows has been exercised, and these are known
to need attention:

- **Paths.** Configs carry `C:\slm-data\...`. Set `SLM_DATA_ROOT` and update `tokenizer_dir`,
  `data.sft_root` and `data.tokenized_root` in the configs you run.
- **Interpreter.** `.venv/Scripts/python.exe` becomes `.venv/bin/python`.
- **Triton.** `triton-windows` is a Windows-only wheel; on Linux install plain `triton` (usually already a
  torch dependency) and drop that line.
- **Attention backend.** `runtime.sdpa_backend: cudnn` is set because the SDPA *flash* backend is
  unavailable on Windows. On Linux, flash is available and likely faster — try `flash`.
- **The Windows gotchas in [runbook.md](runbook.md) §8 mostly disappear**: memory-mapped files can be
  replaced while open, and WDDM's "spill to host RAM instead of OOM" behaviour — which makes a too-large
  batch run 60x slower rather than failing outright — is not a thing.

## 8. Check it works

```bash
python -m pytest                                              # CUDA tests skip without a GPU
python scripts/bench_throughput.py --config configs/model/base_336m.yaml --seq 2048
python -m slm.train.pretrain --config configs/train/m1_tinystories_26m.yaml    # ~1 h on a 4080 SUPER
python -m slm.portal                                          # command center at http://127.0.0.1:8930
```

The portal refuses to start when port 8930 is taken and names the program holding it (`--port 8931` … `8939` are this
project's spares). In the Claude desktop app, start it from the browser preview (`.claude/launch.json`, the `portal`
entry) rather than by hand; see [runbook.md](runbook.md) §7.

M1 is the intended smoke test: a 26M model on TinyStories that exercises the loop, packing,
checkpoint/resume, sampling, evaluation and the report, in about an hour.

## 9. Reproducing a specific model

`configs/train/` is named after the runs. The chain that produced the current model (`m10_rl_336m` final.pt, the
portal chat mode's default checkpoint):

| Stage | Config | Tokens | Wall clock |
|---|---|---|---|
| base, phase 1 | `m8_base_stable_336m.yaml` | 7.5B at 2K | ~73 h |
| base, phase 2 | `m8_base_4k_336m.yaml` | 2.5B at 4K, LR decay | ~27 h |
| prose continuation | `m10_base_prose_336m.yaml` | 2B at 4K, LR re-warm and decay | ~24 h |
| chat SFT | `m10_sft_336m.yaml` | 200M | ~2.8 h |
| reasoning + tools SFT | `m10_tool_336m.yaml` | 120M | ~1.8 h |
| GRPO | `m10_rl_336m.yaml` | 250 steps | ~1.8 h |

The M9 chain on the unmodified base (`m9_sft_336m` → `m9_tool4_336m` → `m9_rl5_336m` → `m9_select_336m` →
`m9_rl6_336m`) is the comparison point for every M10 stage. `scripts/m10_chain.sh` runs the M10 chain with its gates.

Each stage's `init_from` points at the previous stage's checkpoint. A run that *continues* another (rather
than initialising from it) must also inherit optimizer state and data-stream cursors (`m10_base_prose_336m` does,
from `m8_base_4k_336m`'s `latest.pt`) — see
[runbook.md](runbook.md) §5, which exists because getting this wrong silently re-trained 94% of a phase on
data the parent had already seen.

Numbers for every stage are in [results.md](results.md); the narrative, including what went wrong, is in
[log.md](log.md).

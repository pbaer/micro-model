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
python -m slm.data.download cosmopedia  --n-files 2     # synthetic textbooks, ~1.17 GB per file
python -m slm.data.download finemath    --n-files 3     # math web text with LaTeX
python -m slm.data.download tinystories --all           # tiny, only needed for the M1 toy run
python -m slm.data.download python-edu  --all           # ids only -- see below
python -m slm.data.swh      python-edu  --max-files 200000   # fetch contents from Software Heritage S3
```

`python-edu` and `stack-edu-shell` ship *file identifiers*, not code, so the contents come from Software
Heritage in a second pass. That pass is slow (hours for 200K files) and resumable — rerun it if interrupted.
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

Chat, math and tool data:

```bash
python -m slm.data.download smoltalk-smol-magpie-ultra smoltalk-openhermes-100k \
    smoltalk-systemchats-30k smoltalk-smol-constraints smoltalk-everyday-conversations --all
python -m slm.data.download smoltalk-metamathqa-50k gsm8k --all
```

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

Two sources in the base mixture are **generated here, not downloaded**:

```bash
# retrieval: facts inserted into real text, plus key-value ledgers (context curriculum + 3.5% of the base)
python -m slm.data.synth_retrieval --tokenizer C:/slm-data/tokenizer/v1 --name synth-retrieval

# chat and tool-call conversations, folded into the decay phase as pretraining rows
python -m slm.rl.synth_python --tokenizer C:/slm-data/tokenizer/v1 --n 170000
python scripts/sft_to_pretrain.py --out smoltalk-chat smoltalk-smol-magpie-ultra smoltalk-openhermes-100k \
    smoltalk-systemchats-30k smoltalk-smol-constraints smoltalk-everyday-conversations
python scripts/sft_to_pretrain.py --out tool-chat synthetic-python-tools gsm8k-tools
```

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
python -m slm.data.sft gsm8k --tokenizer C:/slm-data/tokenizer/v1 --think-required --max-len 1024 --name gsm8k-reasoning
python -m slm.data.sft gsm8k --tokenizer C:/slm-data/tokenizer/v1 --think-required --tools --max-len 1024 --name gsm8k-tools
python -m slm.data.sft smoltalk-metamathqa-50k --tokenizer C:/slm-data/tokenizer/v1 --think-required --max-len 1024 --name metamathqa-reasoning
python -m slm.data.sft smoltalk-metamathqa-50k --tokenizer C:/slm-data/tokenizer/v1 --think-required --tools --max-len 1024 --name metamathqa-tools
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
python -m slm.portal                                          # command center at http://127.0.0.1:8765
```

M1 is the intended smoke test: a 26M model on TinyStories that exercises the loop, packing,
checkpoint/resume, sampling, evaluation and the report, in about an hour.

## 9. Reproducing a specific model

`configs/train/` is named after the runs. The chain that produced the current model:

| Stage | Config | Tokens | Wall clock |
|---|---|---|---|
| base, phase 1 | `m8_base_stable_336m.yaml` | 7.5B at 2K | ~73 h |
| base, phase 2 | `m8_base_4k_336m.yaml` | 2.5B at 4K, LR decay | ~27 h |
| chat SFT | `m9_sft_336m.yaml` | 200M | ~2.5 h |
| reasoning + tools SFT | `m9_tool4_336m.yaml` | 120M | ~1.7 h |
| GRPO | `m9_rl2_336m.yaml` | 141 steps | ~1.3 h |

Each stage's `init_from` points at the previous stage's checkpoint. A run that *continues* another (rather
than initialising from it) must also inherit optimizer state and data-stream cursors — see
[runbook.md](runbook.md) §5, which exists because getting this wrong silently re-trained 94% of a phase on
data the parent had already seen.

Numbers for every stage are in [results.md](results.md); the narrative, including what went wrong, is in
[log.md](log.md).

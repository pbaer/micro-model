# Small language model, from scratch

Research stack that trains a ~149M-parameter decoder-only Transformer from random initialization on
one RTX 4080 SUPER, then applies instruction SFT, reasoning SFT, verifiable-reward RL, and RoPE
context extension. Everything is in-house except tokenization/dataset utilities.

## Setup (Windows, native)

```bash
pip install uv
uv venv .venv --python 3.13
uv pip install --python .venv/Scripts/python.exe "torch==2.14.0+cu130" --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/Scripts/python.exe "triton-windows==3.8.0.post28"
uv pip install --python .venv/Scripts/python.exe -e .
.venv/Scripts/python.exe -m pytest
```

Data root is `C:\slm-data` (override with `SLM_DATA_ROOT`). Code, configs, runs, and reports live here.

## Pipeline

| Step | Command |
|---|---|
| download raw corpora | `python -m slm.data.download fineweb-edu --n-files 2` (`--list` shows sources) |
| fetch code contents | `python -m slm.data.swh python-edu --max-files 200000` |
| train tokenizer | `python scripts/train_tokenizer.py --out C:/slm-data/tokenizer/v1 --chars 1.5e9` |
| evaluate tokenizer | `python scripts/eval_tokenizer.py C:/slm-data/tokenizer/v1` |
| tokenize to shards | `python -m slm.data.prepare tinystories fineweb-edu --tokenizer C:/slm-data/tokenizer/v1` |
| benchmark throughput | `python scripts/bench_throughput.py --config configs/model/base_149m.yaml` |
| pretrain | `python -m slm.train.pretrain --config configs/train/m1_tinystories_26m.yaml` |
| diagnostics | `python scripts/diagnose.py runs/<run>/checkpoints/best.pt --root ... --source ...` |
| SFT data | `python -m slm.data.sft smoltalk-openhermes-100k --tokenizer C:/slm-data/tokenizer/v1` (`--think-required` for reasoning sets) |
| synthetic reasoning traces | `python -m slm.rl.synth --tokenizer C:/slm-data/tokenizer/v1 --n 40000` |
| instruct / reasoning SFT | `python -m slm.train.pretrain --config configs/train/m4_sft_149m.yaml` (same loop, `data.kind: sft`, `init_from`) |
| GRPO RL | `python -m slm.train.rl --config configs/train/m6_rl_arith_149m.yaml` |
| reasoning benchmark | `python -m slm.eval.reasoning --checkpoint ... --gsm8k 200` |
| long-context needle eval | `python -m slm.eval.long_context --checkpoint ... --lengths 1024 4096 8192 16384` |
| command center (web) | `python -m slm.portal` then open http://127.0.0.1:8765 |

Every run writes `runs/<run>/report.html` (self-contained, auto-refreshing) with ETA, loss curves,
milestone timings, and latest samples. Stop a run with Ctrl-C or by creating `runs/<run>/STOP`;
rerunning the same command resumes from `checkpoints/latest.pt`.

## Layout

```
slm/
  config.py          dataclass configs, YAML + k=v overrides
  model/             attention (GQA, QK-norm, KV cache), rope (linear/NTK/YaRN), mlp (SwiGLU), transformer, loss
  data/              sources registry, download, SWH fetch, tokenizer, prepare (shards), loader (packing, resume)
  train/             pretrain loop (also SFT via data.kind=sft), config, LR schedules, rl.py (GRPO)
  rl/                task generators (disjoint splits), verifier, advantages, objectives, rollouts, synthetic traces
  eval/              generation suite, diagnostics, sampling, reasoning benchmark, long-context needle eval
  portal/            FastAPI + Preact command center: runs, data, tokenizer, model harness (torch worker), architecture
  utils/             checkpoint, logging, HTML report, SDPA backend probe, FLOP/MFU accounting
configs/model/       tiny (tests), sanity_26m (M1), base_149m (M2+)
configs/train/       per-run training configs
scripts/             benchmarks, tokenizer training/eval, diagnostics CLI
tests/               model, data, and trainer (resume equivalence) tests
```

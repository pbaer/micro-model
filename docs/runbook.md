# Runbook: operating the project

Everything operational, in the order you need it. All commands run from the repo root with
`.venv/Scripts/python.exe` (abbreviated `python` below). Data root is `C:\slm-data` (override with
`SLM_DATA_ROOT`).

## 1. Environment

```bash
pip install uv
uv venv .venv --python 3.13
uv pip install --python .venv/Scripts/python.exe "torch==2.14.0+cu130" --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/Scripts/python.exe "triton-windows==3.8.0.post28"
uv pip install --python .venv/Scripts/python.exe -e ".[portal,dev]"
.venv/Scripts/python.exe -m playwright install chromium     # only for tests/e2e and scripts/portal_smoke.py
.venv/Scripts/python.exe -m pytest                            # CUDA tests skip without a GPU
```

Reinstall torch only from the cu130 index; the `triton-windows` wheel is what makes `torch.compile`
work (6× faster than eager here, not optional). `nvidia-smi` must be on `PATH` for GPU telemetry.

## 2. Data preparation

```bash
python -m slm.data.download --list                                   # registered sources
python -m slm.data.download fineweb-edu --n-files 7                   # raw parquet -> C:\slm-data\raw\fineweb-edu
python -m slm.data.download cosmopedia --n-files 2
python -m slm.data.download finemath --n-files 3
python -m slm.data.download python-edu --all && python -m slm.data.swh python-edu --max-files 200000
python -m slm.data.download stack-edu-shell --all && python -m slm.data.swh stack-edu-shell --max-files 50000
python -m slm.data.download tinystories --all
python scripts/train_tokenizer.py --out C:/slm-data/tokenizer/v1 --chars 1.5e9     # once; tokenizer is frozen afterwards
python scripts/eval_tokenizer.py C:/slm-data/tokenizer/v1
python -m slm.data.prepare tinystories cosmopedia finemath python-edu stack-edu-shell --tokenizer C:/slm-data/tokenizer/v1
python -m slm.data.prepare fineweb-edu --tokenizer C:/slm-data/tokenizer/v1 --name fineweb-edu-b        # all downloaded files
python -m slm.data.prepare fineweb-edu --tokenizer C:/slm-data/tokenizer/v1 --min-doc-tokens 4096 --name fineweb-edu-long
```

SFT and reasoning data:

```bash
python -m slm.data.download smoltalk-smol-magpie-ultra --n-files 1     # and the other smoltalk-* subsets, gsm8k, smoltalk-metamathqa-50k
python -m slm.data.sft smoltalk-smol-magpie-ultra smoltalk-openhermes-100k smoltalk-systemchats-30k smoltalk-smol-constraints smoltalk-everyday-conversations --tokenizer C:/slm-data/tokenizer/v1
python -m slm.data.sft gsm8k --tokenizer C:/slm-data/tokenizer/v1 --think-required --max-len 1024 --name gsm8k-reasoning
python -m slm.data.sft smoltalk-metamathqa-50k --tokenizer C:/slm-data/tokenizer/v1 --think-required --max-len 1024 --name metamathqa-reasoning
python -m slm.rl.synth --tokenizer C:/slm-data/tokenizer/v1 --n 40000
```

Rules: stop the portal (or make sure no Documents page is open) before re-tokenizing a source, because
a memory-mapped shard cannot be replaced on Windows. Never change the tokenizer after a model has been
trained with it; a new tokenizer is a new `tokenized/<tag>` tree.

## 3. Before launching a training run

1. **GPU free?** `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` must show no training
   process, and the Overview page must show no live run. Two jobs on the GPU run each other at a
   quarter speed or worse.
2. **Memory budget.** Take `microbatch` from the benchmark table in `docs/results.md` (or run
   `python scripts/bench_throughput.py --config configs/model/base_149m.yaml --seq <len> --max-mb <n>`).
   Peak VRAM must stay under ~14.5 GiB; on Windows PyTorch does not OOM at 16 GiB, it spills into host
   memory and runs 60× slower. The benchmark flags spills.
3. **Config named after the run** in `configs/train/<run_name>.yaml`, `run_name` matching, `init_from`
   pointing at an existing checkpoint (`checkpoints/final.pt` or a `snap_*.pt`), mixture sources present
   under `C:\slm-data\tokenized\v1`.
4. **Sanity on the schedule**: `total_tokens`, `warmup_tokens`, `tokens_per_update` (must be
   `microbatch × seq_len × integer`), `milestone_tokens`, eval cadence. `python scripts/param_count.py
   configs/model/base_149m.yaml` prints the memory budget.
5. **Disk**: each full `latest.pt` is ~1.8 GB for the 149M model (fp32 weights + AdamW), each bf16
   snapshot ~300 MB, one per 100M tokens.

## 4. Launching, monitoring, stopping, resuming

```bash
mkdir -p runs/<run>
python -u -m slm.train.pretrain --config configs/train/<run>.yaml > runs/<run>/train.log 2>&1 &
```

Always `-u` and a redirect to a log file. Never pipe a long job through `grep`/`tail`: buffered output
is lost if the pipeline dies, and the job keeps running invisibly.

- The console prints one line per `log_every_updates`: tokens, loss, LR, grad norm, tok/s, step time,
  VRAM, GPU temperature and power, ETA. After the compile warmup it prints the initial whole-run estimate.
- `runs/<run>/report.html` auto-refreshes; the command center shows the same data live.
- Stop: Ctrl-C in the foreground, or `touch runs/<run>/STOP`. The current update finishes, `latest.pt`
  and the report are written, the process exits. Resume by rerunning the same command (it resumes by
  default; `--fresh` ignores `latest.pt`).
- Watch for: loss spikes or NaN (the run stops itself on non-finite loss), tok/s far below the benchmark
  (another GPU process, or a spill), `GPU HOT` warnings above 80 °C, `data_ms` growing (loader starved).

RL runs: `python -u -m slm.train.rl --config configs/train/<run>.yaml > runs/<run>/train.log 2>&1 &`.
Same STOP/resume semantics; progress is in steps; rollouts land in `runs/<run>/rollouts/`.

## 5. Phase changes (the pipeline)

Each stage starts from the previous stage's checkpoint via `init_from` with a fresh schedule:

| Stage | Config | init_from |
|---|---|---|
| M3a stable continuation | `m3_base_stable_149m.yaml` | `runs/m2_base_149m/checkpoints/snap_800M.pt` (pre-decay snapshot) |
| M3b 8K phase + decay | `m3_base_8k_149m.yaml` | `runs/m3_base_stable_149m/checkpoints/final.pt` |
| M4 instruct SFT | `m4_sft_149m.yaml` | the base `final.pt` |
| M5 reasoning SFT | `m5_reasoning_149m.yaml` | `runs/m4_sft_149m/checkpoints/final.pt` |
| M6 RL stage A | `m6_rl_arith_149m.yaml` | `runs/m5_reasoning_149m/checkpoints/final.pt` |
| M7 16K extension | `m7_ctx16k_149m.yaml` | `runs/m3_base_8k_149m/checkpoints/final.pt` |

`scripts/pipeline_after_m2.sh` shows how to chain evals → M4 → M5 → M6 unattended (each stage waits
for the previous checkpoint). After every base checkpoint run the evaluation set:

```bash
python -u -m slm.eval.lm_eval_wrapper --checkpoint <ckpt> --tasks hellaswag,arc_easy,piqa --batch-size 16 --out runs/<run>/lm_eval.json
python -u scripts/diagnose.py <ckpt> --root C:/slm-data/tokenized/v1 --source fineweb-edu --seq 2048 --batches 8 --mb 4
python -u -m slm.eval.long_context --checkpoint <ckpt> --lengths 1024 2048 4096 8192 --n 16 --out runs/<run>/needle.json   # --batch-tokens 16384 caps the generation batch
python -u -m slm.eval.reasoning --checkpoint <ckpt> --n 100 --gsm8k 200 --out runs/<run>/reasoning_eval.json   # post-M5/M6
```

Evaluations use the GPU; run them between training runs, or accept sharing the GPU for short jobs.

## 6. Run naming and bookkeeping

- `m<milestone>_<role>_<params>`; roles describe what the run produces (`base`, `base_stable`,
  `base_8k`, `sft`, `reasoning`, `rl_arith`, `ctx16k`). Token counts never go in names.
- Exact token counts per checkpoint live in `runs/<run>/checkpoints/index.json`;
  `scripts/backfill_ckpt_index.py` rebuilds it for old runs from `metrics.jsonl`.
- A finished run's outcome goes into `docs/results.md`; incidents and decisions into `docs/log.md`.
- `runs/` is untracked (checkpoints are large); commit configs, code, docs and `artifacts/bench`.

## 7. Command center

```bash
python -m slm.portal                # http://127.0.0.1:8765, opens a browser tab
python -m slm.portal --no-browser --port 8766 --runs-root some/other/runs
python scripts/portal_smoke.py --url http://127.0.0.1:8765        # click through a live portal
python -m pytest tests/e2e -q                                      # hermetic Playwright suite
```

The server never imports torch; the Inference page spawns a worker subprocess on demand and refuses
the GPU while a training run is live unless forced. Static assets are served with no-cache headers, so
a browser reload picks up JS changes; Python changes need a server restart. A tab loaded while the
server was restarting shows a failure panel with a Reload button.

## 8. Windows gotchas (all learned the hard way)

| Symptom | Cause | Rule |
|---|---|---|
| Training at ~1K tok/s, "peak VRAM 20–38 GiB" | WDDM lets PyTorch spill into host memory instead of OOM | Keep peaks ≤ ~14.5 GiB; benchmark marks spills as failures |
| Generation crawls during training or eval | cuDNN attention re-plans for every KV length | All decode paths use `sdpa_context("decode")` (efficient + math) |
| `prepare.py` fails with EINVAL / PermissionError on replace | Another process holds a memmap or open handle | Stop viewers before re-tokenizing; checkpoint writes retry; portal never caches memmaps |
| A run mysteriously runs at a quarter speed | Another GPU job (a benchmark piped through grep kept running) | Check `nvidia-smi` compute apps before and after every job |
| `torch.compile` errors | Missing `triton-windows` or wrong torch build | Reinstall from the cu130 index; keep the pinned versions |
| Bash heredoc with JS content fails to parse | Git Bash quoting | Write patch scripts to a file and run them |
| RNG restore TypeError on resume | CUDA generator state must be a CPU tensor | Handled in `checkpoint.py` |

## 9. Troubleshooting

- **Run stopped, `latest.pt` missing or corrupt**: `latest.prev.pt` is the previous full checkpoint;
  rename it and resume. Snapshots are model-only and can seed a new run via `init_from`.
- **Resume loss differs from the pre-stop curve**: check the run log for `resumed from` and that the
  same config was used; the resume test guarantees equivalence only for identical configs.
- **Portal shows a run at 99% / stale**: `metrics.jsonl` is the source of truth; `finish` records carry
  the exact total. Backfill the checkpoint index if the run predates it.
- **Portal blank**: hard reload; the boot panel names the failing module. If the page is fine but the
  pane looks empty in the desktop app, another floating window may be covering it.
- **Needle eval assertion `sequence N exceeds RoPE table`**: the requested length must leave 16 tokens
  for the answer inside `max_seq_len`; use lengths below the table size or extend the model first.

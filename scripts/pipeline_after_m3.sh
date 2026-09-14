#!/usr/bin/env bash
# Post-base pipeline on the real 8K base checkpoint (M3b):
#   base evals -> M4 instruct SFT -> lm-eval -> M5 reasoning SFT -> reasoning eval -> M6 RL stage A -> stage B
#   -> reasoning evals -> M7 16K extension -> needle eval.
# Each stage starts only if the previous checkpoint exists. Every training run is watched for the WDDM
# host-memory spill (peak VRAM > 14.5 GiB or tok/s collapsing) and stopped via STOP if it happens.
# Stop the whole chain by killing this script; stop one stage with runs/<run>/STOP.
set -u
cd "$(dirname "$0")/.."
P=.venv/Scripts/python.exe
log() { echo "$(date '+%H:%M:%S') $*"; }

# guard_run <run>: after warmup, read the last train record; STOP the run if it is spilling.
guard_run() {
  local run=$1 rec
  sleep 420
  rec=$($P - "$run" <<'PY'
import json, sys
from slm.utils.logging import MetricsLogger
recs = [r for r in MetricsLogger.read(f"runs/{sys.argv[1]}/metrics.jsonl") if r["kind"] == "train"]
if not recs: print("norecord"); sys.exit()
r = recs[-1]; print(f"{r.get('vram_gib', 0):.1f} {r.get('tok_s', 0):.0f}")
PY
)
  log "guard $run: $rec"
  case "$rec" in norecord) return;; esac
  local vram=${rec% *} tps=${rec#* }
  # RL runs log tok/s as 0 (progress is in steps), so the throughput test applies only when it is reported.
  if [ "${vram%.*}" -ge 14 ] || { [ "$tps" -gt 0 ] && [ "$tps" -lt 3000 ]; }; then
    log "SPILL suspected on $run (vram $vram GiB, $tps tok/s): stopping it"
    touch "runs/$run/STOP"
  fi
}

train() {  # train <run> <module> : runs in the background, guards, waits
  local run=$1 mod=$2
  mkdir -p "runs/$run"
  log "start $run"
  $P -u -m "$mod" --config "configs/train/$run.yaml" > "runs/$run/train.log" 2>&1 &
  local pid=$!
  guard_run "$run"
  wait $pid
  log "$run exited $?"
}

BASE=runs/m3_base_8k_149m/checkpoints/final.pt
[ -f "$BASE" ] || { log "no base checkpoint; aborting"; exit 1; }
START=${1:-evals}   # evals | m4 | m5 | m6 | m6b | m7 : resume the chain from this stage
after() { case "$START" in evals) return 0;; esac; local s; for s in evals m4 m5 m6 m6b m7; do [ "$s" = "$START" ] && return 0; [ "$s" = "$1" ] && return 1; done; return 1; }

if after evals; then
log "base evals"
$P -u -m slm.eval.lm_eval_wrapper --checkpoint "$BASE" --tasks hellaswag,arc_easy,piqa --batch-size 16 --out runs/m3_base_8k_149m/lm_eval.json > runs/m3_base_8k_149m/lm_eval.log 2>&1
$P -u scripts/diagnose.py "$BASE" --root C:/slm-data/tokenized/v1 --source fineweb-edu-b --seq 2048 --batches 8 --mb 4 > runs/m3_base_8k_149m/diagnostics.log 2>&1
$P -u -m slm.eval.long_context --checkpoint "$BASE" --lengths 1024 2048 4096 8000 --n 4 --out runs/m3_base_8k_149m/needle.json > runs/m3_base_8k_149m/needle.log 2>&1
log "base evals done"
fi

if after m4; then
train m4_sft_149m slm.train.pretrain
fi
M4=runs/m4_sft_149m/checkpoints/final.pt
[ -f "$M4" ] || { log "M4 produced no final.pt; aborting"; exit 1; }
if after m4; then
$P -u -m slm.eval.lm_eval_wrapper --checkpoint "$M4" --tasks hellaswag,arc_easy,piqa --batch-size 16 --limit 2000 --out runs/m4_sft_149m/lm_eval.json > runs/m4_sft_149m/lm_eval.log 2>&1
fi

if after m5; then
train m5_reasoning_149m slm.train.pretrain
fi
M5=runs/m5_reasoning_149m/checkpoints/final.pt
[ -f "$M5" ] || { log "M5 produced no final.pt; aborting"; exit 1; }
if after m5; then
$P -u -m slm.eval.reasoning --checkpoint "$M5" --n 100 --gsm8k 200 --out runs/m5_reasoning_149m/reasoning_eval.json > runs/m5_reasoning_149m/reasoning_eval.log 2>&1
fi

if after m6; then
train m6_rl_arith_149m slm.train.rl
fi
M6=runs/m6_rl_arith_149m/checkpoints/final.pt
[ -f "$M6" ] || { log "M6 stage A produced no final.pt; aborting"; exit 1; }
if after m6; then
$P -u -m slm.eval.reasoning --checkpoint "$M6" --n 100 --gsm8k 200 --out runs/m6_rl_arith_149m/reasoning_eval.json > runs/m6_rl_arith_149m/reasoning_eval.log 2>&1
fi

if after m6b; then
train m6_rl_multi_149m slm.train.rl
fi
M6B=runs/m6_rl_multi_149m/checkpoints/final.pt
if [ -f "$M6B" ] && after m6b; then
  $P -u -m slm.eval.reasoning --checkpoint "$M6B" --n 100 --gsm8k 200 --out runs/m6_rl_multi_149m/reasoning_eval.json > runs/m6_rl_multi_149m/reasoning_eval.log 2>&1
fi

if after m7; then
train m7_ctx16k_149m slm.train.pretrain
fi
M7=runs/m7_ctx16k_149m/checkpoints/final.pt
if [ -f "$M7" ]; then
  $P -u -m slm.eval.long_context --checkpoint "$M7" --lengths 2048 8000 12000 16000 --n 4 --out runs/m7_ctx16k_149m/needle.json > runs/m7_ctx16k_149m/needle.log 2>&1
  $P -u -m slm.eval.lm_eval_wrapper --checkpoint "$M7" --tasks hellaswag,arc_easy,piqa --batch-size 8 --limit 2000 --out runs/m7_ctx16k_149m/lm_eval.json > runs/m7_ctx16k_149m/lm_eval.log 2>&1
fi
log "PIPELINE_DONE"

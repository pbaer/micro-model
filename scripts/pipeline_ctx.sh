#!/usr/bin/env bash
# Context curriculum, gated: waits for stage 1 (8K retrieval training) to finish, measures it properly
# (needle v2, n=16; lm-eval at the 2000-sample limit; 2K loss), and launches stage 2 (16K) only if the 8K
# gate passes. Stage 2 is then measured the same way. Stage 3 (32K) is a manual decision.
#   bash scripts/pipeline_ctx.sh              # stage 1 -> gate -> stage 2 -> gate
#   bash scripts/pipeline_ctx.sh stage1b      # wait for the tool track, then stage 1b -> gate -> stage 2 -> gate
#   bash scripts/pipeline_ctx.sh stage2       # skip straight to waiting for stage 2
set -u
cd "$(dirname "$0")/.."
P=.venv/Scripts/python.exe
log() { echo "$(date '+%H:%M:%S') $*"; }

wait_for() { while [ ! -f "$1" ]; do sleep 60; done; }

measure() {  # measure <run> <lengths...> -> prints effective context, writes needle_v2.json / lm_eval.json / diagnostics.log
  local run=$1; shift
  local ck="runs/$run/checkpoints/final.pt"
  $P -u -m slm.eval.long_context --checkpoint "$ck" --lengths "$@" --n 16 --out "runs/$run/needle_v2.json" > "runs/$run/needle_v2.log" 2>&1
  $P -u -m slm.eval.lm_eval_wrapper --checkpoint "$ck" --tasks hellaswag,arc_easy,piqa --batch-size 8 --limit 2000 --out "runs/$run/lm_eval.json" > "runs/$run/lm_eval.log" 2>&1
  $P -u scripts/diagnose.py "$ck" --root C:/slm-data/tokenized/v1 --source fineweb-edu-b --seq 2048 --batches 8 --mb 4 --no-ablation > "runs/$run/diagnostics.log" 2>&1
  $P -c "import json; print(json.load(open('runs/$run/needle_v2.json'))['effective_context'])"
}

gate() {  # gate <run> <required effective context>
  local eff; eff=$(measure "$1" "${@:3}")
  log "$1: effective context $eff (need >= $2)"
  [ "$eff" -ge "$2" ]
}

STAGE1=${STAGE1:-m7_ctx8k_retrieval_149m}   # override: STAGE1=<run> bash scripts/pipeline_ctx.sh
if [ "${1:-}" = "stage1b" ]; then
  STAGE1=m7_ctx8k_retrieval2_149m
  log "waiting for the tool track (runs/pipeline_tools.log: TOOLS_PIPELINE_DONE) before stage 1b"
  while ! grep -q "TOOLS_PIPELINE_DONE" runs/pipeline_tools.log 2>/dev/null; do sleep 120; done
  sleep 30
  log "stage 1b (m7_ctx8k_retrieval2_149m)"
  mkdir -p runs/m7_ctx8k_retrieval2_149m
  $P -u -m slm.train.pretrain --config configs/train/m7_ctx8k_retrieval2_149m.yaml > runs/m7_ctx8k_retrieval2_149m/train.log 2>&1
  log "stage 1b exited $?"
fi
if [ "${1:-}" != "stage2" ]; then
  log "waiting for $STAGE1"
  wait_for "runs/$STAGE1/checkpoints/final.pt"
  sleep 30
  if gate "$STAGE1" 8000 1024 2048 4096 6000 8000; then
    log "$STAGE1 passed the 8K gate; starting stage 2 (16K)"
    mkdir -p runs/m7_ctx16k_retrieval_149m
    $P -u -m slm.train.pretrain --config configs/train/m7_ctx16k_retrieval_149m.yaml > runs/m7_ctx16k_retrieval_149m/train.log 2>&1
    log "stage 2 exited $?"
  else
    log "$STAGE1 did NOT pass the 8K gate; stopping here (effective context stays below 8K)"
    log "CTX_PIPELINE_DONE"
    exit 2
  fi
else
  wait_for runs/m7_ctx16k_retrieval_149m/checkpoints/final.pt
  sleep 30
fi

if gate m7_ctx16k_retrieval_149m 16000 2048 4096 8000 12000 16000; then
  log "stage 2 passed the 16K gate; 32K is a manual decision (see docs/roadmap.md)"
else
  log "stage 2 did NOT pass the 16K gate; effective context is what needle_v2.json says"
fi
log "CTX_PIPELINE_DONE"

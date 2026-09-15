#!/usr/bin/env bash
# Sequence for 2026-09-15 (one GPU, never shared):
#   wait for stage 1b (m7_ctx8k_retrieval2_149m, already running) -> full needle/lm-eval/2K-loss measurement and the
#   8K gate -> tool RL second attempt (m6_rl_gsm_tools_149m, from the tool SFT) + full GSM8K eval with tools ->
#   stage 2 (16K) only if stage 1b passed its gate -> 16K measurement.
# The RL goes before stage 2 so the tool result lands early; stage 2 takes ~6.5 h.
set -u
cd "$(dirname "$0")/.."
P=.venv/Scripts/python.exe
log() { echo "$(date '+%H:%M:%S') $*"; }
wait_for() { while [ ! -f "$1" ]; do sleep 60; done; }

measure() {  # measure <run> <lengths...> -> prints effective context
  local run=$1; shift
  local ck="runs/$run/checkpoints/final.pt"
  $P -u -m slm.eval.long_context --checkpoint "$ck" --lengths "$@" --n 16 --out "runs/$run/needle_v2.json" > "runs/$run/needle_v2.log" 2>&1
  $P -u -m slm.eval.lm_eval_wrapper --checkpoint "$ck" --tasks hellaswag,arc_easy,piqa --batch-size 8 --limit 2000 --out "runs/$run/lm_eval.json" > "runs/$run/lm_eval.log" 2>&1
  $P -u scripts/diagnose.py "$ck" --root C:/slm-data/tokenized/v1 --source fineweb-edu-b --seq 2048 --batches 8 --mb 4 --no-ablation > "runs/$run/diagnostics.log" 2>&1
  $P -c "import json; print(json.load(open('runs/$run/needle_v2.json'))['effective_context'])"
}
gate() { local eff; eff=$(measure "$1" "${@:3}"); log "$1: effective context $eff (need >= $2)"; [ "$eff" -ge "$2" ]; }

eval_reasoning() {  # <run> <tools-flag>: best.pt when present (RL), else final.pt
  local run=$1 flag=$2 tag=notools ck="runs/$1/checkpoints/final.pt"
  [ -n "$flag" ] && tag=tools
  [ -f "runs/$run/checkpoints/best.pt" ] && ck="runs/$run/checkpoints/best.pt"
  $P -u -m slm.eval.reasoning --checkpoint "$ck" --n 100 --gsm8k -1 --max-new 320 $flag \
     --out "runs/$run/reasoning_eval_$tag.json" --dump "runs/$run/reasoning_dump_$tag.jsonl" > "runs/$run/reasoning_eval_$tag.log" 2>&1
  log "$run ($tag): $(grep -E 'gsm8k_test|mean accuracy' "runs/$run/reasoning_eval_$tag.log" | tr '\n' ' ')"
}

log "waiting for stage 1b (m7_ctx8k_retrieval2_149m)"
wait_for runs/m7_ctx8k_retrieval2_149m/checkpoints/final.pt
sleep 30
PASS1B=0
if gate m7_ctx8k_retrieval2_149m 8000 1024 2048 4096 6000 8000; then PASS1B=1; log "stage 1b passed the 8K gate"; else log "stage 1b did NOT pass the 8K gate"; fi

log "tool RL (second attempt)"
mkdir -p runs/m6_rl_gsm_tools_149m
$P -u -m slm.train.rl --config configs/train/m6_rl_gsm_tools_149m.yaml > runs/m6_rl_gsm_tools_149m/train.log 2>&1
log "tool RL exited $?"
if [ -f runs/m6_rl_gsm_tools_149m/checkpoints/best.pt ]; then eval_reasoning m6_rl_gsm_tools_149m "--tools"; else log "tool RL produced no best.pt"; fi
log "TOOLS_PIPELINE_DONE"

if [ "$PASS1B" = "1" ]; then
  log "stage 2 (16K)"
  mkdir -p runs/m7_ctx16k_retrieval_149m
  $P -u -m slm.train.pretrain --config configs/train/m7_ctx16k_retrieval_149m.yaml > runs/m7_ctx16k_retrieval_149m/train.log 2>&1
  log "stage 2 exited $?"
  if gate m7_ctx16k_retrieval_149m 16000 2048 4096 8000 12000 16000; then log "stage 2 passed the 16K gate; 32K is a manual decision"; else log "stage 2 did NOT pass the 16K gate"; fi
fi
log "CTX_PIPELINE_DONE"
log "DAY_PIPELINE_DONE"

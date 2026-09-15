#!/usr/bin/env bash
# Tool-use track (calculator), queued behind the context curriculum so the two never share the GPU:
#   wait for CTX_PIPELINE_DONE -> M5 tools SFT (from the real M4) -> reasoning eval with and without tools,
#   full GSM8K test -> M6 tools RL on GSM8K-train prompts -> full GSM8K test with tools.
#   bash scripts/pipeline_tools.sh [now]      # "now" skips the wait (GPU must be free)
set -u
cd "$(dirname "$0")/.."
P=.venv/Scripts/python.exe
log() { echo "$(date '+%H:%M:%S') $*"; }

if [ "${1:-}" != "now" ]; then
  log "waiting for the context pipeline (runs/pipeline_ctx.log: CTX_PIPELINE_DONE)"
  while ! grep -q "CTX_PIPELINE_DONE" runs/pipeline_ctx.log 2>/dev/null; do sleep 120; done
  sleep 60
fi

eval_reasoning() {  # eval_reasoning <run> <tools-flag> ; uses best.pt (RL, held-out best) when present, else final.pt
  local run=$1 flag=$2 tag=${2:-notools} ck="runs/$1/checkpoints/final.pt"
  [ -n "$flag" ] && tag=tools
  [ -f "runs/$run/checkpoints/best.pt" ] && ck="runs/$run/checkpoints/best.pt"
  $P -u -m slm.eval.reasoning --checkpoint "$ck" --n 100 --gsm8k -1 --max-new 320 $flag \
     --out "runs/$run/reasoning_eval_$tag.json" --dump "runs/$run/reasoning_dump_$tag.jsonl" > "runs/$run/reasoning_eval_$tag.log" 2>&1
  log "$run ($tag): $(grep -E 'gsm8k_test|mean accuracy' "runs/$run/reasoning_eval_$tag.log" | tr '\n' ' ')"
}

log "M5 tools SFT"
mkdir -p runs/m5_reasoning_tools_149m
$P -u -m slm.train.pretrain --config configs/train/m5_reasoning_tools_149m.yaml > runs/m5_reasoning_tools_149m/train.log 2>&1
[ -f runs/m5_reasoning_tools_149m/checkpoints/final.pt ] || { log "M5 tools produced no final.pt; aborting"; exit 1; }
eval_reasoning m5_reasoning_tools_149m "--tools"
eval_reasoning m5_reasoning_tools_149m ""
# reference: the no-tool M5 on the same full GSM8K test
[ -f runs/m5_reasoning_149m/reasoning_eval_notools.json ] || eval_reasoning m5_reasoning_149m ""

log "M6 tools RL"
mkdir -p runs/m6_rl_gsm_tools_149m
$P -u -m slm.train.rl --config configs/train/m6_rl_gsm_tools_149m.yaml > runs/m6_rl_gsm_tools_149m/train.log 2>&1
[ -f runs/m6_rl_gsm_tools_149m/checkpoints/best.pt ] || { log "M6 tools RL produced no best.pt; aborting"; exit 1; }
eval_reasoning m6_rl_gsm_tools_149m "--tools"
log "TOOLS_PIPELINE_DONE"

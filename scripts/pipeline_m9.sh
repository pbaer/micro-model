#!/usr/bin/env bash
# M9 post-training on the second base: chat SFT -> reasoning+tools SFT -> GRPO, each gated by its own
# measurements. Survives a session crash; each run resumes from latest.pt if relaunched. Stop a stage with
# runs/<run>/STOP.
set -u
cd "$(dirname "$0")/.."
P=.venv/Scripts/python.exe
log() { echo "$(date '+%H:%M:%S') $*"; }
run() { mkdir -p "runs/$1"; log "start $1"; $P -u -m "$2" --config "configs/train/$1.yaml" > "runs/$1/train.log" 2>&1; log "$1 exited $?"; }
measure() {  # $1 = run name; the gate numbers for the next stage
  R="runs/$1"; CK="$R/checkpoints/final.pt"
  [ -f "$CK" ] || { log "$1 produced no final.pt; aborting"; exit 1; }
  log "measuring $1"
  $P -u -m slm.eval.long_context --checkpoint "$CK" --lengths 1024 2048 3072 4096 --n 64 --batch-tokens 8192 --out $R/needle_v2.json > $R/needle_v2.log 2>&1
  $P -u -m slm.eval.lm_eval_wrapper --checkpoint "$CK" --tasks hellaswag,arc_easy,piqa --batch-size 8 --limit 2000 --out $R/lm_eval_limit2000.json > $R/lm_eval.log 2>&1
  $P -u -m slm.eval.facts --checkpoint "$CK" --out $R/facts.json > $R/facts.log 2>&1
  $P -u -m slm.eval.quality generate --run "$1" --device cuda > $R/quality_gen.log 2>&1
  $P -u -m slm.eval.quality pack --run "$1" --batch 50 >> $R/quality_gen.log 2>&1
}

[ -f runs/m9_sft_336m/checkpoints/final.pt ]  || run m9_sft_336m  slm.train.pretrain
measure m9_sft_336m
[ -f runs/m9_tool_336m/checkpoints/final.pt ] || run m9_tool_336m slm.train.pretrain
measure m9_tool_336m
# reasoning/tool-use accuracy, and the tool-use rates the RL stage is meant to move
$P -u -m slm.eval.reasoning --checkpoint runs/m9_tool_336m/checkpoints/final.pt --n 100 --gsm8k 200 --tools \
   --out runs/m9_tool_336m/reasoning_eval.json > runs/m9_tool_336m/reasoning_eval.log 2>&1
log "M9_SFT_DONE — judge the packets, then launch RL by hand once the gates are read"

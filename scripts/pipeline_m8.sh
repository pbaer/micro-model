#!/usr/bin/env bash
# Second base: phase 1 (2K stable, 7.5B) -> phase 2 (4K decay + chat/tool data, 2.5B) -> measurements.
# Survives session crashes; each run resumes from latest.pt if relaunched. Stop a phase with runs/<run>/STOP.
set -u
cd "$(dirname "$0")/.."
P=.venv/Scripts/python.exe
log() { echo "$(date '+%H:%M:%S') $*"; }
run() { mkdir -p "runs/$1"; log "start $1"; $P -u -m slm.train.pretrain --config "configs/train/$1.yaml" > "runs/$1/train.log" 2>&1; log "$1 exited $?"; }

[ -f runs/m8_base_stable_336m/checkpoints/final.pt ] || run m8_base_stable_336m
[ -f runs/m8_base_stable_336m/checkpoints/final.pt ] || { log "phase 1 produced no final.pt; aborting"; exit 1; }
[ -f runs/m8_base_4k_336m/checkpoints/final.pt ] || run m8_base_4k_336m
CK=runs/m8_base_4k_336m/checkpoints/final.pt
[ -f "$CK" ] || { log "phase 2 produced no final.pt; aborting"; exit 1; }
R=runs/m8_base_4k_336m
log "measuring the base"
$P -u -m slm.eval.long_context --checkpoint "$CK" --lengths 1024 2048 3072 4096 --n 16 --out $R/needle_v2.json > $R/needle_v2.log 2>&1
$P -u -m slm.eval.lm_eval_wrapper --checkpoint "$CK" --tasks hellaswag,arc_easy,piqa --batch-size 8 --out $R/lm_eval.json > $R/lm_eval.log 2>&1
$P -u -m slm.eval.lm_eval_wrapper --checkpoint "$CK" --tasks hellaswag,arc_easy,piqa --batch-size 8 --limit 2000 --out $R/lm_eval_limit2000.json > $R/lm_eval_limit2000.log 2>&1
$P -u scripts/diagnose.py "$CK" --root C:/slm-data/tokenized/v1 --source fineweb-edu-b --seq 2048 --batches 8 --mb 4 > $R/diagnostics.log 2>&1
log "M8_PIPELINE_DONE"

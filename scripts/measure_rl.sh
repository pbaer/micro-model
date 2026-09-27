#!/usr/bin/env bash
# Post-run measurements for an RL checkpoint (best.pt, final.pt or a step snapshot): the same hard suite as
# scripts/measure_stage.sh plus the swarm eval, written next to the run with a <tag> suffix.
#   bash scripts/measure_rl.sh m9_rl6_336m best.pt best
# Judging is separate (docs/quality_eval.md).
set -u
cd "$(dirname "$0")/.."
R="runs/$1"; CK="$R/checkpoints/$2"; T="${3:-$2}"
P=.venv/Scripts/python.exe
[ -f "$CK" ] || { echo "$CK: missing"; exit 1; }
$P -u scripts/swarm_eval.py --checkpoint "$CK" --gsm8k 50 --svamp 50 --k 16 --mode both --out "$R/swarm_eval_$T.json" > "$R/swarm_eval_$T.log" 2>&1
$P -u -m slm.eval.reasoning --checkpoint "$CK" --n 100 --gsm8k 200 --svamp 300 --tools --out "$R/reasoning_$T.json" > "$R/reasoning_$T.log" 2>&1
$P -u -m slm.eval.multiturn --checkpoint "$CK" --out "$R/multiturn_$T.json" > "$R/multiturn_$T.log" 2>&1
$P -u -m slm.eval.facts --checkpoint "$CK" --out "$R/facts_$T.json" > "$R/facts_$T.log" 2>&1
$P -u -m slm.eval.lm_eval_wrapper --checkpoint "$CK" --tasks hellaswag,arc_easy,piqa,lambada_openai,openbookqa,sciq --batch-size 8 --limit 2000 --out "$R/lm_eval_$T.json" > "$R/lm_eval_$T.log" 2>&1
$P -u -m slm.eval.long_context --checkpoint "$CK" --lengths 1024 2048 3072 4096 --n 64 --batch-tokens 8192 --out "$R/needle_$T.json" > "$R/needle_$T.log" 2>&1
$P -u -m slm.eval.quality generate --run "$1" --checkpoints "$2" --device cuda >> "$R/quality_gen.log" 2>&1
$P -u -m slm.eval.quality pack --run "$1" --batch 50 >> "$R/quality_gen.log" 2>&1
echo "MEASURE_RL_DONE $1 $2 $(date '+%H:%M')"

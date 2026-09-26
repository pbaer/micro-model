#!/usr/bin/env bash
# Post-run measurements for one SFT stage: the numbers its gate is read from.
#   bash scripts/measure_stage.sh m9_tool3_336m
# Judging is separate (docs/quality_eval.md): this writes the packets, a judge scores them, then
#   python -m slm.eval.quality ingest --run <run> --scores <file> --judge <name>
set -u
cd "$(dirname "$0")/.."
R="runs/$1"; CK="$R/checkpoints/final.pt"
P=.venv/Scripts/python.exe
[ -f "$CK" ] || { echo "$1: no final.pt"; exit 1; }
$P -u -m slm.eval.long_context --checkpoint "$CK" --lengths 1024 2048 3072 4096 --n 64 --batch-tokens 8192 --out "$R/needle_v2.json" > "$R/needle_v2.log" 2>&1
# lambada/openbookqa/sciq added 2026-09-26: the candidates that had resolution across base -> SFT -> RL.
# Dropped after measuring: MMLU, ARC-Challenge, WinoGrande (chance), BoolQ (below its 62% class prior),
# TriviaQA (floor), ASDiv (lm-eval format mismatch with think spans; SVAMP via slm.eval.reasoning instead).
$P -u -m slm.eval.lm_eval_wrapper --checkpoint "$CK" --tasks hellaswag,arc_easy,piqa,lambada_openai,openbookqa,sciq --batch-size 8 --limit 2000 --out "$R/lm_eval_limit2000.json" > "$R/lm_eval.log" 2>&1
$P -u -m slm.eval.facts --checkpoint "$CK" --out "$R/facts.json" > "$R/facts.log" 2>&1
# reasoning + tool-use rates. SVAMP (300, one-step word problems) sits between the saturated synthetic
# families and GSM8K at the floor -- the band with resolution (2026-09-26)
$P -u -m slm.eval.reasoning --checkpoint "$CK" --n 100 --gsm8k 200 --svamp 300 --tools --out "$R/reasoning_eval.json" > "$R/reasoning_eval.log" 2>&1
$P -u -m slm.eval.quality generate --run "$1" --device cuda >> "$R/quality_gen.log" 2>&1
$P -u -m slm.eval.quality pack --run "$1" --batch 50 >> "$R/quality_gen.log" 2>&1
echo "MEASURE_DONE $1 $(date '+%H:%M')"

#!/usr/bin/env bash
# Every applicable eval for one external comparison model (slm.eval.external), serially, with the limits
# scripts/measure_stage.sh and scripts/measure_rl.sh use for our checkpoints. Weights must already be on disk
# (python -m slm.eval.external download <name>); nothing here touches the network.
#   bash scripts/measure_external.sh smollm2-360m-instruct
# Results land in runs/ext_<name>/ under the file names our runs use, so the Evals tab picks them up.
# n/a, skipped with an echo: chat-only evals (reasoning, multiturn, quality, swarm) for base models. Tool use,
# sandbox verification and the selector are n/a for every external model (null fields inside the files).
# Judging is separate (docs/quality_eval.md): this writes the packets, a judge scores them, then
#   python -m slm.eval.quality ingest --run ext_<name> --scores <file> --judge <name>
set -u
cd "$(dirname "$0")/.."
N="$1"; R="runs/ext_$N"
P=.venv/Scripts/python.exe
$P -m slm.eval.external model-json "$N" || exit 1
CHAT=$($P -c "import sys; from slm.eval.external import get; print(int(get(sys.argv[1]).is_chat))" "$N") || exit 1
$P -u -m slm.eval.lm_eval_wrapper --external "$N" --tasks hellaswag,arc_easy,piqa,lambada_openai,openbookqa,sciq --batch-size 8 --limit 2000 --out "$R/lm_eval_limit2000.json" > "$R/lm_eval.log" 2>&1
# completion form for every model, as measure_stage.sh runs it for every checkpoint of ours
$P -u -m slm.eval.facts --external "$N" --out "$R/facts.json" > "$R/facts.log" 2>&1
# lengths are counted in the model's own tokens; lengths above its position table are recorded as skipped (n/a)
$P -u -m slm.eval.long_context --external "$N" --lengths 1024 2048 3072 4096 --n 64 --batch-tokens 8192 --out "$R/needle_v2.json" > "$R/needle_v2.log" 2>&1
if [ "$CHAT" = "1" ]; then
  # no --tools: the sandbox is our protocol (tool-use fields are null)
  $P -u -m slm.eval.reasoning --external "$N" --n 100 --gsm8k 200 --svamp 300 --out "$R/reasoning_eval.json" > "$R/reasoning_eval.log" 2>&1
  $P -u -m slm.eval.multiturn --external "$N" --out "$R/multiturn.json" > "$R/multiturn.log" 2>&1
  $P -u scripts/swarm_eval.py --external "$N" --gsm8k 50 --svamp 50 --k 16 --out "$R/swarm_eval.json" > "$R/swarm_eval.log" 2>&1
  $P -u -m slm.eval.quality generate --external "$N" --device cuda >> "$R/quality_gen.log" 2>&1
  $P -u -m slm.eval.quality pack --run "ext_$N" --batch 50 >> "$R/quality_gen.log" 2>&1
else
  echo "$N: base model -- reasoning, multiturn, swarm and judged quality are chat evals: n/a (skipped)"
fi
echo "MEASURE_EXTERNAL_DONE $N $(date '+%H:%M')"

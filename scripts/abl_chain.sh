#!/usr/bin/env bash
# Depth-vs-width ablation chain (2026-09-27): four 149M runs back to back, each followed by the four public
# benchmarks on its final.pt, then one table. Trigger checked against a real log: pretrain prints "FINISHED at".
cd "$(dirname "$0")/.."
P=.venv/Scripts/python.exe
for run in m10_abl_wide_149m m10_abl_deep_149m m10_abl_deeper_149m m10_abl_wide_noz_149m; do
  R="runs/$run"; mkdir -p "$R"
  if [ ! -f "$R/checkpoints/final.pt" ]; then
    echo "TRAIN $run $(date '+%H:%M')"
    $P -u -m slm.train.pretrain --config "configs/train/$run.yaml" > "$R/train.log" 2>&1 || { echo "TRAIN_FAILED $run $(date '+%H:%M')"; exit 1; }
  fi
  [ -f "$R/lm_eval_limit2000.json" ] || $P -u -m slm.eval.lm_eval_wrapper --checkpoint "$R/checkpoints/final.pt" --tasks hellaswag,arc_easy,piqa,lambada_openai --batch-size 8 --limit 2000 --out "$R/lm_eval_limit2000.json" > "$R/lm_eval.log" 2>&1
  echo "DONE $run $(date '+%H:%M')"
done
echo "ABL_CHAIN_DONE $(date '+%H:%M')"

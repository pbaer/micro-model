#!/usr/bin/env bash
# Model against model in the competitive arena tasks (docs/results.md §28): each pairing plays twice with the sides
# swapped (tag is asymmetric: chasers vs runners), 3 seeds x 3 games at 8 robots; plus M10's mirror match as baseline.
set -u
P=.venv/Scripts/python.exe
M10=runs/m10_rl_336m/checkpoints/final.pt
M9=runs/m9_rl6_336m/checkpoints/final.pt
OUT=${OUT:-runs/arena/versus}
mkdir -p $OUT
run() {  # name, side A, side B
  $P -u scripts/arena_eval.py --checkpoint "$2" --opponent "$3" --tasks team_relay,tag,gold_rush --agents 8 --seeds 3 --turns 12 \
     --out $OUT/$1.json > $OUT/$1.log 2>&1
  echo "done $1"
}
run m10_vs_m10 $M10 $M10
for X in external:smollm2-360m-instruct external:qwen2.5-0.5b-instruct $M9; do
  tag=$(echo "$X" | sed 's#external:##; s#runs/##; s#/checkpoints/final.pt##')
  run "m10_vs_$tag" $M10 "$X"
  run "${tag}_vs_m10" "$X" $M10
done
echo ALL DONE

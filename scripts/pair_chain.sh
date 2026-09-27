#!/usr/bin/env bash
# The tournament chain (2026-09-27): untrained pairwise accuracy -> pairwise SFT -> trained pairwise accuracy
# -> RL run 7 -> hard suite + swarm eval (both modes) on best.pt and final.pt. Waits for the GPU job before it.
# Trigger patterns were checked against real logs: pretrain prints "FINISHED at <tokens>", rl "FINISHED at step".
cd "$(dirname "$0")/.."
P=.venv/Scripts/python.exe
until [ -f runs/m9_rl5_336m/swarm_eval_s200_both.json ]; do sleep 30; done
echo "PAIR_EVAL_UNTRAINED $(date '+%H:%M')"
$P -u scripts/pair_eval.py --checkpoint runs/m9_rl6_336m/checkpoints/final.pt --n 300 --out runs/m9_rl6_336m/pair_eval_final.json > runs/m9_rl6_336m/pair_eval_final.log 2>&1
[ -f runs/m9_pair_336m/checkpoints/final.pt ] || {
echo "SFT_PAIR $(date '+%H:%M')"
mkdir -p runs/m9_pair_336m
$P -u -m slm.train.pretrain --config configs/train/m9_pair_336m.yaml > runs/m9_pair_336m/train.log 2>&1 || { echo "SFT_FAILED $(date '+%H:%M')"; exit 1; }
}
[ -f runs/m9_pair_336m/checkpoints/final.pt ] || { echo "NO_SFT_FINAL"; exit 1; }
echo "PAIR_EVAL_SFT $(date '+%H:%M')"
$P -u scripts/pair_eval.py --checkpoint runs/m9_pair_336m/checkpoints/final.pt --n 300 --out runs/m9_pair_336m/pair_eval_final.json > runs/m9_pair_336m/pair_eval_final.log 2>&1
$P -u scripts/swarm_eval.py --checkpoint runs/m9_pair_336m/checkpoints/final.pt --gsm8k 50 --svamp 50 --k 16 --mode both --out runs/m9_pair_336m/swarm_eval_final.json > runs/m9_pair_336m/swarm_eval_final.log 2>&1
echo "RL7 $(date '+%H:%M')"
mkdir -p runs/m9_rl7_336m
$P -u -m slm.train.rl --config configs/train/m9_rl7_336m.yaml > runs/m9_rl7_336m/train.log 2>&1 || { echo "RL7_FAILED $(date '+%H:%M')"; exit 1; }
until grep -qE "FINISHED at step|stopped at step" runs/m9_rl7_336m/train.log && [ -f runs/m9_rl7_336m/checkpoints/final.pt ]; do sleep 30; done
echo "RL7_FINISHED $(date '+%H:%M')"
for ck in final best; do
  $P -u scripts/pair_eval.py --checkpoint runs/m9_rl7_336m/checkpoints/$ck.pt --n 300 --out runs/m9_rl7_336m/pair_eval_$ck.json > runs/m9_rl7_336m/pair_eval_$ck.log 2>&1
  bash scripts/measure_rl.sh m9_rl7_336m $ck.pt $ck
done
echo "PAIR_CHAIN_DONE $(date '+%H:%M')"

#!/usr/bin/env bash
# After RL run 6: measure best.pt and final.pt (hard suite + swarm eval + judge packets), serially on the GPU.
cd "$(dirname "$0")/.."
until grep -qE "FINISHED at step|finished at step|stopped at step|collapse guard" runs/m9_rl6_336m/train.log 2>/dev/null && [ -f runs/m9_rl6_336m/checkpoints/final.pt ]; do sleep 60; done
sleep 30
echo "RL6_FINISHED $(date '+%H:%M')"
bash scripts/measure_rl.sh m9_rl6_336m best.pt best
bash scripts/measure_rl.sh m9_rl6_336m final.pt final
# the SFT-only base, so the comparison separates what the format SFT gave from what RL added
.venv/Scripts/python.exe -u scripts/swarm_eval.py --checkpoint runs/m9_select_336m/checkpoints/final.pt --gsm8k 50 --svamp 50 --k 16 --out runs/m9_select_336m/swarm_eval_final.json > runs/m9_select_336m/swarm_eval_final.log 2>&1
echo "AFTER_RL6_DONE $(date '+%H:%M')"

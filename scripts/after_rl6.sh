#!/usr/bin/env bash
# After RL run 6: measure best.pt and final.pt (hard suite + swarm eval + judge packets), serially on the GPU.
cd "$(dirname "$0")/.."
until grep -qE "finished at step|stopped at step" runs/m9_rl6_336m/train.log 2>/dev/null && [ -f runs/m9_rl6_336m/checkpoints/final.pt ]; do sleep 60; done
sleep 30
echo "RL6_FINISHED $(date '+%H:%M')"
bash scripts/measure_rl.sh m9_rl6_336m best.pt best
bash scripts/measure_rl.sh m9_rl6_336m final.pt final
echo "AFTER_RL6_DONE $(date '+%H:%M')"

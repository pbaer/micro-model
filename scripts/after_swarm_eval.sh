#!/usr/bin/env bash
# The goal-5 chain (2026-09-26): baseline swarm eval -> selection data -> selection SFT -> RL run 6.
# Each step waits for the previous one's artefact and stops the chain on a non-zero exit; steps whose artefact
# already exists are skipped, so the chain can be relaunched after a fix (v2 of the SFT, 2026-09-26).
cd "$(dirname "$0")/.."
P=.venv/Scripts/python.exe
until [ -f runs/m9_rl5_336m/swarm_eval_s200.json ]; do sleep 30; done
[ -f /c/slm-data/sft/v1/select-sft/manifest.json ] || {
echo "BUILD_SELECT $(date '+%H:%M')"
$P -u -m slm.rl.synth_select --checkpoint runs/m9_rl5_336m/checkpoints/step_00200.pt --n-gsm8k 600 --n-svamp 250 --n-synth 250 --k 8 --name select-sft > runs/m9_rl5_336m/synth_select.log 2>&1 || { echo "BUILD_FAILED $(date '+%H:%M')"; exit 1; }
}
[ -f /c/slm-data/sft/v1/select-sft/manifest.json ] || { echo "NO_MANIFEST"; exit 1; }
[ -f runs/m9_select_336m/checkpoints/final.pt ] || {
echo "SFT_SELECT $(date '+%H:%M')"
mkdir -p runs/m9_select_336m
$P -u -m slm.train.pretrain --config configs/train/m9_select_336m.yaml > runs/m9_select_336m/train.log 2>&1 || { echo "SFT_FAILED $(date '+%H:%M')"; exit 1; }
}
[ -f runs/m9_select_336m/checkpoints/final.pt ] || { echo "NO_SFT_FINAL"; exit 1; }
echo "RL6 $(date '+%H:%M')"
mkdir -p runs/m9_rl6_336m
$P -u -m slm.train.rl --config configs/train/m9_rl6_336m.yaml > runs/m9_rl6_336m/train.log 2>&1 || { echo "RL6_FAILED $(date '+%H:%M')"; exit 1; }
echo "CHAIN_DONE $(date '+%H:%M')"

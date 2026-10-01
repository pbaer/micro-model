#!/usr/bin/env bash
# The four chat-diagnostic kinds (slm.eval.multiturn: recall, recall_absent, revise, sysrule) at n=64 per kind on the
# M9 output and the three local chat comparison models, so the M10 stage-C numbers have apples-to-apples rows.
cd "$(dirname "$0")/.."
P=.venv/Scripts/python.exe
$P -u -m slm.eval.multiturn --checkpoint runs/m9_rl6_336m/checkpoints/final.pt --n 64 --out runs/m9_rl6_336m/multiturn_v2_final.json > runs/m9_rl6_336m/multiturn_v2_final.log 2>&1
for n in smollm2-360m-instruct qwen2.5-0.5b-instruct smollm2-135m-instruct; do
  $P -u -m slm.eval.multiturn --external "$n" --n 64 --out "runs/ext_$n/multiturn.json" > "runs/ext_$n/multiturn.log" 2>&1
done
echo "CHAT_DIAG_BASELINES_DONE $(date '+%H:%M')"

#!/usr/bin/env bash
# Rebuild the generated SFT sets after the list-answer fix (slm/data/answers.py humanize).
#
# These sets carried their gold as the sandbox produced it, so a list answer reached the natural (non-marker)
# templates as a Python repr: 6.7% of synthetic-python-tools read "The answer is ['tape', 'binder', 'stapler']."
# M9 stage B v3 learned it and answered "List the days of the week" with "So the answer is ['sunday', ...]" --
# judged pattern 4.00 -> 1.67, the largest single piece of that stage's regression.
#
# Same n/seed/corpus as the originals, so the humanize change is the only difference.
cd /d/dev/micro-model
P=.venv/Scripts/python.exe
T=C:/slm-data/tokenizer/v1
set -x
$P -u -m slm.rl.synth_python    --tokenizer $T --n 170000 --seed 0 > /dev/null 2>&1
$P -u -m slm.rl.synth           --tokenizer $T --n 40000  --seed 0 > /dev/null 2>&1
$P -u -m slm.rl.synth           --tokenizer $T --n 40000  --seed 0 --tools > /dev/null 2>&1
$P -u -m slm.rl.synth_multiturn --tokenizer $T --n 20000  --seed 0 > /dev/null 2>&1
set +x
echo "SYNTH_REBUILD_DONE $(date '+%H:%M')"

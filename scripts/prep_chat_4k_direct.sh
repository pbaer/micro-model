#!/usr/bin/env bash
# Chat rehearsal sets for M9 stage B v2: same public conversations as the -4k-think sets, but 45% of them get a
# short prose think span instead of an empty one (slm/data/direct_think.py). Stage B v1 had no example of
# "think in prose, answer, no tool" for ordinary chat, and its model reached for <|python_call|> on 13-17 of the
# 35 judged prompts -- every category that called a tool regressed, every category that did not held or improved.
cd /d/dev/micro-model
P=.venv/Scripts/python.exe
T=C:/slm-data/tokenizer/v1
for s in smol-magpie-ultra openhermes-100k; do
  echo "=== smoltalk-$s"
  $P -u -m slm.data.sft "smoltalk-$s" --tokenizer $T --max-len 4096 --think-required --direct-think 0.45 \
     --name "smoltalk-$s-4k-direct" 2>&1 | grep -v -e FutureWarning -e "obj = getattr" | tail -2
done
echo "CHAT_4K_DIRECT_DONE $(date '+%H:%M')"

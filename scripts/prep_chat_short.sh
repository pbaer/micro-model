#!/usr/bin/env bash
# Short-prompt no-tool rehearsal for M9 stage B v3. Mixture weights are in tokens, but "do I reach for the tool?"
# is decided once per conversation -- and a synthetic tool conversation is ~180 tokens against a chat
# conversation's ~400-1470. v2's mixture was 51:1 in favour of calling the tool on short question-shaped prompts
# even at a 33.5% token share, which is why misfire only moved 0.464 -> 0.429. These sets hold nothing but short,
# non-computational prompts, every one with a prose think span and no tool call, so a ~0.15 token share buys
# ~70k no-tool decisions instead of ~5k.
cd /d/dev/micro-model
P=.venv/Scripts/python.exe
T=C:/slm-data/tokenizer/v1
for s in smol-magpie-ultra openhermes-100k systemchats-30k smol-constraints everyday-conversations; do
  echo "=== smoltalk-$s"
  $P -u -m slm.data.sft "smoltalk-$s" --tokenizer $T --max-len 4096 --think-required \
     --direct-think 1.0 --short-only 20 --name "smoltalk-$s-short" 2>&1 | grep -v -e FutureWarning -e "obj = getattr" | tail -2
done
echo "CHAT_SHORT_DONE $(date '+%H:%M')"

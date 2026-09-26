#!/usr/bin/env bash
# Chat SFT sets at 4K with a mandatory (usually empty) think span -- the M9 stage A data.
#
# The span is opened on every assistant turn and left empty for ordinary chat, so the chat format is
# identical from the first SFT stage through reasoning SFT and RL: the model always emits <|think|>...
# <|/think|>, and later stages only change what goes inside it. Nothing about the format shifts under the
# model mid-chain, which is what made stage A's gains survive into stage B.
#
# Builds the five sets named by configs/train/m9_sft_336m.yaml.
cd "$(dirname "$0")/.."
P=.venv/Scripts/python.exe
T=C:/slm-data/tokenizer/v1
for s in smol-magpie-ultra openhermes-100k systemchats-30k smol-constraints everyday-conversations; do
  echo "=== smoltalk-$s"
  $P -u -m slm.data.sft "smoltalk-$s" --tokenizer $T --max-len 4096 --think-required \
     --name "smoltalk-$s-4k-think" 2>&1 | grep -v -e FutureWarning -e "obj = getattr" | tail -2
done
echo "CHAT_4K_THINK_DONE $(date '+%H:%M')"

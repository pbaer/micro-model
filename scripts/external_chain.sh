#!/usr/bin/env bash
# Score the local comparison models on every suite, serially on the GPU, after the tournament chain is done.
# Chat models first (they run the most evals); then the bases. ~1 h per chat model.
cd "$(dirname "$0")/.."
until grep -q "PAIR_CHAIN_DONE\|RL7_FAILED\|SFT_FAILED" runs/m9_rl6_336m/pair_chain.log 2>/dev/null; do sleep 120; done
echo "EXTERNAL_START $(date '+%H:%M')"
for n in smollm2-360m-instruct qwen2.5-0.5b-instruct smollm2-135m-instruct smollm2-360m qwen2.5-0.5b smollm2-135m gpt2-medium; do
  mkdir -p "runs/ext_$n"
  bash scripts/measure_external.sh "$n" > "runs/ext_$n/measure.log" 2>&1
  tail -1 "runs/ext_$n/measure.log"
done
echo "EXTERNAL_CHAIN_DONE $(date '+%H:%M')"

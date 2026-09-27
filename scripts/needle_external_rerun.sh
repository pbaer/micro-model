#!/usr/bin/env bash
# Re-run the needle eval for the chat comparison models with the chat-template prompt (2026-09-27 fix), after
# the external chain has finished with the GPU. Base models are unaffected (completion form is their native one).
cd "$(dirname "$0")/.."
P=.venv/Scripts/python.exe
until grep -q "EXTERNAL_CHAIN_DONE" runs/external_chain.log 2>/dev/null; do sleep 120; done
for n in smollm2-360m-instruct qwen2.5-0.5b-instruct smollm2-135m-instruct; do
  R="runs/ext_$n"
  [ -f "$R/needle_v2.json" ] && mv "$R/needle_v2.json" "$R/needle_v2_completion_form.json.bak"
  $P -u -m slm.eval.long_context --external "$n" --lengths 1024 2048 3072 4096 --n 64 --batch-tokens 8192 --out "$R/needle_v2.json" > "$R/needle_v2.log" 2>&1
  echo "NEEDLE_RERUN $n $(date '+%H:%M')"
done
echo "NEEDLE_RERUN_DONE $(date '+%H:%M')"

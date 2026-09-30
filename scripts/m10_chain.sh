#!/usr/bin/env bash
# M10 chain (2026-09-30): prose-sprinkle continuation of the M8 base, its floor gate, then the M9 post-training
# recipe replayed (stage A -> B -> C) with the hard suite after each stage. Every step is skipped when its
# artefact exists, so the chain can be relaunched after a fix. Trigger pattern checked against real logs:
# pretrain prints "FINISHED at <tokens>", rl "FINISHED at step".
cd "$(dirname "$0")/.."
P=.venv/Scripts/python.exe
train() {  # train <run> <module>
  local run="$1" mod="$2"; mkdir -p "runs/$run"
  [ -f "runs/$run/checkpoints/final.pt" ] && return 0
  echo "TRAIN $run $(date '+%H:%M')"
  $P -u -m "$mod" --config "configs/train/$run.yaml" > "runs/$run/train.log" 2>&1 || { echo "TRAIN_FAILED $run $(date '+%H:%M')"; exit 1; }
  [ -f "runs/$run/checkpoints/final.pt" ] || { echo "NO_FINAL $run"; exit 1; }
}
until [ -f /c/slm-data/tokenized/v1/gutenberg-pg19/manifest.json ]; do sleep 120; done
train m10_base_prose_336m slm.train.pretrain
# the floor gate: benchmarks, facts and needle on the continued base (compare to m8_base_4k final.pt, results.md §20)
R=runs/m10_base_prose_336m; CK=$R/checkpoints/final.pt
[ -f "$R/lm_eval_limit2000.json" ] || $P -u -m slm.eval.lm_eval_wrapper --checkpoint "$CK" --tasks hellaswag,arc_easy,piqa,lambada_openai,openbookqa,sciq --batch-size 8 --limit 2000 --out "$R/lm_eval_limit2000.json" > "$R/lm_eval.log" 2>&1
[ -f "$R/facts.json" ] || $P -u -m slm.eval.facts --checkpoint "$CK" --out "$R/facts.json" > "$R/facts.log" 2>&1
[ -f "$R/needle_v2.json" ] || $P -u -m slm.eval.long_context --checkpoint "$CK" --lengths 1024 2048 3072 4096 --n 64 --batch-tokens 8192 --out "$R/needle_v2.json" > "$R/needle_v2.log" 2>&1
echo "BASE_GATE_MEASURED $(date '+%H:%M')"
train m10_sft_336m slm.train.pretrain
bash scripts/measure_stage.sh m10_sft_336m
train m10_tool_336m slm.train.pretrain
bash scripts/measure_stage.sh m10_tool_336m
train m10_rl_336m slm.train.rl
bash scripts/measure_rl.sh m10_rl_336m final.pt final
bash scripts/measure_rl.sh m10_rl_336m best.pt best
echo "M10_CHAIN_DONE $(date '+%H:%M')"

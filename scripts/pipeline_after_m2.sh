#!/usr/bin/env bash
# Runs the full post-pretraining pipeline once M2 finishes: evals -> M4 SFT -> M5 reasoning SFT -> RL stage A.
# Each stage only starts if the previous checkpoint exists. Stop any stage with runs/<run>/STOP.
set -u
cd "$(dirname "$0")/.."
P=.venv/Scripts/python.exe
log() { echo "$(date '+%H:%M:%S') $*"; }

until grep -q "M2_EXIT" runs/m2_base_149m_1b/train.log; do sleep 60; done
log "M2 exited"
M2=runs/m2_base_149m_1b/checkpoints/final.pt
[ -f "$M2" ] || { log "no M2 final.pt; aborting"; exit 1; }

log "lm-eval on M2"
$P -u -m slm.eval.lm_eval_wrapper --checkpoint "$M2" --tasks hellaswag,arc_easy,piqa --batch-size 16 --out runs/m2_base_149m_1b/lm_eval.json > runs/m2_base_149m_1b/lm_eval.log 2>&1
log "diagnostics on M2"
$P -u scripts/diagnose.py "$M2" --root C:/slm-data/tokenized/v1 --source fineweb-edu --seq 2048 --batches 8 --mb 4 > runs/m2_base_149m_1b/diagnostics.log 2>&1
log "needle eval on M2 (trained context 2K so far; probes 1K-8K)"
$P -u -m slm.eval.long_context --checkpoint "$M2" --lengths 1024 2048 4096 8192 --n 4 --out runs/m2_base_149m_1b/needle.json > runs/m2_base_149m_1b/needle.log 2>&1

log "M4 instruct SFT"
mkdir -p runs/m4_sft_149m
$P -u -m slm.train.pretrain --config configs/train/sft_149m.yaml > runs/m4_sft_149m/train.log 2>&1
M4=runs/m4_sft_149m/checkpoints/final.pt
[ -f "$M4" ] || { log "M4 produced no final.pt; aborting"; exit 1; }
$P -u -m slm.eval.lm_eval_wrapper --checkpoint "$M4" --tasks hellaswag,arc_easy,piqa --batch-size 16 --limit 2000 --out runs/m4_sft_149m/lm_eval.json > runs/m4_sft_149m/lm_eval.log 2>&1

log "M5 reasoning SFT"
mkdir -p runs/m5_reasoning_149m
$P -u -m slm.train.pretrain --config configs/train/reasoning_sft_149m.yaml > runs/m5_reasoning_149m/train.log 2>&1
M5=runs/m5_reasoning_149m/checkpoints/final.pt
[ -f "$M5" ] || { log "M5 produced no final.pt; aborting"; exit 1; }
log "reasoning benchmark (pre-RL)"
$P -u -m slm.eval.reasoning --checkpoint "$M5" --n 100 --gsm8k 200 --out runs/m5_reasoning_149m/reasoning_eval.json > runs/m5_reasoning_149m/reasoning_eval.log 2>&1

log "M6 RL stage A"
mkdir -p runs/m6_rl_arith_149m
$P -u -m slm.train.rl --config configs/train/rl_arith_149m.yaml > runs/m6_rl_arith_149m/train.log 2>&1
M6=runs/m6_rl_arith_149m/checkpoints/final.pt
if [ -f "$M6" ]; then
  log "reasoning benchmark (post-RL)"
  $P -u -m slm.eval.reasoning --checkpoint "$M6" --n 100 --gsm8k 200 --out runs/m6_rl_arith_149m/reasoning_eval.json > runs/m6_rl_arith_149m/reasoning_eval.log 2>&1
fi
log "PIPELINE_DONE"

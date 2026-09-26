#!/usr/bin/env bash
# Curated backup of the checkpoints that matter, plus everything needed to interpret them.
#
# runs/ is 241 GB and untracked, but a 336M checkpoint is only 0.63 GB: the whole chain that produced the
# current model fits in ~3 GB. This copies each stage's chosen checkpoint with its config, metrics, eval
# results and judged-quality records, so a lost D: drive costs GPU time but not the models or their numbers.
# D: and C: are separate volumes; this is off-drive, NOT off-machine. Real off-site backup is still open.
set -u
cd "$(dirname "$0")/.."
DEST="${1:-C:/slm-backup}"
stamp=$(date '+%Y-%m-%d')
echo "backing up to $DEST/$stamp"
copy_run() {  # copy_run <run> <checkpoint file>
  local run=$1
  local ck=$2
  local d="$DEST/$stamp/$run"
  mkdir -p "$d/checkpoints"
  cp "runs/$run/checkpoints/$ck" "$d/checkpoints/" 2>/dev/null || { echo "  MISSING runs/$run/checkpoints/$ck"; return; }
  for f in checkpoints/index.json run.json metrics.jsonl records.jsonl report.html; do
    [ -f "runs/$run/$f" ] && cp "runs/$run/$f" "$d/$(basename "$f")" 2>/dev/null
  done
  cp runs/$run/*.json "$d/" 2>/dev/null   # needle, lm_eval, facts, reasoning
  [ -d "runs/$run/quality" ] && cp -r "runs/$run/quality" "$d/quality" 2>/dev/null
  [ -f "configs/train/$run.yaml" ] && cp "configs/train/$run.yaml" "$d/"
  echo "  $run/$ck -> $(du -sh "$d" | cut -f1)"
}
copy_run m8_base_4k_336m final.pt     # the base
copy_run m9_sft_336m     final.pt     # stage A, chat SFT
copy_run m9_tool4_336m   final.pt     # stage B, tools SFT
copy_run m9_rl5_336m     step_00200.pt  # stage C, GRPO run 5 step 200 -- the M9 output (2026-09-26, results.md 16e)
copy_run m9_rl5_336m     best.pt      # run 5 best.pt (step 150), kept beside it
copy_run m9_rl4_336m     best.pt      # stage C run 4: the best pure-chat judged checkpoint, kept for comparison
mkdir -p "$DEST/$stamp/_meta"
cp docs/results.md docs/log.md docs/roadmap.md "$DEST/$stamp/_meta/" 2>/dev/null
git rev-parse HEAD > "$DEST/$stamp/_meta/git_commit.txt"
git log --oneline -20 >> "$DEST/$stamp/_meta/git_commit.txt"
cp C:/slm-data/tokenizer/v1/* "$DEST/$stamp/_meta/tokenizer_v1/" 2>/dev/null || { mkdir -p "$DEST/$stamp/_meta/tokenizer_v1"; cp -r C:/slm-data/tokenizer/v1/* "$DEST/$stamp/_meta/tokenizer_v1/"; }
echo "total: $(du -sh "$DEST/$stamp" | cut -f1)"
echo "BACKUP_DONE $(date '+%H:%M')"

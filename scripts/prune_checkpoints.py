"""Delete checkpoints that have no remaining purpose; keep what a future run could start from or compare to.

    .venv/Scripts/python.exe scripts/prune_checkpoints.py            # dry run, prints the plan
    .venv/Scripts/python.exe scripts/prune_checkpoints.py --apply    # delete

runs/ reached 233 GB of .pt files on a 1.9 TB volume that was 97% full. Almost none of it is load-bearing:

  final.pt / best.pt   the model a stage produced. Small (0.63 GB at 336M) and the only thing another run
                       initialises from or is compared against. KEEP, always.
  snap_*.pt step_*.pt  per-milestone weights. They exist so a run's quality curve can be re-derived, but the
                       judged scores are already in quality/summary.json and the metrics in metrics.jsonl,
                       so the curve survives without them. DROP.
  latest.pt            weights + optimizer + data-stream cursors, ~3x the model. Only a resume needs it, and
                       only a run that might be CONTINUED (not just initialised from) needs the cursors --
                       `init_loader_from` points here, and without it a continuation silently re-reads the
                       parent's data (docs/log.md 2026-09-19). KEEP for the runs in CONTINUABLE, drop elsewhere.
  latest.prev.pt       the previous latest, kept only so a crash mid-write cannot lose both. DROP once a run
                       has finished.

Nothing here is recoverable, so the dry run is the default and the plan is printed per run before anything goes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

G = 1073741824

# Runs a future job might continue rather than merely initialise from: pretraining phases whose data-stream
# cursors still matter. Everything else can be started from its final.pt.
CONTINUABLE = {"m8_base_4k_336m"}

# Runs whose weights have no remaining purpose at all, so even final/best goes. Both are kept as run directories
# -- their metrics.jsonl, train.log and report.html are the record, and that is what they are referenced for.
#   m6_rl_gsm_tools_try1_149m  the m6 collapse (entropy -> 5.34, kl 0.277, malformed 0.70). It is the reference
#                              for what a REAL collapse looks like, cited when the m9 guard was fixed, but the
#                              evidence is the metrics, not the weights.
#   m8_base_4k_336m_stub       a 55-second aborted start (524K tokens, val inf), set aside deliberately.
NO_KEEP = {"m6_rl_gsm_tools_try1_149m", "m8_base_4k_336m_stub",
           "m9_pair_336m", "m9_rl7_336m"}  # the tournament ladder (results.md §19): format SFT + 100 RL steps, nothing to reuse

# Checkpoints kept by name beyond final/best: a run's chosen output when it is a milestone snapshot rather than
# best.pt. m9_rl5_336m step 200 is the M9 output (results.md 16e): judged 4.07 and multi-turn recall 0.562
# against best.pt's 3.84 / 0.484, everything else tied.
KEEP_EXTRA = {"m9_rl5_336m": {"step_00200.pt"}, "m9_rl6_336m": {"final.pt"}}


def classify(run: Path) -> tuple[list[Path], list[Path]]:
    """(keep, drop) for one run's checkpoint files."""
    ck = run / "checkpoints"
    if not ck.is_dir():
        return [], []
    keep, drop = [], []
    if run.name in NO_KEEP:
        return [], sorted(ck.glob("*.pt"))
    for f in sorted(ck.glob("*.pt")):
        n = f.name
        if n in ("final.pt", "best.pt") or n in KEEP_EXTRA.get(run.name, set()):
            keep.append(f)
        elif n == "latest.pt" and run.name in CONTINUABLE:
            keep.append(f)
        else:  # snap_*, step_*, latest.prev.pt, and latest.pt outside CONTINUABLE
            drop.append(f)
    return keep, drop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()

    total_keep = total_drop = 0
    plan = []
    for run in sorted(Path(a.runs).iterdir()):
        if not run.is_dir():
            continue
        keep, drop = classify(run)
        if not keep and not drop:
            continue
        ks = sum(f.stat().st_size for f in keep)
        ds = sum(f.stat().st_size for f in drop)
        total_keep += ks
        total_drop += ds
        plan.append((run, keep, drop, ks, ds))

    print(f"{'run':<28}{'keep':>18}{'drop GB':>9}")
    for run, keep, drop, ks, ds in plan:
        names = ", ".join(f.name.replace(".pt", "") for f in keep) or "-- NOTHING --"
        print(f"{run.name:<28}{names:>18}{ds / G:>9.1f}")
        if not keep and run.name not in NO_KEEP:
            print(f"    !! {run.name} would keep no checkpoint at all; skipping its deletions", file=sys.stderr)
    print(f"\nkeep {total_keep / G:.1f} GB, delete {total_drop / G:.1f} GB")

    if not a.apply:
        print("\ndry run; pass --apply to delete")
        return
    freed = 0
    for run, keep, drop, ks, ds in plan:
        if not keep and run.name not in NO_KEEP:  # never strip a run to nothing unless it is listed
            continue
        for f in drop:
            sz = f.stat().st_size
            try:
                f.unlink()
                freed += sz
            except OSError as e:
                print(f"  could not delete {f}: {e}")
    print(f"freed {freed / G:.1f} GB")


if __name__ == "__main__":
    main()

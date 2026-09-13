"""Backfill checkpoints/index.json for runs created before the trainer wrote it (exact tokens from
milestone/eval records; snapshot names are matched through snapshot_name()).

    python scripts/backfill_ckpt_index.py runs/m1_tinystories_26m runs/m2_base_149m
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slm.utils.checkpoint import snapshot_name, update_index  # noqa: E402
from slm.utils.logging import MetricsLogger  # noqa: E402

for run in sys.argv[1:]:
    d = Path(run)
    recs = MetricsLogger.read(d / "metrics.jsonl")
    evals = [r for r in recs if r["kind"] == "eval"]
    miles = [r for r in recs if r["kind"] == "milestone"]
    ck = d / "checkpoints"
    for m in miles:
        name = snapshot_name(m["tokens"])
        if (ck / name).exists():
            last_eval = [e for e in evals if e["tokens"] <= m["tokens"]]
            update_index(ck, name, kind="snapshot", tokens=m["tokens"], update=m.get("update"), val_loss=last_eval[-1]["val_loss"] if last_eval else None)
    if evals:
        best = min(evals, key=lambda e: e["val_loss"])
        if (ck / "best.pt").exists():
            update_index(ck, "best.pt", kind="best", tokens=best["tokens"], update=best.get("update"), val_loss=best["val_loss"])
        if (ck / "final.pt").exists():
            update_index(ck, "final.pt", kind="final", tokens=evals[-1]["tokens"], update=evals[-1].get("update"), val_loss=evals[-1]["val_loss"])
    last_ck = [r for r in recs if r["kind"] == "checkpoint"]
    if last_ck and (ck / "latest.pt").exists():
        update_index(ck, "latest.pt", kind="latest", tokens=last_ck[-1]["tokens"])
    print(run, "->", len(json.loads((ck / "index.json").read_text())), "entries")

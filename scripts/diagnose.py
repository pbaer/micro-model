"""Run the checkpoint diagnostics suite and write diagnostics.{json,html} next to the checkpoint.

    python scripts/diagnose.py runs/m1_tinystories_26m/checkpoints/best.pt --root C:/slm-data/tokenized/v1 \
        --source tinystories --seq 1024 --batches 8 --mb 4 [--no-ablation]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slm.config import ModelConfig, from_dict  # noqa: E402
from slm.data.loader import MixtureSpec, ValLoader  # noqa: E402
from slm.eval.diagnostics import run_diagnostics, save  # noqa: E402
from slm.model import Transformer  # noqa: E402
from slm.utils.sdpa import sdpa_context  # noqa: E402


def load_model(path: Path) -> Transformer:
    ck = torch.load(path, map_location="cuda", weights_only=False)
    mcfg = ck.get("config") if "model" in ck and "n_layers" in ck.get("config", {}) else ck["meta"]["model_config"]
    model = Transformer(from_dict(ModelConfig, mcfg)).cuda()
    sd = {k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--root", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    model = load_model(Path(a.checkpoint))
    vl = ValLoader(MixtureSpec(Path(a.root), {a.source: 1.0}, "val"), a.seq, a.mb, a.seq * a.mb * a.batches)
    batches = list(vl)[: a.batches]
    with sdpa_context("auto"):
        d = run_diagnostics(model, batches, do_ablations=not a.no_ablation)
    out = Path(a.out) if a.out else Path(a.checkpoint).parent.parent / "diagnostics"
    name = Path(a.checkpoint).stem
    save(d, out, name)
    print(f"base loss {d['base_loss']:.4f}; dead MLP units per layer: {[f'{x * 100:.1f}%' for x in d['mlp']['dead_frac']]}")
    print(f"layer skip deltas: {[f'{x:+.3f}' for x in d.get('ablation', {}).get('layer_loss_delta', [])]}")
    print(f"wrote {out / (name + '.html')} ({d['seconds']:.0f}s)")


if __name__ == "__main__":
    main()

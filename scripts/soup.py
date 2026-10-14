"""Weight interpolation between two checkpoints of one lineage: (1 - alpha) * A + alpha * B, written as a weights-only
checkpoint (`model`, `config`, `meta`) that every loader reads like a final.pt.

Meant for a fine-tune and its parent (M11: the arena SFT and the RL stage trained from it, results §31), where the two
sit in one basin and the straight line between them stays a working model.

    python scripts/soup.py --a runs/m11_arena_sft_336m/checkpoints/final.pt --b runs/m11_rl_arena_336m/checkpoints/best.pt --alpha 0.5 --out runs/m11_soup_336m/checkpoints/a050.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--alpha", type=float, required=True, help="weight of --b")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ca = torch.load(a.a, map_location="cpu", weights_only=False)
    cb = torch.load(a.b, map_location="cpu", weights_only=False)
    ma, mb = ca["model"], cb["model"]
    if set(ma) != set(mb) or any(ma[k].shape != mb[k].shape for k in ma):
        raise SystemExit("the checkpoints do not share an architecture")
    runtime = {"max_seq_len", "loss_chunk_size", "grad_checkpointing"}  # not the architecture: B's are kept
    if {k: v for k, v in ca["config"].items() if k not in runtime} != {k: v for k, v in cb["config"].items() if k not in runtime}:
        raise SystemExit("the model configs differ")
    soup = {k: ((1 - a.alpha) * ma[k].float() + a.alpha * mb[k].float()).to(ma[k].dtype) if ma[k].is_floating_point() else ma[k] for k in ma}
    meta = dict(ca.get("meta") or {})
    meta.update({"soup": {"a": a.a, "b": a.b, "alpha": a.alpha}})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": soup, "config": cb["config"], "meta": meta}, a.out)
    print(f"wrote {a.out}: (1 - {a.alpha}) * {a.a} + {a.alpha} * {a.b}")


if __name__ == "__main__":
    main()

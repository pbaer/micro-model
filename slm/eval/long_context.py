"""Long-context evaluation: needle-in-a-haystack and multi-needle retrieval across context lengths.

Distinguishes configured context (RoPE table size), memory-feasible context, and *effective*
context (retrieval accuracy by needle depth and total length).

    python -m slm.eval.long_context --checkpoint runs/<run>/checkpoints/final.pt \
        --lengths 1024 4096 8192 --depths 0.1 0.5 0.9 --n 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from slm.config import ModelConfig, RopeScaling, from_dict
from slm.data.tokenizer import SlmTokenizer
from slm.model import Transformer
from slm.rl.rewards import _NUM_RE  # noqa: PLC2701
from slm.utils.sdpa import sdpa_context

FILLER = [
    "The village market opened early, and the vendors arranged their goods in neat rows along the square.",
    "Rainfall in the valley varies from year to year, which shapes what the farmers decide to plant.",
    "Photosynthesis converts light energy into chemical energy stored in sugars inside plant cells.",
    "The committee met on Tuesday to review the budget and postponed the final vote until spring.",
    "A well-tuned bicycle needs clean gears, a lubricated chain, and tires at the recommended pressure.",
    "Historians disagree about when the bridge was completed, though most accept the later date.",
    "The recipe calls for two cups of flour, a pinch of salt, and enough water to form a soft dough.",
    "Migrating birds navigate using the stars, the Earth's magnetic field, and familiar landmarks.",
]


def build_haystack(tok: SlmTokenizer, total_tokens: int, depth: float, needle: str, question: str, rng: random.Random, reserve: int = 16) -> tuple[list[int], int]:
    """Return prompt ids of ~total_tokens - reserve with the needle at the given depth (0..1).
    `reserve` leaves room for the generated answer inside the model's RoPE table."""
    q_ids = tok.encode("\n\nQuestion: " + question + "\nAnswer:")
    n_ids = tok.encode(" " + needle + " ")
    budget = total_tokens - reserve - len(q_ids) - len(n_ids) - 1
    filler_ids: list[int] = []
    while len(filler_ids) < budget:
        filler_ids.extend(tok.encode(rng.choice(FILLER) + " "))
    filler_ids = filler_ids[:budget]
    pos = int(len(filler_ids) * depth)
    ids = [tok.bos_id, *filler_ids[:pos], *n_ids, *filler_ids[pos:], *q_ids]
    return ids, pos + 1


@torch.no_grad()
def answer(model: Transformer, tok: SlmTokenizer, ids: list[int], max_new: int = 12) -> str:
    x = torch.tensor([ids], device=next(model.parameters()).device)
    with sdpa_context("decode"), torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
        out = model.generate(x, max_new, temperature=0.0, stop_ids=(tok.eos_id,))
    return tok.decode(out[0, len(ids) :].tolist())


def run_needle(model: Transformer, tok: SlmTokenizer, lengths: list[int], depths: list[float], n: int = 5, seed: int = 0, multi: bool = False) -> dict:
    rng = random.Random(seed)
    results = []
    for L in lengths:
        if L > model.cfg.max_seq_len:
            results.append({"length": L, "skipped": "exceeds RoPE table"})
            continue
        for d in depths:
            hits = 0
            for i in range(n):
                secret = rng.randint(100000, 999999)
                needle = f"The secret number is {secret}."
                question = "What is the secret number?"
                if multi:
                    secret2 = rng.randint(100000, 999999)
                    needle = f"The first secret number is {secret}."
                    ids, _ = build_haystack(tok, L, d, needle, "What is the sum of the first and second secret numbers?", rng)
                    extra = tok.encode(f" The second secret number is {secret2}. ")
                    k = rng.randint(1, max(1, len(ids) - 30))
                    ids = ids[:k] + extra + ids[k - len(extra) if k >= len(extra) else k :]  # insert without growing past the budget
                    ids = ids[: L - 16]
                    gold = secret + secret2
                else:
                    ids, _ = build_haystack(tok, L, d, needle, question, rng)
                    gold = secret
                try:
                    out = answer(model, tok, ids)
                except torch.OutOfMemoryError:
                    results.append({"length": L, "depth": d, "oom": True})
                    torch.cuda.empty_cache()
                    break
                m = _NUM_RE.search(out.replace(",", ""))
                hits += int(m is not None and m.group(0) == str(gold))
            results.append({"length": L, "depth": d, "accuracy": hits / n, "n": n, "multi": multi})
    return {"lengths": lengths, "depths": depths, "results": results, "max_seq_len": model.cfg.max_seq_len}


def load(path: str, max_seq_len: int | None = None, rope_scaling: dict | None = None) -> tuple[Transformer, dict]:
    ck = torch.load(path, map_location="cuda", weights_only=False)
    mcfg_d = ck.get("meta", {}).get("model_config") or ck["config"]
    mcfg = from_dict(ModelConfig, mcfg_d)
    model = Transformer(mcfg).cuda().to(torch.bfloat16)
    model.load_state_dict({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in ck["model"].items()})
    if max_seq_len or rope_scaling:
        model.set_rope(max_seq_len or mcfg.max_seq_len, scaling=RopeScaling(**rope_scaling) if rope_scaling else None)
    return model.eval(), ck.get("meta", {})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=r"C:\slm-data\tokenizer\v1")
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 8192])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--multi", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=None, help="extend the RoPE table (with --rope-scaling) to probe beyond the trained context")
    ap.add_argument("--rope-scaling", default=None, help='json, e.g. {"type":"yarn","factor":2,"original_max_seq_len":8192}')
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tok = SlmTokenizer.load(a.tokenizer)
    model, meta = load(a.checkpoint, a.max_seq_len, json.loads(a.rope_scaling) if a.rope_scaling else None)
    res = run_needle(model, tok, a.lengths, a.depths, a.n, multi=a.multi)
    res["checkpoint"] = a.checkpoint
    for r in res["results"]:
        print(r)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())

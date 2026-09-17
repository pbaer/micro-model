"""Long-context evaluation: needle-in-a-haystack retrieval across context lengths and needle depths.

Distinguishes configured context (RoPE table size), memory-feasible context, and *effective* context:
the longest length at which retrieval accuracy stays above a threshold at every depth.

    python -m slm.eval.long_context --checkpoint runs/<run>/checkpoints/final.pt \
        --lengths 1024 2048 4096 8000 --depths 0 0.1 0.25 0.5 0.75 0.9 1 --n 16 --haystack real

Haystacks: `real` (default) is continuous validation text from a tokenized source, with document
boundary tokens removed so the needle and the question sit inside one long document, which is the
situation long context is for. `filler` is the old 8-sentence loop (kept as a control: repetitive text
is much easier because the needle is the only thing that changes).

The trainer runs the same measurement periodically for extension runs (`eval.needle_lengths`) and logs
`needle_<L>` (mean over depths) and `needle_min_<L>` (worst depth) into its eval records.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
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
DEFAULT_DEPTHS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


class FillerHaystack:
    """Repetitive control haystack: the 8 FILLER sentences in random order."""

    name = "filler"

    def __init__(self, tok: SlmTokenizer) -> None:
        self.tok = tok

    def tokens(self, budget: int, rng: random.Random) -> list[int]:
        ids: list[int] = []
        while len(ids) < budget:
            ids.extend(self.tok.encode(rng.choice(FILLER) + " "))
        return ids[:budget]


class RealHaystack:
    """Continuous text from a tokenized split (e.g. C:/slm-data/tokenized/v1/fineweb-edu-b/val): a random
    window with document-boundary specials (<|bos|>/<|eos|>/<|pad|>) removed."""

    name = "real"

    def __init__(self, tok: SlmTokenizer, split_dir: Path) -> None:
        self.paths = sorted(Path(split_dir).glob("shard_*.bin"))
        assert self.paths, f"no shards under {split_dir}"
        self.mm = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.paths]
        self.drop = np.array(sorted({tok.bos_id, tok.eos_id, getattr(tok, "pad_id", tok.eos_id)}), dtype=np.uint16)

    def tokens(self, budget: int, rng: random.Random) -> list[int]:
        mm = self.mm[rng.randrange(len(self.mm))]
        want = int(budget * 1.05) + 64
        start = rng.randrange(0, max(1, len(mm) - want))
        chunk = np.asarray(mm[start : start + want])
        chunk = chunk[~np.isin(chunk, self.drop)]
        while len(chunk) < budget:  # boundary-heavy region: append more
            start = rng.randrange(0, max(1, len(mm) - want))
            more = np.asarray(mm[start : start + want])
            chunk = np.concatenate([chunk, more[~np.isin(more, self.drop)]])
        return chunk[:budget].astype(np.int64).tolist()


def make_haystack(tok: SlmTokenizer, kind: str, split_dir: Path | None) -> FillerHaystack | RealHaystack:
    if kind == "real":
        if split_dir is None or not any(Path(split_dir).glob("shard_*.bin")):
            raise FileNotFoundError(f"real haystack needs a tokenized val split; none at {split_dir}")
        return RealHaystack(tok, split_dir)
    return FillerHaystack(tok)


def build_haystack(tok: SlmTokenizer, total_tokens: int, depth: float, needle: str, question: str, rng: random.Random,
                   haystack: FillerHaystack | RealHaystack | None = None, reserve: int = 16) -> tuple[list[int], int]:
    """Prompt ids of length total_tokens - reserve with the needle sentence at `depth` (0 = right after
    <|bos|>, 1 = right before the question). `reserve` leaves room for the answer inside the RoPE table."""
    haystack = haystack or FillerHaystack(tok)
    q_ids = tok.encode("\n\nQuestion: " + question + "\nAnswer:")
    n_ids = tok.encode(" " + needle + " ")
    budget = total_tokens - reserve - len(q_ids) - len(n_ids) - 1
    filler_ids = haystack.tokens(budget, rng)
    pos = int(len(filler_ids) * depth)
    ids = [tok.bos_id, *filler_ids[:pos], *n_ids, *filler_ids[pos:], *q_ids]
    return ids, pos + 1


@torch.no_grad()
def answer_batch(model: Transformer, tok: SlmTokenizer, prompts: list[list[int]], max_new: int = 12) -> list[str]:
    """Greedy answers for a batch of equal-length prompts (one KV cache, one decode loop). Every prompt a
    needle run builds for a given length has exactly that length, so whole cells batch together."""
    assert prompts and len({len(p) for p in prompts}) == 1, "answer_batch needs equal-length prompts"
    t0 = len(prompts[0])
    x = torch.tensor(prompts, device=next(model.parameters()).device)
    with sdpa_context("decode"), torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
        out = model.generate(x, max_new, temperature=0.0, stop_ids=(tok.eos_id,))
    texts = []
    for row in out[:, t0:].tolist():
        if tok.eos_id in row:  # rows that stopped early keep decoding while the batch runs: cut them back
            row = row[: row.index(tok.eos_id) + 1]
        texts.append(tok.decode(row))
    return texts


def answer(model: Transformer, tok: SlmTokenizer, ids: list[int], max_new: int = 12) -> str:
    return answer_batch(model, tok, [ids], max_new)[0]


def answer_all(model: Transformer, tok: SlmTokenizer, prompts: list[list[int]], max_batch_tokens: int = 16384,
               max_new: int = 12) -> list[str | None]:
    """Answer every prompt, batching equal-length prompts up to `max_batch_tokens` prompt tokens per batch.
    A batch that runs out of memory is retried row by row; rows that still fail come back as None."""
    order: dict[int, list[int]] = {}
    for i, p in enumerate(prompts):
        order.setdefault(len(p), []).append(i)
    out: list[str | None] = [None] * len(prompts)
    for length, idxs in order.items():
        bs = max(1, max_batch_tokens // max(1, length))
        for k in range(0, len(idxs), bs):
            chunk = idxs[k : k + bs]
            try:
                texts = answer_batch(model, tok, [prompts[i] for i in chunk], max_new)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                texts = []
                for i in chunk:
                    try:
                        texts.append(answer_batch(model, tok, [prompts[i]], max_new)[0])
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        texts.append(None)
            for i, t in zip(chunk, texts):
                out[i] = t
    return out


def _make_case(tok: SlmTokenizer, L: int, d: float, rng: random.Random, haystack, multi: bool) -> tuple[list[int], int]:
    """One (prompt ids, gold) case. The rng draws happen in the same order as the unbatched version."""
    secret = rng.randint(100000, 999999)
    needle = f"The secret number is {secret}."
    if not multi:
        ids, _ = build_haystack(tok, L, d, needle, "What is the secret number?", rng, haystack)
        return ids, secret
    secret2 = rng.randint(100000, 999999)
    needle = f"The first secret number is {secret}."
    ids, _ = build_haystack(tok, L, d, needle, "What is the sum of the first and second secret numbers?", rng, haystack)
    extra = tok.encode(f" The second secret number is {secret2}. ")
    k = rng.randint(1, max(1, len(ids) - 30))
    ids = ids[:k] + extra + ids[k - len(extra) if k >= len(extra) else k :]
    return ids[: L - 16], secret + secret2


def run_needle(model: Transformer, tok: SlmTokenizer, lengths: list[int], depths: list[float] | None = None, n: int = 16, seed: int = 0,
               multi: bool = False, haystack: FillerHaystack | RealHaystack | None = None, threshold: float = 0.8, keep_failures: int = 3,
               max_batch_tokens: int = 16384) -> dict:
    """Accuracy per (length, depth) cell plus a per-length summary and the effective context
    (longest length whose worst depth is still >= threshold).

    All cases of one length are generated in batches (`max_batch_tokens` prompt tokens per batch), which is
    what makes a useful `n` affordable inside a training run: decoding 16 rows costs about what 1 row costs."""
    depths = DEFAULT_DEPTHS if depths is None else depths
    haystack = haystack or FillerHaystack(tok)
    rng = random.Random(seed)
    results = []
    was_training = model.training
    model.eval()
    for L in lengths:
        if L > model.cfg.max_seq_len:
            results.append({"length": L, "skipped": "exceeds RoPE table"})
            continue
        cases = [(d, *_make_case(tok, L, d, rng, haystack, multi)) for d in depths for _ in range(n)]
        outs = answer_all(model, tok, [c[1] for c in cases], max_batch_tokens)
        for j, d in enumerate(depths):
            hits, failures, n_oom = 0, [], 0
            for (_, _, gold), out in zip(cases[j * n : (j + 1) * n], outs[j * n : (j + 1) * n]):
                if out is None:
                    n_oom += 1
                    continue
                m = _NUM_RE.search(out.replace(",", ""))
                ok = m is not None and m.group(0) == str(gold)
                hits += int(ok)
                if not ok and len(failures) < keep_failures:
                    failures.append({"gold": gold, "out": out.strip()[:60]})
            if n_oom == n:
                results.append({"length": L, "depth": d, "oom": True})
                continue
            results.append({"length": L, "depth": d, "accuracy": hits / (n - n_oom), "n": n - n_oom, "multi": multi, "failures": failures})
    if was_training:
        model.train()
    summary: dict[int, dict] = {}
    for L in lengths:
        cells = [r["accuracy"] for r in results if r.get("length") == L and "accuracy" in r]
        if cells:
            summary[L] = {"mean": sum(cells) / len(cells), "min": min(cells)}
    effective = max((L for L, s in summary.items() if s["min"] >= threshold), default=0)
    return {"lengths": lengths, "depths": depths, "n": n, "haystack": getattr(haystack, "name", "filler"), "results": results,
            "summary": summary, "threshold": threshold, "effective_context": effective, "max_seq_len": model.cfg.max_seq_len}


def format_table(res: dict) -> str:
    depths = res["depths"]
    lines = ["length  " + "".join(f"d={d:<6}" for d in depths) + " mean   min"]
    for L in res["lengths"]:
        cells = {r["depth"]: r for r in res["results"] if r.get("length") == L and "accuracy" in r}
        if not cells:
            lines.append(f"{L:<7} skipped")
            continue
        s = res["summary"][L] if L in res["summary"] else res["summary"][str(L)]
        lines.append(f"{L:<7} " + "".join(f"{cells[d]['accuracy'] * 100:>5.0f}% " for d in depths if d in cells) + f" {s['mean'] * 100:>4.0f}%  {s['min'] * 100:>4.0f}%")
    lines.append(f"effective context (min over depths >= {res['threshold']:.0%}): {res['effective_context']} · haystack {res['haystack']} · n={res['n']} per cell")
    return "\n".join(lines)


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
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8000])
    ap.add_argument("--depths", type=float, nargs="+", default=DEFAULT_DEPTHS)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--haystack", choices=["real", "filler"], default="real")
    ap.add_argument("--tokenized-root", default=r"C:\slm-data\tokenized\v1")
    ap.add_argument("--haystack-source", default="fineweb-edu-b", help="tokenized source whose val split provides the real haystack")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--multi", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seq-len", type=int, default=None, help="extend the RoPE table (with --rope-scaling) to probe beyond the trained context")
    ap.add_argument("--rope-scaling", default=None, help='json, e.g. {"type":"yarn","factor":2,"original_max_seq_len":8192}')
    ap.add_argument("--batch-tokens", type=int, default=16384, help="prompt tokens per generation batch (higher = faster, more VRAM)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tok = SlmTokenizer.load(a.tokenizer)
    model, meta = load(a.checkpoint, a.max_seq_len, json.loads(a.rope_scaling) if a.rope_scaling else None)
    hs = make_haystack(tok, a.haystack, Path(a.tokenized_root) / a.haystack_source / "val")
    res = run_needle(model, tok, a.lengths, a.depths, a.n, seed=a.seed, multi=a.multi, haystack=hs, threshold=a.threshold, max_batch_tokens=a.batch_tokens)
    res["checkpoint"] = a.checkpoint
    print(format_table(res))
    for r in res["results"]:
        if r.get("failures"):
            print(f"  L={r['length']} d={r['depth']}: e.g. gold {r['failures'][0]['gold']} -> {r['failures'][0]['out']!r}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())

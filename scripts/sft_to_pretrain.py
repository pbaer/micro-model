"""Turn SFT shards (chat-formatted examples <|bos|>...<|eos|>, tokens_*.bin + idx_*.npy) into a pretraining source
(shard_*.bin + shard_*.idx.npy) so chat and tool-call data can be mixed into pretraining (the decay phase of the
second base). Loss masks are dropped: in pretraining every token is a target, including the user turns.

    python scripts/sft_to_pretrain.py --out smoltalk-chat smoltalk-smol-magpie-ultra smoltalk-openhermes-100k ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slm.data.prepare import ShardWriter  # noqa: E402
from slm.data.sources import DATA_ROOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sft-root", default=str(DATA_ROOT / "sft" / "v1"))
    ap.add_argument("--tokenized-root", default=str(DATA_ROOT / "tokenized" / "v1"))
    ap.add_argument("--shard-tokens", type=int, default=50_000_000)
    a = ap.parse_args()
    out = Path(a.tokenized_root) / a.out
    stats = {}
    for split in ("train", "val"):
        w = ShardWriter(out / split, a.shard_tokens)
        for name in a.sources:
            d = Path(a.sft_root) / name / split
            for tp in sorted(d.glob("tokens_*.bin")):
                toks = np.memmap(tp, dtype=np.uint16, mode="r")
                idx = np.load(tp.with_name(tp.name.replace("tokens_", "idx_").replace(".bin", ".npy")))
                bounds = list(idx) + [len(toks)]
                for s, e in zip(bounds[:-1], bounds[1:]):
                    w.add(toks[s:e].tolist())
        w.flush()
        stats[split] = {"tokens": w.total_tokens, "docs": w.total_docs, "shards": w.shard_idx}
    (out / "manifest.json").write_text(json.dumps({"source": a.out, "from_sft": a.sources, **stats}, indent=1), encoding="utf-8")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()

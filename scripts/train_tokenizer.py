"""Train the 32K byte-level BPE tokenizer on a mixture sample matching the pretraining mix.

    python scripts/train_tokenizer.py --out C:/slm-data/tokenizer/v1 --chars 1.5e9 \
        --mix fineweb-edu:0.72 cosmopedia:0.10 finemath:0.06 python-edu:0.09 stack-edu-shell:0.03

Documents are sampled from the locally downloaded raw parquet files of each source, up to the
per-source character budget. Code sources read the Software Heritage content shards.
"""

from __future__ import annotations

import argparse
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slm.data.sources import SOURCES, Source  # noqa: E402
from slm.data.swh import content_dir  # noqa: E402
from slm.data.tokenizer import SlmTokenizer, train_bpe  # noqa: E402


def parquet_files(src: Source) -> list[Path]:
    if src.content_via_swh:
        return sorted(content_dir(src).glob("*.parquet"))
    return sorted(p for p in src.local_dir.rglob("*.parquet") if "validation" not in p.name)


def iter_texts(src: Source, char_budget: float, seed: int, batch_rows: int = 2000) -> Iterator[str]:
    """Yield documents from a source until the character budget is met (round-robin over files)."""
    files = parquet_files(src)
    assert files, f"no local files for {src.name}"
    rng = random.Random(seed)
    rng.shuffle(files)
    got = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            col = pf.read_row_group(rg, columns=[src.text_col]).column(src.text_col).to_pylist()
            rng.shuffle(col)
            for t in col:
                if not t:
                    continue
                yield t
                got += len(t)
                if got >= char_budget:
                    return


def mixture_iterator(mix: dict[str, float], total_chars: float, seed: int) -> Iterator[str]:
    """Interleave sources so the trainer sees a shuffled mixture, not one source after another."""
    iters = {name: iter_texts(SOURCES[name], total_chars * frac, seed) for name, frac in mix.items()}
    names = list(iters)
    weights = [mix[n] for n in names]
    rng = random.Random(seed)
    counts = dict.fromkeys(names, 0)
    while iters:
        name = rng.choices(names, weights=weights)[0]
        try:
            yield next(iters[name])
            counts[name] += 1
        except StopIteration:
            idx = names.index(name)
            names.pop(idx)
            weights.pop(idx)
            del iters[name]
            print(f"  source {name} exhausted after {counts[name]} docs", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--chars", type=float, default=1.5e9)
    ap.add_argument("--mix", nargs="+", default=["fineweb-edu:0.72", "cosmopedia:0.10", "finemath:0.06", "python-edu:0.09", "stack-edu-shell:0.03"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    mix = {k: float(v) for k, v in (m.split(":") for m in args.mix)}
    assert abs(sum(mix.values()) - 1.0) < 1e-6, "mixture must sum to 1"
    print(f"training tokenizer on ~{args.chars / 1e9:.2f}G chars, mix={mix}")
    t0 = time.time()
    tok = train_bpe(mixture_iterator(mix, args.chars, args.seed))
    from slm.data.tokenizer import BPE_VOCAB
    assert tok.get_vocab_size() == BPE_VOCAB, f"BPE vocab {tok.get_vocab_size()} != {BPE_VOCAB}; need more training text"
    st = SlmTokenizer(tok)
    st.save(args.out)
    print(f"saved to {args.out} (vocab {st.vocab_size}, sha256 {st.sha256[:12]}) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

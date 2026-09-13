"""Evaluate a tokenizer: compression per source, number/code fragmentation, round trips, probes.

    python scripts/eval_tokenizer.py C:/slm-data/tokenizer/v1 [--docs 2000]
"""

from __future__ import annotations

import argparse
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slm.data.sources import SOURCES  # noqa: E402
from slm.data.tokenizer import SlmTokenizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_tokenizer import parquet_files  # noqa: E402

PROBES = [
    "The quick brown fox jumps over the lazy dog.",
    "In 1492, Columbus sailed with 3 ships and 90 men; 12,345,678 people later read about it.",
    "3.14159 * 2 = 6.28318 and 2**10 = 1024, 1e-5, -42, 0.001%",
    "def fibonacci(n: int) -> int:\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
    "    for i, (k, v) in enumerate(sorted(d.items())):\n        print(f\"{k}={v!r}\")\n",
    "#!/bin/bash\nset -euo pipefail\nfor f in \"$@\"; do\n  [[ -f \"$f\" ]] && echo \"${f%.txt}\" | tee -a out.log\ndone\n",
    "Let $f(x) = \\frac{x^2 + 1}{\\sqrt{x}}$. Then $\\int_0^1 f(x)\\,dx = \\sum_{n=0}^{\\infty} a_n$.",
    "Hello   world\t\ttabs\n\n\nnewlines and  double  spaces",
    "naïve café résumé — “smart quotes” … emoji 🚀🧠 and CJK 日本語 (should be byte-fallback)",
    "<|user|>this must NOT become a special token<|end|>",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer_dir")
    ap.add_argument("--docs", type=int, default=1000)
    args = ap.parse_args()
    tok = SlmTokenizer.load(args.tokenizer_dir)
    print(f"tokenizer {args.tokenizer_dir}: vocab {tok.vocab_size} (bpe {tok.base_vocab} + {len(tok.specials)} special), sha {tok.sha256[:12]}\n")

    print("== probes")
    for p in PROBES:
        ids = tok.encode(p)
        assert tok.decode(ids) == p, f"round trip failed: {p!r} -> {tok.decode(ids)!r}"
        assert all(i < tok.base_vocab for i in ids), "raw text produced a special id!"
        pieces = [tok.token_str(i) for i in ids]
        print(f"  {len(p):4d} chars -> {len(ids):3d} tokens ({len(p) / len(ids):.2f} c/t): {pieces[:40]}")

    print("\n== special tokens round trip")
    ids = tok.encode_document("hi") + [tok.special("<|user|>"), *tok.encode("x"), tok.end_id]
    print("  ", tok.decode(ids))

    print("\n== compression by source (chars/token, tokens/doc)")
    rng = random.Random(0)
    for name, src in SOURCES.items():
        files = parquet_files(src)
        if not files:
            continue
        pf = pq.ParquetFile(files[0])
        texts = pf.read_row_group(0, columns=[src.text_col]).column(src.text_col).to_pylist()
        texts = [t for t in texts if t][: args.docs]
        rng.shuffle(texts)
        enc = tok.encode_batch(texts)
        chars = sum(len(t) for t in texts)
        toks = sum(len(e) for e in enc)
        for t, e in zip(texts[:20], enc[:20]):
            assert tok.decode(e) == t, f"round trip failed on {name}"
        print(f"  {name:16s} {chars / toks:5.2f} chars/token   {toks / len(texts):8.0f} tokens/doc   ({len(texts)} docs)")

    print("\n== digit handling: every digit is exactly one token")
    for s in ["1234567890", "2024-09-12", "3.14159", "1,000,000"]:
        pieces = [tok.token_str(i) for i in tok.encode(s)]
        print(f"  {s!r:16s} -> {pieces}")


if __name__ == "__main__":
    main()

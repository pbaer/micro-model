"""Tokenize raw parquet corpora into uint16 token shards with document boundaries.

    python -m slm.data.prepare fineweb-edu --tokenizer C:/slm-data/tokenizer/v1 --max-tokens 2e9
    python -m slm.data.prepare tinystories --tokenizer C:/slm-data/tokenizer/v1

Output layout (per tokenizer version, per source):
    C:/slm-data/tokenized/<tok_tag>/<source>/train/shard_00000.bin   uint16 tokens, docs = <|bos|> ... <|eos|>
    C:/slm-data/tokenized/<tok_tag>/<source>/train/shard_00000.idx   int64 doc start offsets (npy)
    C:/slm-data/tokenized/<tok_tag>/<source>/train/shard_00000.src   int32 [n_docs, 3] = (file_index, row_group, row) (npy)
    C:/slm-data/tokenized/<tok_tag>/<source>/val/...                 held-out docs (never trained on)
    C:/slm-data/tokenized/<tok_tag>/<source>/manifest.json

Validation split is decided per document by a hash of its text, so it is stable across reruns
and independent of file order. Language policy: English prose, Python, shell. Prose sources use
their own language columns plus an ASCII-letter-ratio backstop; code sources use a non-ASCII cap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from slm.data.sources import SOURCES, TOKENIZED_DIR, Source
from slm.data.swh import content_dir
from slm.data.tokenizer import SlmTokenizer

SHARD_TOKENS = 100_000_000  # 200 MB per shard as uint16
MIN_DOC_TOKENS = 16
MAX_DOC_TOKENS = 65536


def raw_files(src: Source) -> list[Path]:
    if src.content_via_swh:
        return sorted(content_dir(src).glob("*.parquet"))
    return sorted(p for p in src.local_dir.rglob("*.parquet") if "validation" not in p.name)


def ascii_letter_ratio(t: str) -> float:
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return 1.0
    return sum(c.isascii() for c in letters) / len(letters)


def keep_doc(src: Source, row: dict) -> bool:
    t = row.get(src.text_col) or ""
    if len(t) < 64:
        return False
    lang = row.get("language")
    if lang is not None and src.kind != "code" and lang != "en":
        return False
    sample = t[:4000]
    if src.kind == "code":
        return sum(c.isascii() for c in sample) / len(sample) >= 0.9
    return ascii_letter_ratio(sample) >= 0.95


def is_val(text: str, val_permille: int) -> bool:
    h = hashlib.sha1(text[:2048].encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(h[:4], "little") % 1000 < val_permille


class ShardWriter:
    def __init__(self, out_dir: Path, shard_tokens: int) -> None:
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_tokens = shard_tokens
        self.buf = np.empty(shard_tokens, dtype=np.uint16)
        self.n = 0
        self.doc_starts: list[int] = []
        self.doc_srcs: list[tuple[int, int, int]] = []  # (file_index into manifest["files"], row_group, row)
        self.shard_idx = 0
        self.total_tokens = 0
        self.total_docs = 0

    def add(self, ids: list[int], src_row: tuple[int, int, int] = (-1, -1, -1)) -> None:
        if self.n + len(ids) > self.shard_tokens:
            self.flush()
        self.doc_starts.append(self.n)
        self.doc_srcs.append(src_row)
        self.buf[self.n : self.n + len(ids)] = ids
        self.n += len(ids)
        self.total_tokens += len(ids)
        self.total_docs += 1

    def flush(self) -> None:
        if self.n == 0:
            return
        self.buf[: self.n].tofile(self.out_dir / f"shard_{self.shard_idx:05d}.bin")
        np.save(self.out_dir / f"shard_{self.shard_idx:05d}.idx.npy", np.array(self.doc_starts, dtype=np.int64))
        np.save(self.out_dir / f"shard_{self.shard_idx:05d}.src.npy",
                np.array(self.doc_srcs, dtype=np.int32).reshape(len(self.doc_srcs), 3))
        self.shard_idx += 1
        self.n = 0
        self.doc_starts = []
        self.doc_srcs = []


def prepare(src: Source, tok: SlmTokenizer, out_root: Path, max_tokens: float, val_permille: int, batch_docs: int = 512, min_doc_tokens: int = MIN_DOC_TOKENS, name: str | None = None) -> dict:
    files = raw_files(src)
    assert files, f"no raw files for {src.name}"
    out = out_root / (name or src.name)
    train = ShardWriter(out / "train", SHARD_TOKENS)
    val = ShardWriter(out / "val", SHARD_TOKENS // 10)
    cols = [src.text_col] + [c for c in ("language",) if c]  # language may not exist; filtered below
    t0 = time.time()
    n_seen = n_dropped = 0
    stop = False
    for fi, f in enumerate(files):
        pf = pq.ParquetFile(f)
        avail = [c for c in cols if c in pf.schema_arrow.names]
        for rg in range(pf.num_row_groups):
            rows = pf.read_row_group(rg, columns=avail).to_pylist()
            for i in range(0, len(rows), batch_docs):
                batch = rows[i : i + batch_docs]
                n_seen += len(batch)
                kept = [(i + j, r) for j, r in enumerate(batch) if keep_doc(src, r)]
                n_dropped += len(batch) - len(kept)
                texts = [r[src.text_col] for _, r in kept]
                for (row_i, _), text, ids in zip(kept, texts, tok.encode_batch(texts)):
                    if not (min_doc_tokens <= len(ids) <= MAX_DOC_TOKENS):
                        n_dropped += 1
                        continue
                    doc = [tok.bos_id, *ids, tok.eos_id]
                    (val if is_val(text, val_permille) else train).add(doc, (fi, rg, row_i))
                if train.total_tokens >= max_tokens:
                    stop = True
                    break
            if stop:
                break
            el = time.time() - t0
            print(
                f"[{src.name}] {f.name} rg{rg}: train {train.total_tokens / 1e6:.0f}M tok, val {val.total_tokens / 1e6:.1f}M, "
                f"docs {train.total_docs + val.total_docs}, dropped {n_dropped}/{n_seen}, {train.total_tokens / el / 1e6:.2f}M tok/s",
                flush=True,
            )
        if stop:
            break
    train.flush()
    val.flush()
    manifest = {
        "source": src.name, "name": name or src.name, "kind": src.kind, "tokenizer_sha256": tok.sha256, "min_doc_tokens": min_doc_tokens,
        "train_tokens": train.total_tokens, "train_docs": train.total_docs, "train_shards": train.shard_idx,
        "val_tokens": val.total_tokens, "val_docs": val.total_docs, "val_shards": val.shard_idx,
        "docs_seen": n_seen, "docs_dropped": n_dropped, "val_permille": val_permille,
        "files": [str(p) for p in files], "sidecar": "src", "seconds": time.time() - t0,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[{src.name}] done: {manifest['train_tokens'] / 1e6:.1f}M train / {manifest['val_tokens'] / 1e6:.1f}M val tokens in {manifest['seconds'] / 60:.1f} min")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-tokens", type=float, default=float("inf"))
    ap.add_argument("--val-permille", type=int, default=5)
    ap.add_argument("--min-doc-tokens", type=int, default=MIN_DOC_TOKENS, help="keep only documents with at least this many tokens (long-context phase)")
    ap.add_argument("--name", default=None, help="output source name (default: source name); e.g. fineweb-edu-long")
    a = ap.parse_args()
    for s in a.sources:
        if SOURCES[s].custom_prepare:  # its own selection rules and document handling (e.g. books > MAX_DOC_TOKENS)
            raise SystemExit(f"{s} is prepared by its own module: python -m {SOURCES[s].custom_prepare} prepare --tokenizer {a.tokenizer}")
    tok = SlmTokenizer.load(a.tokenizer)
    out_root = TOKENIZED_DIR / Path(a.tokenizer).name
    for s in a.sources:
        prepare(SOURCES[s], tok, out_root, a.max_tokens, a.val_permille, min_doc_tokens=a.min_doc_tokens, name=a.name)


if __name__ == "__main__":
    sys.exit(main())

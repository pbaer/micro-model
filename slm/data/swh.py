"""Fetch file contents for id-only code datasets (python-edu, stack-edu) from Software Heritage S3.

Contents live at s3://softwareheritage/content/<blob_id> (gzip, anonymous access). We read the
parquet id files, fetch with a thread pool, and write text parquet shards next to them.

    python -m slm.data.swh python-edu --max-files 200000
    python -m slm.data.swh stack-edu-shell --max-files 50000
"""

from __future__ import annotations

import argparse
import gzip
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from botocore import UNSIGNED
from botocore.config import Config

from slm.data.sources import SOURCES, Source

_s3 = None


def s3():
    global _s3
    if _s3 is None:
        _s3 = boto3.client(
            "s3",
            config=Config(signature_version=UNSIGNED, max_pool_connections=128, retries={"max_attempts": 5}),
        )
    return _s3


def fetch_blob(blob_id: str) -> str | None:
    try:
        obj = s3().get_object(Bucket="softwareheritage", Key=f"content/{blob_id}")
        return gzip.decompress(obj["Body"].read()).decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None


def content_dir(src: Source) -> Path:
    return src.local_dir / "content"


def existing_blob_ids(src: Source) -> set[str]:
    done: set[str] = set()
    for p in sorted(content_dir(src).glob("*.parquet")):
        done.update(pq.read_table(p, columns=["blob_id"]).column("blob_id").to_pylist())
    return done


def fetch_source(src: Source, max_files: int, workers: int = 64, shard_rows: int = 20000, min_score: float = 0.0) -> None:
    assert src.content_via_swh
    id_files = sorted(p for p in src.local_dir.glob("*.parquet"))
    assert id_files, f"no id parquet files under {src.local_dir}; run slm.data.download first"
    tbl = pa.concat_tables([pq.read_table(p) for p in id_files])
    cols = tbl.column_names
    score_col = "score" if "score" in cols else ("int_score" if "int_score" in cols else None)
    if score_col:
        idx = pa.compute.sort_indices(tbl, sort_keys=[(score_col, "descending")])
        tbl = tbl.take(idx)
        if min_score > 0:
            tbl = tbl.filter(pa.compute.greater_equal(tbl[score_col], min_score))
    done = existing_blob_ids(src)
    ids = [b for b in tbl.column("blob_id").to_pylist() if b not in done][: max(0, max_files - len(done))]
    print(f"[{src.name}] {len(done)} already fetched, fetching {len(ids)} more (of {tbl.num_rows} ids) with {workers} threads")
    out_dir = content_dir(src)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = len(list(out_dir.glob("*.parquet")))
    meta_cols = [c for c in ("repo_name", "path", "language", "license_type", score_col) if c and c in cols]
    meta = {r["blob_id"]: r for r in tbl.select(["blob_id", *meta_cols]).to_pylist()} if meta_cols else {}
    buf: list[dict] = []
    t0 = time.time()
    n_ok = n_fail = 0

    def flush():
        nonlocal buf, shard_idx
        if not buf:
            return
        pq.write_table(pa.Table.from_pylist(buf), out_dir / f"content-{shard_idx:05d}.parquet", compression="zstd")
        shard_idx += 1
        buf = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_blob, b): b for b in ids}
        for i, f in enumerate(as_completed(futs)):
            b = futs[f]
            text = f.result()
            if text is None:
                n_fail += 1
                continue
            n_ok += 1
            row = {"blob_id": b, "text": text}
            row.update({k: v for k, v in meta.get(b, {}).items() if k != "blob_id"})
            buf.append(row)
            if len(buf) >= shard_rows:
                flush()
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i + 1}/{len(ids)} ok={n_ok} fail={n_fail} {rate:.0f} files/s eta {(len(ids) - i - 1) / rate / 60:.1f} min", flush=True)
    flush()
    print(f"[{src.name}] done: ok={n_ok} fail={n_fail} in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--max-files", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--min-score", type=float, default=0.0)
    a = ap.parse_args()
    fetch_source(SOURCES[a.source], a.max_files, a.workers, min_score=a.min_score)


if __name__ == "__main__":
    sys.exit(main())

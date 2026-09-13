"""Download the first K parquet files of a registered source into DATA_ROOT/raw/<source>.

    python -m slm.data.download fineweb-edu --n-files 1
    python -m slm.data.download tinystories --all
    python -m slm.data.download --list
"""

from __future__ import annotations

import argparse
import fnmatch
import sys

from huggingface_hub import HfApi, hf_hub_download

from slm.data.sources import RAW_DIR, SOURCES, Source


def list_remote_files(src: Source) -> list[tuple[str, int]]:
    api = HfApi()
    files = api.list_repo_tree(src.repo, repo_type="dataset", recursive=True, path_in_repo=src.pattern.split("/*")[0])
    out = []
    for f in files:
        path = getattr(f, "path", None)
        if path and fnmatch.fnmatch(path, src.pattern):
            out.append((path, getattr(f, "size", 0) or 0))
    return sorted(out)


def download(src: Source, n_files: int | None, start: int = 0) -> list[str]:
    files = list_remote_files(src)
    if n_files is not None:
        files = files[start : start + n_files]
    total = sum(s for _, s in files) / 1e9
    print(f"[{src.name}] downloading {len(files)} files ({total:.2f} GB) -> {src.local_dir}", flush=True)
    paths = []
    for i, (path, size) in enumerate(files):
        p = hf_hub_download(src.repo, path, repo_type="dataset", local_dir=str(src.local_dir))
        print(f"  [{i + 1}/{len(files)}] {path} ({size / 1e9:.2f} GB) ok", flush=True)
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?")
    ap.add_argument("--n-files", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list or not args.source:
        for s in SOURCES.values():
            files = list_remote_files(s)
            print(f"{s.name:16s} {s.repo:32s} {len(files):4d} files {sum(x for _, x in files) / 1e9:8.2f} GB  {s.notes}")
        return
    src = SOURCES[args.source]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download(src, None if args.all else args.n_files, args.start)


if __name__ == "__main__":
    sys.exit(main())

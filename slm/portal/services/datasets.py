"""Raw parquet and tokenized shard access for the data browser (row-group addressed, no scans)."""

from __future__ import annotations

import hashlib
import json
import random
import threading
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from slm.data.sources import SOURCES, Source
from slm.data.swh import content_dir


class DiskCache:
    def __init__(self, d: Path) -> None:
        self.dir = Path(d)

    def key(self, *parts) -> str:
        return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:20]

    def get(self, k: str):
        p = self.dir / f"{k}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
        return None

    def put(self, k: str, v) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"{k}.json").write_text(json.dumps(v), encoding="utf-8")


# ------------------------------------------------------------------------------- raw parquet
class RawSource:
    def __init__(self, src: Source) -> None:
        self.src = src
        self._meta: dict[str, dict] = {}
        self.lock = threading.Lock()

    def files(self) -> list[dict]:
        if self.src.content_via_swh:
            paths = sorted(content_dir(self.src).glob("*.parquet"))
        else:
            paths = sorted(p for p in self.src.local_dir.rglob("*.parquet") if content_dir(self.src) not in p.parents)
        out = []
        for i, p in enumerate(paths):
            m = self._file_meta(p)
            out.append({"index": i, "path": str(p), "name": p.name, "rows": m["rows"], "row_groups": m["row_groups"], "bytes": p.stat().st_size, "columns": m["columns"], "id_only": self.src.content_via_swh and "text" not in m["columns"]})
        return out

    def _file_meta(self, p: Path) -> dict:
        k = str(p)
        with self.lock:
            if k not in self._meta:
                pf = pq.ParquetFile(p)
                self._meta[k] = {"rows": pf.metadata.num_rows, "row_groups": pf.num_row_groups, "columns": pf.schema_arrow.names,
                                 "rg_rows": [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]}
            return self._meta[k]

    def _path(self, file_index: int) -> Path:
        return Path(self.files()[file_index]["path"])

    def docs(self, file_index: int, rg: int, offset: int = 0, limit: int = 50, preview_chars: int = 200) -> dict:
        p = self._path(file_index)
        pf = pq.ParquetFile(p)
        cols = [c for c in pf.schema_arrow.names if c != "prompt"]
        tbl = pf.read_row_group(rg, columns=cols)
        rows = tbl.slice(offset, limit).to_pylist()
        text_col = self.src.text_col if self.src.text_col in cols else None
        out = []
        for i, r in enumerate(rows):
            t = r.get(text_col, "") if text_col else ""
            meta = {k: (str(v)[:80] if not isinstance(v, (int, float)) else v) for k, v in r.items() if k != text_col}
            out.append({"row": offset + i, "chars": len(t or ""), "preview": (t or "")[:preview_chars], "meta": meta})
        return {"file": file_index, "rg": rg, "rg_rows": tbl.num_rows, "offset": offset, "docs": out}

    def doc(self, file_index: int, rg: int, row: int) -> dict:
        p = self._path(file_index)
        pf = pq.ParquetFile(p)
        cols = [c for c in pf.schema_arrow.names if c != "prompt"]
        r = pf.read_row_group(rg, columns=cols).slice(row, 1).to_pylist()[0]
        text = r.pop(self.src.text_col, "") if self.src.text_col in r else ""
        return {"file": file_index, "rg": rg, "row": row, "text": text, "meta": {k: (str(v) if not isinstance(v, (int, float)) else v) for k, v in r.items()}}

    def sample(self, n: int = 10, seed: int = 0) -> list[dict]:
        files = self.files()
        if not files:
            return []
        rng = random.Random(seed)
        out = []
        for _ in range(n):
            f = rng.choice(files)
            m = self._file_meta(Path(f["path"]))
            rg = rng.randrange(m["row_groups"])
            row = rng.randrange(m["rg_rows"][rg])
            d = self.doc(f["index"], rg, row)
            d["preview"] = d["text"][:300]
            del d["text"]
            out.append(d)
        return out


# --------------------------------------------------------------------------- tokenized shards
class TokenizedSplit:
    def __init__(self, split_dir: Path) -> None:
        self.dir = Path(split_dir)
        self.paths = sorted(self.dir.glob("shard_*.bin"))
        self._mm: dict[int, np.memmap] = {}
        self._idx: dict[int, np.ndarray] = {}
        self.lock = threading.Lock()

    def shards(self) -> list[dict]:
        out = []
        for i, p in enumerate(self.paths):
            n_tok = p.stat().st_size // 2
            out.append({"shard": i, "name": p.name, "tokens": n_tok, "docs": int(self.idx(i).shape[0])})
        return out

    def mm(self, i: int) -> np.memmap:
        # Not cached on purpose: a live mapping would block re-tokenization from overwriting the
        # shard on Windows. Opening a memmap is a cheap syscall; slices are copied out immediately.
        return np.memmap(self.paths[i], dtype=np.uint16, mode="r")

    def idx(self, i: int) -> np.ndarray:
        with self.lock:
            if i not in self._idx:
                self._idx[i] = np.load(self.paths[i].with_name(self.paths[i].name.replace(".bin", ".idx.npy")))
            return self._idx[i]

    def doc_bounds(self, shard: int, doc: int) -> tuple[int, int]:
        idx = self.idx(shard)
        start = int(idx[doc])
        end = int(idx[doc + 1]) if doc + 1 < len(idx) else len(self.mm(shard))
        return start, end

    def docs(self, shard: int, offset: int = 0, limit: int = 100) -> dict:
        idx = self.idx(shard)
        n = len(self.mm(shard))
        out = []
        for d in range(offset, min(offset + limit, len(idx))):
            s = int(idx[d])
            e = int(idx[d + 1]) if d + 1 < len(idx) else n
            out.append({"doc": d, "start": s, "length": e - s})
        return {"shard": shard, "n_docs": int(len(idx)), "docs": out}

    def doc(self, shard: int, doc: int) -> dict:
        s, e = self.doc_bounds(shard, doc)
        return {"shard": shard, "doc": doc, "start": s, "length": e - s, "ids": self.mm(shard)[s:e].astype(int).tolist()}

    def window(self, shard: int, start: int, length: int) -> dict:
        mm = self.mm(shard)
        start = max(0, min(start, len(mm) - 1))
        ids = mm[start : start + length].astype(int).tolist()
        idx = self.idx(shard)
        lo = int(np.searchsorted(idx, start, side="left"))
        hi = int(np.searchsorted(idx, start + length, side="left"))
        bounds = [int(x) - start for x in idx[lo:hi]]
        first_doc = lo - 1 if lo > 0 and int(idx[lo - 1]) < start else lo
        return {"shard": shard, "start": start, "ids": ids, "doc_starts": bounds, "first_doc": max(0, first_doc), "shard_tokens": int(len(mm))}

    def stats(self, cache: DiskCache | None = None) -> dict:
        key = cache.key("tokstats", self.dir, [(p.stat().st_mtime, p.stat().st_size) for p in self.paths]) if cache else None
        if cache and (c := cache.get(key)):
            return c
        lengths = []
        total = 0
        for i, p in enumerate(self.paths):
            idx = self.idx(i)
            n = p.stat().st_size // 2
            total += n
            ends = np.append(idx[1:], n)
            lengths.append(ends - idx)
        L = np.concatenate(lengths) if lengths else np.zeros(0, dtype=np.int64)
        edges = [0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 10**9]
        hist = np.histogram(L, bins=edges)[0].tolist() if len(L) else [0] * (len(edges) - 1)
        q = np.percentile(L, [10, 50, 90, 99]).tolist() if len(L) else [0, 0, 0, 0]
        res = {"docs": int(len(L)), "tokens": int(total), "mean_len": float(L.mean()) if len(L) else 0.0, "p10": q[0], "p50": q[1], "p90": q[2], "p99": q[3],
               "max_len": int(L.max()) if len(L) else 0, "hist_edges": edges[:-1], "hist": hist,
               "tokens_by_bin": [int(L[(L >= a) & (L < b)].sum()) for a, b in zip(edges[:-1], edges[1:])] if len(L) else hist,
               "docs_over_4k": int((L >= 4096).sum()) if len(L) else 0, "tokens_over_4k": int(L[L >= 4096].sum()) if len(L) else 0,
               "docs_over_8k": int((L >= 8192).sum()) if len(L) else 0, "tokens_over_8k": int(L[L >= 8192].sum()) if len(L) else 0}
        if cache:
            cache.put(key, res)
        return res


class DataCatalog:
    def __init__(self, data_root: Path, cache_dir: Path) -> None:
        self.data_root = Path(data_root)
        self.cache = DiskCache(cache_dir)
        self._raw: dict[str, RawSource] = {}
        self._splits: dict[tuple, TokenizedSplit] = {}
        self.lock = threading.Lock()

    @property
    def tokenized_root(self) -> Path:
        return self.data_root / "tokenized"

    def tags(self) -> list[str]:
        r = self.tokenized_root
        return sorted(d.name for d in r.iterdir() if d.is_dir()) if r.exists() else []

    def manifests(self, tag: str) -> dict[str, dict]:
        out = {}
        d = self.tokenized_root / tag
        if d.exists():
            for s in sorted(d.iterdir()):
                m = s / "manifest.json"
                if m.exists():
                    out[s.name] = json.loads(m.read_text(encoding="utf-8"))
        return out

    def sft_manifests(self) -> dict[str, dict[str, dict]]:
        out: dict[str, dict[str, dict]] = {}
        root = self.data_root / "sft"
        if root.exists():
            for tag in sorted(d for d in root.iterdir() if d.is_dir()):
                out[tag.name] = {}
                for s in sorted(tag.iterdir()):
                    m = s / "manifest.json"
                    if m.exists():
                        out[tag.name][s.name] = json.loads(m.read_text(encoding="utf-8"))
        return out

    def overview(self) -> dict:
        tags = self.tags()
        mani = {t: self.manifests(t) for t in tags}
        sources = []
        for name, src in SOURCES.items():
            raw = self.raw(name)
            files = raw.files()
            sources.append({
                "name": name, "repo": src.repo, "kind": src.kind, "license": src.license, "notes": src.notes, "content_via_swh": src.content_via_swh,
                "raw_files": len(files), "raw_bytes": sum(f["bytes"] for f in files), "raw_rows": sum(f["rows"] for f in files),
                "tokenized": {t: mani[t].get(name) for t in tags if mani[t].get(name)},
            })
        # Tokenized sources that are not registry entries (derived sets such as fineweb-edu-long,
        # fineweb-edu-b, or synthetic data) must still be browsable.
        known = {s["name"] for s in sources}
        for t in tags:
            for name, m in mani[t].items():
                if name in known:
                    continue
                known.add(name)
                sources.append({
                    "name": name, "repo": f"derived from {m.get('source', '?')}", "kind": m.get("kind", "derived"), "license": "", "content_via_swh": False,
                    "notes": f"derived tokenized set (min_doc_tokens={m.get('min_doc_tokens', '-')}); raw = {m.get('source', '?')}",
                    "raw_files": 0, "raw_bytes": 0, "raw_rows": 0, "tokenized": {tt: mani[tt].get(name) for tt in tags if mani[tt].get(name)},
                })
        return {"tags": tags, "sources": sources, "sft": self.sft_manifests()}

    def raw(self, name: str) -> RawSource:
        with self.lock:
            if name not in self._raw:
                self._raw[name] = RawSource(SOURCES[name])
            return self._raw[name]

    def split(self, tag: str, source: str, split: str) -> TokenizedSplit:
        key = (tag, source, split)
        with self.lock:
            if key not in self._splits:
                d = self.tokenized_root / tag / source / split
                if not d.is_dir():
                    raise KeyError(key)
                self._splits[key] = TokenizedSplit(d)
            return self._splits[key]

    def mixture(self, train_cfg) -> dict:
        tag = Path(train_cfg.data.tokenized_root).name
        mani = self.manifests(tag)
        total = train_cfg.schedule.total_tokens
        w = sum(train_cfg.data.mixture.values())
        rows = []
        for name, weight in train_cfg.data.mixture.items():
            m = mani.get(name, {})
            avail = m.get("train_tokens", 0)
            planned = total * weight / w
            rows.append({"source": name, "weight": weight / w, "available_tokens": avail, "planned_tokens": planned, "epochs": planned / avail if avail else None, "val_tokens": m.get("val_tokens", 0)})
        return {"tag": tag, "total_tokens": total, "seq_len": train_cfg.data.seq_len, "rows": rows}

"""Raw parquet and tokenized shard access for the data browser (row-group addressed, no scans)."""

from __future__ import annotations

import hashlib
import json
import random
import threading
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from slm.data.chat import rows_to_messages
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

    def _render(self, r: dict) -> tuple[str, list[dict] | None, dict]:
        """(display text, normalized chat messages or None, metadata) for one raw row."""
        kind = self.src.kind
        if kind in ("chat", "math_cot", "math_qa"):
            msgs = rows_to_messages(self.src, r)
            if msgs is None:  # would be dropped by the SFT pipeline; still show the raw content
                raw = r.get("messages") or [{"role": "user", "content": str(r.get("question", ""))}, {"role": "assistant", "content": str(r.get("answer", ""))}]
                msgs_disp, msgs = raw, None
            else:
                msgs_disp = msgs
            parts = []
            for m in msgs_disp:
                parts.append(f"[{m.get('role')}]")
                if m.get("think") is not None:
                    parts.append("<think>\n" + m["think"] + "\n</think>")
                parts.append(m.get("content", ""))
                parts.append("")
            text = "\n".join(parts).rstrip()
            meta = {k: (str(v)[:200] if not isinstance(v, (int, float)) else v) for k, v in r.items() if k not in ("messages", "question", "answer")}
            if msgs is None:
                meta["note"] = "row would be dropped by the SFT pipeline (unsupported roles/empty content/no numeric answer)"
            return text, msgs, meta
        text = r.get(self.src.text_col) or ""
        meta = {k: (str(v)[:200] if not isinstance(v, (int, float)) else v) for k, v in r.items() if k != self.src.text_col}
        return str(text), None, meta

    def docs(self, file_index: int, rg: int, offset: int = 0, limit: int = 50, preview_chars: int = 200) -> dict:
        p = self._path(file_index)
        pf = pq.ParquetFile(p)
        cols = [c for c in pf.schema_arrow.names if c != "prompt"]
        tbl = pf.read_row_group(rg, columns=cols)
        rows = tbl.slice(offset, limit).to_pylist()
        out = []
        for i, r in enumerate(rows):
            text, msgs, meta = self._render(r)
            out.append({"row": offset + i, "chars": len(text), "preview": text[:preview_chars], "meta": {k: (str(v)[:80] if not isinstance(v, (int, float)) else v) for k, v in meta.items()}, "turns": len(msgs) if msgs else None})
        return {"file": file_index, "rg": rg, "rg_rows": tbl.num_rows, "offset": offset, "docs": out}

    def doc(self, file_index: int, rg: int, row: int) -> dict:
        p = self._path(file_index)
        pf = pq.ParquetFile(p)
        cols = [c for c in pf.schema_arrow.names if c != "prompt"]
        r = pf.read_row_group(rg, columns=cols).slice(row, 1).to_pylist()[0]
        text, msgs, meta = self._render(r)
        return {"file": file_index, "rg": rg, "row": row, "text": text, "messages": msgs, "meta": meta}

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
        self._idx_sig: dict[int, tuple] = {}  # (mtime, size) of the .idx.npy the cached index came from
        self._dir_sig = self._dir_signature()
        self.lock = threading.Lock()

    def _dir_signature(self) -> tuple:
        try:
            return (self.dir.stat().st_mtime, len(list(self.dir.glob("shard_*.bin"))))
        except OSError:
            return ()

    def refresh(self) -> None:
        """Shards can be replaced under the same names (data swaps at a resume): re-list and drop stale indexes."""
        sig = self._dir_signature()
        if sig != self._dir_sig:
            with self.lock:
                self.paths = sorted(self.dir.glob("shard_*.bin"))
                self._idx.clear()
                self._idx_sig.clear()
                self._dir_sig = sig

    def shards(self) -> list[dict]:
        self.refresh()
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
        self.refresh()
        with self.lock:
            ip = self.paths[i].with_name(self.paths[i].name.replace(".bin", ".idx.npy"))
            st = ip.stat()
            sig = (st.st_mtime, st.st_size)
            if self._idx_sig.get(i) != sig:  # first use, or the file was replaced
                self._idx[i] = np.load(ip)
                self._idx_sig[i] = sig
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


# ------------------------------------------------------------------- manifest dialects
def normalize_manifest(name: str, m: dict) -> dict:
    """One shape for the four manifest writers, so every consumer reads the same keys.

    prepare.py          train_tokens/train_docs/... + files + source
    sft.py              train_examples/train_tokens/train_targets + max_len/think_required/tools
    sft_to_pretrain.py  train: {tokens, docs, shards}  + from_sft   (source == the output name)
    synth_retrieval.py / rl.synth.py   name + train_tokens, backdrop / tasks, no raw source
    """
    src = m.get("source")
    if "from_sft" in m:  # chat/tool SFT sets converted into a pretraining stream
        tr, va = m.get("train") or {}, m.get("val") or {}
        return {"kind": "tokenized", "made_by": "sft_to_pretrain", "from": list(m["from_sft"]), "raw_source": None,
                "train_tokens": int(tr.get("tokens", 0)), "train_docs": int(tr.get("docs", 0)), "train_shards": int(tr.get("shards", 0)),
                "val_tokens": int(va.get("tokens", 0)), "val_docs": int(va.get("docs", 0)), "val_shards": int(va.get("shards", 0)),
                "targets": None, "manifest": m}
    if "train_examples" in m:  # an SFT set (chat examples + loss mask), whether from parquet or templated
        made = "rl.synth" if "tasks" in m else "sft"
        return {"kind": "sft", "made_by": made, "from": list(m.get("tasks", [])) if made == "rl.synth" else ([src] if src else []), "raw_source": src,
                "train_tokens": int(m.get("train_tokens", 0)), "train_docs": int(m.get("train_examples", 0)), "train_shards": int(m.get("train_shards", 0)),
                "val_tokens": int(m.get("val_tokens", 0)), "val_docs": int(m.get("val_examples", 0)), "val_shards": int(m.get("val_shards", 0)),
                "targets": int(m.get("train_targets", 0)), "manifest": m}
    if "backdrop" in m:  # synthetic retrieval, templated over another tokenized set
        return {"kind": "tokenized", "made_by": "synth_retrieval", "from": [Path(str(m["backdrop"])).parent.name], "raw_source": None,
                "train_tokens": int(m.get("train_tokens", 0)), "train_docs": int(m.get("train_docs", 0)), "train_shards": int(m.get("train_shards", 0)),
                "val_tokens": int(m.get("val_tokens", 0)), "val_docs": int(m.get("val_docs", 0)), "val_shards": int(m.get("val_shards", 0)),
                "targets": None, "manifest": m}
    # prepare.py: `source` is the registry source, `name` the output set (they differ for fineweb-edu-long etc.)
    return {"kind": "tokenized", "made_by": "prepare", "from": [src] if src and src != name else [], "raw_source": src if src != name else src,
            "train_tokens": int(m.get("train_tokens", 0)), "train_docs": int(m.get("train_docs", 0)), "train_shards": int(m.get("train_shards", 0)),
            "val_tokens": int(m.get("val_tokens", 0)), "val_docs": int(m.get("val_docs", 0)), "val_shards": int(m.get("val_shards", 0)),
            "targets": None, "raw_files": len(m.get("files") or []), "manifest": m}


def _prov_line(name: str, p: dict) -> str:
    """One-line 'where did this come from' for a prepared set."""
    m = p["manifest"]
    if p["made_by"] == "sft_to_pretrain":
        return f"{len(p['from'])} SFT sets in chat format: " + ", ".join(p["from"])
    if p["made_by"] == "synth_retrieval":
        return f"templated needle/ledger documents over {p['from'][0] if p['from'] else '?'} (seed {m.get('seed', '-')}, {m.get('min_len', '-')}–{m.get('max_len', '-')} tokens)"
    if p["made_by"] == "rl.synth":
        return "templated reasoning traces for tasks " + ", ".join(p["from"])
    if p["made_by"] == "sft":
        return f"chat-formatted from {p['raw_source']} (max_len {m.get('max_len', '-')}, think {'required' if m.get('think_required') else 'optional'}{', tools' if m.get('tools') else ''})"
    raw = p.get("raw_source") or "?"
    extra = f", min_doc_tokens {m['min_doc_tokens']}" if m.get("min_doc_tokens") else ""
    return f"tokenized from {raw}" + (f" ({p.get('raw_files', 0)} parquet files{extra})" if p.get("raw_files") else extra)


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
        # fineweb-edu-b, or synthetic data) must still be browsable. Provenance comes from the
        # normalizer, not from manifest["source"], which two writers set to the *output* name.
        known = {s["name"] for s in sources}
        for t in tags:
            for name, m in mani[t].items():
                if name in known:
                    continue
                known.add(name)
                p = normalize_manifest(name, m)
                sources.append({
                    "name": name, "repo": _prov_line(name, p), "kind": m.get("kind", "derived"), "license": "", "content_via_swh": False,
                    "notes": _prov_line(name, p), "made_by": p["made_by"], "from": p["from"],
                    "raw_files": 0, "raw_bytes": 0, "raw_rows": 0, "tokenized": {tt: mani[tt].get(name) for tt in tags if mani[tt].get(name)},
                })
        for s in sources:
            s["prepared"] = {t: normalize_manifest(s["name"], mani[t][s["name"]]) for t in tags if mani[t].get(s["name"])}
        return {"tags": tags, "sources": sources, "sft": self.sft_manifests()}

    def prepared(self, tag: str) -> dict[str, dict]:
        """Normalized prepared artifacts for one tokenizer tag: tokenized sets plus SFT sets."""
        out = {n: normalize_manifest(n, m) for n, m in self.manifests(tag).items()}
        for n, m in self.sft_manifests().get(tag, {}).items():
            out.setdefault(f"sft:{n}", normalize_manifest(n, m))
        return out

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

    # ------------------------------------------------------------------- recipes
    def _row(self, name: str, weight: float, wsum: float, total: int, prep: dict, kind: str) -> dict:
        p = prep.get(name)
        avail = p["train_tokens"] if p else 0
        planned = total * weight / wsum if wsum else 0.0
        raw_src = (p or {}).get("raw_source")
        return {"source": name, "weight": weight / wsum if wsum else 0.0, "planned_tokens": planned, "available_tokens": avail,
                "epochs": planned / avail if avail else None, "val_tokens": (p or {}).get("val_tokens", 0),
                "prepared": p, "provenance": _prov_line(name, p) if p else None, "missing": p is None, "kind": kind,
                "raw": {"source": raw_src, "files": (p or {}).get("raw_files", 0)} if raw_src and raw_src in SOURCES else None}

    def recipe(self, cfg: dict, stage: str, cache: dict | None = None) -> dict:
        """The data section of one training config or run, resolved against what is on disk.

        `cfg` is a plain dict (yaml under TrainConfig defaults, or run.json's `config`), never a
        dataclass: RL configs have a different shape and must not go through TrainConfig at all.
        """
        if stage == "rl":
            return self._rl_recipe(cfg)
        cache = {} if cache is None else cache
        data, sched = cfg.get("data") or {}, cfg.get("schedule") or {}
        is_sft = (data.get("kind") or "pretrain") == "sft"
        tok_root = data.get("tokenized_root") or ""
        root = (data.get("sft_root") or "") if is_sft else tok_root
        tag, tok_tag = Path(root).name, Path(tok_root).name
        if ("tok", tok_tag) not in cache:
            cache[("tok", tok_tag)] = {n: normalize_manifest(n, m) for n, m in self.manifests(tok_tag).items()}
        if is_sft and ("sft", tag) not in cache:
            cache[("sft", tag)] = {n: normalize_manifest(n, m) for n, m in self.sft_manifests().get(tag, {}).items()}
        tok_prep = cache[("tok", tok_tag)]
        prep = cache[("sft", tag)] if is_sft else tok_prep
        mixture = {k: float(v) for k, v in (data.get("mixture") or {}).items()}
        wsum = sum(mixture.values())
        avail_total = sum((prep.get(n) or {}).get("train_tokens", 0) for n in mixture)
        total, note = int(sched.get("total_tokens") or 0), None
        if float(sched.get("epochs") or 0) > 0:  # the trainer resolves epochs against the loader at start (pretrain.py)
            total = int(float(sched["epochs"]) * avail_total)
            note = f"{sched['epochs']} epochs x {avail_total / 1e6:.0f}M tokens in the mixture"
        rows = [self._row(n, w, wsum, total, prep, "sft" if is_sft else "tokenized") for n, w in mixture.items()]
        rows.sort(key=lambda r: -r["weight"])
        ev = {k: float(v) for k, v in (data.get("extra_val_mixture") or {}).items()}
        evs = sum(ev.values())
        extra_val = [self._row(n, w, evs, 0, tok_prep, "tokenized") for n, w in ev.items()]
        extra_val.sort(key=lambda r: -r["weight"])
        return {"stage": "sft" if is_sft else "pretrain", "tag": tag, "tokenizer_tag": tok_tag, "root": root, "seq_len": int(data.get("seq_len") or 0),
                "total_tokens": total, "total_note": note, "available_tokens": avail_total, "schedule": sched,
                "init_from": cfg.get("init_from") or "", "rows": rows, "extra_val": extra_val, "extra_val_tokens": int(data.get("val_tokens") or 0), "rl": None}

    def _rl_recipe(self, cfg: dict) -> dict:
        from slm.train.rl_config import REWARD_RULES  # torch-free

        tasks = list(cfg.get("tasks") or [])
        counts: dict[str, int] = {}
        for t in tasks:
            counts[t] = counts.get(t, 0) + 1
        n = len(tasks) or 1
        rl = {"tasks": sorted(counts), "task_weights": {k: v / n for k, v in counts.items()}, "task_counts": counts,
              "reward_rule": REWARD_RULES.get(cfg.get("reward_scheme", "binary"), "?"),
              **{k: cfg.get(k) for k in ("n_train_prompts", "n_heldout_prompts", "group_size", "prompts_per_step", "max_new_tokens",
                                         "think_required", "tools", "max_tool_calls", "reward_scheme", "kl_coef", "kl_kind", "clip_eps",
                                         "temperature", "top_p", "entropy_stop", "kl_stop", "total_steps", "lr", "seed")}}
        return {"stage": "rl", "tag": Path(cfg.get("tokenizer_dir") or "").name, "tokenizer_tag": Path(cfg.get("tokenizer_dir") or "").name,
                "root": "", "seq_len": 0, "total_tokens": 0, "total_note": f"{cfg.get('total_steps')} steps x {cfg.get('prompts_per_step')} prompts x {cfg.get('group_size')} rollouts",
                "available_tokens": 0, "schedule": {}, "init_from": cfg.get("init_from") or "", "rows": [], "extra_val": [], "extra_val_tokens": 0, "rl": rl}

    def mixture(self, train_cfg) -> dict:  # legacy shape, kept for one release (see docs/command_center.md)
        from slm.config import to_dict

        r = self.recipe(to_dict(train_cfg), "sft" if train_cfg.data.kind == "sft" else "pretrain")
        return {"tag": r["tag"], "total_tokens": r["total_tokens"], "seq_len": r["seq_len"], "rows": r["rows"]}

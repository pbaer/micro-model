from __future__ import annotations

import json
from pathlib import Path

import anyio
from fastapi import APIRouter, HTTPException, Request

from slm.config import to_dict
from slm.train.config import load_train_config
from slm.train.rl_config import load_rl_config  # torch-free copy of RlConfig, so RL yaml never 500s

router = APIRouter(prefix="/api/data")

# ------------------------------------------------------------------------- recipes
# A recipe is the data section of one training config ("config:<path>", the plan) or one run
# ("run:<name>", what actually happened, from runs/<run>/run.json).


def _is_rl(d: dict) -> bool:
    return "tasks" in d or "total_steps" in d or "group_size" in d


def _config_dict(path: Path) -> tuple[dict, str]:
    """Full config dict with defaults applied, plus its stage. RL yaml never goes through TrainConfig."""
    from slm.config import load_yaml

    raw = load_yaml(str(path))
    if _is_rl(raw):
        return to_dict(load_rl_config(str(path))), "rl"
    cfg = to_dict(load_train_config(str(path)))
    return cfg, "sft" if (cfg.get("data") or {}).get("kind") == "sft" else "pretrain"


def _configs(request: Request) -> list[Path]:
    root = Path(request.app.state.settings.configs_root) / "train"
    return sorted(root.glob("*.yaml")) if root.exists() else []


def _run_cfg(request: Request, name: str) -> tuple[dict, str, dict]:
    try:
        reader = request.app.state.runs.get(name)
    except KeyError:
        raise HTTPException(404, f"unknown run {name!r}") from None
    meta = reader.meta()
    cfg = meta.get("config") or {}
    stage = {"grpo": "rl", "rl": "rl", "sft": "sft"}.get(meta.get("stage") or "", "rl" if _is_rl(cfg) else "pretrain")
    return cfg, stage, meta


def _plan_differs(run_cfg: dict, cfg_path: Path | None) -> bool | None:
    """The yaml under the run's name may have been edited after launch (M3a's data swap)."""
    if cfg_path is None or not cfg_path.exists():
        return None
    try:
        plan, _ = _config_dict(cfg_path)
    except Exception:
        return None
    keys = ("data", "init_from", "tasks", "reward_scheme", "group_size", "prompts_per_step", "n_train_prompts")
    if any(json.dumps(plan.get(k), sort_keys=True, default=str) != json.dumps(run_cfg.get(k), sort_keys=True, default=str) for k in keys):
        return True
    a, b = plan.get("schedule") or {}, run_cfg.get("schedule") or {}
    # epochs configs have total_tokens resolved at start, so compare it only when epochs is off
    if float(a.get("epochs") or 0) != float(b.get("epochs") or 0):
        return True
    return not float(a.get("epochs") or 0) and int(a.get("total_tokens") or 0) != int(b.get("total_tokens") or 0)


@router.get("/recipes")
async def recipes(request: Request) -> list[dict]:
    def build() -> list[dict]:
        cat = request.app.state.data
        cache: dict = {}
        by_name = {p.stem: p for p in _configs(request)}
        summaries = {s["run_name"]: s for s in request.app.state.runs.summaries()}
        out = []
        for name, s in summaries.items():
            cfg, stage, meta = _run_cfg(request, name)
            r = cat.recipe(cfg, stage, cache)
            p = by_name.get(name)
            out.append({**r, "id": f"run:{name}", "kind": "run", "run_name": name, "config_path": str(p).replace("\\", "/") if p else None,
                        "status": s.get("status"), "tokens": s.get("tokens"), "plan_differs": _plan_differs(cfg, p),
                        "rows": [{"source": x["source"], "weight": x["weight"]} for x in r["rows"]]})
        for p in by_name.values():
            if p.stem in summaries:
                continue
            try:
                cfg, stage = _config_dict(p)
            except Exception as e:  # a broken yaml must not blank the whole list
                out.append({"id": f"config:{str(p).replace(chr(92), '/')}", "kind": "config", "run_name": p.stem, "stage": "?", "status": "error", "error": str(e), "rows": []})
                continue
            r = cat.recipe(cfg, stage, cache)
            out.append({**r, "id": f"config:{str(p).replace(chr(92), '/')}", "kind": "config", "run_name": p.stem, "config_path": str(p).replace("\\", "/"),
                        "status": "not started", "tokens": None, "plan_differs": None,
                        "rows": [{"source": x["source"], "weight": x["weight"]} for x in r["rows"]]})
        order = {"pretrain": 0, "sft": 1, "rl": 2, "?": 3}
        out.sort(key=lambda r: (order.get(r.get("stage"), 9), r["run_name"]))
        return out

    return await anyio.to_thread.run_sync(build)


@router.get("/recipe")
async def recipe(request: Request, id: str) -> dict:
    kind, _, ref = id.partition(":")
    if kind == "run":
        cfg, stage, meta = _run_cfg(request, ref)
        reader = request.app.state.runs.get(ref)
        p = Path(request.app.state.settings.configs_root) / "train" / f"{ref}.yaml"
        extra = {"kind": "run", "run_name": ref, "config_path": str(p).replace("\\", "/") if p.exists() else None,
                 "status": reader.summary().get("status"), "tokens": reader.summary().get("tokens"),
                 "started": meta.get("started"), "plan_differs": _plan_differs(cfg, p if p.exists() else None),
                 "stream_sources": reader.stream_sources()}
    elif kind == "config":
        p = Path(ref)
        if not p.exists():
            raise HTTPException(404, "config not found")
        cfg, stage = _config_dict(p)
        extra = {"kind": "config", "run_name": p.stem, "config_path": str(p).replace("\\", "/"), "status": "not started", "tokens": None,
                 "plan_differs": None, "stream_sources": None}
    else:
        raise HTTPException(400, "id must be run:<name> or config:<path>")
    r = await anyio.to_thread.run_sync(lambda: request.app.state.data.recipe(cfg, stage))
    return {**r, **extra, "id": id}


@router.get("/sources")
async def sources(request: Request) -> dict:
    return await anyio.to_thread.run_sync(request.app.state.data.overview)


@router.get("/configs")
def configs(request: Request) -> list[str]:
    root = Path(request.app.state.settings.configs_root) / "train"
    return sorted(str(p).replace("\\", "/") for p in root.glob("*.yaml")) if root.exists() else []


@router.get("/mixture")
async def mixture(request: Request, config: str) -> dict:
    p = Path(config)
    if not p.exists():
        raise HTTPException(404, "config not found")
    cfg = load_train_config(str(p))
    return await anyio.to_thread.run_sync(lambda: request.app.state.data.mixture(cfg))


@router.get("/raw/{source}/files")
async def raw_files(request: Request, source: str) -> list[dict]:
    try:
        raw = request.app.state.data.raw(source)
    except KeyError:
        raise HTTPException(404, "unknown source") from None
    return await anyio.to_thread.run_sync(raw.files)


@router.get("/raw/{source}/docs")
async def raw_docs(request: Request, source: str, file: int = 0, rg: int = 0, offset: int = 0, limit: int = 50) -> dict:
    raw = request.app.state.data.raw(source)
    return await anyio.to_thread.run_sync(lambda: raw.docs(file, rg, offset, min(limit, 500)))


@router.get("/raw/{source}/doc")
async def raw_doc(request: Request, source: str, file: int, rg: int, row: int) -> dict:
    raw = request.app.state.data.raw(source)
    return await anyio.to_thread.run_sync(lambda: raw.doc(file, rg, row))


@router.get("/raw/{source}/sample")
async def raw_sample(request: Request, source: str, n: int = 10, seed: int = 0) -> list[dict]:
    raw = request.app.state.data.raw(source)
    return await anyio.to_thread.run_sync(lambda: raw.sample(min(n, 50), seed))


def _split(request: Request, tag: str, source: str, split: str):
    try:
        return request.app.state.data.split(tag, source, split)
    except KeyError:
        raise HTTPException(404, "no such tokenized split") from None


@router.get("/tokenized/{tag}/{source}/{split}/shards")
async def shards(request: Request, tag: str, source: str, split: str) -> list[dict]:
    return await anyio.to_thread.run_sync(_split(request, tag, source, split).shards)


@router.get("/tokenized/{tag}/{source}/{split}/docs")
async def docs(request: Request, tag: str, source: str, split: str, shard: int = 0, offset: int = 0, limit: int = 100) -> dict:
    s = _split(request, tag, source, split)
    return await anyio.to_thread.run_sync(lambda: s.docs(shard, offset, min(limit, 1000)))


@router.get("/tokenized/{tag}/{source}/{split}/doc")
async def doc(request: Request, tag: str, source: str, split: str, shard: int, doc: int) -> dict:
    s = _split(request, tag, source, split)
    d = await anyio.to_thread.run_sync(lambda: s.doc(shard, doc))
    reg = request.app.state.tokenizers
    d["pieces"] = reg.pieces(tag, d["ids"])
    d["text"] = reg.get(tag).decode(d["ids"], skip_special=True)
    return d


@router.get("/tokenized/{tag}/{source}/{split}/window")
async def window(request: Request, tag: str, source: str, split: str, shard: int, start: int, length: int = 1024) -> dict:
    s = _split(request, tag, source, split)
    d = await anyio.to_thread.run_sync(lambda: s.window(shard, start, min(length, 16384)))
    d["pieces"] = request.app.state.tokenizers.pieces(tag, d["ids"])
    return d


# --------------------------------------------------------------------------- SFT shards
def _sft(request: Request, tag: str, name: str, split: str):
    try:
        return request.app.state.data.sft_split(tag, name, split)
    except KeyError:
        raise HTTPException(404, "no such sft split") from None


@router.get("/sft/{tag}/{name}/{split}/shards")
async def sft_shards(request: Request, tag: str, name: str, split: str) -> list[dict]:
    return await anyio.to_thread.run_sync(_sft(request, tag, name, split).shards)


@router.get("/sft/{tag}/{name}/{split}/examples")
async def sft_examples(request: Request, tag: str, name: str, split: str, shard: int = 0, offset: int = 0, limit: int = 100) -> dict:
    s = _sft(request, tag, name, split)
    return await anyio.to_thread.run_sync(lambda: s.examples(shard, offset, min(limit, 1000)))


@router.get("/sft/{tag}/{name}/{split}/example")
async def sft_example(request: Request, tag: str, name: str, split: str, shard: int, ex: int) -> dict:
    s = _sft(request, tag, name, split)
    d = await anyio.to_thread.run_sync(lambda: s.example(shard, ex))
    reg = request.app.state.tokenizers
    d["pieces"] = [{**p, "loss": m} for p, m in zip(reg.pieces(tag, d["ids"]), d["mask"])]
    d["text"] = reg.get(tag).decode(d["ids"], skip_special=True)
    return d


@router.get("/sft/{tag}/{name}/{split}/window")
async def sft_window(request: Request, tag: str, name: str, split: str, shard: int, start: int, length: int = 1024) -> dict:
    """One packed SFT training row: seq_len consecutive tokens, the loss mask, and example boundaries."""
    s = _sft(request, tag, name, split)
    d = await anyio.to_thread.run_sync(lambda: s.window(shard, start, min(length, 16384)))
    d["pieces"] = [{**p, "loss": m} for p, m in zip(request.app.state.tokenizers.pieces(tag, d["ids"]), d["mask"])]
    return d


@router.get("/sft/{tag}/{name}/{split}/stats")
async def sft_stats(request: Request, tag: str, name: str, split: str) -> dict:
    s = _sft(request, tag, name, split)
    return await anyio.to_thread.run_sync(lambda: s.stats(request.app.state.data.cache))


@router.post("/raw/{source}/trace")
async def raw_trace(request: Request, source: str, body: dict) -> dict:
    """Run one raw parquet row through the preparation it would get, and say what happened to it.

    There is no stored row -> document mapping for pretraining shards (prepare.py records none), so
    this re-derives the result: same tokenizer, same filters, exact by construction. For SFT sets the
    stored example in the shard is the ground truth and this is explanatory only (the marker/natural
    style is drawn from a per-set RNG that cannot be replayed for a single row).
    """
    from slm.data.prepare import MAX_DOC_TOKENS, is_val, keep_doc

    cat, reg = request.app.state.data, request.app.state.tokenizers
    tag = body.get("tag") or ""
    raw = cat.raw(source)
    file, rg, row = int(body.get("file", 0)), int(body.get("rg", 0)), int(body.get("row", 0))

    def work() -> dict:
        from slm.data.chat import format_chat, rows_to_messages
        from slm.data.sources import SOURCES

        import pyarrow.parquet as pq

        p = Path(raw.files()[file]["path"])
        pf = pq.ParquetFile(p)
        cols = [c for c in pf.schema_arrow.names if c != "prompt"]
        r = pf.read_row_group(rg, columns=cols).slice(row, 1).to_pylist()[0]
        tok = reg.get(tag)
        src = SOURCES[source]
        if body.get("sft_set"):
            m = cat.sft_manifests().get(tag, {}).get(body["sft_set"], {})
            msgs = rows_to_messages(src, r)
            if msgs is None:
                return {"kept": False, "reason": "rows_to_messages dropped the row (unsupported roles, empty content, or no numeric answer)", "pipeline": "sft"}
            enc = format_chat(tok, msgs, think_required=bool(m.get("think_required")), tools=bool(m.get("tools")))
            too_long = len(enc.ids) > int(m.get("max_len") or 10**9)
            return {"kept": not too_long, "pipeline": "sft", "split": None,
                    "reason": f"longer than max_len {m.get('max_len')} ({len(enc.ids)} tokens): dropped, never truncated" if too_long else f"kept: {len(enc.ids)} tokens, {sum(enc.loss_mask)} loss targets",
                    "ids": enc.ids, "n_tokens": len(enc.ids), "n_target": sum(enc.loss_mask),
                    "pieces": [{"id": i, "piece": tok.token_str(i), "special": i >= tok.base_vocab, "loss": lm} for i, lm in zip(enc.ids, enc.loss_mask)]}
        text = r.get(src.text_col) or ""
        if not keep_doc(src, r):
            lang = r.get("language")
            why = "shorter than 64 characters" if len(text) < 64 else (f"language {lang!r} != en" if lang is not None and lang != "en" else "too few ASCII letters for English prose")
            return {"kept": False, "reason": f"dropped by keep_doc: {why}", "pipeline": "pretrain"}
        ids = tok.encode_document(text)
        min_tok = int(body.get("min_doc_tokens") or 16)
        if not (min_tok <= len(ids) <= MAX_DOC_TOKENS):
            return {"kept": False, "reason": f"dropped: {len(ids)} tokens outside [{min_tok}, {MAX_DOC_TOKENS}]", "pipeline": "pretrain", "n_tokens": len(ids)}
        split = "val" if is_val(text, int(body.get("val_permille") or 5)) else "train"
        return {"kept": True, "pipeline": "pretrain", "split": split, "n_tokens": len(ids), "ids": ids,
                "reason": f"kept -> {split} ({len(ids)} tokens incl. bos/eos; the split is a hash of the first 2048 characters)",
                "pieces": reg.pieces(tag, ids)}

    try:
        return await anyio.to_thread.run_sync(work)
    except KeyError:
        raise HTTPException(404, "unknown source or tokenizer") from None


@router.get("/tokenized/{tag}/{source}/{split}/stats")
async def stats(request: Request, tag: str, source: str, split: str) -> dict:
    s = _split(request, tag, source, split)
    return await anyio.to_thread.run_sync(lambda: s.stats(request.app.state.data.cache))

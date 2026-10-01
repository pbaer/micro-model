from __future__ import annotations

import json
from pathlib import Path

import anyio
from fastapi import APIRouter, HTTPException, Request

from slm.config import to_dict
from slm.portal.services.tokenizer import text_runs
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


@router.get("/rl/prompts")
async def rl_prompts(request: Request, id: str, split: str = "train", offset: int = 0, limit: int = 20) -> dict:
    """The deterministic prompt list of an RL recipe, rendered as the exact generation prompt.

    `make_tasks` is seeded, so this is the same list the trainer builds; slm.rl.tasks, slm.data.answers
    and slm.data.chat import no torch, so it runs in the portal process.
    """
    kind, _, ref = id.partition(":")
    if kind == "run":
        cfg, stage, _ = _run_cfg(request, ref)
    elif kind == "config":
        if not Path(ref).exists():
            raise HTTPException(404, "config not found")
        cfg, stage = _config_dict(Path(ref))
    else:
        raise HTTPException(400, "id must be run:<name> or config:<path>")
    if stage != "rl":
        raise HTTPException(400, "not an RL recipe")

    def work() -> dict:
        from slm.data.chat import format_chat
        from slm.rl.tasks import make_tasks, prompt_messages

        names = list(cfg.get("tasks") or [])
        n = int(cfg.get("n_train_prompts" if split == "train" else "n_heldout_prompts") or 0)
        tasks = make_tasks(names, n, split, seed=int(cfg.get("seed") or 0))
        tag = Path(cfg.get("tokenizer_dir") or "").name
        tok = request.app.state.tokenizers.get(tag)
        out = []
        for t in tasks[offset : offset + min(limit, 100)]:
            enc = format_chat(tok, prompt_messages(t), add_generation_prompt=True, think_required=bool(cfg.get("think_required")))
            out.append({"prompt_id": t.id, "task": t.task, "prompt": t.prompt, "gold": t.answer,
                        "ids": enc.ids, "n_tokens": len(enc.ids), "pieces": request.app.state.tokenizers.pieces(tag, enc.ids),
                        "runs": request.app.state.tokenizers.runs(tag, enc.ids)})
        return {"split": split, "n": len(tasks), "offset": offset, "tag": tag, "prompts": out}

    try:
        return await anyio.to_thread.run_sync(work)
    except KeyError as e:
        raise HTTPException(404, f"tokenizer or task not found: {e}") from None


@router.get("/chain")
async def chain(request: Request, run: str) -> dict:
    """Cumulative data exposure along the init_from chain ending at `run`, one row per stage.

    Per-source tokens are the loader's own counts when the run's last `checkpoint` record carries a
    `sources` block (scaled when the child loaded a mid-run checkpoint), and `tokens_used x weight`
    otherwise; `actual` says which, per row. RL stages contribute no mixture (the model trains on its
    own samples), so they show as tokens with no per-source split.
    """

    def build() -> dict:
        idx, cat = request.app.state.runs, request.app.state.data
        entries = idx.chain(run)
        if not entries:
            raise HTTPException(404, f"unknown run {run!r}")
        cache: dict = {}
        rows, totals = [], {}
        for e in entries:
            cfg, stage, _ = _run_cfg(request, e["run"])
            r = cat.recipe(cfg, stage, cache)
            used, own = int(e["tokens_used"]), int(e["own_tokens"] or 0)
            streams = idx.get(e["run"]).stream_sources()
            if streams and own:
                frac = used / own  # a mid-run checkpoint means only part of what the loader consumed
                per = {k: float((v or {}).get("tokens", 0)) * frac for k, v in streams.items()}
                actual = True
            else:
                per, actual = {x["source"]: used * x["weight"] for x in r["rows"]}, False
            for k, v in per.items():
                totals[k] = totals.get(k, 0.0) + v
            rows.append({**e, "seq_len": r["seq_len"], "tag": r["tokenizer_tag"], "per_source": per,
                         "actual": actual, "n_sources": len(r["rows"]), "rl": r["rl"] is not None})
        order = sorted(totals, key=lambda k: -totals[k])
        return {"run": run, "runs": rows, "sources": order, "totals": totals,
                "cumulative_tokens": sum(int(e["tokens_used"]) for e in entries),
                "any_expected": any(not x["actual"] and x["per_source"] for x in rows)}

    return await anyio.to_thread.run_sync(build)


@router.get("/sources")
async def sources(request: Request) -> dict:
    return await anyio.to_thread.run_sync(request.app.state.data.overview)


@router.get("/source/{name}")
async def source(request: Request, name: str) -> dict:
    """One catalog entry: raw files, prepared artifacts per tag, provenance both ways, and the
    recipes whose mixture names it (with the weight each gives it)."""

    def build() -> dict:
        from slm.data.sources import SOURCES
        from slm.portal.services.datasets import _prov_line, normalize_manifest

        cat = request.app.state.data
        prepared: dict[str, dict] = {}
        parents: list[str] = []
        for tag in cat.tags():
            m = cat.manifests(tag).get(name)
            if m:
                p = normalize_manifest(name, m)
                prepared[f"tokenized:{tag}"] = {**p, "tag": tag, "provenance": _prov_line(name, p)}
                parents += [x for x in p["from"] if x and x != name]
        for tag, sets in cat.sft_manifests().items():
            if name in sets:
                p = normalize_manifest(name, sets[name])
                prepared[f"sft:{tag}"] = {**p, "tag": tag, "provenance": _prov_line(name, p)}
                parents += [x for x in p["from"] if x and x != name]
        src = SOURCES.get(name)
        children = []
        for tag in cat.tags():
            for other, m in cat.manifests(tag).items():
                if other != name and name in normalize_manifest(other, m)["from"]:
                    children.append(other)
        for tag, sets in cat.sft_manifests().items():
            for other, m in sets.items():
                if other != name and name in normalize_manifest(other, m)["from"]:
                    children.append(other)
        if src is None and not prepared:
            raise HTTPException(404, f"unknown source {name!r}")
        return {"name": name, "kind": src.kind if src else (list(prepared.values())[0]["kind"] if prepared else "?"),
                "license": src.license if src else "", "repo": src.repo if src else "",
                "notes": src.notes if src else (list(prepared.values())[0]["provenance"] if prepared else ""),
                "raw_files": cat.raw(name).files() if src else [], "prepared": prepared,
                "parents": sorted(set(parents)), "children": sorted(set(children)),
                "browsable": bool(src) or any(p["kind"] == "tokenized" for p in prepared.values())}

    out = await anyio.to_thread.run_sync(build)
    out["used_by"] = [{"recipe_id": r["id"], "run_name": r["run_name"], "stage": r.get("stage"), "kind": r["kind"], "weight": x["weight"]}
                      for r in await recipes(request) for x in (r.get("rows") or []) if x["source"] == name]
    return out


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
    d["runs"] = reg.runs(tag, d["ids"])  # the text view: decoded text with the reserved tokens kept as their own runs
    return d


@router.get("/tokenized/{tag}/{source}/{split}/window")
async def window(request: Request, tag: str, source: str, split: str, shard: int, start: int, length: int = 1024) -> dict:
    s = _split(request, tag, source, split)
    d = await anyio.to_thread.run_sync(lambda: s.window(shard, start, min(length, 16384)))
    d["pieces"] = request.app.state.tokenizers.pieces(tag, d["ids"])
    d["runs"] = request.app.state.tokenizers.runs(tag, d["ids"])
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
    d["runs"] = reg.runs(tag, d["ids"], d["mask"])
    return d


@router.get("/sft/{tag}/{name}/{split}/window")
async def sft_window(request: Request, tag: str, name: str, split: str, shard: int, start: int, length: int = 1024) -> dict:
    """One packed SFT training row: seq_len consecutive tokens, the loss mask, and example boundaries."""
    s = _sft(request, tag, name, split)
    d = await anyio.to_thread.run_sync(lambda: s.window(shard, start, min(length, 16384)))
    d["pieces"] = [{**p, "loss": m} for p, m in zip(request.app.state.tokenizers.pieces(tag, d["ids"]), d["mask"])]
    d["runs"] = request.app.state.tokenizers.runs(tag, d["ids"], d["mask"])
    return d


@router.get("/sft/{tag}/{name}/{split}/stats")
async def sft_stats(request: Request, tag: str, name: str, split: str) -> dict:
    s = _sft(request, tag, name, split)
    return await anyio.to_thread.run_sync(lambda: s.stats(request.app.state.data.cache))


def _trace_book(books_jsonl: Path, r: dict, tok, reg, tag: str, head_tokens: int = 2048) -> dict:
    """A PG-19 book's fate comes from a corpus-level ranking (dialogue density until the token target), which one
    row cannot replay, so the trace reads the per-book verdict slm.data.gutenberg wrote next to the manifest."""
    from slm.data.gutenberg import RULE_TEXT, normalize, strip_boilerplate

    rec = None
    if books_jsonl.exists():
        with open(books_jsonl, encoding="utf-8") as f:
            for line in f:
                b = json.loads(line)
                if b["book_id"] == r.get("book_id"):
                    rec = b
                    break
    if rec is None:
        return {"kept": False, "pipeline": "pretrain", "reason": f"book {r.get('book_id')} is not in {books_jsonl.name} for tag {tag!r} (not prepared yet)"}
    sig = (f"dialogue {rec['dialogue']:.3f}, caps {rec['caps_share']:.3f}, verse {rec['verse_share']:.3f}, "
           f"archaic {rec['archaic_per_1k']:.2f}/1K words, {rec['words']:,} words")
    if not rec["selected"]:
        return {"kept": False, "pipeline": "pretrain", "n_tokens": rec["tokens"],
                "reason": f"dropped by rule '{rec['verdict']}': {RULE_TEXT.get(rec['verdict'], '')} ({sig})"}
    ids = [tok.bos_id, *tok.encode(normalize(strip_boilerplate(r.get("text") or ""))[: head_tokens * 8])][:head_tokens]
    return {"kept": True, "pipeline": "pretrain", "split": rec["split"], "n_tokens": rec["tokens"], "ids": ids,
            "reason": f"kept -> {rec['split']}: {rec['tokens']:,} tokens in {rec['segments']} documents ({sig}); showing the first {len(ids)} tokens",
            "pieces": reg.pieces(tag, ids), "runs": text_runs(tok, ids)}


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
                    "pieces": [{"id": i, "piece": tok.token_str(i), "special": i >= tok.base_vocab, "loss": lm} for i, lm in zip(enc.ids, enc.loss_mask)],
                    "runs": text_runs(tok, enc.ids, enc.loss_mask)}
        text = r.get(src.text_col) or ""
        if src.custom_prepare == "slm.data.gutenberg":  # selection is corpus-level: read the verdict prepare recorded
            return _trace_book(cat.tokenized_root / tag / source / "books.jsonl", r, tok, reg, tag)
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
                "pieces": reg.pieces(tag, ids), "runs": text_runs(tok, ids)}

    try:
        return await anyio.to_thread.run_sync(work)
    except KeyError:
        raise HTTPException(404, "unknown source or tokenizer") from None


@router.get("/tokenized/{tag}/{source}/{split}/stats")
async def stats(request: Request, tag: str, source: str, split: str) -> dict:
    s = _split(request, tag, source, split)
    return await anyio.to_thread.run_sync(lambda: s.stats(request.app.state.data.cache))

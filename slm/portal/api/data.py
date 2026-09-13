from __future__ import annotations

from pathlib import Path

import anyio
from fastapi import APIRouter, HTTPException, Request

from slm.train.config import load_train_config

router = APIRouter(prefix="/api/data")


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
    d["pieces"] = request.app.state.tokenizers.pieces(tag, d["ids"])
    return d


@router.get("/tokenized/{tag}/{source}/{split}/window")
async def window(request: Request, tag: str, source: str, split: str, shard: int, start: int, length: int = 1024) -> dict:
    s = _split(request, tag, source, split)
    d = await anyio.to_thread.run_sync(lambda: s.window(shard, start, min(length, 16384)))
    d["pieces"] = request.app.state.tokenizers.pieces(tag, d["ids"])
    return d


@router.get("/tokenized/{tag}/{source}/{split}/stats")
async def stats(request: Request, tag: str, source: str, split: str) -> dict:
    s = _split(request, tag, source, split)
    return await anyio.to_thread.run_sync(lambda: s.stats(request.app.state.data.cache))

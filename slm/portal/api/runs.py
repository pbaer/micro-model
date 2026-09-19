from __future__ import annotations

import asyncio
import json

import anyio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

router = APIRouter(prefix="/api/runs")


def _reader(request: Request, run: str):
    try:
        return request.app.state.runs.get(run)
    except KeyError:
        raise HTTPException(404, f"unknown run {run!r}") from None


@router.get("")
async def list_runs(request: Request) -> list[dict]:
    return await anyio.to_thread.run_sync(request.app.state.runs.summaries)


@router.get("/{run}")
async def get_run(request: Request, run: str) -> dict:
    r = _reader(request, run)
    summary = await anyio.to_thread.run_sync(r.summary)
    meta = r.meta()
    return {"summary": summary, "meta": {k: v for k, v in meta.items() if k != "config"}, "config": meta.get("config", {}), "model_config": meta.get("model_config", {})}


@router.get("/{run}/series")
async def get_series(request: Request, run: str, max_points: int = 1500) -> dict:
    r = _reader(request, run)
    return await anyio.to_thread.run_sync(lambda: r.series(max_points))


@router.get("/{run}/events")
async def get_events(request: Request, run: str, limit: int = 50) -> list[dict]:
    r = _reader(request, run)
    return await anyio.to_thread.run_sync(lambda: r.events(limit))


@router.get("/{run}/checkpoints")
async def get_checkpoints(request: Request, run: str) -> list[dict]:
    r = _reader(request, run)
    return await anyio.to_thread.run_sync(r.checkpoints)


@router.get("/{run}/rollouts")
async def get_rollouts(request: Request, run: str, step: int | None = None, offset: int = 0, limit: int = 20) -> dict:
    """RL rollouts of one step: prompt, completion, reward, parsed answer, malformed flag.

    `pieces` is prompt + completion in one strip with `prompt_len` marking where the model's own
    tokens (the only policy targets) begin; the per-token logprob arrays are dropped by the reader.
    """
    from pathlib import Path

    r = _reader(request, run)
    d = await anyio.to_thread.run_sync(lambda: r.rollouts(step, offset, min(limit, 100)))
    tag = Path((r.meta().get("config") or {}).get("tokenizer_dir") or "").name
    reg = request.app.state.tokenizers
    for x in d["rollouts"]:
        ids = list(x.get("prompt_ids") or []) + list(x.get("completion_ids") or [])
        try:
            x["pieces"] = reg.pieces(tag, ids)
        except (KeyError, FileNotFoundError):
            x["pieces"] = None
        x["prompt_len"] = len(x.get("prompt_ids") or [])
    return d


@router.get("/{run}/samples")
async def get_samples(request: Request, run: str) -> list[dict]:
    r = _reader(request, run)
    return await anyio.to_thread.run_sync(r.samples)


@router.get("/{run}/samples/{tokens}")
async def get_sample(request: Request, run: str, tokens: int) -> dict:
    r = _reader(request, run)
    try:
        return await anyio.to_thread.run_sync(lambda: r.sample(tokens))
    except FileNotFoundError:
        raise HTTPException(404, "no such sample") from None


@router.get("/{run}/quality")
async def get_quality(request: Request, run: str) -> dict:
    r = _reader(request, run)
    q = await anyio.to_thread.run_sync(r.quality)
    return q or {"run": run, "checkpoints": []}


@router.get("/{run}/quality/{tokens}")
async def get_quality_detail(request: Request, run: str, tokens: int) -> dict:
    r = _reader(request, run)
    try:
        return await anyio.to_thread.run_sync(lambda: r.quality_detail(tokens))
    except FileNotFoundError:
        raise HTTPException(404, "no quality outputs for that checkpoint") from None


@router.get("/{run}/report")
async def get_report(request: Request, run: str):
    r = _reader(request, run)
    p = r.dir / "report.html"
    if not p.exists():
        raise HTTPException(404, "no report.html yet")
    return FileResponse(p, media_type="text/html")


@router.get("/{run}/live")
async def live(request: Request, run: str, poll_s: float = 2.0):
    """SSE: every metrics record appended after the connection opens, plus 15 s heartbeats."""
    r = _reader(request, run)
    await anyio.to_thread.run_sync(r.refresh)
    seen = len(r.records)

    async def gen():
        nonlocal seen
        idle = 0.0
        yield f"event: hello\ndata: {json.dumps({'records': seen})}\n\n"
        while True:
            if await request.is_disconnected():
                return
            await anyio.to_thread.run_sync(r.refresh)
            if len(r.records) > seen:
                for rec in r.records[seen:]:
                    yield f"event: record\ndata: {json.dumps(rec)}\n\n"
                seen = len(r.records)
                idle = 0.0
            else:
                idle += poll_s
                if idle >= 15:
                    yield "event: heartbeat\ndata: {}\n\n"
                    idle = 0.0
            await asyncio.sleep(poll_s)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

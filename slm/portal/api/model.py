from __future__ import annotations

import asyncio
import json
import queue
import threading
import uuid
from pathlib import Path

import anyio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from slm.portal.api.system import gpu_info

router = APIRouter(prefix="/api/model")


class LoadRequest(BaseModel):
    checkpoint: str
    device: str = "auto"  # auto | cuda | cpu
    dtype: str = "bf16"
    tokenizer_tag: str | None = None
    force_cuda: bool = False


class GenerateRequest(BaseModel):
    slots: list[str] = ["A"]
    mode: str = "completion"  # completion | chat
    text: str = ""
    messages: list[dict] | None = None
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    max_new_tokens: int = 128
    seed: int | None = 1234
    logprobs_topk: int = 5
    think_required: bool = False
    tools: bool = False  # Python tool available (chat mode): calls run in the conversation's session
    session_id: str | None = None  # conversation id for REPL state; reset via /sessions/{id}/reset
    max_tool_calls: int = 8


class ScoreRequest(BaseModel):
    slot: str = "A"
    mode: str = "completion"
    text: str = ""
    messages: list[dict] | None = None


class DiagRequest(BaseModel):
    slot: str = "A"
    source: str
    seq_len: int = 1024
    n_batches: int = 4
    mb: int = 2
    ablations: bool = True
    out_dir: str | None = None


def _gpu_decision(request: Request, wanted: str, force: bool) -> tuple[str, str]:
    """Return (device, reason) honoring the gpu policy and live training runs."""
    s = request.app.state.settings
    if wanted == "cpu" or s.gpu_policy == "cpu":
        return "cpu", "cpu requested"
    live = request.app.state.runs.live_runs()
    g = gpu_info()
    free = (g["total_gib"] - g["used_gib"]) if g.get("available") else 0.0
    if not g.get("available"):
        return "cpu", "no GPU visible"
    if live and not force and wanted != "cuda":
        return "cpu", f"training run live ({', '.join(live)}); loading on CPU to protect it (force to override)"
    if free < 2.5 and not force:
        return "cpu", f"only {free:.1f} GiB VRAM free"
    return "cuda", "cuda" + (" (forced)" if force else "")


@router.get("/status")
def status(request: Request) -> dict:
    w = request.app.state.worker
    if not w.alive():
        return {"worker": False, "slots": {"A": {"slot": "A", "loaded": False}, "B": {"slot": "B", "loaded": False}}, "gpu": gpu_info(), "live_runs": request.app.state.runs.live_runs()}
    st = w.call("status")
    st["worker"] = True
    st["gpu"] = gpu_info()
    st["live_runs"] = request.app.state.runs.live_runs()
    return st


@router.post("/worker/stop")
def worker_stop(request: Request) -> dict:
    request.app.state.worker.stop()
    return {"worker": False}


@router.get("/checkpoints")
def checkpoints(request: Request) -> list[dict]:
    out = []
    ri = request.app.state.runs
    for name in ri.names():
        r = ri.get(name)
        for c in r.checkpoints():
            if c["kind"] in ("latest_prev",):
                continue
            c["run"] = name
            out.append(c)
    return out


@router.post("/slots/{slot}/load")
async def load(request: Request, slot: str, body: LoadRequest) -> dict:
    if slot not in ("A", "B"):
        raise HTTPException(400, "slot must be A or B")
    p = Path(body.checkpoint)
    if not p.exists() or p.suffix != ".pt":
        raise HTTPException(404, "checkpoint not found")
    device, reason = _gpu_decision(request, body.device, body.force_cuda)
    w = request.app.state.worker
    try:
        info = await anyio.to_thread.run_sync(lambda: w.call("load", slot=slot, checkpoint=str(p), device=device, dtype=body.dtype, tokenizer_tag=body.tokenizer_tag))
    except RuntimeError as e:
        raise HTTPException(500, str(e)) from None
    info["device_reason"] = reason
    return info


@router.post("/slots/{slot}/unload")
async def unload(request: Request, slot: str) -> dict:
    w = request.app.state.worker
    if not w.alive():
        return {"slot": slot, "loaded": False}
    return await anyio.to_thread.run_sync(lambda: w.call("unload", slot=slot))


@router.post("/score")
async def score(request: Request, body: ScoreRequest) -> dict:
    w = request.app.state.worker
    try:
        return await anyio.to_thread.run_sync(lambda: w.call("score", slot=body.slot, mode=body.mode, text=body.text, messages=body.messages))
    except RuntimeError as e:
        raise HTTPException(500, str(e)) from None


@router.post("/generate")
async def generate(request: Request, body: GenerateRequest):
    """SSE stream. For two slots, events carry `slot`; the streams run one after the other with the same seed."""
    w = request.app.state.worker
    streams = request.app.state.streams
    sid = uuid.uuid4().hex[:12]
    cancel = threading.Event()
    streams[sid] = cancel
    q: queue.Queue = queue.Queue()

    def run():
        try:
            for slot in body.slots:
                kw = body.model_dump(exclude={"slots"})
                for ev in w.stream("generate", cancel_flag=cancel, slot=slot, **kw):
                    ev["slot"] = slot
                    q.put(ev)
                    if ev["event"] == "error":
                        break
                if cancel.is_set():
                    break
        except Exception as e:  # noqa: BLE001
            q.put({"event": "error", "error": str(e)})
        finally:
            q.put(None)

    threading.Thread(target=run, daemon=True).start()

    async def gen():
        yield f"event: start\ndata: {json.dumps({'stream_id': sid})}\n\n"
        while True:
            try:
                ev = q.get_nowait()
            except queue.Empty:
                if await request.is_disconnected():
                    cancel.set()
                await asyncio.sleep(0.01)
                continue
            if ev is None:
                break
            yield f"event: {ev['event']}\ndata: {json.dumps(ev)}\n\n"
        streams.pop(sid, None)
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/sessions/{sid}/reset")
async def reset_session(request: Request, sid: str) -> dict:
    """Forget the Python session (variables) of a conversation."""
    w = request.app.state.worker
    if not w.alive():
        return {"session_id": sid, "reset": True, "worker": False}
    return await anyio.to_thread.run_sync(lambda: w.call("reset_session", session_id=sid))


@router.post("/streams/{sid}/cancel")
def cancel(request: Request, sid: str) -> dict:
    ev = request.app.state.streams.get(sid)
    if ev is None:
        return {"cancelled": False}
    ev.set()
    return {"cancelled": True}


@router.post("/diagnostics")
async def diagnostics(request: Request, body: DiagRequest) -> dict:
    w = request.app.state.worker
    root = str(Path(request.app.state.settings.data_root) / "tokenized" / "v1")
    tags = request.app.state.data.tags()
    if tags:
        root = str(Path(request.app.state.settings.data_root) / "tokenized" / tags[-1])
    try:
        d = await anyio.to_thread.run_sync(lambda: w.call("diagnostics", slot=body.slot, tokenized_root=root, source=body.source, seq_len=body.seq_len, n_batches=body.n_batches, mb=body.mb, ablations=body.ablations))
    except RuntimeError as e:
        raise HTTPException(500, str(e)) from None
    if body.out_dir:
        from slm.eval.diagnostics import save

        save(d, Path(body.out_dir), "diagnostics")
    return d

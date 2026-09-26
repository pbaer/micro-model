from __future__ import annotations

import asyncio
import json
import queue
import threading
import uuid
from pathlib import Path
from typing import Literal

import anyio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from slm.utils.stage import run_stage  # noqa: F401 (re-exported; the portal and the quality eval share it)
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
    functions: list[dict] | None = None  # declared functions (chat mode): {name, signature, comment} per entry


class SwarmRequest(BaseModel):
    """Swarm inference (slm.swarm.swarm_answer): k samples, collapsed by answer, one selector pass."""
    slot: Literal["A", "B"] = "A"
    text: str = ""  # the task prompt
    k: int = Field(16, ge=1, le=64)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    max_new_tokens: int = Field(512, ge=1, le=4096)
    max_calls: int = Field(6, ge=0, le=16)
    seed: int | None = None  # None: the worker draws one and reports it in result.meta.seed
    budget_tokens: int = Field(2400, ge=200, le=8192)  # selector prompt budget
    max_groups: int = Field(12, ge=1, le=64)
    answer_suffix: bool = True  # append slm.data.answers.SUFFIX (the '#### <number>' instruction) to the sampling prompt


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
    for info in st.get("slots", {}).values():  # stage/run of each loaded checkpoint, for the chat-mode warning
        if info.get("checkpoint"):
            info.update(_stage_of(request, info["checkpoint"]))
    return st


def _stage_of(request: Request, checkpoint: str) -> dict:
    p = Path(checkpoint)
    run_name = p.parent.parent.name if p.parent.name == "checkpoints" else None
    try:
        return {"stage": run_stage(request.app.state.runs.get(run_name).meta()) if run_name else "base", "run": run_name}
    except (KeyError, OSError):
        return {"stage": "base", "run": run_name}


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
        stage = run_stage(r.meta())
        for c in r.checkpoints():
            if c["kind"] in ("latest_prev",):
                continue
            c["run"] = name
            c["stage"] = stage
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
    info.update(_stage_of(request, str(p)))  # which kind of model this is, so the UI can warn about chat mode on a base checkpoint
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

    def produce(q: queue.Queue, cancel: threading.Event) -> None:
        for slot in body.slots:
            kw = body.model_dump(exclude={"slots"})
            for ev in w.stream("generate", cancel_flag=cancel, slot=slot, **kw):
                ev["slot"] = slot
                q.put(ev)
                if ev["event"] == "error":
                    break
            if cancel.is_set():
                break

    return _sse(request, produce)


@router.post("/swarm")
async def swarm(request: Request, body: SwarmRequest):
    """Swarm inference on one slot (slm.swarm), as SSE: `stage` events (sampling, collapsed with the groups,
    selecting), then `done` with the whole SwarmResult dict. Cancel (POST /streams/{id}/cancel) takes effect at
    the next stage boundary; the k samples are one batch."""
    if not body.text.strip():
        raise HTTPException(422, "text (the task prompt) is empty")
    w = request.app.state.worker

    def produce(q: queue.Queue, cancel: threading.Event) -> None:
        for ev in w.stream("swarm", cancel_flag=cancel, **body.model_dump()):
            ev["slot"] = body.slot
            q.put(ev)
            if ev["event"] == "error":
                break

    return _sse(request, produce)


def _sse(request: Request, produce) -> StreamingResponse:
    """Run `produce(queue, cancel_event)` on a thread and stream what it puts on the queue as SSE: a `start` event
    with the stream id (for POST /streams/{id}/cancel), each event under its own name, then `end`. A client that
    disconnects sets the cancel event."""
    streams = request.app.state.streams
    sid = uuid.uuid4().hex[:12]
    cancel = threading.Event()
    streams[sid] = cancel
    q: queue.Queue = queue.Queue()

    def run():
        try:
            produce(q, cancel)
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

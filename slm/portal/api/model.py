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
from pydantic import BaseModel, Field, model_validator

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
    """Swarm inference (slm.swarm.swarm_answer): k samples, collapsed by answer, then one selector pass and/or a pairwise
    single-elimination bracket over the distinct answers (`mode`)."""
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
    mode: Literal["select", "tournament", "both"] = "both"  # selector prompt, pairwise bracket, or both (final follows the bracket)
    pair_budget_tokens: int = Field(1200, ge=200, le=8192)  # pairwise prompt budget (tournament)
    max_entrants: int = Field(16, ge=2, le=64)  # the bracket takes the first max_entrants groups (evidence order)


class ArenaRequest(BaseModel):
    """The arena (slm.arena.world): an n x n grid of robots, each a conversation with the model, one batched generation
    per turn. The ranges are the harness's (`ARENA_LIMITS`); key_door pairs robots, a relay needs a source and a sink."""
    slot: Literal["A", "B"] = "A"
    task: Literal["key_door", "relay", "triangulate"] = "key_door"
    n: int = Field(8, ge=4, le=16)  # grid side
    n_agents: int = Field(4, ge=1, le=32)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    turns: int = Field(12, ge=1, le=50)
    max_new_tokens: int = Field(128, ge=16, le=1024)  # per robot per turn
    max_calls: int = Field(4, ge=0, le=16)  # tool calls per robot per turn
    max_history: int = Field(1, ge=0, le=10)  # prior (observation, action) exchanges kept in each robot's conversation (World default)
    stop_when_done: bool = True

    @model_validator(mode="after")
    def _robots_fit_task(self):
        if self.task == "key_door" and self.n_agents % 2:
            raise ValueError("key_door needs an even number of robots (pairs of key holder and door opener)")
        if self.task == "relay" and self.n_agents < 2:
            raise ValueError("relay needs at least 2 robots (a source and a sink)")
        return self


# what each arena task asks of the robots, written from the task docstrings in slm/arena/world.py
ARENA_TASKS = {
    "key_door": {
        "title": "Key and door",
        "description": "Robots work in pairs. The key holder (R1, R3, ...) can read a secret 4-digit code with read_key(); its partner "
                       "(R2, R4, ...) has open_door(code), which works only while standing on the pair's door cell. The key holder must "
                       "tell the code with say(...) while the partner is within comm range, and the partner must walk to the door and "
                       "open it. Each pair starts within one cell of each other, so turn 1 can already talk; with more robots there "
                       "is one door per pair. Success: every door open.",
        "roles": "even ids: read_key(); odd ids: open_door(code)", "min_agents": 2, "even_agents": True, "movement": True},
    "relay": {
        "title": "Relay",
        "description": "The robots stand in a line two cells apart and cannot usefully move. The first robot knows a code (read_code()), "
                       "the last one must submit(code) it, and each robot hears only its neighbours, so the code has to hop robot to "
                       "robot: a pure communication task, a chain of say(...) calls. Success: the last robot submits the right code; "
                       "the minimum is one turn per hop.",
        "roles": "R1: read_code(); last robot: submit(code); the rest only relay", "min_agents": 2, "even_agents": False, "movement": False},
    "triangulate": {
        "title": "Triangulate",
        "description": "Something is buried at a secret cell. Every robot's sense() returns its own Manhattan distance to it; only R1 has "
                       "dig(), which works only on the target cell. The others must report their position and distance to R1 with "
                       "say(...), and R1 must work out the cell from the distances, walk there and dig. The target is hidden on the grid "
                       "until it is dug up. Hard: it needs arithmetic over several messages.",
        "roles": "everyone: sense(); R1: dig()", "min_agents": 1, "even_agents": False, "movement": True},
}


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
        if info.get("checkpoint") and not info.get("external"):
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




def external_models() -> list[dict]:
    """The local open-weight comparison models (slm.eval.external), as checkpoint entries of the group "external models":
    `path` is what LoadRequest.checkpoint takes (`external:<name>`), `available` whether the weights are on disk. The
    registry module is torch-free; importing it forces the Hugging Face offline switches on (nothing here goes online)."""
    from slm.eval import external as E

    return [{"path": E.checkpoint_label(m.name), "name": m.name, "kind": "external", "group": "external models", "external": True,
             "stage": "external", "run": None, "hf_id": m.hf_id, "params": m.params, "license": m.license, "is_chat": m.is_chat,
             "context": m.max_positions, "train_tokens": m.train_tokens, "notes": m.notes, "available": m.available(),
             "local_dir": str(m.local_dir), "tokens": None, "val_loss": None} for m in E.EXTERNAL_MODELS.values()]


@router.get("/checkpoints")
def checkpoints(request: Request) -> list[dict]:
    """Our checkpoints (every run's, stage-tagged), then the external comparison models (`external: true`)."""
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
    return out + external_models()


@router.post("/slots/{slot}/load")
async def load(request: Request, slot: str, body: LoadRequest) -> dict:
    """Load one of our checkpoints (a .pt path) or an external comparison model (`external:<name>`) into a slot."""
    if slot not in ("A", "B"):
        raise HTTPException(400, "slot must be A or B")
    external = body.checkpoint.startswith("external:")
    if external:
        name = body.checkpoint.split(":", 1)[1]
        entry = next((m for m in external_models() if m["name"] == name), None)
        if entry is None:
            raise HTTPException(404, f"unknown external model {name!r}")
        if not entry["available"]:
            raise HTTPException(404, f"{name}: no weights under {entry['local_dir']}; run `python -m slm.eval.external download {name}` once")
        ck = body.checkpoint
    else:
        p = Path(body.checkpoint)
        if not p.exists() or p.suffix != ".pt":
            raise HTTPException(404, "checkpoint not found")
        ck = str(p)
    device, reason = _gpu_decision(request, body.device, body.force_cuda)
    w = request.app.state.worker
    try:
        info = await anyio.to_thread.run_sync(lambda: w.call("load", slot=slot, checkpoint=ck, device=device, dtype=body.dtype, tokenizer_tag=body.tokenizer_tag))
    except RuntimeError as e:
        raise HTTPException(500, str(e)) from None
    info["device_reason"] = reason
    if not external:
        info.update(_stage_of(request, ck))  # which kind of model this is, so the UI can warn about chat mode on a base checkpoint
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
    selecting, then tournament: round 0 = the seeded entrants, one event per decided round with its matches), then
    `done` with the whole SwarmResult dict. Cancel (POST /streams/{id}/cancel) takes effect at the next stage boundary
    or bracket round; the k samples are one batch."""
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


@router.get("/arena/tasks")
def arena_tasks() -> list[dict]:
    """The arena's tasks: key, title, a one-paragraph description, roles, comm range and sight (from the task classes)."""
    from slm.arena.world import TASKS  # torch-free at import

    return [{"key": k, **ARENA_TASKS[k], "comm_range": TASKS[k].comm_range, "sight": TASKS[k].sight} for k in sorted(TASKS) if k in ARENA_TASKS]


@router.post("/arena")
async def arena(request: Request, body: ArenaRequest):
    """The arena on one slot (slm.arena.world via Harness.arena), as SSE: `start` (once with the stream id, then once
    with the initial state, the robots' system prompts and tools), one `turn` per turn (records, state, messages),
    `done` with the result and transcript. Cancel (POST /streams/{id}/cancel) takes effect between turns."""
    w = request.app.state.worker

    def produce(q: queue.Queue, cancel: threading.Event) -> None:
        for ev in w.stream("arena", cancel_flag=cancel, **body.model_dump()):
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

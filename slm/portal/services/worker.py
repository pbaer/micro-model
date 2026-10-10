"""Torch worker subprocess and its client. The web server never imports torch; all model work
happens here, so idle VRAM is zero and a crash cannot take the portal down."""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any

STREAM_METHODS = {"generate", "swarm", "arena"}


class WorkerDied(RuntimeError):
    """The worker process exited while a request was waiting on it."""


def describe_exit(code: int | None) -> str:
    """A worker exit code in words. Windows reports NTSTATUS values; 0xC0000142 (a DLL failed to initialise) is what a
    child gets when it is spawned into a console that no longer exists, e.g. a portal started from a shell session
    that has since closed."""
    if code is None:
        return "still running"
    u = code & 0xFFFFFFFF
    known = {0xC0000142: "a DLL failed to initialise -- typically a portal started from a console that has since closed; "
                         "restart the portal from a live terminal",
             0xC0000005: "access violation (a native crash)", 0xC0000409: "stack buffer overrun / fast-fail (a native crash)",
             0xC000013A: "interrupted (Ctrl+C / console closed)", 1: "Python exited with an error"}
    hint = known.get(u, "killed or crashed")
    return f"exit code {code} (0x{u:08X}): {hint}" if u > 255 else f"exit code {code}: {hint}"


def worker_main(conn, tokenizer_root: str) -> None:  # pragma: no cover - runs in the child
    import torch  # noqa: F401

    from slm.portal.services.harness import Harness
    from slm.utils.sdpa import sdpa_context

    h = Harness(Path(tokenizer_root))
    with sdpa_context("decode"):
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                return
            rid, method, kw = msg["id"], msg["method"], msg.get("kwargs", {})
            if method == "exit":
                conn.send({"id": rid, "event": "done", "data": None})
                return
            try:
                fn = getattr(h, method)
                if method in STREAM_METHODS:
                    cancelled = {"v": False}

                    def should_stop() -> bool:
                        while conn.poll():
                            m2 = conn.recv()
                            if m2.get("method") == "cancel":
                                cancelled["v"] = True
                        return cancelled["v"]

                    for ev in fn(**kw, should_stop=should_stop):
                        conn.send({"id": rid, "event": ev.pop("event"), "data": ev})
                    conn.send({"id": rid, "event": "end", "data": None})
                else:
                    conn.send({"id": rid, "event": "result", "data": fn(**kw)})
            except Exception as e:  # noqa: BLE001
                conn.send({"id": rid, "event": "error", "data": {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}})


class WorkerClient:
    def __init__(self, tokenizer_root: Path, idle_timeout_s: float = 900.0, target=None) -> None:
        self.tokenizer_root = str(tokenizer_root)
        self.idle_timeout_s = idle_timeout_s
        self.target = target or worker_main  # tests pass a stub that dies on purpose
        self.proc: mp.Process | None = None
        self.conn = None
        self.lock = threading.Lock()
        self.last_used = time.time()
        self._rid = 0
        self.busy = False

    # ------------------------------------------------------------- lifecycle
    def alive(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    def start(self) -> None:
        if self.alive():
            return
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        self.proc = ctx.Process(target=self.target, args=(child, self.tokenizer_root), daemon=True)
        self.proc.start()
        # The child has its own copy of this end now. Keeping ours open meant a worker that died left recv() waiting
        # forever (no EOF while any handle to the other end is open) -- the chat page's endless "loading" (2026-10-09).
        child.close()
        self.conn = parent
        self.last_used = time.time()

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            if self.alive():
                self.conn.send({"id": -1, "method": "exit"})
                self.proc.join(timeout=5)
            if self.proc.is_alive():
                self.proc.terminate()
        finally:
            self.proc = None
            self.conn = None

    def maybe_idle_stop(self) -> bool:
        # Only while no request holds the lock: without it, a request could slip in between the busy check and stop().
        if not self.lock.acquire(blocking=False):
            return False
        try:
            if self.alive() and not self.busy and time.time() - self.last_used > self.idle_timeout_s:
                self.stop()
                return True
            return False
        finally:
            self.lock.release()

    def _recv(self, timeout: float) -> Any:
        """The next message, or None after `timeout` seconds; raises WorkerDied when the worker has exited (and resets
        the client, so the next request starts a fresh worker)."""
        try:
            if self.conn.poll(timeout):
                return self.conn.recv()
        except (EOFError, OSError, BrokenPipeError):
            pass  # the pipe closed under us: the worker is gone; fall through to report how it ended
        else:
            if self.proc is not None and self.proc.is_alive():
                return None
        proc = self.proc
        if proc is not None:
            proc.join(timeout=2)
        code = proc.exitcode if proc is not None else None
        self.proc, self.conn = None, None
        raise WorkerDied(f"the model worker exited ({describe_exit(code)})")

    # ------------------------------------------------------------------ rpc
    def _next_id(self) -> int:
        self._rid += 1
        return self._rid

    def call(self, method: str, **kwargs: Any) -> Any:
        with self.lock:
            self.start()
            self.busy = True
            try:
                rid = self._next_id()
                self.conn.send({"id": rid, "method": method, "kwargs": kwargs})
                while True:
                    msg = self._recv(0.5)  # never a bare blocking recv: a dead worker must surface as an error
                    if msg is None or msg["id"] != rid:
                        continue
                    if msg["event"] == "error":
                        raise RuntimeError(msg["data"]["error"])
                    return msg["data"]
            finally:
                self.busy = False
                self.last_used = time.time()

    def stream(self, method: str, cancel_flag: threading.Event | None = None, **kwargs: Any) -> Iterator[dict]:
        with self.lock:
            self.start()
            self.busy = True
            try:
                rid = self._next_id()
                self.conn.send({"id": rid, "method": method, "kwargs": kwargs})
                while True:
                    if cancel_flag is not None and cancel_flag.is_set():
                        self.conn.send({"id": rid, "method": "cancel"})
                        cancel_flag = None
                    try:
                        msg = self._recv(0.05)
                    except WorkerDied as e:
                        yield {"event": "error", "error": str(e)}
                        return
                    if msg is None or msg["id"] != rid:
                        continue
                    if msg["event"] == "error":
                        yield {"event": "error", **msg["data"]}
                        return
                    if msg["event"] == "end":
                        return
                    yield {"event": msg["event"], **(msg["data"] or {})}
            finally:
                self.busy = False
                self.last_used = time.time()

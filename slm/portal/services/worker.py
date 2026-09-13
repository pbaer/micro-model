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

STREAM_METHODS = {"generate"}


def worker_main(conn, tokenizer_root: str) -> None:  # pragma: no cover - runs in the child
    import torch  # noqa: F401

    from slm.portal.services.harness import Harness
    from slm.utils.sdpa import sdpa_context

    h = Harness(Path(tokenizer_root))
    with sdpa_context("auto"):
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
    def __init__(self, tokenizer_root: Path, idle_timeout_s: float = 900.0) -> None:
        self.tokenizer_root = str(tokenizer_root)
        self.idle_timeout_s = idle_timeout_s
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
        self.proc = ctx.Process(target=worker_main, args=(child, self.tokenizer_root), daemon=True)
        self.proc.start()
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
        if self.alive() and not self.busy and time.time() - self.last_used > self.idle_timeout_s:
            self.stop()
            return True
        return False

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
                    msg = self.conn.recv()
                    if msg["id"] != rid:
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
                    if not self.conn.poll(0.05):
                        continue
                    msg = self.conn.recv()
                    if msg["id"] != rid:
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

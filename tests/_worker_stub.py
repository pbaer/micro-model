"""Worker targets for tests/test_portal_worker.py (importable in the spawned child: pytest puts tests/ on sys.path)."""

import os
import time


def dies_at_once(conn, tokenizer_root):
    os._exit(7)


def dies_mid_request(conn, tokenizer_root):
    conn.recv()  # take the request, then vanish without answering (a native crash or a kill looks like this)
    time.sleep(0.2)
    os._exit(3)


def echoes(conn, tokenizer_root):
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg["method"] == "exit":
            conn.send({"id": msg["id"], "event": "done", "data": None})
            return
        conn.send({"id": msg["id"], "event": "result", "data": {"echo": msg["method"]}})

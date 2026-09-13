"""SSE live tail test against a real uvicorn server in a thread (ASGI test transports buffer whole
responses, so an open-ended event stream cannot be tested through them)."""

import json
import socket
import threading
import time

import httpx
import uvicorn

from slm.portal.app import create_app
from slm.portal.settings import PortalSettings
from test_portal_runs import make_run


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_live_stream_delivers_appended_records(tmp_path):
    d = make_run(tmp_path / "runs", "gamma")
    app = create_app(PortalSettings(runs_root=tmp_path / "runs", open_browser=False))
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        got = []
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10) as c, c.stream("GET", "/api/runs/gamma/live?poll_s=0.1") as resp:
            it = resp.iter_lines()
            assert next(it).startswith("event: hello")
            with open(d / "metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"kind": "train", "time": time.time(), "tokens": 99999, "update": 99, "loss": 1.0}) + "\n")
            deadline = time.time() + 10
            for line in it:
                if line.startswith("data:") and "99999" in line:
                    got.append(line)
                    break
                assert time.time() < deadline, "no record within 10 s"
        assert got
    finally:
        server.should_exit = True
        th.join(timeout=5)

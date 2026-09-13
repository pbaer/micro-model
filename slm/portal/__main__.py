"""python -m slm.portal [--port 8765] [--runs-root runs] [--no-browser] [--reload]"""

from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

import uvicorn

from slm.portal.settings import PortalSettings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--gpu-policy", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--reload", action="store_true", help="dev: auto-reload on code changes")
    a = ap.parse_args()
    s = PortalSettings(runs_root=Path(a.runs_root), host=a.host, port=a.port, gpu_policy=a.gpu_policy, open_browser=not a.no_browser)
    if a.data_root:
        s.data_root = Path(a.data_root)
    url = f"http://{s.host}:{s.port}/"
    if s.open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"command center: {url}")
    if a.reload:
        uvicorn.run("slm.portal.app:create_app", factory=True, host=s.host, port=s.port, reload=True, log_level="warning")
    else:
        from slm.portal.app import create_app

        uvicorn.run(create_app(s), host=s.host, port=s.port, log_level="warning")


if __name__ == "__main__":
    main()

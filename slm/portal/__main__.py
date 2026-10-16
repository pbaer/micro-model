"""python -m slm.portal [--port 8930] [--runs-root runs] [--no-browser] [--reload]

Port 8930 is the portal's (slm.portal.settings.PORT_RANGE: 8930-8939 are this project's). When the port is taken the
portal stops and names the program holding it rather than moving to another port: chat conversations are stored per
origin in the browser, so a silent port change makes them look lost."""

from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

import uvicorn

from slm.portal.settings import PORT_RANGE, PORTAL_PORT, PortalSettings


def port_holder(host: str, port: int) -> str | None:
    """None when `host:port` can be bound; otherwise a description of who holds it (best effort)."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return None
        except OSError:
            pass
    try:
        import psutil

        for c in psutil.net_connections(kind="tcp"):
            if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN and c.pid:
                p = psutil.Process(c.pid)
                return f"pid {c.pid}: {' '.join(p.cmdline())[:160]}"
    except Exception:  # noqa: BLE001 - permissions or a vanished process: the refusal still stands
        pass
    return "another program"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=PORTAL_PORT, help=f"default {PORTAL_PORT}; this project's range is {PORT_RANGE.start}-{PORT_RANGE.stop - 1}")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--gpu-policy", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--default-chat", default=None, help="checkpoint the chat mode loads on first use, as <run>/<file> (default: the settings' m11_rl_arena_neg_336m/final.pt)")
    ap.add_argument("--reload", action="store_true", help="dev: auto-reload on code changes")
    a = ap.parse_args()
    s = PortalSettings(runs_root=Path(a.runs_root), host=a.host, port=a.port, gpu_policy=a.gpu_policy, open_browser=not a.no_browser)
    if a.data_root:
        s.data_root = Path(a.data_root)
    if a.default_chat:
        s.default_chat_checkpoint = a.default_chat
    if not a.reload and (holder := port_holder(s.host, s.port)):
        spare = next((p for p in PORT_RANGE if p != s.port and port_holder(s.host, p) is None), None)
        raise SystemExit(f"port {s.port} is in use by {holder}.\n"
                         + (f"Use a spare port of this project's range: python -m slm.portal --port {spare}" if spare else
                            f"Every port of this project's range ({PORT_RANGE.start}-{PORT_RANGE.stop - 1}) is in use."))
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

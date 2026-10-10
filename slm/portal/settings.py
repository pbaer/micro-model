from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from slm.data.sources import DATA_ROOT

# This project's ports (2026-10-09): 8930 is the portal, 8931-8939 spares (a second portal on another runs root, a smoke
# server). Chosen clear of the common dev ports and of 87xx, where another local app (a map viewer) serves on 8765.
# Windows reserves 5357, 50000-50059 and blocks above 60000 on this machine; none overlap.
PORTAL_PORT = 8930
PORT_RANGE = range(8930, 8940)


@dataclass
class PortalSettings:
    runs_root: Path = field(default_factory=lambda: Path("runs"))
    data_root: Path = field(default_factory=lambda: DATA_ROOT)
    configs_root: Path = field(default_factory=lambda: Path("configs"))
    cache_dir: Path = field(default_factory=lambda: Path("artifacts/portal-cache"))
    host: str = "127.0.0.1"
    port: int = PORTAL_PORT
    gpu_policy: str = "auto"  # auto | cuda | cpu
    worker_idle_timeout_s: float = 900.0
    open_browser: bool = True
    # the Inference tab's chat mode loads this on first use: "<run>/<file>" (or a path); when it is not on disk the
    # newest RL / reasoning / SFT final.pt stands in (slm.portal.api.model.mark_default_chat)
    default_chat_checkpoint: str = "m10_rl_336m/final.pt"

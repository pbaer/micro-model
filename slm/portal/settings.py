from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from slm.data.sources import DATA_ROOT


@dataclass
class PortalSettings:
    runs_root: Path = field(default_factory=lambda: Path("runs"))
    data_root: Path = field(default_factory=lambda: DATA_ROOT)
    configs_root: Path = field(default_factory=lambda: Path("configs"))
    cache_dir: Path = field(default_factory=lambda: Path("artifacts/portal-cache"))
    host: str = "127.0.0.1"
    port: int = 8765
    gpu_policy: str = "auto"  # auto | cuda | cpu
    worker_idle_timeout_s: float = 900.0
    open_browser: bool = True

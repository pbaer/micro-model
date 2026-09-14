"""GPU health telemetry via nvidia-smi (torch-free): one-shot query plus a background sampler.

The trainers attach the latest sample (temperature, power, utilization) to every train record and
warn on the console when the GPU runs hot. Sampling happens on a daemon thread so the training loop
never blocks on the nvidia-smi subprocess (~50-100 ms per call on Windows).
"""

from __future__ import annotations

import subprocess
import threading
import time

FIELDS = ["gpu_temp_c", "gpu_power_w", "gpu_util"]
_QUERY = "name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks_throttle_reasons.active"


def _num(x: str) -> float | None:
    x = x.strip()
    if x in ("", "[N/A]", "N/A", "[Not Supported]"):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def parse_query(line: str) -> dict:
    """Parse one CSV line of `nvidia-smi --query-gpu=<_QUERY> --format=csv,noheader,nounits`."""
    parts = [p.strip() for p in line.split(",")]
    parts += [""] * (8 - len(parts))
    name, used, total, util, power, temp, clock, throttle = parts[:8]
    try:
        throttle_mask = int(throttle, 16) if throttle.lower().startswith("0x") else (int(throttle) if throttle.isdigit() else 0)
    except ValueError:
        throttle_mask = 0
    return {
        "available": True, "name": name,
        "used_gib": (_num(used) or 0.0) / 1024, "total_gib": (_num(total) or 0.0) / 1024,
        "util": _num(util) or 0.0, "power_w": _num(power), "temp_c": _num(temp), "clock_mhz": _num(clock),
        # bit 0x1 is "GPU idle"; anything else (power cap 0x4, thermal 0x20/0x40, HW slowdown 0x8) is a real throttle
        "throttled": bool(throttle_mask & ~0x1),
    }


def query_gpu(timeout: float = 5.0) -> dict:
    """Current GPU state, or {"available": False, "error": ...} when nvidia-smi is unusable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            text=True, timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).strip().splitlines()[0]
        return parse_query(out)
    except Exception as e:  # noqa: BLE001 (missing binary, timeout, odd output)
        return {"available": False, "error": str(e)[:200]}


class GpuSampler:
    """Polls nvidia-smi every `interval_s` seconds on a daemon thread.

    `record()` returns the fields to attach to a train record; `hot_warning()` returns a message the
    first time the temperature crosses `warn_temp_c` and then at most every `warn_every_s` while hot
    (None otherwise), so a hot GPU is loud but not spammy.
    """

    def __init__(self, interval_s: float = 2.0, warn_temp_c: float = 80.0, warn_every_s: float = 300.0) -> None:
        self.interval_s = max(0.5, interval_s)
        self.warn_temp_c = warn_temp_c
        self.warn_every_s = warn_every_s
        self.latest: dict = {}
        self.max_temp_c: float | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_warn = 0.0

    def start(self) -> "GpuSampler":
        self._sample()  # synchronous first sample so the first log line already has values
        if self.latest.get("available") and self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="gpu-sampler", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        g = query_gpu()
        with self._lock:
            if g.get("available"):
                self.latest = g
                t = g.get("temp_c")
                if t is not None:
                    self.max_temp_c = t if self.max_temp_c is None else max(self.max_temp_c, t)
            elif not self.latest:
                self.latest = g

    def record(self) -> dict:
        with self._lock:
            g = self.latest
        return {"gpu_temp_c": g.get("temp_c"), "gpu_power_w": g.get("power_w"), "gpu_util": g.get("util") if g.get("available") else None}

    def console_suffix(self) -> str:
        r = self.record()
        if r["gpu_temp_c"] is None and r["gpu_power_w"] is None:
            return ""
        t = f"{r['gpu_temp_c']:.0f}C" if r["gpu_temp_c"] is not None else "?C"
        p = f"{r['gpu_power_w']:.0f}W" if r["gpu_power_w"] is not None else "?W"
        return f" | {t} {p}"

    def hot_warning(self, now: float | None = None) -> str | None:
        with self._lock:
            g = self.latest
        t = g.get("temp_c")
        if t is None or t < self.warn_temp_c:
            return None
        now = time.time() if now is None else now
        if now - self._last_warn < self.warn_every_s:
            return None
        self._last_warn = now
        extra = " (driver reports throttling)" if g.get("throttled") else ""
        return f"GPU HOT: {t:.0f} C at {g.get('power_w') or 0:.0f} W, {g.get('clock_mhz') or 0:.0f} MHz{extra}; check airflow/fan curve"

"""GPU telemetry: nvidia-smi parsing, the background sampler, and the hot-GPU warning rate limit."""

import time

from slm.utils import metrics as M
from slm.utils.gpu import GpuSampler, parse_query, query_gpu


def test_parse_query_line():
    g = parse_query("NVIDIA GeForce RTX 4080 SUPER, 12345, 16376, 98, 281.55, 71, 2505, 0x0000000000000004")
    assert g["available"] and g["name"].endswith("4080 SUPER")
    assert abs(g["used_gib"] - 12345 / 1024) < 1e-6 and g["util"] == 98 and g["power_w"] == 281.55 and g["temp_c"] == 71
    assert g["clock_mhz"] == 2505 and g["throttled"] is True  # 0x4 = software power cap
    idle = parse_query("X, 0, 16376, 0, [N/A], [N/A], 210, 0x0000000000000001")
    assert idle["power_w"] is None and idle["temp_c"] is None and idle["throttled"] is False  # 0x1 = idle, not a throttle


def test_sampler_records_and_warns(monkeypatch):
    fake = {"t": 60.0}
    monkeypatch.setattr("slm.utils.gpu.query_gpu", lambda timeout=5.0: {"available": True, "temp_c": fake["t"], "power_w": 250.0, "util": 97.0, "clock_mhz": 2500, "throttled": False})
    s = GpuSampler(interval_s=0.5, warn_temp_c=80.0, warn_every_s=60.0).start()
    try:
        assert s.record() == {"gpu_temp_c": 60.0, "gpu_power_w": 250.0, "gpu_util": 97.0}
        assert s.console_suffix() == " | 60C 250W" and s.hot_warning() is None
        fake["t"] = 84.0
        s._sample()
        w = s.hot_warning(now=1000.0)
        assert w and "84 C" in w and s.hot_warning(now=1010.0) is None and s.hot_warning(now=1100.0)  # rate-limited while hot
        assert s.max_temp_c == 84.0
    finally:
        s.stop()


def test_sampler_without_gpu(monkeypatch):
    monkeypatch.setattr("slm.utils.gpu.query_gpu", lambda timeout=5.0: {"available": False, "error": "no nvidia-smi"})
    s = GpuSampler().start()
    assert s.record() == {"gpu_temp_c": None, "gpu_power_w": None, "gpu_util": None} and s.console_suffix() == "" and s.hot_warning() is None
    assert s._thread is None  # nothing to poll


def test_metrics_carry_gpu_fields():
    now = time.time()
    recs = [{"kind": "train", "time": now + i, "tokens": 1000 * (i + 1), "update": i + 1, "loss": 3.0, "vram_gib": 12.0, "gpu_temp_c": 70.0 + i, "gpu_power_w": 280.0} for i in range(3)]
    recs.append({"kind": "warn", "time": now + 3, "tokens": 3000, "msg": "GPU HOT: 82 C"})
    ser = M.series(recs)
    assert ser["train"]["gpu_temp_c"] == [70.0, 71.0, 72.0] and ser["train"]["gpu_power_w"] == [280.0] * 3
    summ = M.summary(recs, {"config": {"schedule": {"total_tokens": 10_000}}})
    assert summ["gpu_temp_c"] == 72.0 and summ["gpu_temp_max_c"] == 72.0 and summ["gpu_power_w"] == 280.0
    assert "warn" in M.EVENT_KINDS


def test_query_gpu_real_or_absent():
    g = query_gpu()
    if g.get("available"):
        assert g["total_gib"] > 0 and (g["temp_c"] is None or 0 < g["temp_c"] < 120)
    else:
        assert "error" in g


def test_summary_elapsed_survives_a_stop_and_resume():
    t0 = time.time() - 1000
    recs = [{"kind": "start", "time": t0, "tokens": 0, "msg": "fresh"},
            {"kind": "train", "time": t0 + 100, "tokens": 1000, "update": 1, "loss": 3.0, "elapsed_s": 100.0},
            {"kind": "stop", "time": t0 + 101, "tokens": 1000, "elapsed_s": 101.0, "msg": "stopped"},
            {"kind": "resume", "time": t0 + 500, "tokens": 1000, "msg": "resumed"},
            {"kind": "train", "time": t0 + 700, "tokens": 3000, "update": 3, "loss": 2.9, "elapsed_s": 301.0}]
    s = M.summary(recs, {"config": {"schedule": {"total_tokens": 10_000}}})
    assert s["status"] == "running" and 301.0 + 250 <= s["elapsed_s"] <= 301.0 + 350  # counter + time since the last record, not the stale stop
    done = recs + [{"kind": "finish", "time": t0 + 800, "tokens": 4000, "elapsed_s": 400.0, "msg": "finished"}]
    assert M.summary(done, {"config": {"schedule": {"total_tokens": 10_000}}})["elapsed_s"] == 400.0

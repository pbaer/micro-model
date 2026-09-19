"""Needle sweep over a run's snapshots: resumable JSON, chart-ready series, and the portal payload (CPU, tiny model)."""

import json
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient

from slm.config import ModelConfig, load_config, to_dict
from slm.data.tokenizer import SlmTokenizer, train_bpe
from slm.eval import needle_sweep as NS
from slm.eval.long_context import RealHaystack
from slm.model import Transformer
from slm.portal.app import PortalSettings, create_app
from slm.utils.checkpoint import save_snapshot
from slm.utils.logging import MetricsLogger


def _run_with_snapshots(root: Path, tok: SlmTokenizer, name: str = "sweep") -> Path:
    d = root / name
    (d / "checkpoints").mkdir(parents=True)
    tok_dir = root / "tok"
    tok.save(tok_dir)
    (d / "run.json").write_text(json.dumps({"run_name": name, "stage": "pretrain", "config": {"tokenizer_dir": str(tok_dir), "data": {"kind": "pretrain"}, "schedule": {"total_tokens": 300}}}), encoding="utf-8")
    cfg = load_config(ModelConfig, "configs/model/tiny.yaml")
    cfg.vocab_size, cfg.max_seq_len = tok.vocab_size, 256
    idx = {}
    for i, tokens in enumerate((100, 200, 300)):
        torch.manual_seed(i)
        m = Transformer(cfg)
        save_snapshot(d / "checkpoints" / f"snap_{tokens}.pt", m, to_dict(cfg), {"tokens": tokens, "model_config": to_dict(cfg)}, dtype=torch.float32)
        idx[f"snap_{tokens}.pt"] = {"kind": "snapshot", "tokens": tokens}
    (d / "checkpoints" / "index.json").write_text(json.dumps(idx), encoding="utf-8")
    lg = MetricsLogger(d)
    lg.log("start", msg="go")
    lg.log("train", tokens=100, update=1, loss=3.0, lr=1e-4, grad_norm=1.0, tok_s=1.0, tok_s_ema=1.0, step_ms=1.0, fwd_ms=0, bwd_ms=0, opt_ms=0, data_ms=0, vram_gib=1.0, elapsed_s=1, eta_s=1)
    lg.close()
    return d


def _fake_val(root: Path, tok: SlmTokenizer) -> Path:
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 200, size=20000, dtype=np.uint16)
    ids[::50] = tok.eos_id
    ids[1::50] = tok.bos_id
    v = root / "src" / "val"
    v.mkdir(parents=True)
    ids.tofile(v / "shard_00000.bin")
    np.save(v / "shard_00000.idx.npy", np.arange(1, 20000, 50, dtype=np.int64))
    return v


def test_sweep_is_resumable_and_served(tmp_path):
    tok = SlmTokenizer(train_bpe(["The secret number is 123456. Question: What is the secret number? Answer: " * 30, "the village market opened early " * 30], vocab_size=320))
    d = _run_with_snapshots(tmp_path / "runs", tok)
    hay = RealHaystack(tok, _fake_val(tmp_path, tok))
    logs = []
    data = NS.sweep(d, [128, 256], [0.0, 1.0], n=2, device="cpu", haystack=hay, log=logs.append)
    assert [c["tokens"] for c in data["checkpoints"]] == [100, 200, 300] and data["n"] == 2 and data["lengths"] == [128, 256]
    c = data["checkpoints"][0]
    assert set(c["summary"]) == {"128", "256"} and len(c["cells"]) == 4 and c["effective"] in (0, 128, 256)
    assert (d / "needle_sweep.json").exists() and json.loads((d / "needle_sweep.json").read_text(encoding="utf-8"))["checkpoints"][2]["checkpoint"] == "snap_300.pt"
    # a second call does nothing; a change of settings starts over
    again = NS.sweep(d, [128, 256], [0.0, 1.0], n=2, device="cpu", haystack=hay, log=logs.append)
    assert again == data and any("0 to do" in line for line in logs)
    fresh = NS.sweep(d, [128], [0.0, 1.0], n=1, device="cpu", haystack=hay, log=logs.append)
    assert fresh["n"] == 1 and fresh["lengths"] == [128] and len(fresh["checkpoints"]) == 3
    ser = NS.series(fresh)
    assert ser["tokens"] == [100, 200, 300] and set(ser["mean"]) == {"128"} and len(ser["min"]["128"]) == 3
    # the portal's series payload carries it, torch-free
    app = create_app(PortalSettings(runs_root=tmp_path / "runs", open_browser=False))
    s = TestClient(app).get("/api/runs/sweep/series").json()
    assert s["needle_sweep"]["n"] == 1 and s["needle_sweep"]["tokens"] == [100, 200, 300]

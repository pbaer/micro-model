"""Worker-based model harness tests on CPU with the tiny config and a synthetic tokenizer."""

import json
import threading

import torch
from fastapi.testclient import TestClient

from slm.config import ModelConfig, load_config, to_dict
from slm.data.tokenizer import SlmTokenizer, train_bpe
from slm.model import Transformer
from slm.portal.app import create_app
from slm.portal.settings import PortalSettings
from slm.utils.checkpoint import save_snapshot


def _world(tmp_path):
    tok_dir = tmp_path / "tokenizer" / "t1"
    tok = SlmTokenizer(train_bpe(["hello world " * 50, "the cat sat on the mat " * 50], vocab_size=300))
    tok.save(tok_dir)
    cfg = load_config(ModelConfig, "configs/model/tiny.yaml")
    cfg.vocab_size = tok.vocab_size
    m = Transformer(cfg)
    ck = tmp_path / "runs" / "r" / "checkpoints" / "snap_1K.pt"
    save_snapshot(ck, m, to_dict(cfg), {"tokens": 1000, "val_loss": 3.0, "tokenizer_sha256": tok.sha256})
    (tmp_path / "runs" / "r" / "run.json").write_text(json.dumps({"run_name": "r", "config": {"schedule": {"total_tokens": 1}}}))
    (tmp_path / "runs" / "r" / "metrics.jsonl").write_text("")
    return ck


def test_worker_load_generate_score_cancel(tmp_path):
    ck = _world(tmp_path)
    app = create_app(PortalSettings(runs_root=tmp_path / "runs", data_root=tmp_path, gpu_policy="cpu", open_browser=False))
    c = TestClient(app)
    try:
        st = c.get("/api/model/status").json()
        assert st["worker"] is False
        cks = c.get("/api/model/checkpoints").json()
        assert cks and cks[0]["name"] == "snap_1K.pt"
        info = c.post("/api/model/slots/A/load", json={"checkpoint": str(ck), "device": "auto"}).json()
        assert info["device"] == "cpu" and info["tokenizer_matched"] is True and info["params"] > 0
        st = c.get("/api/model/status").json()
        assert st["worker"] is True and st["slots"]["A"]["name"] == "snap_1K.pt"

        events = []
        with c.stream("POST", "/api/model/generate", json={"slots": ["A"], "mode": "completion", "text": "hello", "max_new_tokens": 6, "seed": 7, "temperature": 0.9}) as r:
            for line in r.iter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:]))
        toks = [e for e in events if e.get("event") == "token"]
        assert len(toks) <= 6 and all(isinstance(t["logprob"], float) and t["logprob"] <= 0 for t in toks)
        assert all(len(t["topk"]) == 5 and t["topk"][0]["logprob"] >= t["topk"][-1]["logprob"] for t in toks)
        done = [e for e in events if e.get("event") == "done"]
        assert done and done[0]["n"] == len(toks)
        # same seed -> same tokens
        events2 = []
        with c.stream("POST", "/api/model/generate", json={"slots": ["A"], "mode": "completion", "text": "hello", "max_new_tokens": 6, "seed": 7, "temperature": 0.9}) as r:
            for line in r.iter_lines():
                if line.startswith("data:"):
                    events2.append(json.loads(line[5:]))
        assert [t["id"] for t in toks] == [t["id"] for t in events2 if t.get("event") == "token"]

        sc = c.post("/api/model/score", json={"slot": "A", "mode": "completion", "text": "hello world"}).json()
        assert sc["n"] == len(sc["tokens"]) and sc["tokens"][0]["logprob"] is None and sc["ppl"] > 0
        chat = c.post("/api/model/score", json={"slot": "A", "mode": "chat", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}).json()
        assert any(t["target"] for t in chat["tokens"]) and not chat["tokens"][1]["target"]

        # two slots side by side: load B too, stream both
        c.post("/api/model/slots/B/load", json={"checkpoint": str(ck), "device": "cpu"})
        slots_seen = set()
        with c.stream("POST", "/api/model/generate", json={"slots": ["A", "B"], "text": "the cat", "max_new_tokens": 3, "temperature": 0.0}) as r:
            for line in r.iter_lines():
                if line.startswith("data:"):
                    e = json.loads(line[5:])
                    if e.get("event") == "token":
                        slots_seen.add(e["slot"])
        assert slots_seen == {"A", "B"}
        assert c.post("/api/model/slots/B/unload").json()["loaded"] is False
        assert c.post("/api/model/worker/stop").json()["worker"] is False
        assert c.get("/api/model/status").json()["worker"] is False
    finally:
        app.state.worker.stop()

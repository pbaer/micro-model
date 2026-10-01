"""Worker-based model harness tests on CPU with the tiny config and a synthetic tokenizer, and external slots (a stubbed
HfChatModel behind an in-process worker, plus one real CPU smoke test through the worker subprocess)."""

import json
import threading

import pytest
import torch
from fastapi.testclient import TestClient

import slm.eval.external as E
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
        # the text view draws reserved tokens as chips from these flags, not from the piece's shape
        assert [t["piece"] for t in chat["tokens"] if t["special"]][:2] == ["<|bos|>", "<|user|>"]
        assert all(t["special"] == (t["piece"].startswith("<|") and t["piece"].endswith("|>")) for t in chat["tokens"])
        evc = _sse_events(c, {"slots": ["A"], "mode": "chat", "messages": [{"role": "user", "content": "hi"}], "max_new_tokens": 2, "temperature": 0.0})
        pr = evc[0]
        assert pr["event"] == "prompt" and len(pr["special"]) == len(pr["pieces"]) == pr["n"]
        assert [p for p, f in zip(pr["pieces"], pr["special"]) if f][:3] == ["<|bos|>", "<|user|>", "<|end|>"] and pr["pieces"][-1] in ("<|assistant|>", "<|think|>") and pr["special"][-1]
        assert not any(f for p, f in zip(pr["pieces"], pr["special"]) if p == "hi")

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


# ------------------------------------------------------------------------------------------ external slots
def _sse_events(c, body, url="/api/model/generate"):
    out = []
    with c.stream("POST", url, json=body) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("data:"):
                out.append(json.loads(line[5:]))
    return out[1:-1]  # drop start / end


def test_external_slot_list_load_generate_and_refusals(tmp_path, monkeypatch):
    """Stubbed HfChatModel behind an in-process worker: the external group in /checkpoints, loading, streaming
    completion and chat, and the refusals (base model in chat, our tool protocol, scoring)."""
    from _ext_stub import InProcWorker, patch_external

    from slm.portal.services.harness import Harness

    loads = patch_external(monkeypatch)
    (tmp_path / "runs").mkdir()
    app = create_app(PortalSettings(runs_root=tmp_path / "runs", data_root=tmp_path, gpu_policy="cpu", open_browser=False))
    h = Harness(tmp_path / "tokenizer")
    app.state.worker = InProcWorker(h)
    c = TestClient(app)

    ext = [x for x in c.get("/api/model/checkpoints").json() if x.get("external")]
    assert [x["name"] for x in ext] == list(E.EXTERNAL_MODELS)
    for x in ext:
        m = E.get(x["name"])
        assert x["path"] == f"external:{m.name}" and x["group"] == "external models" and x["stage"] == "external"
        assert (x["hf_id"], x["params"], x["license"], x["is_chat"], x["context"]) == (m.hf_id, m.params, m.license, m.is_chat, m.max_positions)
        assert x["available"] is (m.name != "gpt2-medium")
    r = c.post("/api/model/slots/A/load", json={"checkpoint": "external:gpt2-medium", "device": "cpu"})
    assert r.status_code == 404 and "download" in r.json()["detail"]
    assert c.post("/api/model/slots/A/load", json={"checkpoint": "external:llama-7b", "device": "cpu"}).status_code == 404
    assert loads == [], "nothing reaches the worker for a missing or unknown model"

    info = c.post("/api/model/slots/A/load", json={"checkpoint": "external:smollm2-135m-instruct", "device": "cpu"}).json()
    assert loads == [("smollm2-135m-instruct", "cpu", torch.float32)], "fp32 on cpu"
    assert info["external"] is True and info["stage"] == "external" and info["device"] == "cpu" and info["dtype"] == "float32"
    assert (info["hf_id"], info["params"], info["license"], info["is_chat"], info["context"]) == (
        "HuggingFaceTB/SmolLM2-135M-Instruct", 134_515_008, "Apache-2.0", True, 8192)
    st = c.get("/api/model/status").json()
    assert st["slots"]["A"]["external"] and st["slots"]["A"]["stage"] == "external", "the run/stage lookup leaves an external slot alone"

    # completion: prompt event, one token event per id with the stub's pieces and log-probs, then done
    evs = _sse_events(c, {"slots": ["A"], "mode": "completion", "text": "The capital of France is", "max_new_tokens": 32, "seed": 5})
    assert evs[0]["event"] == "prompt" and evs[0]["n"] == len("The capital of France is") and "".join(evs[0]["pieces"]) == "The capital of France is"
    assert evs[0]["special"] == [False] * evs[0]["n"], "per-id special flags travel with an external prompt too"
    toks = [e for e in evs if e["event"] == "token"]
    assert "".join(t["piece"] for t in toks) == "Paris.<|im_end|>" and [t["special"] for t in toks] == [False] * 6 + [True]
    assert all(t["logprob"] == -0.25 and len(t["topk"]) == 2 and t["slot"] == "A" and not t["inserted"] for t in toks)
    done = evs[-1]
    assert done["event"] == "done" and done["n"] == 7 and done["reason"] == "stop" and done["external"] and "assistant" not in done
    stub = h.slots["A"].ext
    assert stub.log[-1] == ("stream", len("The capital of France is"), 32, 0.8, 5)

    # chat through the model's own template: a well-formed turn with no think span and no ids to carry
    evs = _sse_events(c, {"slots": ["A"], "mode": "chat", "messages": [{"role": "user", "content": "hi", "ids": [1, 2]}], "max_new_tokens": 16})
    assert "".join(evs[0]["pieces"]) == "[user]hi[assistant]"
    a = evs[-1]["assistant"]
    assert a == {"think": None, "answer": "Paris.", "terminated": True, "malformed": False, "well_formed": True, "ids": None, "n_calls": 0, "tool_errors": 0}
    evs = _sse_events(c, {"slots": ["A"], "mode": "chat", "messages": [{"role": "user", "content": "hi"}], "max_new_tokens": 3})
    assert evs[-1]["reason"] == "length" and not evs[-1]["assistant"]["well_formed"]

    # our protocol is refused, one message per option set
    for opt, needle in (({"tools": True}, "Python tool"), ({"functions": [{"name": "f", "signature": "def f()", "comment": "x"}]}, "declared functions"),
                        ({"think_required": True}, "think")):
        evs = _sse_events(c, {"slots": ["A"], "mode": "chat", "messages": [{"role": "user", "content": "hi"}], **opt})
        assert [e["event"] for e in evs] == ["error"] and needle in evs[0]["error"] and "external model" in evs[0]["error"], opt
    r = c.post("/api/model/score", json={"slot": "A", "mode": "completion", "text": "hello"})
    assert r.status_code == 500 and "n/a for an external slot" in r.json()["detail"]

    # a base model: completion works, chat is refused (no template is invented)
    c.post("/api/model/slots/B/load", json={"checkpoint": "external:smollm2-135m", "device": "cpu"})
    evs = _sse_events(c, {"slots": ["B"], "mode": "chat", "messages": [{"role": "user", "content": "hi"}]})
    assert [e["event"] for e in evs] == ["error"] and "base model" in evs[0]["error"] and "completion mode" in evs[0]["error"]
    evs = _sse_events(c, {"slots": ["A", "B"], "mode": "completion", "text": "x", "max_new_tokens": 4, "temperature": 0.0})
    assert {e["slot"] for e in evs if e["event"] == "token"} == {"A", "B"}, "two external slots side by side"

    assert c.post("/api/model/slots/A/unload").json()["loaded"] is False and h.slots["A"].ext is None
    assert c.get("/api/model/status").json()["slots"]["A"] == {"slot": "A", "loaded": False}


@pytest.mark.skipif(not E.get("smollm2-135m-instruct").available(), reason="smollm2-135m-instruct weights not on disk")
def test_external_slot_real_cpu_smoke(tmp_path):
    """The real worker subprocess and the real SmolLM2-135M-Instruct on the CPU: load, then a 16-token greedy chat reply."""
    (tmp_path / "runs").mkdir()
    app = create_app(PortalSettings(runs_root=tmp_path / "runs", data_root=tmp_path, gpu_policy="cpu", open_browser=False))
    c = TestClient(app)
    try:
        info = c.post("/api/model/slots/A/load", json={"checkpoint": "external:smollm2-135m-instruct", "device": "cpu"}).json()
        assert info["external"] and info["device"] == "cpu" and info["dtype"] == "float32" and info["name"] == "smollm2-135m-instruct"
        evs = _sse_events(c, {"slots": ["A"], "mode": "chat", "messages": [{"role": "user", "content": "What is the capital of France?"}],
                              "max_new_tokens": 16, "temperature": 0.0})
        assert evs[0]["event"] == "prompt" and "<|im_start|>" in "".join(evs[0]["pieces"])
        assert len(evs[0]["special"]) == evs[0]["n"] and all(f for p, f in zip(evs[0]["pieces"], evs[0]["special"]) if p == "<|im_start|>")
        toks = [e for e in evs if e["event"] == "token"]
        done = evs[-1]
        assert 0 < len(toks) <= 16 and done["event"] == "done" and done["n"] == len(toks)
        assert all(t["logprob"] <= 0 and len(t["topk"]) == 5 for t in toks)
        assert done["assistant"]["answer"].strip() and done["assistant"]["think"] is None
    finally:
        app.state.worker.stop()

"""Swarm mode of the inference page, without a GPU: request validation and the SSE shape of /api/model/swarm
against a stub worker, the Harness.swarm stages with scripted sampling/selection, and one real (tiny, CPU)
pass through the whole pipeline."""

import json

import pytest
from fastapi.testclient import TestClient

import slm.swarm as S
from slm.config import ModelConfig, load_config, to_dict
from slm.data.answers import SUFFIX
from slm.data.tokenizer import SlmTokenizer, train_bpe
from slm.model import Transformer
from slm.portal.app import create_app
from slm.portal.services import harness as H
from slm.portal.services.worker import STREAM_METHODS
from slm.portal.settings import PortalSettings
from slm.utils.checkpoint import save_snapshot


class StubWorker:
    """Records the stream call and replays a fixed event list, like WorkerClient.stream does."""

    def __init__(self, events):
        self.events, self.calls = events, []

    def alive(self):
        return False

    def stop(self):
        pass

    def maybe_idle_stop(self):
        return False

    def stream(self, method, cancel_flag=None, **kw):
        self.calls.append((method, kw))
        yield from (dict(e) for e in self.events)


def _client(tmp_path, events):
    (tmp_path / "runs").mkdir(exist_ok=True)
    app = create_app(PortalSettings(runs_root=tmp_path / "runs", data_root=tmp_path, gpu_policy="cpu", open_browser=False))
    app.state.worker = StubWorker(events)
    return app, TestClient(app)


def _sse(c, body):
    out = []
    with c.stream("POST", "/api/model/swarm", json=body) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("data:"):
                out.append(json.loads(line[5:]))
    return out


def test_swarm_endpoint_validates_and_streams_stages(tmp_path):
    result = {"final": "12", "groups": [], "candidates": [], "meta": {"seed": 5}}
    events = [{"event": "stage", "stage": "sampling", "k": 4}, {"event": "stage", "stage": "collapsed", "groups": []},
              {"event": "stage", "stage": "selecting"}, {"event": "done", "stage": "done", "result": result}]
    app, c = _client(tmp_path, events)
    assert "swarm" in STREAM_METHODS
    for bad in ({"text": "q", "k": 65}, {"text": "q", "k": 0}, {"text": "q", "slot": "C"}, {"text": "q", "top_p": 0},
                {"text": "q", "temperature": -1}, {"text": "q", "max_calls": 99}, {"text": "   "}):
        assert c.post("/api/model/swarm", json=bad).status_code == 422, bad
    assert app.state.worker.calls == [], "an invalid request never reaches the worker"

    evs = _sse(c, {"text": "What is 3 * 4?", "k": 4, "max_new_tokens": 64})
    assert evs[0]["stream_id"] and evs[-1] == {}  # start ... end
    body = [e for e in evs[1:-1]]
    assert [e.get("stage") for e in body] == ["sampling", "collapsed", "selecting", "done"]
    assert all(e["slot"] == "A" for e in body) and body[-1]["result"]["final"] == "12"
    method, kw = app.state.worker.calls[0]
    assert method == "swarm" and kw["text"] == "What is 3 * 4?" and kw["k"] == 4 and kw["max_new_tokens"] == 64
    assert kw["answer_suffix"] is True and kw["seed"] is None and kw["budget_tokens"] == 2400 and kw["max_groups"] == 12
    assert app.state.streams == {}, "the stream id is released at the end"

    app.state.worker.events = [{"event": "error", "error": "RuntimeError: slot B is empty"}]
    evs = _sse(c, {"text": "q", "slot": "B", "answer_suffix": False})
    assert [e["event"] for e in evs[1:-1]] == ["error"] and evs[1]["slot"] == "B"
    assert app.state.worker.calls[-1][1]["answer_suffix"] is False


# ------------------------------------------------------------------------------------------ harness (in-process)
def _world(tmp_path):
    tok = SlmTokenizer(train_bpe(["x = 6 x*2 2*3 #### 6 ok " * 60, "the cat sat on the mat " * 40], vocab_size=300))
    tok.save(tmp_path / "tokenizer" / "t1")
    cfg = load_config(ModelConfig, "configs/model/tiny.yaml")
    cfg.vocab_size = tok.vocab_size
    ck = tmp_path / "runs" / "r" / "checkpoints" / "snap_1K.pt"
    save_snapshot(ck, Transformer(cfg), to_dict(cfg), {"tokens": 1000, "val_loss": 3.0, "tokenizer_sha256": tok.sha256})
    h = H.Harness(tmp_path / "tokenizer")
    h.load("A", str(ck), device="cpu", dtype="fp32")
    return h, tok


def _cand(i, parsed, from_tool=False):
    return S.Candidate(idx=i, think=f"think {i}", answer=f"#### {parsed}", parsed=parsed, key=S.answer_key(parsed), terminated=True,
                       n_calls=int(from_tool), n_errors=0, calls=[["3*4", parsed]] if from_tool else [], from_tool=from_tool, n_tokens=20)


def test_harness_swarm_stages_scripted(tmp_path, monkeypatch):
    h, tok = _world(tmp_path)
    seen = {}

    def fake_sample(model, tok_, messages, k, temperature, top_p, max_new_tokens, max_calls, seed):
        seen.update(messages=messages, k=k, seed=seed, max_calls=max_calls)
        return [_cand(0, "10"), _cand(1, "10"), _cand(2, "12", from_tool=True), _cand(3, None)]

    def fake_select(model, tok_, messages):
        seen["selector_prompt"] = messages[0]["content"]
        return "check <<3*4=12>>", "#### 12", "12", 1

    monkeypatch.setattr(S, "sample_candidates", fake_sample)
    monkeypatch.setattr(S, "select", fake_select)
    evs = list(h.swarm("A", "What is 3 * 4?", k=4, seed=None, answer_suffix=True))
    assert [e.get("stage") for e in evs] == ["sampling", "collapsed", "selecting", "done"]
    assert seen["messages"][0]["content"] == "What is 3 * 4?" + SUFFIX and isinstance(seen["seed"], int)
    col = evs[1]
    assert [g["answer"] for g in col["groups"]] == ["12", "10"] and col["majority"] == "10" and col["verified_majority"] == "12"
    assert col["n_candidates"] == 4 and col["n_parsed"] == 3 and col["n_verified"] == 1
    assert evs[2]["groups_in_prompt"] == 2 and evs[2]["prompt_tokens"] > 0
    assert seen["selector_prompt"].startswith("What is 3 * 4?\n\n"), "the selector sees the task without the suffix"
    res = evs[-1]["result"]
    assert res["final"] == "12" and res["selector_calls"] == 1 and res["meta"]["selector_parsed"] and res["meta"]["seed"] == seen["seed"]
    assert [c["verified"] for c in res["candidates"]] == [False, False, True, False]
    json.dumps(res)  # crosses the worker pipe and the SSE stream

    # a cancel after sampling skips the selector; final falls back to the verified majority
    evs = list(h.swarm("A", "What is 3 * 4?", k=4, seed=3, answer_suffix=False, should_stop=lambda: True))
    assert [e.get("stage") for e in evs] == ["sampling", "collapsed", "done"]
    res = evs[-1]["result"]
    assert res["meta"]["cancelled"] and res["final"] == "12" and res["selector_messages"] and res["selector_think"] is None
    assert seen["messages"][0]["content"] == "What is 3 * 4?" and seen["seed"] == 3

    with pytest.raises(RuntimeError, match="empty"):
        list(h.swarm("B", "q"))


def test_harness_swarm_real_tiny_model(tmp_path):
    """The real sampler and selector on a random tiny model: garbage answers, but every stage runs and the result
    is complete and serializable."""
    h, _ = _world(tmp_path)
    evs = list(h.swarm("A", "What is 2 * 3?", k=2, max_new_tokens=6, max_calls=1, seed=1))
    assert evs[0]["stage"] == "sampling" and evs[-1]["event"] == "done"
    res = evs[-1]["result"]
    assert res["k"] == 2 and len(res["candidates"]) == 2 and all("verified" in c and c["n_tokens"] <= 6 for c in res["candidates"])
    assert json.dumps(res)
    again = list(h.swarm("A", "What is 2 * 3?", k=2, max_new_tokens=6, max_calls=1, seed=1))[-1]["result"]
    assert [c["answer"] for c in again["candidates"]] == [c["answer"] for c in res["candidates"]], "same seed, same samples"

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
                {"text": "q", "temperature": -1}, {"text": "q", "max_calls": 99}, {"text": "   "}, {"text": "q", "mode": "vote"},
                {"text": "q", "max_entrants": 1}, {"text": "q", "max_entrants": 65}, {"text": "q", "pair_budget_tokens": 100}):
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
    assert kw["mode"] == "both" and kw["pair_budget_tokens"] == 1200 and kw["max_entrants"] == 16
    _sse(c, {"text": "q", "mode": "tournament", "max_entrants": 2, "pair_budget_tokens": 800})
    kw = app.state.worker.calls[-1][1]
    assert kw["mode"] == "tournament" and kw["max_entrants"] == 2 and kw["pair_budget_tokens"] == 800
    _sse(c, {"text": "q", "mode": "select", "max_entrants": 64})
    assert app.state.worker.calls[-1][1]["mode"] == "select"
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
    evs = list(h.swarm("A", "What is 3 * 4?", k=4, seed=None, answer_suffix=True, mode="select"))
    assert [e.get("stage") for e in evs] == ["sampling", "collapsed", "selecting", "done"]
    assert seen["messages"][0]["content"] == "What is 3 * 4?" + SUFFIX and isinstance(seen["seed"], int)
    col = evs[1]
    assert [g["answer"] for g in col["groups"]] == ["12", "10"] and col["majority"] == "10" and col["verified_majority"] == "12"
    assert col["n_candidates"] == 4 and col["n_parsed"] == 3 and col["n_verified"] == 1
    assert evs[2]["groups_in_prompt"] == 2 and evs[2]["prompt_tokens"] > 0
    assert seen["selector_prompt"].startswith("What is 3 * 4?\n\n"), "the selector sees the task without the suffix"
    res = evs[-1]["result"]
    assert res["final"] == "12" and res["selector_calls"] == 1 and res["meta"]["selector_parsed"] and res["meta"]["seed"] == seen["seed"]
    assert res["tournament"] is None and res["rounds"] == [] and res["meta"]["mode"] == "select"
    assert [c["verified"] for c in res["candidates"]] == [False, False, True, False]
    json.dumps(res)  # crosses the worker pipe and the SSE stream

    # a cancel after sampling skips the selector; final falls back to the verified majority
    evs = list(h.swarm("A", "What is 3 * 4?", k=4, seed=3, answer_suffix=False, mode="select", should_stop=lambda: True))
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
    assert res["meta"]["mode"] == "both"
    if len(res["groups"]) >= 2:  # the real compare_batch ran: a complete bracket with a champion among the groups
        assert res["rounds"] and res["tournament"] in [g["answer"] for g in res["groups"]] and res["final"] == res["tournament"]
    again = list(h.swarm("A", "What is 2 * 3?", k=2, max_new_tokens=6, max_calls=1, seed=1))[-1]["result"]
    assert [c["answer"] for c in again["candidates"]] == [c["answer"] for c in res["candidates"]], "same seed, same samples"


# ------------------------------------------------------------------------------------------ tournament (harness)
def _bracket_cands():
    """Five distinct answers in evidence order 1 (verified), 2 (support 2), 3, 4, 5: the bracket of tests/test_swarm.py."""
    return [_cand(0, "1", from_tool=True), _cand(1, "2"), _cand(2, "2"), _cand(3, "3"), _cand(4, "4"), _cand(5, "5")]


def _scripted(monkeypatch, seen):
    """sample_candidates -> the five-answer pool; compare_batch -> picks the larger number, no pick for a pair holding '4'
    (so the evidence fallback decides it); select -> '#### 1'."""
    monkeypatch.setattr(S, "sample_candidates", lambda *a, **kw: _bracket_cands())

    def fake_compare(model, tok_, task_prompt, pairs, budget_tokens=1200, max_new_tokens=96):
        seen.setdefault("compare", []).append(([(x.key, y.key) for x, y in pairs], task_prompt, budget_tokens, max_new_tokens))
        return [None if "4" in (x.key, y.key) else (0 if float(x.key) > float(y.key) else 1) for x, y in pairs]

    def fake_select(model, tok_, messages):
        seen["selected"] = True
        return "pick", "#### 1", "1", 0

    monkeypatch.setattr(S, "compare_batch", fake_compare)
    monkeypatch.setattr(S, "select", fake_select)
    return fake_compare


def test_harness_swarm_tournament_rounds_stream_and_match_the_library(tmp_path, monkeypatch):
    h, _ = _world(tmp_path)
    seen = {}
    fake_compare = _scripted(monkeypatch, seen)
    evs = list(h.swarm("A", "What is it?", k=6, seed=2, mode="tournament", pair_budget_tokens=900))
    assert [(e.get("stage"), e.get("round")) for e in evs] == [("sampling", None), ("collapsed", None), ("tournament", 0), ("tournament", 1),
                                                              ("tournament", 2), ("tournament", 3), ("done", None)]
    assert "selected" not in seen, "tournament mode runs no selector"
    seeded = evs[2]
    assert seeded["entrants"] == ["1", "2", "3", "4", "5"] and seeded["matches"] == [] and seeded["n_rounds_expected"] == 3
    r1 = evs[3]
    assert set(r1) >= {"event", "stage", "round", "n_rounds_expected", "matches", "entrants", "byes", "seconds"}
    assert r1["entrants"] == ["1", "2", "3", "4", "5"] and r1["byes"] == ["3"] and r1["n_rounds_expected"] == 3
    assert r1["matches"] == [{"a": "1", "b": "5", "swapped": False, "pick": 1, "winner": "5"},
                             {"a": "4", "b": "2", "swapped": True, "pick": None, "winner": "2"}], "second pair swapped; no pick -> evidence (support 2)"
    assert evs[4]["entrants"] == ["5", "2", "3"] and evs[4]["byes"] == ["2"] and evs[4]["matches"] == [
        {"a": "5", "b": "3", "swapped": False, "pick": 0, "winner": "5"}]
    assert evs[5]["entrants"] == ["5", "2"] and evs[5]["byes"] == [] and evs[5]["matches"][0]["winner"] == "5"
    assert seen["compare"][0][1:] == ("What is it?", 900, 96), "pairwise prompts see the task without the suffix, the pair budget, 96 new tokens"

    res = evs[-1]["result"]
    assert res["tournament"] == "5" and res["final"] == "5" and res["meta"]["mode"] == "tournament" and res["meta"]["tournament_complete"]
    assert res["selector_messages"] == [] and res["meta"]["selector_final"] is None and res["meta"]["n_entrants"] == 5
    assert res["rounds"] == [e["matches"] for e in evs[3:6]]
    groups = S.collapse(_bracket_cands())
    champ, lib_rounds = S.tournament(None, None, "What is it?", groups, compare=lambda pairs: fake_compare(None, None, "q", pairs))
    assert res["rounds"] == lib_rounds and res["tournament"] == champ.answer, "the streamed bracket is the library's bracket"
    json.dumps(res)


def test_harness_swarm_both_modes_and_entrant_cap(tmp_path, monkeypatch):
    h, _ = _world(tmp_path)
    seen = {}
    _scripted(monkeypatch, seen)
    evs = list(h.swarm("A", "What is it?", k=6, seed=2))  # default mode: both
    assert [e.get("stage") for e in evs] == ["sampling", "collapsed", "selecting", "tournament", "tournament", "tournament", "tournament", "done"]
    res = evs[-1]["result"]
    assert seen["selected"] and res["meta"]["selector_final"] == "1" and res["tournament"] == "5"
    assert res["final"] == "5", "in both modes final follows the bracket; the selector's pick is kept for comparison"

    evs = list(h.swarm("A", "What is it?", k=6, seed=2, mode="tournament", max_entrants=2))
    rounds = [e for e in evs if e.get("stage") == "tournament"]
    assert rounds[0]["entrants"] == ["1", "2"] and rounds[0]["n_rounds_expected"] == 1 and len(rounds) == 2
    assert evs[-1]["result"]["tournament"] == "2" and evs[-1]["result"]["meta"]["n_entrants"] == 2

    evs = list(h.swarm("A", "What is it?", k=6, seed=2, mode="select"))
    assert "tournament" not in [e.get("stage") for e in evs] and evs[-1]["result"]["final"] == "1"
    with pytest.raises(RuntimeError, match="mode"):
        list(h.swarm("A", "q", mode="vote"))


def test_harness_swarm_tournament_cancel_at_round_boundary(tmp_path, monkeypatch):
    h, _ = _world(tmp_path)
    seen = {}
    _scripted(monkeypatch, seen)
    calls = {"n": 0}

    def stop_after(n):
        def should_stop():
            calls["n"] += 1
            return calls["n"] > n
        return should_stop

    # tournament mode: the check before round 1 passes, the one before round 2 cancels
    evs = list(h.swarm("A", "What is it?", k=6, seed=2, mode="tournament", should_stop=stop_after(1)))
    assert [(e.get("stage"), e.get("round")) for e in evs] == [("sampling", None), ("collapsed", None), ("tournament", 0), ("tournament", 1), ("done", None)]
    res = evs[-1]["result"]
    assert res["meta"]["cancelled"] and not res["meta"]["tournament_complete"] and len(seen["compare"]) == 1
    assert len(res["rounds"]) == 1 and res["tournament"] is None
    assert res["final"] == res["verified_majority"] == "1", "no champion: the verified majority, as in the library"

    # both: a cancel before the selector skips the bracket too
    calls["n"], seen = 0, {}
    _scripted(monkeypatch, seen)
    evs = list(h.swarm("A", "What is it?", k=6, seed=2, mode="both", should_stop=stop_after(0)))
    assert [e.get("stage") for e in evs] == ["sampling", "collapsed", "done"] and "compare" not in seen and "selected" not in seen
    assert evs[-1]["result"]["meta"]["cancelled"] and evs[-1]["result"]["final"] == "1"

    # both: selector runs (check 1), round 1 runs (check 2), cancel before round 2
    calls["n"], seen = 0, {}
    _scripted(monkeypatch, seen)
    evs = list(h.swarm("A", "What is it?", k=6, seed=2, mode="both", should_stop=stop_after(2)))
    assert [e.get("stage") for e in evs] == ["sampling", "collapsed", "selecting", "tournament", "tournament", "done"]
    res = evs[-1]["result"]
    assert seen["selected"] and len(res["rounds"]) == 1 and res["tournament"] is None and res["meta"]["selector_final"] == "1"


# ------------------------------------------------------------------------------------------ external slot
def test_harness_swarm_external_slot_has_no_verification(tmp_path, monkeypatch):
    """A stubbed external chat model: k replies in one batch through its own template, '#### <answer>' parsed and
    collapsed as for ours, selector and pairwise prompts through the same template, greedy -- and verification n/a
    everywhere (no sandbox): null n_verified / verified_majority / per-candidate verified, the plain majority as fallback."""
    from _ext_stub import patch_external

    patch_external(monkeypatch)
    h = H.Harness(tmp_path / "tokenizer")
    h.load("A", "external:smollm2-135m-instruct", device="cpu")
    stub = h.slots["A"].ext
    evs = list(h.swarm("A", "What is 3 * 4?", k=4, max_new_tokens=48, max_calls=6, seed=9, temperature=0.9))
    assert [(e.get("stage"), e.get("round")) for e in evs] == [("sampling", None), ("collapsed", None), ("selecting", None), ("tournament", 0),
                                                              ("tournament", 1), ("done", None)]
    assert all(e["verification"] == "n/a" and e["external"] for e in evs[:3])
    sample = stub.log[0]
    assert sample[:5] == ("batch", 4, 48, 0.9, 9) and sample[5] == ["What is 3 * 4?" + SUFFIX] * 4, "k copies of task + suffix, one batch, seeded"
    col = evs[1]
    assert [g["answer"] for g in col["groups"]] == ["10", "12"] and col["majority"] == "10"
    assert col["n_verified"] is None and col["verified_majority"] is None and col["n_parsed"] == 3
    assert col["groups"][0]["rationale"] == "Ten." and col["groups"][1]["rationale"] == "Twelve, from 3 * 4.", "the shortest member reply, up to its #### line"
    sel = [x for x in stub.log if x[0] == "chat"]
    assert len(sel) == 1 and sel[0][1].startswith("What is 3 * 4?\n\n" + S.SELECT_INTRO) and sel[0][3] == 0.0, "selector: greedy, own template"
    assert "- Answer: 12 (agreed by 1 attempt; not computed with code)\n  Reasoning: Twelve, from 3 * 4." in sel[0][1]
    pair = [x for x in stub.log if x[0] == "batch"][-1]
    assert pair[1] == 1 and pair[2] == 96 and pair[3] == 0.0 and S.PAIR_ASK in pair[5][0], "one greedy pairwise batch per round"
    assert evs[4]["matches"] == [{"a": "10", "b": "12", "swapped": False, "pick": 1, "winner": "12"}]

    res = evs[-1]["result"]
    assert res["tournament"] == "12" and res["final"] == "12" and res["meta"]["selector_final"] == "12" and res["majority"] == "10"
    assert res["verified_majority"] is None and all(c["verified"] is None for c in res["candidates"])
    assert res["meta"]["external"] and res["meta"]["verification"] == "n/a" and res["meta"]["max_calls"] == 0
    assert res["meta"]["checkpoint"] == "smollm2-135m-instruct" and res["meta"]["prompt_tokens"] == len(res["selector_messages"][0]["content"])
    assert res["candidates"][3]["parsed"] is None and res["candidates"][0]["think"] is None and res["selector_calls"] == 0
    json.dumps(res)

    # select mode with an unparsable selector reply: the plain majority (there is no verified one)
    stub.selector_reply = "I am not sure."
    res = list(h.swarm("A", "What is 3 * 4?", k=4, seed=9, mode="select"))[-1]["result"]
    assert res["final"] == "10" and not res["meta"]["selector_parsed"] and res["tournament"] is None

    # a base model: swarm is n/a (it samples chat replies, and no template is invented)
    h.load("B", "external:smollm2-135m", device="cpu")
    with pytest.raises(RuntimeError, match="base model"):
        list(h.swarm("B", "What is 3 * 4?", k=2))

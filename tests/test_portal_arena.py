"""The Arena tab without a GPU: Harness.arena with a scripted generator (event shape per turn, the messages list, stop at
done, cancel, refusals), one real turn of a tiny CPU model through `arena_generate`, and /api/model/arena (validation,
SSE shape against a stub worker) plus /api/model/arena/tasks."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from slm.arena.world import TASKS, World
from slm.config import ModelConfig, load_config, to_dict
from slm.data.tokenizer import SlmTokenizer, train_bpe
from slm.model import Transformer
from slm.portal.app import create_app
from slm.portal.services import harness as H
from slm.portal.services.worker import STREAM_METHODS
from slm.portal.settings import PortalSettings
from slm.tools.functions import parse_defs
from slm.tools.loop import ToolCompletion
from slm.utils.checkpoint import save_snapshot


# ------------------------------------------------------------------------------------------ harness (in-process)
@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("arena")
    text = ("You are robot R1 on a grid. move north south east west say look read key open door code turn heard sight "
            "the code is 1234 5678 9012 status moving to the door ") * 40
    tok = SlmTokenizer(train_bpe([text, "def move(direction: str) -> str " * 30], vocab_size=400))
    tok.save(tmp / "tokenizer" / "t1")
    cfg = load_config(ModelConfig, "configs/model/tiny.yaml")
    cfg.vocab_size, cfg.max_seq_len = tok.vocab_size, 4096
    ck = tmp / "runs" / "r" / "checkpoints" / "snap_1K.pt"
    save_snapshot(ck, Transformer(cfg), to_dict(cfg), {"tokens": 1000, "val_loss": 3.0, "tokenizer_sha256": tok.sha256})
    h = H.Harness(tmp / "tokenizer")
    h.load("A", str(ck), device="cpu", dtype="fp32")
    return h, tok


def _tc(tok, think: str, answer: str, calls: list) -> ToolCompletion:
    """A generated turn as sample_with_tools returns it: think, </think>, the status line, <|end|>."""
    ids = tok.encode(think) + [tok.special("<|/think|>")] + tok.encode(answer) + [tok.end_id]
    return ToolCompletion(ids=ids, gen_mask=[1] * len(ids), n_calls=len(calls), calls=calls, termination="stop")


def _key_door_script(tok, seen: dict):
    """Robot R1 reads the key and says it; R2 walks to the door and opens it with the code it heard."""

    def generate(runner, prompts, sessions):
        w = runner.world
        seen.setdefault("prompts", []).append(prompts)
        door = next(iter(w.task.doors))
        out = []
        for a, s in zip(w.agents, sessions):
            env = s.functions  # the session holds only this turn's tool (one tool per turn, enforced at execution)
            if a.id == 0:
                if "read_key" in env:
                    res = env["read_key"](); seen["key"] = res.split()[-1].rstrip(".")
                    out.append(_tc(tok, "read it", "#### " + seen["key"], [("read_key()", res)]))
                elif "say" in env:
                    said = env["say"](f"the code is {seen['key']}")
                    out.append(_tc(tok, "say it", "Told R2.", [(f"say('the code is {seen['key']}')", said)]))
                else:
                    name = next(iter(env)); out.append(_tc(tok, name, "ok", [(f"{name}('east')" if name == "move" else f"{name}()", env[name]("east") if name == "move" else env[name]())]))
                continue
            heard = [m for m in a.inbox if "code is" in m]
            if heard:
                seen["code"] = heard[-1].split()[-1].strip("'\".")
            if "move" in env:
                dx, dy = door[0] - a.x, door[1] - a.y
                d = "east" if dx > 0 else "west" if dx < 0 else "south" if dy > 0 else "north"
                out.append(_tc(tok, "walk", "Moving.", [(f"move({d!r})", env["move"](d))]))
            elif "open_door" in env:
                out.append(_tc(tok, "open", "Opening.", [(f"open_door({seen.get('code')!r})", env["open_door"](seen.get("code")))]))
            else:
                name = next(iter(env)); out.append(_tc(tok, name, "ok", [(f"{name}()", env[name]())]))
        return out

    return generate


def test_harness_arena_turn_events_messages_and_stop_at_done(harness, monkeypatch):
    h, tok = harness
    seen = {}
    monkeypatch.setattr(H, "arena_generate", _key_door_script(tok, seen))
    evs = list(h.arena("A", task="key_door", n=5, n_agents=2, seed=7, turns=20, max_new_tokens=64))
    assert "arena" in STREAM_METHODS
    kinds = [e["event"] for e in evs]
    assert kinds[0] == "start" and kinds[-1] == "done" and set(kinds[1:-1]) == {"turn"}

    start = evs[0]
    assert start["state"]["turn"] == 0 and start["state"]["task"] == "key_door" and len(start["state"]["agents"]) == 2
    assert start["meta"]["comm_range"] == 3 and start["meta"]["sight"] == 2 and start["meta"]["device"] == "cpu"
    r1, r2 = start["robots"]
    assert [t["name"] for t in r1["tools"]] == ["move", "say", "look", "read_key"]
    assert [t["name"] for t in r2["tools"]] == ["move", "say", "look", "open_door"]
    assert r2["tools"][-1]["signature"] == "def open_door(code: str) -> str" and r2["tools"][-1]["comment"]
    assert r1["system_prompt"].startswith("You are robot R1") and "read_key()" in r1["system_prompt"]

    turns = evs[1:-1]
    # the prompts the model saw: one per robot, each opening with the declared-function blocks of that robot's turn
    p1 = seen["prompts"][0]
    assert len(p1) == 2 and all(isinstance(i, int) for i in p1[0])
    for j, rob in enumerate((r1, r2)):
        declared = [d.name for d in parse_defs(h.slots["A"].tok, p1[j])]
        assert declared and set(declared) <= {t["name"] for t in rob["tools"]}
        assert declared == turns[0]["records"][j]["declared"], "the record names the tools declared that turn"

    assert [t["turn"] for t in turns] == list(range(1, len(turns) + 1))
    for t in turns:
        assert set(t) >= {"turn", "seconds", "records", "state", "messages"} and t["state"]["turn"] == t["turn"]
        assert [r["agent"] for r in t["records"]] == ["R1", "R2"] and [r["agent_id"] for r in t["records"]] == [0, 1]
        for r in t["records"]:
            assert r["turn"] == t["turn"] and r["observation"].strip() and r["n_calls"] == len(r["calls"])
            assert "status" not in r or r["status"].startswith(f"Turn {t['turn']}."), "the world's view of the robot (not shown to it)"
            assert r["raw"].startswith("<|think|>") and "<|/think|>" in r["raw"] and r["raw"].endswith("<|end|>")
            assert r["think"] is not None and r["answer"]
    assert turns[0]["messages"] == [], "turn 1: the key holder reads the key (one tool per turn); nobody speaks yet"
    m = turns[1]["messages"]
    code = seen["code"]
    r1_pos = [turns[1]["state"]["agents"][0]["x"], turns[1]["state"]["agents"][0]["y"]]
    assert m == [{"speaker": "R1", "speaker_id": 0, "pos": r1_pos, "text": f"the code is {code}", "heard_by": ["R2"], "heard_by_ids": [1]}]
    assert "Heard by: R2" in turns[1]["records"][0]["calls"][0][1]
    assert "R1 said: 'the code is" in turns[2]["records"][1].get("status", turns[2]["records"][1]["observation"]), "delivered at the next turn"
    # every record carries its inbox: what was delivered to that robot at the start of the turn
    assert all(r["heard"] == [] for r in turns[0]["records"]) and all(r["heard"] == [] for r in turns[1]["records"])
    assert turns[2]["records"][1]["heard"] == [f"R1 said: 'the code is {code}'"] and turns[2]["records"][0]["heard"] == []
    assert _heard_matches_status(turns)
    assert all(all(x["speaker"] == "R1" for x in t["messages"]) for t in turns), "only the key holder ever speaks"
    assert sum(len(t["messages"]) for t in turns) >= 1

    res = evs[-1]["result"]
    assert res["score"]["success"] and res["turns"] == len(turns) < 20, "stop_when_done ends the episode at success"
    assert any("opened door 1" in e for e in res["events"]) and res["cancelled"] is False
    assert len(res["transcript"]) == 2 * res["turns"] and len(res["messages"]) == res["turns"], "one message list per turn"
    assert sum(len(m) for m in res["messages"]) >= 1 and res["messages"][0] == []
    assert res["meta"]["task"] == "key_door" and res["meta"]["n_agents"] == 2 and res["meta"]["turns"] == 20
    json.dumps(evs)  # crosses the worker pipe and the SSE stream

    # without stop_when_done every turn is played
    evs = list(h.arena("A", task="key_door", n=5, n_agents=2, seed=7, turns=res["turns"] + 2, stop_when_done=False))
    assert evs[-1]["result"]["turns"] == res["turns"] + 2 and sum(e["event"] == "turn" for e in evs) == res["turns"] + 2


def _heard_matches_status(turns) -> bool:
    """`heard` is the same inbox the world's status line reports (the viewer falls back to parsing it)."""
    for t in turns:
        for r in t["records"]:
            tail = r["status"].split(". Heard: ", 1)[1].rstrip(".")
            if tail != ("; ".join(r["heard"]) if r["heard"] else "nothing"):
                return False
    return True


def test_harness_arena_cancel_between_turns(harness, monkeypatch):
    h, tok = harness
    monkeypatch.setattr(H, "arena_generate", _key_door_script(tok, {}))
    calls = {"n": 0}

    def stop_after_one():
        calls["n"] += 1
        return calls["n"] > 1

    evs = list(h.arena("A", task="key_door", n=8, n_agents=2, seed=1, turns=10, should_stop=stop_after_one))
    assert [e["event"] for e in evs] == ["start", "turn", "done"]
    res = evs[-1]["result"]
    assert res["cancelled"] is True and res["turns"] == 1 and len(res["transcript"]) == 2
    evs = list(h.arena("A", task="relay", n=10, n_agents=4, seed=0, turns=5, should_stop=lambda: True))
    assert [e["event"] for e in evs] == ["start", "done"] and evs[-1]["result"]["turns"] == 0 and evs[-1]["result"]["cancelled"]


def test_harness_arena_refusals(harness):
    h, _ = harness
    for kw, msg in (({"task": "key_door", "n_agents": 3}, "even number"), ({"task": "relay", "n_agents": 1}, "at least 2"),
                    ({"task": "maze"}, "unknown arena task"), ({"n": 3}, "n=3"), ({"n": 17}, "n=17"), ({"n_agents": 33, "task": "triangulate"}, "n_agents"),
                    ({"turns": 0}, "turns"), ({"turns": 51}, "turns"), ({"max_new_tokens": 8}, "max_new_tokens")):
        with pytest.raises(RuntimeError, match=msg):
            list(h.arena("A", **kw))
    with pytest.raises(RuntimeError, match="slot B is empty"):
        list(h.arena("B"))

    class FakeExt:
        name = "smollm2-135m-instruct"
    h.slots["B"].ext = FakeExt()
    try:
        with pytest.raises(RuntimeError, match="n/a for an external slot"):
            list(h.arena("B", task="relay", n_agents=2))
    finally:
        h.slots["B"].ext = None


def test_arena_messages_follow_delivery():
    w = World("key_door", n=8, n_agents=4, seed=2)
    a, b, c, d = w.agents
    a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y = 0, 0, 1, 1, 7, 7, 0, 3
    env = {x.id: {f.name: f.impl for f in w.decls[x.id]} for x in w.agents}
    env[0]["say"]("door 1 code 1234")
    env[2]["say"]("anyone?")
    msgs = H.arena_messages(w)
    assert msgs == [{"speaker": "R1", "speaker_id": 0, "pos": [0, 0], "text": "door 1 code 1234", "heard_by": ["R2", "R4"], "heard_by_ids": [1, 3]},
                    {"speaker": "R3", "speaker_id": 2, "pos": [7, 7], "text": "anyone?", "heard_by": [], "heard_by_ids": []}]
    w.deliver()
    assert b.inbox and d.inbox and not c.inbox, "heard_by is exactly who deliver() reaches"


def test_arena_real_turn_on_cpu(harness):
    """No scripting: one turn of a tiny CPU model through arena_generate (Runner._generate would need cuda)."""
    h, _ = harness
    evs = list(h.arena("A", task="triangulate", n=4, n_agents=2, seed=3, turns=1, max_new_tokens=16, max_calls=1, max_history=0))
    assert [e["event"] for e in evs] == ["start", "turn", "done"]
    t = evs[1]
    assert len(t["records"]) == 2 and all(isinstance(r["raw"], str) or r["raw"] is None for r in t["records"])
    assert evs[-1]["result"]["turns"] == 1
    json.dumps(evs)


# ------------------------------------------------------------------------------------------ API (stub worker)
class StubWorker:
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
    with c.stream("POST", "/api/model/arena", json=body) as r:
        assert r.status_code == 200
        name = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                out.append((name, json.loads(line[5:])))
    return out


def test_arena_endpoint_validates_and_streams(tmp_path):
    state = {"turn": 0, "n": 5, "task": "key_door", "agents": [], "objects": [], "score": {}, "events": []}
    events = [{"event": "start", "state": state, "robots": [], "meta": {}},
              {"event": "turn", "turn": 1, "seconds": 0.1, "records": [], "state": {**state, "turn": 1}, "messages": [{"speaker": "R1", "text": "hi", "heard_by": ["R2"]}]},
              {"event": "done", "result": {"turns": 1, "score": {"success": True}}}]
    app, c = _client(tmp_path, events)
    for bad in ({"n": 3}, {"n": 17}, {"n_agents": 0}, {"n_agents": 34}, {"turns": 0}, {"turns": 51}, {"task": "maze"}, {"slot": "C"},
                {"task": "key_door", "n_agents": 3}, {"task": "relay", "n_agents": 1}, {"max_new_tokens": 8}, {"max_calls": 17},
                {"max_history": 11}, {"seed": -1}):
        assert c.post("/api/model/arena", json=bad).status_code == 422, bad
    assert app.state.worker.calls == [], "an invalid request never reaches the worker"

    evs = _sse(c, {"task": "relay", "n": 10, "n_agents": 4, "turns": 6})
    assert evs[0][0] == "start" and evs[0][1]["stream_id"] and evs[-1] == ("end", {})
    body = evs[1:-1]
    assert [n for n, _ in body] == ["start", "turn", "done"] and all(d["slot"] == "A" for _, d in body)
    assert body[1][1]["messages"][0]["heard_by"] == ["R2"] and body[2][1]["result"]["score"]["success"]
    method, kw = app.state.worker.calls[0]
    assert method == "arena" and kw == {"slot": "A", "task": "relay", "n": 10, "n_agents": 4, "seed": 0, "turns": 6, "max_new_tokens": 128,
                                        "max_calls": 4, "max_history": 1, "stop_when_done": True}
    _sse(c, {"task": "triangulate", "n_agents": 1, "n": 16, "turns": 50, "slot": "B", "stop_when_done": False, "max_new_tokens": 1024})
    kw = app.state.worker.calls[-1][1]
    assert kw["slot"] == "B" and kw["n_agents"] == 1 and kw["stop_when_done"] is False and kw["n"] == 16
    _sse(c, {"task": "key_door", "n_agents": 32})
    assert app.state.worker.calls[-1][1]["n_agents"] == 32
    assert app.state.streams == {}

    app.state.worker.events = [{"event": "error", "error": "RuntimeError: the arena is n/a for an external slot"}]
    evs = _sse(c, {"task": "relay", "n_agents": 2})
    assert [n for n, _ in evs[1:-1]] == ["error"] and "external" in evs[1][1]["error"]


def test_arena_tasks_endpoint_and_nav(tmp_path):
    _, c = _client(tmp_path, [])
    tasks = c.get("/api/model/arena/tasks").json()
    assert [t["key"] for t in tasks] == sorted(TASKS)
    for t in tasks:
        assert t["title"] and len(t["description"]) > 150 and t["roles"]
        assert t["comm_range"] == TASKS[t["key"]].comm_range and t["sight"] == TASKS[t["key"]].sight
    kd = next(t for t in tasks if t["key"] == "key_door")
    assert kd["even_agents"] and kd["min_agents"] == 2
    assert {"id": "arena", "label": "Arena"} in c.get("/api/meta").json()["pages"]


def test_arena_js_parses_and_is_routed():
    static = Path("slm/portal/static")
    src = (static / "app.js").read_text(encoding="utf-8")
    assert 'from "./pages/arena.js"' in src and 'page === "arena"' in src
    arena = (static / "pages" / "arena.js").read_text(encoding="utf-8")
    assert "/api/model/arena" in arena and "TextWithSpecials" in arena
    assert "/api/model/arena/episodes" in arena and "/api/model/arena/world" in arena, "viewer mode"
    views = (static / "components" / "arena_views.js").read_text(encoding="utf-8")
    assert all(f"export function {x}" in views for x in ("Timeline", "MessageLog", "SummaryStrip", "normalizeEpisode", "analyze", "heardOf"))
    arena += views
    cards = (static / "components" / "cards.js").read_text(encoding="utf-8")
    import re

    for k in set(re.findall(r'k="(arena[a-z_]*)"', arena)):
        assert f"  {k}: {{" in cards, f"info card {k} missing"
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    for f in (static / "app.js", static / "pages" / "arena.js", static / "pages" / "model.js", static / "components" / "cards.js",
              static / "components" / "arena_views.js"):
        tmp = Path(tempfile.gettempdir()) / "slm_arena_check.mjs"
        shutil.copy(f, tmp)
        r = subprocess.run([node, "--check", str(tmp)], capture_output=True, text=True)
        assert r.returncode == 0, f"{f}: {r.stderr[:400]}"


# ------------------------------------------------------------------------------------------ viewer: episode files and the rebuilt world
def _cli_episode(task="relay", n=10, agents=4, seed=0):
    """A CLI-shaped episode (python -m slm.arena --out) from a real World and a scripted generator: whoever holds say or
    submit uses it, the rest call their one declared tool."""
    from slm.arena.world import Runner

    def generate(prompts, sessions):
        out = []
        for s in sessions:
            name = next(iter(s.functions))
            res = s.functions[name]("the code is 1234") if name in ("say", "submit", "move") else s.functions[name]()
            out.append({"think": name, "answer": "ok", "calls": [[f"{name}()", str(res)]]})
        return out

    w = World(task, n, agents, seed)
    return Runner(w, generate=generate).run(3, stop_when_done=False)


def test_arena_episodes_list_read_and_path_safety(tmp_path, monkeypatch):
    from slm.portal.services import arena_files as AF

    _, c = _client(tmp_path, [])
    assert c.get("/api/model/arena/episodes").json() == [], "no runs/arena directory yet"
    d = tmp_path / "runs" / "arena"
    d.mkdir()
    cli = _cli_episode()
    (d / "cli_relay.json").write_text(json.dumps(cli), encoding="utf-8")
    portal = {"request": {"task": "key_door"}, "meta": {"task": "key_door", "n": 8, "n_agents": 2, "seed": 3, "checkpoint": "final.pt"}, "robots": [],
              "start": {"turn": 0}, "turns": [{"turn": 1}], "result": {"turns": 1, "score": {"success": True}}}
    (d / "tab_download.json").write_text(json.dumps(portal), encoding="utf-8")
    (d / "eval_sweep.json").write_text(json.dumps({"checkpoint": "x.pt", "turns": 16, "results": []}), encoding="utf-8")
    (d / "broken.json").write_text("{not json", encoding="utf-8")
    (d / "notes.txt").write_text("not listed", encoding="utf-8")
    (d / "sub").mkdir()
    (d / "sub" / "deep.json").write_text("{}", encoding="utf-8")
    (tmp_path / "runs" / "secret.json").write_text(json.dumps({"secret": 1}), encoding="utf-8")

    eps = {e["name"]: e for e in c.get("/api/model/arena/episodes").json()}
    assert set(eps) == {"cli_relay.json", "tab_download.json", "eval_sweep.json", "broken.json"}, "top-level *.json only"
    assert eps["cli_relay.json"] | {"size": 0, "mtime": 0} == {"name": "cli_relay.json", "size": 0, "mtime": 0, "kind": "cli", "task": "relay", "n": 10,
                                                                  "agents": 4, "seed": 0, "turns": 3, "success": False}
    assert eps["tab_download.json"]["kind"] == "portal" and eps["tab_download.json"]["agents"] == 2 and eps["tab_download.json"]["success"] is True
    assert eps["eval_sweep.json"]["kind"] == "summary" and eps["broken.json"]["kind"] == "unreadable"

    assert c.get("/api/model/arena/episodes/cli_relay.json").json() == json.loads(json.dumps(cli))
    assert c.get("/api/model/arena/episodes/tab_download.json").json()["meta"]["seed"] == 3
    assert c.get("/api/model/arena/episodes/missing.json").status_code == 404
    for bad in ("..%2Fsecret.json", "..%5Csecret.json", "notes.txt", ".hidden.json", "a..b.json", "%2E%2E%2Fsecret.json", "sub%2Fdeep.json"):
        r = c.get(f"/api/model/arena/episodes/{bad}")
        assert r.status_code in (400, 404) and set(r.json()) == {"detail"}, (bad, r.status_code, r.text)  # never the file outside
    for bad in ("../secret.json", "..\\secret.json", "C:\\x.json", "/etc/x.json", "sub/deep.json", "x.json\x00", ""):
        with pytest.raises(ValueError):
            AF.episode_path(tmp_path / "runs", bad)
    monkeypatch.setattr(AF, "MAX_EPISODE_BYTES", 100)
    assert c.get("/api/model/arena/episodes/cli_relay.json").status_code == 413


def test_arena_world_endpoint_rebuilds_the_start(tmp_path):
    _, c = _client(tmp_path, [])
    v = c.get("/api/model/arena/world", params={"task": "key_door", "n": 8, "n_agents": 4, "seed": 5}).json()
    w = World("key_door", 8, 4, 5)
    assert v["state"] == json.loads(json.dumps(w.state())), "World is deterministic: the start a CLI episode was played from"
    assert v["meta"] == {"task": "key_door", "n": 8, "n_agents": 4, "seed": 5, "comm_range": 3, "sight": 2}
    assert [r["name"] for r in v["robots"]] == ["R1", "R2", "R3", "R4"] and v["robots"][1]["tools"][-1]["name"] == "open_door"
    assert v["robots"][0]["system_prompt"] == w.system_prompt(w.agents[0])
    # the CLI transcript's turn-1 positions agree with the rebuilt start (relay robots never move)
    cli = _cli_episode("relay", 10, 4, 0)
    start = c.get("/api/model/arena/world", params={"task": "relay", "n": 10, "n_agents": 4, "seed": 0}).json()["state"]
    t1 = {r["agent"]: r["pos"] for r in cli["transcript"] if r["turn"] == 1}
    assert all(list(t1[a["name"]]) == [a["x"], a["y"]] for a in start["agents"])
    for bad in ({"task": "key_door", "n_agents": 3}, {"task": "relay", "n_agents": 1}, {"task": "maze"}, {"task": "relay", "n": 3}, {"task": "relay", "n_agents": 33}):
        assert c.get("/api/model/arena/world", params=bad).status_code == 422, bad

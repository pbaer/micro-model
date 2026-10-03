"""The arena without a model: tools act on the world, messages travel by range, tasks score, the runner batches."""

import json

import pytest

from slm.arena.world import TASKS, KeyDoor, Relay, Runner, Triangulate, World, dist


def _env(world, agent):
    from slm.tools.functions import functions_env

    return functions_env(world.decls[agent.id])


def test_move_say_look_and_range_delivery():
    w = World("key_door", n=6, n_agents=2, seed=1)
    a, b = w.agents
    a.x, a.y, b.x, b.y = 0, 0, 5, 5
    ea, eb = _env(w, a), _env(w, b)
    assert "moved north" not in ea["move"]("north") and a.pos == (0, 0), "the grid edge blocks"
    assert "moved east" in ea["move"]("east") and a.pos == (1, 0)
    assert "Unknown direction" in ea["move"]("up")
    out = ea["say"]("hello")
    assert "nobody" in out, "b is 9 cells away, comm range 3"
    w.deliver()
    assert b.inbox == [], "out of range: nothing heard"
    b.x, b.y = 3, 0
    ea["say"]("the code is 1234")
    w.deliver()
    assert b.inbox == ["R1 said: 'the code is 1234'"], "delivered at the next observation, with the speaker's name"
    assert "R2 at (3, 0)" in ea["look"]()
    assert "You heard: R1 said" in w.observation(b)


def test_key_door_reads_tells_and_opens():
    w = World("key_door", n=6, n_agents=2, seed=3)
    k, d = w.agents
    assert k.tools[-1] == "read_key" and d.tools[-1] == "open_door"
    code = w.task.codes[0]
    assert code in _env(w, k)["read_key"]()
    door = next(iter(w.task.doors))
    ed = _env(w, d)
    assert "Move there first" in ed["open_door"](code)
    d.x, d.y = door
    assert "Wrong code" in ed["open_door"]("0000")
    assert "Task complete" in ed["open_door"](code)
    assert w.task.score(w)["success"] and w.task.score(w)["opened"] == 1


def test_relay_chain_and_submit():
    w = World("relay", n=10, n_agents=4, seed=0)
    xs = [a.x for a in w.agents]
    assert xs == [0, 2, 4, 6] and all(dist(w.agents[i].pos, w.agents[i + 1].pos) == 2 for i in range(3)), "two cells apart: only neighbours hear"
    assert w.agents[0].tools[-1] == "read_code" and w.agents[-1].tools[-1] == "submit" and w.agents[1].tools == ["move", "say", "look"]
    code = w.task.code
    _env(w, w.agents[0])["say"](code)
    w.deliver()
    assert w.agents[1].inbox and not w.agents[2].inbox and not w.agents[3].inbox
    assert "not the code" in _env(w, w.agents[3])["submit"]("1")
    assert "Correct" in _env(w, w.agents[3])["submit"](code) and w.task.score(w)["success"]


def test_triangulate_sense_and_dig():
    w = World("triangulate", n=6, n_agents=3, seed=2)
    digger = w.agents[0]
    assert "dig" in digger.tools and "dig" not in w.agents[1].tools and all("sense" in a.tools for a in w.agents)
    e = _env(w, digger)
    assert e["sense"]() == dist(digger.pos, w.task.target)
    digger.x, digger.y = w.task.target
    assert e["sense"]() == 0 and "found it" in e["dig"]() and w.task.score(w)["success"]


def test_messages_carry_system_prompt_declarations_and_truncated_history():
    w = World("key_door", n=8, n_agents=4, seed=5, max_history=2)
    a = w.agents[1]
    for i in range(5):
        a.history += [{"role": "user", "content": f"obs {i}"}, {"role": "assistant", "content": f"act {i}"}]
    msgs = w.messages(a)
    assert [m["content"] for m in msgs[:-1]] == ["obs 3", "act 3", "obs 4", "act 4"], "only the last two exchanges are kept"
    assert msgs[-1]["role"] == "user" and "Turn 1." in msgs[-1]["content"] and msgs[-1]["content"].endswith("'#### <answer>'.")
    fresh = World("key_door", n=8, n_agents=4, seed=5)
    first = fresh.messages(fresh.agents[1])
    assert len(first) == 1 and first[0]["content"].startswith("You are robot") and "Door 1" in first[0]["content"], "the briefing opens turn 1"
    names = [d.name for d in w.decls[a.id]]
    assert names == ["move", "say", "look", "open_door"]


def test_runner_with_a_scripted_model_solves_key_door():
    """A scripted 'model' that reads the key, says it, and opens the door: the runner's plumbing end to end."""
    w = World("key_door", n=5, n_agents=2, seed=7)
    k, d = w.agents
    door = next(iter(w.task.doors))

    class TC:
        def __init__(self, calls, answer):
            self.calls, self.ids, self.answer, self.think = calls, None, answer, ""

    heard = {}  # the real model keeps what it heard in its chat history; the script keeps it here

    def scripted(prompts, sessions):
        out = []
        for a, s in zip(w.agents, sessions):
            env = s.functions
            for m in a.inbox:
                if "code" in m:
                    heard[a.id] = m.split()[-1].strip("'\".")
            if a is k:
                res = env["read_key"](); code = res.split()[-1].rstrip(".")
                out.append({"calls": [["read_key()", res], ["say('the code is %s')" % code, env["say"](f"the code is {code}")]], "answer": "Told my partner the code.", "think": ""})
            else:
                code = heard.get(a.id)
                if a.pos != door:
                    dx, dy = door[0] - a.x, door[1] - a.y
                    direction = "east" if dx > 0 else "west" if dx < 0 else "south" if dy > 0 else "north"
                    out.append({"calls": [["move(%r)" % direction, env["move"](direction)]], "answer": "Moving to the door.", "think": ""})
                else:
                    out.append({"calls": [["open_door(%r)" % code, env["open_door"](code)]], "answer": "Opening.", "think": ""})
        return out

    r = Runner(w, model=None, tok=None, generate=scripted)
    res = r.run(turns=12)
    assert res["score"]["success"], res
    assert res["turns"] <= 10 and len(res["transcript"]) == 2 * res["turns"]
    assert any("opened door 1" in e for e in res["events"])
    json.dumps(res)  # serialisable for the portal and the eval


@pytest.mark.parametrize("task", sorted(TASKS))
def test_every_task_asks_a_tool_question_each_turn(task):
    w = World(task, n=8, n_agents=4, seed=3)
    for a in w.agents:
        q = w.task.question(w, a)
        assert "?" in q and any(t in q for t in a.tools), (task, a.name, q)


@pytest.mark.parametrize("task", sorted(TASKS))
def test_every_task_builds_for_up_to_32_agents(task):
    w = World(task, n=12, n_agents=32, seed=11)
    assert len(w.agents) == 32 and all(0 <= a.x < 12 and 0 <= a.y < 12 for a in w.agents)
    assert all(w.decls[a.id] for a in w.agents)
    s = w.state(); json.dumps(s)
    assert s["score"]["success"] is False

"""RL on arena robot turns (2026-10-11; results §29 found that under a fair prompt M10 cannot play the arena).

One task is one robot's turn in an arena episode, under a fair prompt: every tool the robot has is declared and the
question gives the situation (or asks what the right tool answers, or names the tool and leaves its argument), never the
call. The episode up to that turn is played by an oracle that makes the call the scaffold would have spelled out (with
some noise, so the states vary), so the state is one a robot can really be in. The reward is judged from the robot's
first tool call, as the arena counts only one per turn:

    1.0   the right tool, and the right fact in its argument (the code, the name), or a step that really closes (or, for
          a runner, opens) the distance to its goal, or a right tool that takes no argument
    0.25  the right tool with the wrong argument or a step the wrong way
    0.0   no tool call, or the wrong tool

A task carries only a JSON-safe spec of its world (`meta["arena"]`); its declared functions replay that world afresh on
every call (`replay`), so the group's samples never share a world (the trainer registers one set of impls for all of
them) and a task drawn again later starts from the same state. Held out of training: triangulate and team_relay.
"""
from __future__ import annotations

import dataclasses
import random
import re

from slm.rl.rewards import Verdict
from slm.rl.tasks import Task

TRAIN_TASKS = ("key_door", "relay", "tag", "gold_rush")
HELDOUT_TASKS = ("triangulate", "team_relay")  # never drawn in training: the generalization check
# the prompt shapes trained on (slm.arena.world PROMPTS fields): three without worked examples (the skill must not
# need them) and the portal's fair prompt with them (ask_examples), whose turns M10 already gets right now and then
STYLES = [
    ({"tools": "all", "brief": "none", "question": "ask", "ending": ""}, 3),
    ({"tools": "all", "brief": "none", "question": "situation", "ending": " What do you do?"}, 3),
    ({"tools": "turn", "brief": "none", "question": "hint", "ending": ""}, 2),
    ({"tools": "all", "brief": "none", "question": "ask", "ending": "", "demos": [2, 0, 1]}, 3),
]
NOISE = 0.2  # the oracle's share of random actions while playing up to the turn: states a real robot also reaches


def _size(task: str, rng: random.Random) -> tuple[int, int]:
    if task == "key_door":
        return rng.randint(6, 10), rng.choice([2, 4, 6, 8])
    if task == "relay":
        k = rng.randint(3, 7)
        return 2 * k + 2, k
    if task == "team_relay":
        k = rng.choice([4, 6, 8])
        return max(8, k), k
    if task == "triangulate":
        return rng.randint(6, 9), rng.randint(3, 6)
    return rng.randint(6, 10), rng.randint(3, 8)  # tag, gold_rush


def replay(spec: dict):
    """The world of `spec` at its turn, with that turn's messages delivered: (world, robot). Deterministic."""
    from slm.arena.world import Runner, World, expected_calls

    w = World(spec["task"], spec["n"], spec["k"], spec["seed"], max_history=0, prompt="scaffold")
    rng = random.Random(spec["play_seed"])
    r = None

    def oracle(prompts, sessions):
        out = []
        for sess in sessions:
            a = next(x for x in w.agents if r.sessions[x.id] is sess)
            exp = expected_calls(w.turn_question(a)[0])
            calls = []
            if rng.random() < NOISE:  # a random step, or nothing
                if rng.random() < 0.7 and "move" in sess.functions:
                    d = rng.choice(["north", "south", "east", "west"])
                    calls.append([f'move("{d}")', str(sess.functions["move"](d))])
            elif exp:
                call = rng.choice(exp)
                name, args = re.match(r"\w+", call).group(0), re.findall(r'"([^"]*)"', call)
                if name in sess.functions:
                    calls.append([call, str(sess.functions[name](*args))])
            out.append({"calls": calls, "answer": "", "think": ""})
        return out
    r = Runner(w, generate=oracle, retry_temperature=None)
    r.run(spec["turn"])
    w.deliver()
    w.prompt = dict(spec["style"])
    return w, w.agents[spec["agent"]]


def _fresh_impl(spec: dict, name: str):
    def call(*args, **kw):
        from slm.tools.functions import functions_env

        w, a = replay(spec)
        a.memory.pop("_turn", None)
        w.turn_question(a)
        return functions_env(w.turn_tools(a))[name](*args, **kw)
    return call


def make_arena_task(rng: random.Random, task: str) -> Task | None:
    """One robot-turn task of `task`, or None when the drawn turn asks for no call (the generator retries)."""
    from slm.arena.world import expected_calls

    n, k = _size(task, rng)
    style = rng.choices([s for s, _ in STYLES], weights=[w for _, w in STYLES])[0]
    if rng.random() < 0.5:
        style = {**style, "names": "domain"}
    spec = {"task": task, "n": n, "k": k, "seed": rng.randrange(10**6), "play_seed": rng.randrange(10**6), "turn": rng.randint(0, 8),
            "agent": rng.randrange(k), "style": style}
    try:
        w, a = replay(spec)
    except ValueError:  # a size the task cannot be built for
        return None
    msgs = w.messages(a)
    q, names = w.turn_question(a)
    exp = expected_calls(q)
    if not exp:
        return None
    if exp == ["look()"] and rng.random() < 0.8:  # a robot waiting to hear something: most relay turns, kept sparse
        return None
    fact = None
    if task == "team_relay" and len(exp) > 1 and a.memory.get("own_msg"):  # two heard codes: its own team's is the fact
        fact = re.search(r"\d{4}", a.memory["own_msg"]).group(0)
    decls = [dataclasses.replace(d, impl=_fresh_impl(spec, d.name)) for d in w.turn_tools(a)]
    goal = a.memory.get("_goal")
    judge = {"question": q, "tools": [d.name for d in decls], "n": n, "fact": fact,
             "goal": [list(goal[0]), [list(c) for c in goal[1]], goal[2]] if goal else None}
    return Task(id="", prompt=msgs[-1]["content"], answer=exp[0], task=f"arena_{task}",
                meta={"verifier": "arena", "answer_style": "free", "functions": decls, "arena": spec, "judge": judge, "family": f"arena.{task}"})


def arena_verdict(task: Task, calls: list) -> Verdict:
    """The reward fraction of a robot turn, from its first call of a declared tool (see the module docstring)."""
    from slm.arena.world import DIRS, _judge, _step, canonical, dist

    j = task.meta["judge"]
    names = set(j["tools"])
    first = None
    for code, result in calls:
        if any(re.search(rf"\b{re.escape(nm)}\s*\(", str(code)) for nm in names):
            first = [code, result]
            break
    if first is None:
        return Verdict(False, None, "no tool call", fraction=0.0)
    v = _judge(j["question"], [first], j["tools"])
    if not v["tool_ok"]:
        return Verdict(False, first[0], f"wrong tool (expected {v['expected']})", fraction=0.0)
    goal = j.get("goal")
    if re.search(r"\bmove\s*\(", canonical(first[0])):
        # a move needs a direction: move("pen") is not a step the wrong way but no step at all. Scored 0.25 in try 1
        # (2026-10-12), every movement turn became move("scissors") / move("folder") by step 100, a sure 0.25
        m = re.search(r"move\s*\(\s*(?:direction\s*=\s*)?['\"]\s*(north|south|east|west)\s*['\"]\s*\)", str(first[0]).lower())
        if not m:
            return Verdict(False, first[0], "move without a direction", fraction=0.0)
        if not goal:  # a free move (the digger's search): any real step
            return Verdict(True, first[0], "a step", fraction=1.0)
        pos, cells, sense = tuple(goal[0]), [tuple(c) for c in goal[1]], goal[2]
        new = _step(pos, m.group(1), j["n"]) if m.group(1) in DIRS else pos
        before, after = min(dist(pos, c) for c in cells), min(dist(new, c) for c in cells)
        good = new != pos and (after < before if sense == "towards" else after > before)
        return Verdict(good, first[0], f"step {'closes' if sense == 'towards' else 'opens'} the distance" if good else "step the wrong way",
                       fraction=1.0 if good else 0.25)
    if j.get("fact"):  # the team code it must pick of the two it heard
        ok = re.search(rf"\b{j['fact']}\b", str(first[0])) is not None
        return Verdict(ok, first[0], "own team's code" if ok else "not its own team's code", fraction=1.0 if ok else 0.25)
    if v["call_ok"] is not None:
        return Verdict(bool(v["call_ok"]), first[0], "right fact" if v["call_ok"] else "wrong argument", fraction=1.0 if v["call_ok"] else 0.25)
    return Verdict(True, first[0], "right tool", fraction=1.0)


def generator(task: str):
    def gen(rng: random.Random):
        return make_arena_task(rng, task)
    return gen

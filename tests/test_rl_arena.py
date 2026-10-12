"""RL on arena robot turns (slm.rl.arena): tasks, isolation, the verdict, and one rollout end to end."""
import random

import pytest

from slm.data.tokenizer import SlmTokenizer, train_bpe
from slm.rl.arena import HELDOUT_TASKS, TRAIN_TASKS, arena_verdict, make_arena_task, replay
from slm.rl.rollout import rollout_group
from slm.rl.tasks import make_tasks, task_key
from tests.test_rl_families import _script, _tiny_model


def _tasks(task, n=40, seed=0):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        t = make_arena_task(rng, task)
        if t is not None:
            out.append(t)
    return out


def test_tasks_are_fair_prompts_with_every_tool_or_the_named_one():
    for task in TRAIN_TASKS + HELDOUT_TASKS:
        for t in _tasks(task, 25):
            assert t.task == f"arena_{task}" and t.meta["verifier"] == "arena" and t.meta["answer_style"] == "free"
            assert "Think step by step" in t.prompt and "call " not in t.prompt.split("\n")[0].lower().replace("call one of", "x"), t.prompt
            names = [d.name for d in t.meta["functions"]]
            st = t.meta["arena"]["style"]
            assert len(names) >= (2 if st["tools"] == "all" else 1) and all(d.impl is not None for d in t.meta["functions"])


def test_replay_is_deterministic_and_every_call_starts_from_the_same_world():
    t = next(x for x in _tasks("gold_rush", 80) if x.answer == "grab()")
    w1, a1 = replay(t.meta["arena"])
    w2, a2 = replay(t.meta["arena"])
    assert w1.state() == w2.state() and a1.pos == a2.pos
    grab = next(d for d in t.meta["functions"] if d.name in ("grab", "pick_up_coin")).impl
    first, again = grab(), grab()
    assert "picked up" in first and first == again, "a second sample's grab finds the coin still there"


def test_the_verdict_from_the_first_tool_call():
    door = next(x for x in _tasks("key_door", 120) if x.answer.startswith("say(") and x.meta["arena"]["style"].get("names") != "domain")
    code = door.answer.split("is ")[1].rstrip('")')
    assert arena_verdict(door, []).fraction == 0.0
    assert arena_verdict(door, [["x = 3", "3"]]).fraction == 0.0, "plain code is not a tool call"
    assert arena_verdict(door, [["look()", "R2 at (1, 1)"]]).fraction == 0.0, "the wrong tool"
    assert arena_verdict(door, [[f'say("it is {code}")', "You said"]]).fraction == 1.0
    assert arena_verdict(door, [['say("hello")', "You said"]]).fraction == 0.0, "the right tool without the fact earns nothing"
    assert arena_verdict(door, [['say("hello")', "x"], [f'say("{code}")', "x"]]).fraction == 0.0, "only the first tool call counts"

    mover = next(x for x in _tasks("tag", 120) if x.meta["judge"]["goal"])
    pos, cells, sense = mover.meta["judge"]["goal"]
    from slm.arena.world import _step, dist

    n = mover.meta["judge"]["n"]
    scores = {}
    for d in ("north", "south", "east", "west"):
        new = _step(tuple(pos), d, n)
        good = new != tuple(pos) and ((min(dist(new, tuple(c)) for c in cells) < min(dist(tuple(pos), tuple(c)) for c in cells)) if sense == "towards"
                                      else (min(dist(new, tuple(c)) for c in cells) > min(dist(tuple(pos), tuple(c)) for c in cells)))
        scores[d] = arena_verdict(mover, [[f'move("{d}")', "You set off"]]).fraction
        assert scores[d] == (1.0 if good else 0.25), (d, pos, cells, sense)


def test_team_relay_scores_its_own_team_code():
    t = next(x for x in _tasks("team_relay", 400) if x.meta["judge"]["fact"])
    own = t.meta["judge"]["fact"]
    tool = t.meta["judge"]["tools"][-1] if t.answer.startswith(("submit", "hand_in")) else "say"
    arg = f'"{own}"' if tool != "say" else f'"the code is {own}"'
    name = next(n for n in t.meta["judge"]["tools"] if n in ("say", "submit", "hand_in") and (n == "say") == (tool == "say"))
    assert arena_verdict(t, [[f"{name}({arg})", "ok"]]).fraction == 1.0
    assert arena_verdict(t, [[f'{name}("{"0000" if own != "0000" else "1111"}")', "ok"]]).fraction == 0.0


def test_families_split_train_and_heldout_disjointly():
    names = [f"arena_{t}" for t in TRAIN_TASKS]
    tr, he = make_tasks(names, 120, "train", 7), make_tasks(names, 30, "heldout", 7)
    assert not ({task_key(t) for t in tr} & {task_key(t) for t in he})
    assert {t.task for t in tr} == set(names)


@pytest.fixture(scope="module")
def tok():
    text = ('look() move("north") say("the code is 1234") read_key() #### R4 at (3, 3) ' * 30)
    return SlmTokenizer(train_bpe([text * 20, "the quick brown fox " * 40], vocab_size=420))


def test_a_rollout_is_rewarded_by_its_call(tok, monkeypatch):
    task = next(t for t in _tasks("relay", 200) if t.answer == "look()" and "look" in [d.name for d in t.meta["functions"]])
    call = [tok.special("<|python_call|>"), *tok.encode("look()"), tok.special("<|/python_call|>")]
    finish = [tok.special("<|/think|>"), *tok.encode("#### R4"), tok.end_id]
    _script(monkeypatch, [call, finish])
    g = rollout_group(_tiny_model(tok), tok, task, group_size=2, max_new_tokens=300, temperature=0.0, think_required=True,
                      tools=True, reward_scheme="binary", reward_schemes={"arena": "fraction"})
    for r in g:
        assert r.tool_calls == 1 and r.tool_results[0][0] == "look()" and r.fraction == 1.0 and r.reward == 1.0 and r.correct


def test_a_move_without_a_direction_earns_nothing():
    """Try 1 (2026-10-12) learned move("pen") on every movement turn when it still paid 0.25."""
    mover = next(x for x in _tasks("gold_rush", 120) if x.meta["judge"]["goal"])
    for code in ('move("pen")', 'move("scissors")', "move()", 'move("up")', 'x = "north"\nmove(x)'):
        assert arena_verdict(mover, [[code, "Unknown direction"]]).fraction == 0.0, code
    assert arena_verdict(mover, [["move(direction='north')", "ok"]]).fraction in (0.25, 1.0)

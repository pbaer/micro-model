"""The competitive arena tasks (team_relay, tag, gold_rush) without a model: a scripted player that answers each turn's
question by parsing it, perfectly or deliberately wrong, so the mechanics and the scoring are checked before any GPU
time. A perfect player must be able to win; a wrong one must be scored as wrong."""

import json
import re

import pytest

from slm.arena.world import DIRS, Runner, World, dist

POS = r"\((\d+), (\d+)\)"


def _pt(m, i=1):
    return (int(m.group(i)), int(m.group(i + 1)))


def _toward(frm, to):
    dx, dy = to[0] - frm[0], to[1] - frm[1]
    if abs(dx) >= abs(dy) and dx:
        return "east" if dx > 0 else "west"
    return "south" if dy > 0 else "north"


def _away(frm, other, n):
    best, bd = None, -1
    for d, (dx, dy) in DIRS.items():
        nx, ny = frm[0] + dx, frm[1] + dy
        if 0 <= nx < n and 0 <= ny < n and dist((nx, ny), other) > bd:
            best, bd = d, dist((nx, ny), other)
    return best


def player(n, wrong_team=None):
    """A scripted model: answers the question in the last user turn the way a perfect robot would. `wrong_team`: robots
    of that team always pick the rival's code."""

    def generate(prompts, sessions):
        out = []
        for msgs, s in zip(prompts, sessions):
            d = decide(msgs[-1]["content"], n, wrong_team)
            c = [f"{d[0]}({', '.join(repr(a) for a in d[1])})", s.functions[d[0]](*d[1])] if d else None
            out.append({"calls": [c] if c else [], "answer": "ok", "think": ""})
        return out

    return generate


def decide(q, n, wrong_team=None):
    """(tool, args) the perfect robot calls for question `q`, or None."""
    if m := re.search(r"Use (\w+)\(\) to", q):
        return m.group(1), []
    if "own team's code" in q:
        team = re.search(r"You are on team (\w+)", q).group(1)
        heard = re.findall(r"\(team (\w+)\) said '[^']*?(\d{4})", q)
        pick = [code for t, code in heard if (t == team) != (team == wrong_team)] or [code for t, code in heard]
        return ("submit", [pick[0]]) if "submit(" in q else ("say", [f"the code is {pick[0]}"])
    if m := re.search(r"You are at " + POS + r"\. The nearest runner, R\d+, is at " + POS, q):
        return "move", [_toward(_pt(m), _pt(m, 3))]
    if m := re.search(r"You are at " + POS + r"\. The nearest chaser, R\d+, is at " + POS, q):
        return "move", [_away(_pt(m), _pt(m, 3), n)]
    if m := re.search(r"You are at " + POS + r"\. Coins lie at (.*?)\. Which", q):
        me = _pt(m)
        coins = [(int(a), int(b)) for a, b in re.findall(POS, m.group(3))]
        return "move", [_toward(me, min(coins, key=lambda p: (dist(me, p), p)))]
    if m := re.search(r"Call (tag|grab)\((?:\"([^\"]+)\")?\)", q):
        return m.group(1), ([m.group(2)] if m.group(2) else [])
    if m := re.search(r'call (say|tag|move|submit)\("([^"]+)"\)', q, re.I):  # the one call the question names, with its value
        return m.group(1), [m.group(2)]
    return None


def _run(task, n, k, seed, turns, **kw):
    w = World(task, n, k, seed)
    res = Runner(w, generate=player(n, **kw)).run(turns)
    json.dumps(res)
    return w, res


# ---------------------------------------------------------------- team_relay
def test_team_relay_perfect_teams_tie_and_never_pass_the_rival_code():
    w, res = _run("team_relay", 12, 8, seed=1, turns=12)
    sc = res["score"]
    assert sc["success"] and sc["winner"] == "tie" and sc["red_finished"] == sc["blue_finished"] == res["turns"], sc
    assert sc["rival_code_passes"] == 0 and sc["own_code_passes"] > 0 and sc["rival_code_rate"] == 0.0
    assert any("team red submitted" in e for e in res["events"]) and any("team blue submitted" in e for e in res["events"])


def test_team_relay_a_team_that_passes_the_rival_code_loses_and_is_counted():
    w, res = _run("team_relay", 12, 8, seed=2, turns=14, wrong_team="blue")
    sc = res["score"]
    assert sc["winner"] == "red" and sc["verdict"] == "team red wins" and sc["blue_finished"] is None, sc
    assert sc["rival_code_passes"] >= 1 and w.task.wrong["blue"] >= 1 and w.task.wrong["red"] == 0


def test_team_relay_overhears_the_rival_and_asks_without_naming_the_code():
    w = World("team_relay", 10, 6, seed=4)
    red_relay, blue_beside = w.agents[2], w.agents[3]
    assert dist(red_relay.pos, blue_beside.pos) == 1 <= w.task.comm_range, "the rival robot beside it is in range"
    red_relay.memory.update(own_msg="R1 (team red) said 'the code is 1111'", rival_msg="R4 (team blue) said 'the code is 2222'")
    q, tools = w.task.question(w, red_relay)
    # both codes offered as complete calls, the choice left to the model (the team labels are in the heard lines)
    assert tools == ["say"] and 'say("the code is 1111")' in q and 'say("the code is 2222")' in q and "..." not in q, q


# ---------------------------------------------------------------- tag
def test_tag_perfect_chasers_catch_still_runners_and_every_chaser_move_counts_as_good():
    w = World("tag", 7, 4, seed=5)
    runners = [a for a in w.agents if w.task.side(a) == "runner"]

    def gen(prompts, sessions):  # runners stand still: no call at all
        out = player(7)(prompts, sessions)
        return [o if w.task.side(a) == "chaser" else {"calls": [], "answer": "", "think": ""} for a, o in zip(w.agents, out)]

    res = Runner(w, generate=gen).run(30)
    sc = res["score"]
    assert sc["success"] and sc["tagged"] == len(runners) == sc["runners"] and sc["verdict"].startswith("the chasers win"), sc
    assert sc["chaser_good_moves"] == 1.0 and sc["chaser_moves"] > 0
    assert all(a.name in w.task.tagged for a in runners) and all(w.task.agent_info(a)["tagged"] for a in runners)


def test_tag_runners_score_moves_that_open_the_distance():
    w, res = _run("tag", 12, 6, seed=6, turns=6)
    sc = res["score"]
    # a perfect runner opens the distance unless a wall leaves no way to (then every move is blocked or no better)
    assert sc["runner_moves"] > 0 and sc["runner_good_moves"] is not None and sc["runner_good_moves"] >= 0.8, sc


def test_tag_tool_refuses_far_and_unknown_targets():
    w = World("tag", 10, 3, seed=7)
    chaser, runner = w.agents[0], w.agents[1]
    from slm.tools.functions import functions_env

    env = functions_env(w.decls[chaser.id])
    runner.x, runner.y = chaser.x, min(9, chaser.y + 5) if chaser.y < 5 else chaser.y - 5
    assert "too far" in env["tag"](runner.name)
    assert "not a runner" in env["tag"]("R1") and "not a runner" in env["tag"]("nobody")
    runner.x, runner.y = chaser.x, chaser.y
    assert "You tagged" in env["tag"](f"robot {runner.name}") and "already tagged" in env["tag"](runner.name)


# ---------------------------------------------------------------- gold_rush
def test_gold_rush_perfect_players_collect_every_coin_heading_for_a_nearest_one():
    w, res = _run("gold_rush", 8, 4, seed=8, turns=40)
    sc = res["score"]
    assert sc["success"] and sc["collected"] == sc["coins"] and sum(w.task.got.values()) == sc["coins"], sc
    assert sc["nearest_coin_moves"] == 1.0 and sc["any_coin_moves"] == 1.0
    assert sc["verdict"] and sc["leader_coins"] == max(w.task.got.values())


def test_gold_rush_lists_coins_shuffled_and_a_coin_goes_to_the_first_grabber():
    from slm.tools.functions import functions_env

    w = World("gold_rush", 8, 3, seed=9)
    a, b = w.agents[0], w.agents[1]
    coin = next(iter(w.task.coins()))
    a.x, a.y = coin
    b.x, b.y = coin
    assert "picked up" in functions_env(w.decls[a.id])["grab"]() and "no coin" in functions_env(w.decls[b.id])["grab"]()
    orders = set()
    for seed in range(12):  # the nearest coin is not always listed first
        w2 = World("gold_rush", 10, 2, seed=seed)
        r = w2.agents[0]
        q, _ = w2.task.question(w2, r)
        if "Coins lie at" in q:
            listed = [tuple(map(int, p)) for p in re.findall(POS, q.split("Coins lie at", 1)[1])]
            nearest = min(listed, key=lambda p: (dist(r.pos, p), p))
            orders.add(listed.index(nearest))
    assert len(orders) > 1, "the nearest coin's place in the list varies"


# ---------------------------------------------------------------- rules
@pytest.mark.parametrize("task,n,k", [("team_relay", 10, 5), ("team_relay", 10, 2), ("team_relay", 8, 12), ("tag", 8, 1), ("gold_rush", 8, 1)])
def test_robot_count_rules(task, n, k):
    with pytest.raises(ValueError):
        World(task, n, k, 0)


def test_moves_are_offered_as_complete_calls_in_a_shuffled_order():
    seen = set()
    for seed in range(10):
        w = World("tag", 10, 4, seed=seed)
        q, _ = w.task.question(w, w.agents[0])
        if "Call one of" in q:
            seen.add(q.split("Call one of ", 1)[1].split(". What")[0])
    assert len(seen) > 1


def test_scores_report_the_chance_rate_of_each_decision():
    w, res = _run("tag", 10, 6, seed=3, turns=4)
    sc = res["score"]
    assert 0 < sc["chaser_chance"] <= 0.5 and 0 < sc["runner_chance"] <= 0.75, sc  # 1-2 of 4 steps close; 1-3 open
    w, res = _run("gold_rush", 8, 4, seed=4, turns=4)
    assert 0 < res["score"]["nearest_chance"] <= 0.5 and res["score"]["invalid_moves"] == 0


def test_an_invalid_move_is_counted_as_invalid_not_as_a_wrong_direction():
    w = World("gold_rush", 8, 2, seed=5)
    r = w.agents[0]
    w.turn_question(r)
    w.task.after_turn(w, r, {"calls": [['move("marker")', "Unknown direction 'marker'. Use north, south, east or west."]]})
    sc = w.task.score(w)
    assert sc["moves"] == 1 and sc["invalid_moves"] == 1 and sc["nearest_coin_moves"] == 0.0


# ---------------------------------------------------------------- two models against each other
def test_text_action_reads_the_final_line_else_the_last_offered_call():
    from slm.arena.world import text_action

    assert text_action('I could move("north") or move("east").\n#### move("west")', ["move"]) == ("move", ["west"])
    assert text_action('move("north") is wrong, so I pick move("east").', ["move"]) is None, "two options named, none settled on"
    assert text_action('The final answer is: 4. Move "south"', ["move"]) == ("move", ["south"]), "the prose form on a final-answer line"
    assert text_action('1. Move "east"\n2. Move "north"\n3. Move "west"', ["move"]) is None, "restating the list is not a choice"
    assert text_action('I will go: move("west")', ["move"]) == ("move", ["west"]), "exactly one call named"
    assert text_action("say('the code is 1234')", ["say"]) == ("say", ["the code is 1234"])
    assert text_action("Call grab() now. #### grab()", ["grab"]) == ("grab", [])
    assert text_action('tag(R5)', ["tag"]) == ("tag", ["R5"])
    assert text_action("I would go east.", ["move"]) is None, "no offered call written: no action"
    assert text_action('look("x")', ["move"]) is None, "only the offered tools count"


def test_two_players_split_the_sides_and_a_text_player_acts_by_writing_calls():
    from slm.arena.world import Player

    n, k = 12, 8
    w = World("team_relay", n, k, seed=1)
    seen = {"A": [], "B": []}
    perfect = player(n)

    def native(prompts, sessions):
        seen["A"].append(len(prompts))
        return perfect(prompts, sessions)

    def text(prompts, sessions):  # an "external" model: plain-text answers; the runner parses and executes the call
        seen["B"].append(len(prompts))
        outs = []
        for msgs in prompts:
            d = decide(msgs[-1]["content"], n)
            outs.append(f"Let me think.\n#### {d[0]}({', '.join(chr(34) + a + chr(34) for a in d[1])})" if d else "I am not sure.")
        return outs

    runner = Runner(w, players={"A": Player("ours", native=True, generate=native), "B": Player("external", native=False, generate=text)}, batch_size=3)
    res = runner.run(14)
    assert all(w.player_of[a.id] == ("A" if w.task.team(a) == "red" else "B") for a in w.agents)
    recs = res["transcript"]
    assert {r["player"] for r in recs} == {"A", "B"}
    b_calls = [r for r in recs if r["player"] == "B" and r["calls"]]
    assert b_calls and all(isinstance(r["answer"], str) and r["think"] is None for r in b_calls), "text players: answer = the text, no think span"
    assert res["score"]["success"] and res["score"]["winner"] == "tie" and res["score"]["rival_code_passes"] == 0, res["score"]
    assert max(seen["A"] + seen["B"]) <= 3 and len(seen["A"]) > 1, f"batch_size bounds every generation call: {seen}"
    assert all(s["player"] in ("A", "B") for s in w.state()["agents"])


def test_one_player_plays_every_robot_even_in_a_competitive_task():
    w = World("tag", 8, 4, seed=2)
    Runner(w, generate=player(8))
    assert set(w.player_of.values()) == {"A"}, "no opponent: a mirror match, side B robots are the same model"


def test_gold_rush_reports_coins_per_side():
    w, res = _run("gold_rush", 8, 4, seed=8, turns=40)
    sc = res["score"]
    assert sc["coins_A"] + sc["coins_B"] == sc["collected"] == sc["coins"] and sc["winner_side"] in ("A", "B", "tie")


# ---------------------------------------------------------------- tool binding and history (a gold rush on 2026-10-10)
@pytest.mark.parametrize("task", ["key_door", "relay", "triangulate", "team_relay", "tag", "gold_rush"])
def test_a_stray_argument_never_replaces_the_robot_a_tool_acts_for(task):
    """grab("pick") once bound the tool's robot to the string "pick" (AttributeError). Every tool must ignore extra
    positional arguments and keep acting for its own robot."""
    from slm.tools.functions import functions_env

    w = World(task, 12, 4, seed=1)
    for a in w.agents:
        env = functions_env(w.decls[a.id])
        for name, fn in env.items():
            import inspect

            params = [p for p in inspect.signature(fn).parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD]
            args = ["east" if name == "move" else "the code is 1234" if name in ("say", "submit", "open_door") else "R2"][: len(params)]
            out = fn(*args, "pick", 99)  # two stray extra arguments
            assert isinstance(out, (str, int)) and "AttributeError" not in str(out), (task, name, out)


def test_grab_with_a_stray_argument_takes_the_coin():
    from slm.tools.functions import functions_env

    w = World("gold_rush", 8, 2, seed=3)
    a = w.agents[0]
    a.x, a.y = next(iter(w.task.coins()))
    assert "picked up" in functions_env(w.decls[a.id])["grab"]("pick")


def test_failed_turns_kept_by_default_and_retried_with_sampling():
    """Greedy decoding repeats a failure when the prompt does not change. Measured (results §28): keeping the failed turn
    and sampling the robot's next turn decided every team relay with the fewest failed calls; dropping it doubled them.
    A turn with no call counts as failed for the retry. `failed_history="drop"` stays available."""
    w = World("gold_rush", 8, 2, seed=3)
    a = w.agents[0]
    a.x, a.y = next(iter(w.task.coins()))
    script = iter([[["get('pick')", "error: line 1: NameError: get is not defined"]], [], [["grab()", "You picked up the coin"]]])
    temps = []

    r = Runner(w, generate=None)

    def gen(prompts, sessions):
        temps.append(r.current_temperature)
        calls = next(script) if len(prompts) == 1 or temps else []
        return [{"calls": calls, "answer": "x", "think": ""} for _ in prompts]

    r.generate = gen
    assert r.failed_history == "keep" and r.retry_temperature == 0.8
    r.batch_size = None
    w.player_of = {0: "A", 1: "A"}
    r.step()
    assert len(a.history) == 2, "kept by default"
    assert a.memory["_failed"] is True
    r.step()
    assert 0.8 in temps, "the robot that failed is sampled on its next turn"
    assert a.memory["_failed"] is True, "a call-less turn also counts as failed for the retry"

    w2 = World("gold_rush", 8, 2, seed=3)
    r2 = Runner(w2, generate=lambda p, s_: [{"calls": [["x()", "error: boom"]], "answer": "", "think": ""} for _ in p], failed_history="drop")
    r2.step()
    assert w2.agents[0].history == [], "drop: the failed exchange is not kept"
    import pytest as _pt
    with _pt.raises(ValueError):
        Runner(World("tag", 8, 3, 0), generate=lambda p, s_: [], failed_history="sometimes")

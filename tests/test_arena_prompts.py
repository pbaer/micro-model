"""The arena's prompt styles (slm.arena.world PROMPTS): what a robot's prompt declares, says and spells out."""
import re

import pytest

from slm.arena.world import DEMOS, PROMPTS, Runner, World, _judge, expected_calls, native_messages, prompt_style

TASKS = [("key_door", 8, 6), ("relay", 14, 6), ("triangulate", 8, 6), ("team_relay", 8, 6), ("tag", 8, 6), ("gold_rush", 8, 6)]
CALL = re.compile(r"\b[a-z_]+\(")


def _questions(task, n, k, prompt, turns=4):
    """Every robot's user turn over a few oracle-played turns (the scaffold's spelled-out call is made)."""
    w = World(task, n, k, seed=3, prompt=prompt)
    seen = []
    r = None

    def oracle(prompts, sessions):
        out = []
        for sess in sessions:
            a = next(x for x in w.agents if r.sessions[x.id] is sess)
            seen.append((w.messages(a), w.turn_tools(a), [d.name for d in w.decls[a.id]]))
            exp = expected_calls(w.turn_question(a)[0])
            calls = []
            if exp:
                name, args = re.match(r"\w+", exp[0]).group(0), re.findall(r'"([^"]*)"', exp[0])
                if name in sess.functions:
                    calls.append([exp[0], str(sess.functions[name](*args))])
            out.append({"calls": calls, "answer": "ok", "think": ""})
        return out
    r = Runner(w, generate=oracle, retry_temperature=None)
    r.run(turns)
    return w, seen


@pytest.mark.parametrize("task,n,k", TASKS)
def test_a_situation_question_names_no_call_and_every_tool_is_declared(task, n, k):
    w, seen = _questions(task, n, k, "situation_sys")
    assert seen
    for msgs, tools, has in seen:
        q = msgs[-1]["content"].split("\nThink step by step")[0]
        assert not CALL.search(q), f"{task}: a situation question spells out no call: {q!r}"
        assert "What do you do?" in q
        assert [d.name for d in tools] == has and {"move", "say", "look"} <= set(has), "every tool the robot has"
        assert msgs[0]["role"] == "system" and msgs[0]["content"].startswith("You are robot R")


def test_every_tool_the_robot_has_is_declared_and_the_scaffold_declares_one():
    w = World("gold_rush", 8, 4, seed=0, prompt="situation_sys")
    a = w.agents[0]
    assert [d.name for d in w.turn_tools(a)] == ["move", "say", "look", "grab"]
    w2 = World("gold_rush", 8, 4, seed=0, prompt="scaffold")
    assert len(w2.turn_tools(w2.agents[0])) == 1 and w2.messages(w2.agents[0])[0]["role"] == "user"


def test_the_scaffold_question_is_unchanged_and_the_briefing_can_lead_the_question():
    w = World("relay", 14, 6, seed=0, prompt="scaffold")
    assert w.observation(w.agents[0]).startswith("What is the secret code? Use read_code() to find out.")
    u = World("relay", 14, 6, seed=0, prompt="situation_user")
    q = u.observation(u.agents[0])
    assert q.startswith("You are robot R1") and "Find out the code. What do you do?" in q and u.messages(u.agents[0])[0]["role"] == "user"


def test_worked_examples_come_first_and_become_ids_for_our_checkpoint():
    from slm.data.chat import format_chat
    from slm.data.tokenizer import SlmTokenizer

    w = World("gold_rush", 8, 2, seed=0, prompt={"tools": "all", "brief": "system", "question": "situation", "ending": " What do you do?", "demo": 2})
    msgs = w.messages(w.agents[0])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[1]["content"].startswith(DEMOS[0]["situation"]) and msgs[2]["demo"] is DEMOS[0]
    tok = SlmTokenizer.load(r"C:\slm-data\tokenizer\v1")
    nat = native_messages(tok, msgs)
    assert "ids" in nat[2] and "demo" not in nat[2]
    text = tok.decode(format_chat(tok, nat, add_generation_prompt=True, think_required=True, functions=w.turn_tools(w.agents[0])).ids)
    assert '<|python_call|>move("north")<|/python_call|>' in text and "<|python_result|>You moved north to (2, 4).<|/python_result|>" in text


def test_judge_scores_the_tool_and_the_fact_in_its_argument():
    q = 'The key code is 7311. Tell robot R2: call say("the code is 7311"). What does say() return?'
    tools = ["move", "say", "look", "read_key"]
    assert _judge(q, [['say("I know it: 7311")', "You said ..."]], tools) == {"expected": ['say("the code is 7311")'], "tool_ok": True, "tool_chance": 0.25, "call_ok": True}
    assert _judge(q, [['say("hello")', "You said ..."]], tools)["call_ok"] is False
    assert _judge(q, [['move("east")', "You moved ..."]], tools)["tool_ok"] is False
    assert _judge(q, [], tools)["tool_ok"] is False
    mv = _judge('Which step? Call one of move("west"), move("east"). What does move() return?', [['move("north")', "x"]], tools)
    assert mv["tool_ok"] is True and mv["call_ok"] is None, "a move's direction is scored by the task, not here"


def test_prompt_style_checks_its_fields():
    assert prompt_style("scaffold")["question"] == "call" and set(PROMPTS) >= {"scaffold", "situation_sys"}
    with pytest.raises(ValueError):
        prompt_style("nope")
    with pytest.raises(ValueError):
        prompt_style({"tools": "some"})


def test_domain_names_alias_the_tools_and_map_back_for_the_task_and_the_judge():
    from slm.arena.world import canonical

    w = World("relay", 14, 6, seed=0, prompt="ask_examples")
    src = w.agents[0]
    tools = w.turn_tools(src)
    assert "secret_code" in [d.name for d in tools] and "read_code" not in [d.name for d in tools]
    assert any(d.signature.startswith("def secret_code(") for d in tools)
    assert canonical("x = secret_code()\nsay(x)") == "x = read_code()\nsay(x)"

    def gen(prompts, sessions):  # the source calls the alias; the others do nothing
        return [{"calls": [["secret_code()", s_.functions["secret_code"]()]], "answer": "", "think": ""} if "secret_code" in s_.functions
                else {"calls": [], "answer": "", "think": ""} for s_ in sessions]
    rec = Runner(w, generate=gen).step()["records"][0]
    assert rec["tool_ok"] is True and src.memory.get("code") == w.task.code, "the task read the alias call as read_code()"

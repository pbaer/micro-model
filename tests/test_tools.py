"""Calculator tool, tool protocol (markup <-> special tokens, loss masks), the batched tool loop, and the RL
mask that excludes inserted results."""

import pytest
import torch

from slm.data.chat import format_chat, parse_assistant
from slm.data.tokenizer import SlmTokenizer, train_bpe
from slm.tools import PySession, ToolError, calc, render_tools, run_tool, split_markup
from slm.tools.protocol import hoist_calls
from slm.tools.loop import ToolCompletion, sample_with_tools


@pytest.fixture(scope="module")
def tok():
    return SlmTokenizer(train_bpe(["print(12*35) = 420 The secret number is 123456. Question: What? Answer: 7 " * 30, "he has 10 - 2 = 8 trees a = 5 " * 30], vocab_size=320))


def test_calculator():
    assert calc("198+209") == "407" and calc("5*.50") == "2.5" and calc("(12+7)*3-5") == "52" and calc("48/4") == "12"
    assert calc("10/3") == "3.333333" and calc("2^10") == "1024" and calc("$1,250 - 250") == "1000" and calc("-3*-4") == "12" and calc("7 % 3") == "1"
    for bad in ("", "1/0", "2**200", "import os", "x+1", "(1+2", "3.5.1"):
        with pytest.raises(ToolError):
            calc(bad)
    assert run_tool("print(6*7)") == ("42", True) and run_tool("6*7") == ("42", True)
    assert run_tool("shell: ls")[1] is False and run_tool("1/0")[0].startswith("error: line 1: division by zero")


def test_markup_roundtrip_and_masks(tok):
    spans = split_markup("He has 10 - 2 = <<10-2=8>>8 trees. <<1/0=?>> ok <<<a = 4\nprint(a * 2)>>> and <<a+1=5>>")
    assert [s.kind for s in spans] == ["text", "tool", "text", "text", "text", "tool", "text", "tool"] and spans[1].result == "8" and spans[3].text == "1/0 = ?"
    assert spans[2].text == " trees. "  # the dataset's echoed "8" after the markup is dropped: the result span carries it
    assert [s.text for s in split_markup("costs $<<13-11=2>>2. Then <<2*2=4>>4 total") if s.kind == "text"] == ["costs $", ". Then ", " total"]
    assert spans[1].code == "10-2" and spans[5].code == "a = 4\nprint(a * 2)" and spans[5].result == "8" and spans[7].result == "5"  # state carried to the next call
    msgs = [{"role": "user", "content": "How many?"}, {"role": "assistant", "think": "10 - 2 = <<10-2=8>>8 trees", "content": "#### 8"}]
    enc = format_chat(tok, msgs, think_required=True, tools=True)
    plain = format_chat(tok, msgs, think_required=True, tools=False)
    call_open, call_close, res_open, res_close = (tok.special(s) for s in ("<|python_call|>", "<|/python_call|>", "<|python_result|>", "<|/python_result|>"))
    assert call_open in enc.ids and res_close in enc.ids and call_open not in plain.ids
    i_call, i_res = enc.ids.index(call_open), enc.ids.index(res_open)
    j_call, j_res = enc.ids.index(call_close), enc.ids.index(res_close)
    assert all(enc.loss_mask[k] == 1 for k in range(i_call, j_call + 1)), "the call is a target"
    assert all(enc.loss_mask[k] == 0 for k in range(i_res, j_res + 1)), "the result is environment-written"
    assert enc.loss_mask[j_res + 1] == 1  # prose after the result is a target again
    labels = {s[2] for s in enc.segments}
    assert {"python_call", "python_result"} <= labels
    # rendering the assistant turn brings the markup back (with the tool's own result format)
    a_start = enc.ids.index(tok.special("<|assistant|>")) + 1
    parsed = parse_assistant(tok, enc.ids[a_start:])
    assert parsed["think"] == "10 - 2 = <<10-2=8>> trees" and parsed["answer"] == "#### 8" and not parsed["malformed"]
    # unclosed spans render as far as they go
    assert render_tools(tok, [call_open, *tok.encode("1+1")]) == "<<<1+1"
    # markup in the answer (outside think) is NOT a tool call: it stays literal text
    enc3 = format_chat(tok, [{"role": "user", "content": "q"}, {"role": "assistant", "think": "<<2*3=6>>", "content": "#### <<2*3=6>>"}], think_required=True, tools=True)
    assert enc3.ids.count(call_open) == 1 and enc3.ids.index(call_open) < enc3.ids.index(tok.special("<|/think|>"))
    # a call in the answer part of a generated turn is malformed
    gen = [*tok.encode("x"), tok.special("<|/think|>"), call_open, *tok.encode("1+1"), call_close, tok.end_id]
    assert parse_assistant(tok, gen)["malformed"]
    # a multi-turn conversation shares one session: the second turn can use the first turn's variable
    convo = [{"role": "user", "content": "a?"}, {"role": "assistant", "think": "<<<a = 5\nprint(a)>>>", "content": "#### 5"},
             {"role": "user", "content": "double it"}, {"role": "assistant", "think": "<<a*2=10>>", "content": "#### 10"}]
    enc2 = format_chat(tok, convo, think_required=True, tools=True)
    a2 = [k for k, x in enumerate(enc2.ids) if x == tok.special("<|assistant|>")][1] + 1
    assert parse_assistant(tok, enc2.ids[a2:])["think"] == "<<a*2=10>>"


def test_tool_loop_with_scripted_sampler(tok, monkeypatch):
    """Drive the loop with a fake sampler so the bookkeeping is tested without a trained model."""
    t = {s: tok.special(s) for s in ("<|python_call|>", "<|/python_call|>", "<|python_result|>", "<|/python_result|>")}
    call = [t["<|python_call|>"], *tok.encode("12*35"), t["<|/python_call|>"]]
    bad_call = [t["<|python_call|>"], *tok.encode("x"), t["<|/python_call|>"]]
    script = {0: [call, tok.encode(" so 420") + [tok.end_id]], 1: [tok.encode("no tools") + [tok.end_id]], 2: [bad_call, call, tok.encode(" done") + [tok.end_id]]}
    state = {"round": {}}

    def fake_sample(model, tk, prompts, n, temperature, top_p=1.0, top_k=0, generator=None, stop=None):
        out = []
        for p in prompts:
            row = p[0]  # prompt[0] encodes the row id in this test
            k = state["round"].get(row, 0)
            state["round"][row] = k + 1
            out.append(script[row][k][:n])
        return out

    monkeypatch.setattr("slm.rl.rollout.sample_completions", fake_sample)

    class M:  # minimal stand-in for the model
        class cfg:
            max_seq_len = 512

    prompts = [[0, 5, 6], [1, 5, 6], [2, 5, 6]]
    outs = sample_with_tools(M(), tok, prompts, max_new_tokens=64, temperature=0.0, max_calls=8)
    assert [o.n_calls for o in outs] == [1, 0, 2] and [o.n_errors for o in outs] == [0, 0, 1] and all(o.termination == "stop" for o in outs)
    r0 = outs[0]
    assert len(r0.ids) == len(r0.gen_mask) and sum(1 - m for m in r0.gen_mask) == len([t["<|python_result|>"], *tok.encode("420"), t["<|/python_result|>"]])
    assert render_tools(tok, r0.ids) == "<<12*35=420>> so 420" and r0.calls == [("12*35", "420")]
    assert "error:" in render_tools(tok, outs[2].ids) and render_tools(tok, outs[2].ids).endswith("<<12*35=420>> done")
    # a call after </think> is refused without running it
    state["round"].clear()
    script[0] = [[tok.special("<|/think|>"), *call]]
    outs = sample_with_tools(M(), tok, [[0, 5, 6]], max_new_tokens=64, temperature=0.0, max_calls=8)
    assert outs[0].termination == "tool_outside_think" and outs[0].n_calls == 0 and outs[0].calls == []
    # max_calls stops a row that keeps calling
    state["round"].clear()
    script[0] = [call] * 5
    outs = sample_with_tools(M(), tok, [[0, 5, 6]], max_new_tokens=64, temperature=0.0, max_calls=2)
    assert outs[0].termination == "max_calls" and outs[0].n_calls == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rl_mask_excludes_tool_results(tmp_path, tok):
    """The GRPO minibatch mask must be 0 on inserted tool-result tokens."""
    from slm.rl.rollout import Rollout

    r = Rollout(prompt_id="p", task="t", prompt="q", gold="1", prompt_ids=[1, 2, 3], completion_ids=[4, 5, 6, 7, 8], text="", think=None, answer="", parsed=None,
                correct=True, reward=1.0, verifier="", malformed=False, termination="stop", n_tokens=5, old_logprobs=[0.0] * 5, ref_logprobs=[0.0] * 5,
                advantage=1.0, gen_mask=[1, 1, 0, 0, 1])
    P, L = len(r.prompt_ids), len(r.prompt_ids) + r.n_tokens
    mask = torch.zeros((1, L - 1), device="cuda")
    mask[0, P - 1 : P - 1 + r.n_tokens] = torch.tensor(r.gen_mask, dtype=torch.float32, device="cuda")
    assert mask[0].tolist() == [0, 0, 1, 1, 0, 0, 1]


def test_session_persists_and_reward_needs_real_tool_work():
    from slm.rl.rewards import Verdict, answer_from_tool, reward_from_verdict

    s = PySession()
    assert s.run("total = 12 * 35") == "" and s.run("print(total - 7)") == "413" and s.run("total") == "420"
    s.reset()
    with pytest.raises(ToolError):
        s.run("print(total)")
    assert answer_from_tool("413", [("print(12*35-7)", "413")]) and answer_from_tool("413", [("x = 420\nprint(x - 7)", "413")])
    assert not answer_from_tool("413", [("print(413)", "413")]) and not answer_from_tool("413", [("x", "error: NameError")]) and not answer_from_tool("413", [("print(1+1)", "2")])
    ok, bad = Verdict(True, "413", "numeric compare"), Verdict(False, "5", "numeric compare")
    assert reward_from_verdict(ok, False, "tool", True) == 1.0 and reward_from_verdict(ok, False, "tool", False) == 0.5 and reward_from_verdict(bad, False, "tool", True) == 0.0
    assert reward_from_verdict(ok, True, "tool", True) == 0.0  # malformed (e.g. a call outside think) earns nothing


def test_synth_tool_traces_verify():
    """Every synthetic trace's tool results must equal what the calculator computes (correct by construction)."""
    import random

    from slm.rl.synth import trace_for_tools
    from slm.rl.tasks import make_tasks

    for name in ("arith1", "arith2", "arith2mul", "arith_multi", "algebra", "word"):
        for t in make_tasks([name], 20, "train", 3):
            tr = trace_for_tools(t)
            tool_spans = [s for s in split_markup(tr) if s.kind == "tool"]
            assert tool_spans, (name, tr)
            assert all(s.code for s in tool_spans)
            assert tool_spans[-1].result == t.answer, (name, tr, tool_spans[-1].result)  # the sandbox reproduces the gold answer
            assert not tr.rstrip(".").endswith(">>" + t.answer), "traces must not echo the tool result"
    random.Random(0)


def test_multiturn_tool_conversations_are_consistent(tok):
    """Every follow-up reuses `total` from the session; the sandbox result must equal the gold answer in each turn."""
    import random

    from slm.rl.synth_multiturn import conversation
    from slm.tools import PySession, split_markup

    rng = random.Random(5)
    for _ in range(200):
        msgs = conversation(rng)
        sess = PySession()
        assert msgs[0]["role"] == "user" and len(msgs) >= 4 and len(msgs) % 2 == 0
        for m in msgs:
            if m["role"] != "assistant":
                continue
            spans = [s for s in split_markup(m["think"], sess) if s.kind == "tool"]
            gold = m["content"].split("#### ")[1] if m["content"].startswith("#### ") else m["content"].rstrip(".").split()[-1]
            assert len(spans) == 1 and spans[0].result == gold, (msgs, spans)


def test_declared_functions_roundtrip(tok):
    """Declared functions: masked def blocks after <|bos|>, calls that resolve at conversion time,
    parse_defs as the inverse, and the plain NameError hint for a name that was never declared."""
    from slm.tools.functions import FunctionDecl, as_decls, functions_env, parse_defs, render_defs

    def unit_price(item):
        return {"pen": 2.5, "book": 12.0}[item]

    decl = FunctionDecl("unit_price", "def unit_price(item: str) -> float:", "Catalogue price of an item in dollars. Use it instead of guessing.", unit_price)
    assert decl.signature == "def unit_price(item: str) -> float"  # the body-less def line is normalized (no trailing colon)
    with pytest.raises(ValueError):
        FunctionDecl("unit_price", "unit_price(item)", "not a def line")
    msgs = [{"role": "user", "content": "two pens?"}, {"role": "assistant", "think": "a pen costs <<unit_price('pen')=2.5>>2.5 so <<2*2.5=5>>", "content": "#### 5"}]
    enc = format_chat(tok, msgs, think_required=True, tools=True, functions=[decl])
    d_open, d_close = tok.special("<|python_def|>"), tok.special("<|/python_def|>")
    j = enc.ids.index(d_close)
    assert enc.ids[0] == tok.bos_id and enc.ids[1] == d_open and tok.special("<|python_comment|>") in enc.ids
    assert all(m == 0 for m in enc.loss_mask[1 : j + 1]), "the whole declaration block is environment-written"
    assert [s for s in enc.segments if s[2] == "python_def"] == [(1, j + 1, "python_def")]
    k = enc.ids.index(tok.special("<|python_call|>"))
    assert k > j and enc.loss_mask[k] == 1  # the call is still a target
    a = enc.ids.index(tok.special("<|assistant|>")) + 1
    parsed = parse_assistant(tok, enc.ids[a:])
    assert parsed["think"] == "a pen costs <<unit_price('pen')=2.5>> so <<2*2.5=5>>" and not parsed["malformed"]
    # the declarations can travel with the conversation instead, as a leading message (dicts, no impl)
    enc2 = format_chat(tok, [{"role": "functions", "decls": [decl.as_dict()]}, *msgs], think_required=True)
    assert enc2.ids[: j + 1] == enc.ids[: j + 1] and as_decls([decl.as_dict()])[0].impl is None
    # parse_defs is the inverse (impl is not recoverable); it ignores everything outside the blocks
    back = parse_defs(tok, enc.ids)
    assert [(f.name, f.signature, f.comment) for f in back] == [(decl.name, decl.signature, decl.comment)]
    assert render_defs(tok, back) == render_defs(tok, [decl])
    assert parse_defs(tok, [d_open, *tok.encode("def f()")]) == []  # unterminated block
    # calls go through the normal path: same result rendering, same hints; an undeclared name is a NameError
    sess = PySession(functions=functions_env([decl]))
    assert run_tool("unit_price('book')", sess) == ("12", True)
    assert run_tool("total = unit_price('pen') * 4\nprint(total)", sess) == ("10", True)
    out, ok = run_tool("unit_cost('pen')", sess)
    assert not ok and "NameError" in out and "not defined" in out
    out, ok = run_tool("unit_price('hat')", sess)  # the function itself raised
    assert not ok and "KeyError" in out
    # declared but not implemented (the portal sends dicts): the error says so instead of looking like a typo
    out, ok = run_tool("unit_price('pen')", PySession(functions=functions_env([decl.as_dict()])))
    assert not ok and "declared but has no implementation" in out


def test_tool_loop_with_declared_function(tok, monkeypatch):
    """sample_with_tools registers the declared functions in every row's session."""
    from slm.tools.functions import FunctionDecl

    t = {s: tok.special(s) for s in ("<|python_call|>", "<|/python_call|>")}
    call = [t["<|python_call|>"], *tok.encode("unit_price('pen')*2"), t["<|/python_call|>"]]
    script = {0: [call, tok.encode(" so 5") + [tok.end_id]]}
    state = {"round": {}}

    def fake_sample(model, tk, prompts, n, temperature, top_p=1.0, top_k=0, generator=None, stop=None):
        out = []
        for p in prompts:
            row = p[0]
            k = state["round"].get(row, 0)
            state["round"][row] = k + 1
            out.append(script[row][k][:n])
        return out

    monkeypatch.setattr("slm.rl.rollout.sample_completions", fake_sample)

    class M:
        class cfg:
            max_seq_len = 512

    decl = FunctionDecl("unit_price", "def unit_price(item: str) -> float", "Catalogue price in dollars.", lambda item: 2.5)
    outs = sample_with_tools(M(), tok, [[0, 5, 6]], max_new_tokens=64, temperature=0.0, functions=[decl])
    assert outs[0].calls == [("unit_price('pen')*2", "5")] and outs[0].n_errors == 0
    state["round"].clear()
    outs = sample_with_tools(M(), tok, [[0, 5, 6]], max_new_tokens=64, temperature=0.0)  # without the declaration the name is unknown
    assert outs[0].n_errors == 1 and "NameError" in outs[0].calls[0][1]


def test_hoist_calls_puts_the_computation_before_a_stated_result():
    """In GSM8K traces the answer often precedes its own calculation; as tool data that teaches the model the call is
    decorative. The hoist moves the computation first, so the number only appears after the tool result."""
    assert hoist_calls("He eats 32 from the largest pizzas because 2 x 16 = <<2*16=32>>32") == "2 x 16 = <<2*16=32>>. He eats 32 from the largest pizzas."
    assert hoist_calls("He saved up $110 total because 95 + 15 = <<95+15=110>>110") == "95 + 15 = <<95+15=110>>. He saved up $110 total."
    # a line whose prose does not state the result is untouched (the expression before the call is fine: it is the plan)
    same = "She is making 12 x 6 = <<12*6=72>> ounces of water."
    assert hoist_calls(same) == same
    assert hoist_calls("Twice 8 is <<8*2=16>>.") == "Twice 8 is <<8*2=16>>."
    # multi-line traces: only the offending lines move
    tr = "A: 4 apples because 2 + 2 = <<2+2=4>>4\nB gets <<4*3=12>> pears."
    assert hoist_calls(tr) == "2 + 2 = <<2+2=4>>. A: 4 apples.\nB gets <<4*3=12>> pears."


def test_stored_tool_results_replay_from_the_sandbox():
    """Every stored <|python_result|> in the real tool sets must be exactly what the sandbox produces when the calls
    are replayed in order (sampled here; scripts/verify_tool_results.py does the whole sets). Guards against the
    hint text or the result formatting drifting away from the data the model was trained on."""
    from pathlib import Path

    root = Path(r"C:\slm-data\sft\v1")
    if not root.exists():
        pytest.skip("no data root")
    import scripts.verify_tool_results as V

    tok = SlmTokenizer.load(r"C:\slm-data\tokenizer\v1")
    impls = V.service_impls()
    for name in ("gsm8k-tools", "synthetic-python-tools", "synthetic-multiturn-tools"):
        if not (root / name).exists():
            continue
        s = V.verify(root / name, tok, impls, max_conversations=150)
        assert s["calls"] > 0 and s["mismatches"] == 0, (name, s["examples"])

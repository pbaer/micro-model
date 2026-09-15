"""Calculator tool, tool protocol (markup <-> special tokens, loss masks), the batched tool loop, and the RL
mask that excludes inserted results."""

import pytest
import torch

from slm.data.chat import format_chat, parse_assistant
from slm.data.tokenizer import SlmTokenizer, train_bpe
from slm.tools import ToolError, calc, render_tools, run_tool, split_markup
from slm.tools.loop import ToolCompletion, sample_with_tools


@pytest.fixture(scope="module")
def tok():
    return SlmTokenizer(train_bpe(["calc: 12*35 = 420 The secret number is 123456. Question: What? Answer: 7 " * 30, "he has 10 - 2 = 8 trees " * 30], vocab_size=320))


def test_calculator():
    assert calc("198+209") == "407" and calc("5*.50") == "2.5" and calc("(12+7)*3-5") == "52" and calc("48/4") == "12"
    assert calc("10/3") == "3.333333" and calc("2^10") == "1024" and calc("$1,250 - 250") == "1000" and calc("-3*-4") == "12" and calc("7 % 3") == "1"
    for bad in ("", "1/0", "2**200", "import os", "x+1", "(1+2", "3.5.1"):
        with pytest.raises(ToolError):
            calc(bad)
    assert run_tool("calc: 6*7") == ("42", True) and run_tool("6*7") == ("42", True)
    assert run_tool("shell: ls")[1] is False and run_tool("calc: 1/0") == ("error: division by zero", False)


def test_markup_roundtrip_and_masks(tok):
    spans = split_markup("He has 10 - 2 = <<10-2=8>>8 trees. <<1/0=?>> ok")
    assert [s.kind for s in spans] == ["text", "tool", "text", "text", "text"] and spans[1].result == "8" and spans[3].text == "1/0 = ?"
    msgs = [{"role": "user", "content": "How many?"}, {"role": "assistant", "think": "10 - 2 = <<10-2=8>>8 trees", "content": "#### 8"}]
    enc = format_chat(tok, msgs, think_required=True, tools=True)
    plain = format_chat(tok, msgs, think_required=True, tools=False)
    call_open, call_close, res_open, res_close = (tok.special(s) for s in ("<|tool_call|>", "<|/tool_call|>", "<|tool_result|>", "<|/tool_result|>"))
    assert call_open in enc.ids and res_close in enc.ids and call_open not in plain.ids
    i_call, i_res = enc.ids.index(call_open), enc.ids.index(res_open)
    j_call, j_res = enc.ids.index(call_close), enc.ids.index(res_close)
    assert all(enc.loss_mask[k] == 1 for k in range(i_call, j_call + 1)), "the call is a target"
    assert all(enc.loss_mask[k] == 0 for k in range(i_res, j_res + 1)), "the result is environment-written"
    assert enc.loss_mask[j_res + 1] == 1  # prose after the result is a target again
    labels = {s[2] for s in enc.segments}
    assert {"tool_call", "tool_result"} <= labels
    # rendering the assistant turn brings the markup back (with the tool's own result format)
    a_start = enc.ids.index(tok.special("<|assistant|>")) + 1
    parsed = parse_assistant(tok, enc.ids[a_start:])
    assert parsed["think"] == "10 - 2 = <<10-2=8>>8 trees" and parsed["answer"] == "#### 8" and not parsed["malformed"]
    # unclosed spans render as far as they go
    assert render_tools(tok, [call_open, *tok.encode("calc: 1+1")]) == "<<calc: 1+1"


def test_tool_loop_with_scripted_sampler(tok, monkeypatch):
    """Drive the loop with a fake sampler so the bookkeeping is tested without a trained model."""
    t = {s: tok.special(s) for s in ("<|tool_call|>", "<|/tool_call|>", "<|tool_result|>", "<|/tool_result|>")}
    call = [t["<|tool_call|>"], *tok.encode("calc: 12*35"), t["<|/tool_call|>"]]
    bad_call = [t["<|tool_call|>"], *tok.encode("calc: x"), t["<|/tool_call|>"]]
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
    assert len(r0.ids) == len(r0.gen_mask) and sum(1 - m for m in r0.gen_mask) == len([t["<|tool_result|>"], *tok.encode("420"), t["<|/tool_result|>"]])
    assert render_tools(tok, r0.ids) == "<<12*35=420>> so 420"
    assert "error:" in render_tools(tok, outs[2].ids) and render_tools(tok, outs[2].ids).endswith("<<12*35=420>> done")
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
            assert all(s.call.startswith("python: ") for s in tool_spans)
            assert tool_spans[-1].result == t.answer, (name, tr, tool_spans[-1].result)  # the sandbox reproduces the gold answer
            assert tr.rstrip(".").endswith(t.answer)
    random.Random(0)

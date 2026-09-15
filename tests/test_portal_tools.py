"""Inference harness with the Python tool: a scripted sampler drives a tool call through the in-process
Harness (no worker), checking the tool event, inserted-token flags, REPL state across generate calls with
the same session id, the well-formed assistant summary, and verbatim assistant ids in format_chat."""

import json

import pytest

from slm.config import ModelConfig, load_config, to_dict
from slm.data.chat import format_chat
from slm.data.tokenizer import SlmTokenizer, train_bpe
from slm.model import Transformer
from slm.portal.services import harness as H
from slm.utils.checkpoint import save_snapshot


def _world(tmp_path):
    tok_dir = tmp_path / "tokenizer" / "t1"
    tok = SlmTokenizer(train_bpe(["x = 6 x*2 2*3 #### 6 ok " * 60, "the cat sat on the mat " * 40], vocab_size=300))
    tok.save(tok_dir)
    cfg = load_config(ModelConfig, "configs/model/tiny.yaml")
    cfg.vocab_size = tok.vocab_size
    ck = tmp_path / "runs" / "r" / "checkpoints" / "snap_1K.pt"
    save_snapshot(ck, Transformer(cfg), to_dict(cfg), {"tokens": 1000, "val_loss": 3.0, "tokenizer_sha256": tok.sha256})
    return tok, ck


def _scripted(monkeypatch, tok, script: list[int]):
    """Replace sampling with a fixed token script (then <|end|> forever)."""
    import torch

    it = iter(script)

    def fake(logits, temperature=1.0, top_p=1.0, top_k=0, generator=None):
        nid = next(it, tok.end_id)
        return torch.tensor([nid], device=logits.device)

    monkeypatch.setattr(H, "sample_next", fake)


def test_harness_tool_call_session_and_wellformed(tmp_path, monkeypatch):
    tok, ck = _world(tmp_path)
    h = H.Harness(tmp_path / "tokenizer")
    h.load("A", str(ck), device="cpu", dtype="fp32")
    sp = {s: tok.special(s) for s in ("<|python_call|>", "<|/python_call|>", "<|python_result|>", "<|/python_result|>", "<|/think|>")}
    # turn 1: define x with a call, then use it in a second call; answer well-formed
    script = [sp["<|python_call|>"], *tok.encode("x = 6"), sp["<|/python_call|>"], *tok.encode(" ok "), sp["<|python_call|>"], *tok.encode("x*2"), sp["<|/python_call|>"],
              sp["<|/think|>"], *tok.encode("#### 12"), tok.end_id]
    _scripted(monkeypatch, tok, script)
    msgs = [{"role": "user", "content": "q"}]
    evs = list(h.generate("A", mode="chat", messages=msgs, temperature=0.0, max_new_tokens=64, think_required=True, tools=True, session_id="conv1"))
    tools = [e for e in evs if e["event"] == "tool"]
    assert [t["code"] for t in tools] == ["x = 6", "x*2"] and tools[1]["result"] == "12" and tools[1]["ok"] and tools[0]["result"].startswith("(no output")
    toks = [e for e in evs if e["event"] == "token"]
    ins = [t for t in toks if t["inserted"]]
    assert ins and all(t["logprob"] is None for t in ins) and all(not t["inserted"] or True for t in toks)
    assert ins[0]["id"] == sp["<|python_result|>"] and ins[-1]["id"] == sp["<|/python_result|>"]
    done = evs[-1]
    assert done["event"] == "done" and done["reason"] == "stop" and len(done["calls"]) == 2
    a = done["assistant"]
    assert a["well_formed"] and a["answer"] == "#### 12" and a["think"] == "<<<x = 6>>>=(no output; use print(...) to show a value, or end with a bare expression) ok <<x*2=12>>"
    assert a["ids"] == [e["id"] for e in toks] and a["n_calls"] == 2
    # turn 2 in the same conversation: the verbatim assistant ids are reused (no re-execution), and x is still defined
    msgs2 = msgs + [{"role": "assistant", "think": a["think"], "content": a["answer"], "ids": a["ids"]}, {"role": "user", "content": "again"}]
    _scripted(monkeypatch, tok, [sp["<|python_call|>"], *tok.encode("x*3"), sp["<|/python_call|>"], sp["<|/think|>"], *tok.encode("#### 18"), tok.end_id])
    evs2 = list(h.generate("A", mode="chat", messages=msgs2, temperature=0.0, max_new_tokens=64, think_required=True, tools=True, session_id="conv1"))
    prompt = evs2[0]
    assert prompt["event"] == "prompt" and a["ids"] == prompt["ids"][prompt["ids"].index(tok.special("<|assistant|>")) + 1 :][: len(a["ids"])]
    t2 = [e for e in evs2 if e["event"] == "tool"]
    assert t2 and t2[0]["result"] == "18" and h.session("conv1").n_calls == 3
    # a different conversation does not see x; reset forgets it
    _scripted(monkeypatch, tok, [sp["<|python_call|>"], *tok.encode("x"), sp["<|/python_call|>"], sp["<|/think|>"], *tok.encode("#### 0"), tok.end_id])
    evs3 = list(h.generate("A", mode="chat", messages=msgs, temperature=0.0, max_new_tokens=64, think_required=True, tools=True, session_id="conv2"))
    assert [e for e in evs3 if e["event"] == "tool"][0]["result"].startswith("error:")
    h.reset_session("conv1")
    assert "conv1" not in h.sessions
    # a call after </think> is refused and the turn is not well-formed
    _scripted(monkeypatch, tok, [*tok.encode("t"), sp["<|/think|>"], sp["<|python_call|>"], *tok.encode("1+1"), sp["<|/python_call|>"], tok.end_id])
    evs4 = list(h.generate("A", mode="chat", messages=msgs, temperature=0.0, max_new_tokens=64, think_required=True, tools=True, session_id="conv3"))
    d4 = evs4[-1]
    assert d4["reason"] == "tool_outside_think" and not d4["assistant"]["well_formed"] and not [e for e in evs4 if e["event"] == "tool"]


def test_format_chat_uses_generated_ids_verbatim():
    tok = SlmTokenizer(train_bpe(["hello world " * 30], vocab_size=300))
    gen = [tok.special("<|think|>"), *tok.encode("hello"), tok.special("<|/think|>"), *tok.encode("world")]  # no <|end|>: one is appended
    enc = format_chat(tok, [{"role": "user", "content": "hi"}, {"role": "assistant", "ids": gen, "content": "IGNORED"}], add_generation_prompt=True)
    i = enc.ids.index(tok.special("<|assistant|>")) + 1
    assert enc.ids[i : i + len(gen)] == gen and enc.ids[i + len(gen)] == tok.end_id and all(enc.loss_mask[k] == 1 for k in range(i, i + len(gen) + 1))
    assert tok.encode("IGNORED")[0] not in enc.ids[i:]
    assert json.dumps(enc.ids)  # serializable

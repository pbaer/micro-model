"""Test doubles for external slots in the portal (imported by test_portal_model.py and test_portal_swarm.py):
a character-level stand-in for slm.eval.external.HfChatModel, a patcher that makes `load_external` return it, and
an in-process worker (WorkerClient's interface over a Harness in this process, so the patch is seen)."""

from __future__ import annotations

import re

import slm.eval.external as E
from slm import swarm as S


class StubTokenizer:
    """ids are code points; 0 is the end-of-turn token."""

    all_special_ids = [0]

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(("" if skip_special_tokens else "<|im_end|>") if i == 0 else chr(i) for i in ids)


class StubHf:
    """The HfChatModel surface the harness uses. Streams `stream_text` then the stop token; batch chat replies come
    from `replies` (cycled) for sampling prompts, the larger number for a pairwise prompt, `selector_reply` for
    a selector prompt."""

    def __init__(self, name: str, is_chat: bool, device: str = "cpu", stream_text: str = "Paris.", replies=None, selector_reply="#### 12"):
        self.name, self.is_chat, self.device = name, is_chat, device
        self.entry = E.get(name)
        self.tokenizer = StubTokenizer()
        self.eos_ids = {0}
        self.max_positions = 4096
        self.stream_text = stream_text
        self.replies = replies or ["I add them. #### 10", "Ten. #### 10", "Twelve, from 3 * 4. #### 12", "no idea"]
        self.selector_reply = selector_reply
        self.log: list[tuple] = []

    # encoding
    def encode_text(self, text):
        return [ord(c) for c in text]

    encode_plain = encode_text

    def render_chat(self, messages):
        if not self.is_chat:
            raise ValueError(f"{self.name} is a base model: no chat template is applied to it")
        return "".join(f"[{m['role']}]{m['content']}" for m in messages) + "[assistant]"

    def encode_chat(self, messages):
        return self.encode_text(self.render_chat(messages))

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    # generation
    def stream_ids(self, prompt, max_new_tokens, temperature=0.0, top_p=1.0, top_k=0, seed=0, stop_ids=None, logprobs_topk=0):
        self.log.append(("stream", len(prompt), max_new_tokens, temperature, seed))
        for t in ([ord(c) for c in self.stream_text] + [0])[:max_new_tokens]:
            yield {"id": t, "logprob": -0.25, "rank": 0, "entropy": 0.5, "topk": [(t, -0.25), (32, -2.0)][: max(0, logprobs_topk)]}

    def _reply(self, messages, j):
        content = messages[-1]["content"]
        if S.PAIR_ASK in content:
            a, b = (float(x) for x in re.findall(r"^Answer [AB]: (\S+) \(", content, re.M))
            return "#### A" if a > b else "#### B"
        if S.SELECT_ASK in content:
            return self.selector_reply
        return self.replies[j % len(self.replies)]

    def batch_generate_chat(self, list_of_messages, max_new_tokens=256, temperature=0.0, top_p=1.0, top_k=0, seed=0, stop=None, batch_size=32):
        for m in list_of_messages:
            self.render_chat(m)  # a base model refuses, as the real one does
        self.log.append(("batch", len(list_of_messages), max_new_tokens, temperature, seed, [m[-1]["content"] for m in list_of_messages]))
        return [E.Generation(t, [ord(c) for c in t] + [0], True, len(t) + 1) for t in (self._reply(m, j) for j, m in enumerate(list_of_messages))]

    def generate_chat(self, messages, max_new_tokens=256, temperature=0.0, top_p=1.0, top_k=0, seed=0, stop=None):
        self.log.append(("chat", messages[-1]["content"], max_new_tokens, temperature))
        return self.batch_generate_chat([messages], max_new_tokens, temperature, top_p, top_k, seed, stop)[0].text


def patch_external(monkeypatch, missing=("gpt2-medium",), **stub_kw) -> list:
    """`load_external` returns a StubHf; every registered model is on disk except `missing`. Returns the load log."""
    loads = []

    def fake_load(name, device="cuda", dtype=None):
        loads.append((name, device, dtype))
        return StubHf(name, E.get(name).is_chat, device, **stub_kw)

    monkeypatch.setattr(E, "load_external", fake_load)
    monkeypatch.setattr(E.ExternalModel, "available", lambda self: self.name not in missing)
    return loads


class InProcWorker:
    """WorkerClient's call / stream over an in-process Harness: exceptions become RuntimeError / an error event."""

    def __init__(self, harness) -> None:
        self.h, self._alive = harness, False

    def alive(self):
        return self._alive

    def stop(self):
        self._alive = False

    def maybe_idle_stop(self):
        return False

    def call(self, method, **kw):
        self._alive = True
        try:
            return getattr(self.h, method)(**kw)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"{type(e).__name__}: {e}") from None

    def stream(self, method, cancel_flag=None, **kw):
        self._alive = True
        try:
            for ev in getattr(self.h, method)(**kw, should_stop=lambda: bool(cancel_flag is not None and cancel_flag.is_set())):
                yield dict(ev)
        except Exception as e:  # noqa: BLE001
            yield {"event": "error", "error": f"{type(e).__name__}: {e}"}

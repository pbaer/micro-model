"""Chat formatting with our reserved special tokens, plus the SFT loss mask.

Format (see CLAUDE.md):
    <|bos|><|system|>...<|end|><|user|>...<|end|><|assistant|><|think|>...<|/think|>answer<|end|><|eos|>

Loss mask: 1 on assistant-turn content tokens and their closing <|end|> (the stop signal), 0 on
everything else, including <|eos|> (the model cannot know whether another turn follows).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from slm.data.tokenizer import SlmTokenizer

ROLES = ("system", "user", "assistant")


@dataclass
class ChatEncoding:
    ids: list[int]
    loss_mask: list[int]
    # (start, end, label) spans over ids, for visualization: role tokens, content, think, specials
    segments: list[tuple[int, int, str]] = field(default_factory=list)


def format_chat(
    tok: SlmTokenizer,
    messages: list[dict],
    add_generation_prompt: bool = False,
    bos: bool = True,
    eos: bool = True,
    think_required: bool = False,
) -> ChatEncoding:
    ids: list[int] = []
    mask: list[int] = []
    segs: list[tuple[int, int, str]] = []

    def push(seq: list[int], m: int, label: str) -> None:
        segs.append((len(ids), len(ids) + len(seq), label))
        ids.extend(seq)
        mask.extend([m] * len(seq))

    if bos:
        push([tok.bos_id], 0, "bos")
    for msg in messages:
        role = msg["role"]
        assert role in ROLES, f"unknown role {role}"
        push([tok.special(f"<|{role}|>")], 0, f"role:{role}")
        target = 1 if role == "assistant" else 0
        if role == "assistant":
            think = msg.get("think")
            if think is not None or think_required:
                push([tok.special("<|think|>")], target, "think_open")
                push(tok.encode(think or ""), target, "think")
                push([tok.special("<|/think|>")], target, "think_close")
        push(tok.encode(msg.get("content", "")), target, f"content:{role}")
        push([tok.end_id], target, "end")
    if add_generation_prompt:
        push([tok.special("<|assistant|>")], 0, "role:assistant")
        if think_required:
            push([tok.special("<|think|>")], 0, "think_open")
    elif eos:
        push([tok.eos_id], 0, "eos")
    return ChatEncoding(ids, mask, segs)


def parse_assistant(tok: SlmTokenizer, ids: list[int], think_expected: bool = True) -> dict:
    """Split a generated assistant turn into think / answer text.

    `ids` are the tokens generated after the assistant role token. When the generation prompt already
    ended with <|think|> (think_required prompts), the completion starts inside the think span, so
    the opening tag may legitimately be absent; the closing <|/think|> is what separates the spans.
    """
    think_open, think_close = tok.special("<|think|>"), tok.special("<|/think|>")
    stop = {tok.end_id, tok.eos_id}
    body: list[int] = []
    terminated = False
    for i in ids:
        if i in stop:
            terminated = True
            break
        body.append(i)
    if body and body[0] == think_open:
        body = body[1:]
        had_open = True
    else:
        had_open = False
    if think_close in body:
        j = body.index(think_close)
        think, answer = body[:j], body[j + 1 :]
        malformed = think_close in answer or think_open in answer  # nested / repeated tags
    elif think_expected:
        think, answer = None, body  # no closing tag: treat everything as the answer, flag it
        malformed = True
    else:
        think, answer = None, body
        malformed = had_open  # opened a think span without closing it
    return {
        "think": tok.decode(think, skip_special=True) if think is not None else None,
        "answer": tok.decode(answer, skip_special=True),
        "terminated": terminated,
        "malformed": bool(malformed) or not terminated,
    }

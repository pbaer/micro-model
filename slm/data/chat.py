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


def parse_assistant(tok: SlmTokenizer, ids: list[int]) -> dict:
    """Split a generated assistant turn into think / answer text (ids after <|assistant|>)."""
    think_open, think_close = tok.special("<|think|>"), tok.special("<|/think|>")
    stop = {tok.end_id, tok.eos_id}
    body = []
    for i in ids:
        if i in stop:
            break
        body.append(i)
    think, answer = None, body
    if body and body[0] == think_open:
        if think_close in body:
            j = body.index(think_close)
            think, answer = body[1:j], body[j + 1 :]
        else:
            think, answer = body[1:], []
    return {
        "think": tok.decode(think, skip_special=True) if think is not None else None,
        "answer": tok.decode(answer, skip_special=True),
        "terminated": any(i in stop for i in ids),
        "malformed": bool(body) and body[0] != think_open or (think is not None and think_close not in body),
    }

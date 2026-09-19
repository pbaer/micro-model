"""Final-answer style for verifiable tasks.

The `#### <answer>` marker exists for programmatic verification. The model must only produce it when the user
asks for it, so every training set with verifiable answers is mixed: with probability `p_marker` the user turn
carries the instruction (`SUFFIX`) and the answer is `#### X`; otherwise the user turn is left as asked and the
answer is a natural sentence ("So the answer is 624."). RL rollouts and the reasoning benchmark always prompt
with the instruction (`slm.rl.tasks.prompt_messages`), so the verifier stays strict there.
"""

from __future__ import annotations

import random
import re

SUFFIX = "\nThink step by step, then give the final answer on its own line as '#### <number>'."

_NATURAL_NUMERIC = [  # "that makes 42" reads fine; "that makes eraser" does not
    "So the answer is {x}.",
    "The answer is {x}.",
    "That gives {x}.",
    "That makes {x}.",
    "So it comes to {x}.",
    "The result is {x}.",
]
_NATURAL_TEXT = [
    "The answer is {x}.",
    "So the answer is {x}.",
    "It is {x}.",
    "That would be {x}.",
    "So it's {x}.",
]
_NUMERIC_RE = re.compile(r"^[-+]?[\d,]*\.?\d+(e[-+]?\d+)?%?$")


def is_numeric_answer(x: str) -> bool:
    return bool(_NUMERIC_RE.match(x.strip().replace(" ", "")))


def marker_answer(x: str) -> str:
    return f"#### {x}"


def natural_answer(x: str, rng: random.Random) -> str:
    return rng.choice(_NATURAL_NUMERIC if is_numeric_answer(x) else _NATURAL_TEXT).format(x=x)


def apply_style(messages: list[dict], rng: random.Random, p_marker: float = 0.5, final: str | None = None) -> list[dict]:
    """Rewrite a conversation in place: every assistant turn whose content is `#### X` (or `final`) gets the
    marker style with the instruction added to the preceding user turn, or the natural style without it. One
    coin flip per conversation so a multi-turn dialogue is consistent."""
    use_marker = rng.random() < p_marker
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        content = m.get("content", "")
        x = final if final is not None else (content[5:].strip() if content.startswith("#### ") else None)
        if x is None:
            continue
        user = messages[i - 1] if i > 0 and messages[i - 1]["role"] == "user" else None
        if use_marker:
            m["content"] = marker_answer(x)
            if user is not None and SUFFIX.strip() not in user["content"]:
                user["content"] = user["content"].rstrip() + SUFFIX
        else:
            m["content"] = natural_answer(x, rng)
            if user is not None and SUFFIX.strip() in user["content"]:
                user["content"] = user["content"].replace(SUFFIX, "").rstrip()
    return messages

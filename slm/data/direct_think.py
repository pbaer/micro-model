"""Short, content-free think spans for ordinary chat turns: the "no tool needed" signal.

Stage A prepared every chat conversation with a MANDATORY but EMPTY think span, and stage B then taught the
model to put reasoning and `<|python_call|>` spans inside it. The result (2026-09-21) was a routing failure,
not a capability loss: on the judged suite, every category whose items emitted a tool call regressed
(pattern -2.42, qa -2.00, definition -1.00, facts -0.62) and every category with zero tool calls held or
improved (narrative +0.50, prose -0.08). "What is the capital of France?" produced an invented
`city_population(...)` call and the answer "That would be Aldershaw."

The cause is that the mixture contained no example of the third case. The model saw "empty think, answer"
(chat rehearsal) and "think with a tool call, answer" (tools), so once stage B moved it off empty think spans
the only non-empty span it knew how to write was one containing code. Short, question-shaped, data-free
prompts ("What is the opposite of big?") look exactly like the tool families' prompts on the surface, which is
why those are the ones that break.

So these lines fill the gap: a brief prose think that opens the span and does NOT reach for code. They carry
no knowledge on purpose -- the answer still comes from the public chat data, so this teaches the routing
decision without teaching the suite's facts. `_ROUTE` picks a group from the shape of the user's question and
`line()` picks a phrasing inside it; a caller applies them to a fraction of conversations and leaves the rest
with an empty span, so the model keeps both non-tool behaviours.
"""

from __future__ import annotations

import random
import re

# Two kinds of line. PLAIN just starts thinking in prose (the common case, and the one that generalises).
# DECLINE names the decision explicitly; it is the direct negative example for the failure above, kept to a
# minority share so "no code" does not itself become the dominant think token in chat context.
_PLAIN = {
    "why": [
        "The user wants an explanation. Let me lay out the reason plainly.",
        "This asks why something happens, so I should explain the cause.",
        "I should give the reason and keep it understandable.",
        "Let me think about what actually causes this and say it simply.",
    ],
    "list": [
        "The user wants a list. Let me recall the items and give them in order.",
        "This is a listing task. I should name each item, in the usual order.",
        "Let me set out the items one by one.",
    ],
    "define": [
        "The user is asking for a definition. Let me state what it means.",
        "This asks what something is, so a clear definition is what is wanted.",
        "Let me define the term and add a short example.",
    ],
    "howto": [
        "The user wants to know how to do this. Let me describe the steps.",
        "This is a how-to question. I should walk through it in order.",
        "Let me think through the steps and describe them.",
    ],
    "write": [
        "The user wants something written. Let me think about tone and shape first.",
        "This is a writing task. Let me plan what to say before I say it.",
        "Let me decide on the structure, then write it.",
    ],
    "": [
        "Let me think about what the user is asking for.",
        "Let me consider this and answer directly.",
        "I should work out what is being asked and respond to it.",
        "Let me think this through before answering.",
    ],
}

_DECLINE = [
    "This is a question I can answer from what I know; no computation is needed.",
    "There is nothing to compute here, so I will answer directly.",
    "No code is needed for this one. I can answer it as it stands.",
    "This does not need any calculation; I just have to answer the question.",
    "I do not need to run anything here. Let me answer from knowledge.",
    "No data to work through here, so a direct answer is right.",
]

_ROUTE = (
    ("why", ("why ", "how come", "what causes", "what makes")),
    ("list", ("list ", "name the", "name all", "give me a list", "what are the")),
    ("define", ("what is a", "what is an", "what is the meaning", "define ", "what does", "who is", "who was")),
    ("howto", ("how do i", "how do you", "how can i", "how to ", "what are the steps")),
    ("write", ("write ", "compose ", "draft ", "rewrite ", "summarize ", "summarise ", "translate ", "tell me a story")),
)


# A prompt that plausibly wants computation must NOT become a "no code is needed" example: that would teach the
# model to decline on exactly the turns the tool exists for. Chat sets are full of such prompts (a probability
# question from openhermes drew "No code is needed for this one" in the first probe run), so these are dropped
# from the direct-think treatment entirely and keep their empty span.
# Stems match as prefixes (no trailing boundary: "calculat" must reach "calculate"); whole phrases are anchored
# on both sides so "count" does not fire on "country".
_COMPUTE = re.compile(
    r"\b(calculat|comput|evaluat|multipl|divid|percentag|solv|equation|probabilit"
    r"|derivativ|integral|factorial|remaind|modulo|convert|sort|revers|arithmetic)"
    r"|\b(how many|how much|add up|percent of|round to|in order|sequence|next number"
    r"|count the|count how|total of|sum of|average of|product of"
    r"|what is the (sum|total|average|mean|product|median|difference|result))\b",
    re.I,
)
_LITERAL = re.compile(r"[\[{]\s*[-\w'\"]+\s*[,:]|\d\s*[-+*/^]\s*\d|\d{3,}")


def is_computational(user_text: str) -> bool:
    """Would a `<|python_call|>` be a reasonable move here? Deliberately over-inclusive."""
    t = user_text[:600]
    return bool(_COMPUTE.search(t) or _LITERAL.search(t)) or sum(c.isdigit() for c in t) >= 8


def group_of(user_text: str) -> str:
    """Which family of phrasings fits this question. Shape only -- never the subject matter."""
    t = " ".join(user_text.lower().split())[:120]
    for name, prefixes in _ROUTE:
        if any(p in t for p in prefixes):
            return name
    return ""


def line(user_text: str, rng: random.Random, decline_frac: float = 0.4) -> str | None:
    """One short think span for an ordinary chat turn, or None if the turn might want a tool."""
    if is_computational(user_text):
        return None
    if rng.random() < decline_frac:
        return rng.choice(_DECLINE)
    return rng.choice(_PLAIN[group_of(user_text)])

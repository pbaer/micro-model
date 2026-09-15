"""Answer parsing and programmatic verification. Robustness here IS the RL system's safety."""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

_ANSWER_RE = re.compile(r"####\s*([^\n]*)")
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def _to_number(s: str) -> Fraction | None:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if not s:
        return None
    try:
        if "/" in s:
            n, d = s.split("/", 1)
            return Fraction(int(n), int(d))
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def parse_final_answer(text: str) -> str | None:
    """The LAST '#### <answer>' line wins; the answer is the first number-like token after it."""
    hits = _ANSWER_RE.findall(text)
    if not hits:
        return None
    m = _NUM_RE.search(hits[-1])
    return m.group(0) if m else hits[-1].strip() or None


@dataclass
class Verdict:
    correct: bool
    parsed: str | None
    reason: str


def verify_numeric(answer_text: str, gold: str) -> Verdict:
    p = parse_final_answer(answer_text)
    if p is None:
        return Verdict(False, None, "no '####' answer line")
    a, g = _to_number(p), _to_number(gold)
    if a is None or g is None:
        return Verdict(p.strip() == gold.strip(), p, "string compare")
    return Verdict(a == g, p, "numeric compare")


_LITERAL_CALL_RE = re.compile(r"^\s*(?:print\(\s*)?-?[\d.,]+\s*\)?\s*$")


def answer_from_tool(parsed: str | None, calls: list[tuple[str, str]]) -> bool:
    """True when the final answer equals a number some tool call produced, and that call did real work
    (not a bare literal like print(42), which would let the model launder a mental answer)."""
    if parsed is None:
        return False
    a = _to_number(parsed)
    if a is None:
        return False
    for code, result in calls:
        if not code or _LITERAL_CALL_RE.match(code) or result.startswith("error:"):
            continue
        for m in _NUM_RE.finditer(result.replace(",", "")):
            if _to_number(m.group(0)) == a:
                return True
    return False


def reward_from_verdict(v: Verdict, malformed: bool, scheme: str = "binary", from_tool: bool = False) -> float:
    """binary: 1/0. signed: +1/-1. shaped: 1 correct, 0 wrong-but-parsable, -0.5 malformed/unparsable.
    tool: 1 if correct AND the answer came out of a Python call, 0.5 if correct without one, 0 otherwise —
    the incentive to compute with the tool rather than in the head."""
    if scheme == "binary":
        return 1.0 if v.correct else 0.0
    if scheme == "tool":
        return (1.0 if from_tool else 0.5) if v.correct else 0.0
    if scheme == "signed":
        return 1.0 if v.correct else -1.0
    if scheme == "shaped":
        if v.correct:
            return 1.0
        return -0.5 if (malformed or v.parsed is None) else 0.0
    raise ValueError(scheme)

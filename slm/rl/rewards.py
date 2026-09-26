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


def parse_final_span(text: str) -> str | None:
    """The LAST '#### <answer>' line, whole and untouched — the non-numeric path (a word, a list, a
    boolean), where taking the first number out of the line would be exactly wrong."""
    hits = _ANSWER_RE.findall(text)
    return (hits[-1].strip() or None) if hits else None


@dataclass
class Verdict:
    correct: bool
    parsed: str | None
    reason: str
    fraction: float | None = None  # partial credit (constraints: the share satisfied); None = binary verdict


def verify_numeric(answer_text: str, gold: str, strict: bool = True) -> Verdict:
    """strict=True expects the '#### <answer>' marker (prompts that asked for it: RL, benchmarks). strict=False
    accepts the last number in the text, for answers to bare questions written as a sentence."""
    p = parse_final_answer(answer_text)
    if p is None and not strict:
        nums = _NUM_RE.findall(answer_text.replace(",", ""))
        p = nums[-1] if nums else None
    if p is None:
        return Verdict(False, None, "no '####' answer line")
    a, g = _to_number(p), _to_number(gold)
    if a is None or g is None:
        return Verdict(p.strip() == gold.strip(), p, "string compare")
    return Verdict(a == g, p, "numeric compare")


_LISTISH_RE = re.compile(r"^[\[(].*[\])]$", re.DOTALL)


def _norm_token(s: str) -> str:
    """A word answer, stripped of the packaging a model puts around it: quotes, trailing punctuation,
    repeated whitespace, case."""
    s = " ".join(s.split()).strip("\"'`*")
    return s.strip(" \t.,;:!?\"'`*").casefold()


def _same_token(a: str, b: str) -> bool:
    if a == b:
        return True
    na, nb = _to_number(a), _to_number(b)
    return na is not None and nb is not None and na == nb  # 5 == 5.0, 1,200 == 1200


def _items(s: str) -> list[str]:
    body = s.strip()
    if _LISTISH_RE.match(body):
        body = body[1:-1]
    return [x for x in (_norm_token(p) for p in body.split(",")) if x]


def exact_match(parsed: str, gold: str) -> bool:
    """Gold-shaped comparison: a list compares item by item (order and length count, brackets and quotes
    do not), anything else compares as one normalized token (numbers numerically)."""
    if _LISTISH_RE.match(gold.strip()) or ("," in gold and "," in parsed):
        p, g = _items(parsed), _items(gold)
        return len(p) == len(g) and all(_same_token(x, y) for x, y in zip(p, g))
    return _same_token(_norm_token(parsed), _norm_token(gold))


def verify_exact(answer_text: str, gold: str, strict: bool = True) -> Verdict:
    """Non-numeric gold (a word, a list, a boolean): the final answer span must match it. Tolerant of the
    packaging (case, surrounding punctuation, quotes, list brackets and spacing), strict about content —
    a wrong item, a wrong word or an extra token fails."""
    p = parse_final_span(answer_text)
    if p is None and not strict:
        lines = [ln.strip() for ln in answer_text.strip().splitlines() if ln.strip()]
        p = lines[-1] if lines else None
    if p is None:
        return Verdict(False, None, "no '####' answer line")
    return Verdict(exact_match(p, gold), p, "exact compare")


def verify_constraints(answer_text: str, gold: str) -> Verdict:
    """Instruction-following tasks: `correct` is "every constraint satisfied", `fraction` is the share."""
    from slm.rl.constraints import check_constraints, parse_specs

    specs = parse_specs(gold)
    if not specs:
        return Verdict(False, None, "no constraint spec")
    ok, failed = check_constraints(answer_text, specs)
    reason = "constraints: all satisfied" if not failed else "constraints failed: " + ",".join(failed)
    return Verdict(ok == len(specs), f"{ok}/{len(specs)} satisfied", reason, ok / len(specs))


# The natural-answer templates from slm.data.answers, as a chat answer must NOT look. "So the answer is 2." to
# "Who wrote Hamlet?" is what M9 stage C produced after RL had seen only math, tool and constraint families:
# the whole policy drifted toward verifier-shaped output. A one-sentence answer that is nothing but a template
# opener and a short filler is that drift; real prose that happens to start "It is ..." runs longer than this.
_TEMPLATE_RE = re.compile(
    r"^\s*(so the answer is|the answer is|that gives|that makes|so it comes to|the result is|it is|that would be|so it's)"
    r"\b[^.\n]{0,40}\.?\s*$", re.I)


def verify_plain(answer_text: str) -> Verdict:
    """An ordinary chat answer: present, not a `####` line, not a verifier template. There is no gold -- the
    reward is for answering like a chatbot, and the `plain` scheme adds "and without calling the tool"."""
    a = answer_text.strip()
    if not a:
        return Verdict(False, None, "empty answer")
    if "####" in a:
        return Verdict(False, a[:60], "verifier marker in a chat answer")
    if _TEMPLATE_RE.match(a):
        return Verdict(False, a[:60], "verifier-shaped template in a chat answer")
    return Verdict(True, a[:60], "plain chat answer")


def verify_answer(answer_text: str, gold: str, kind: str = "auto", strict: bool = True) -> Verdict:
    """The single entry point the rollout path uses. "auto" dispatches on the gold's shape: a numeric gold
    gets the numeric comparison it always had, anything else the exact one."""
    from slm.data.answers import is_numeric_answer

    if kind == "constraints":
        return verify_constraints(answer_text, gold)
    if kind == "plain":
        return verify_plain(answer_text)
    if kind == "numeric" or (kind == "auto" and is_numeric_answer(gold)):
        return verify_numeric(answer_text, gold, strict)
    if kind not in ("auto", "exact"):
        raise ValueError(kind)
    return verify_exact(answer_text, gold, strict)


_LITERAL_CALL_RE = re.compile(r"""^\s*(?:print\(\s*)?(?:-?[\d.,]+|'[^']*'|"[^"]*"|\[[^][]*\])\s*\)?\s*$""")


def answer_from_tool(parsed: str | None, calls: list[tuple[str, str]]) -> bool:
    """True when the final answer is something a tool call produced, and that call did real work (not a
    bare literal like print(42) or print([1, 2]), which would let the model launder a mental answer).

    A number may appear anywhere in the result (GSM8K traces print sentences); a non-numeric answer (a
    word, a list) must be the whole result, which is what the sandbox prints for such a program."""
    if parsed is None:
        return False
    a = _to_number(parsed)
    for code, result in calls:
        if not code or _LITERAL_CALL_RE.match(code) or result.startswith("error:"):
            continue
        if a is None:
            if exact_match(parsed, result):
                return True
            continue
        for m in _NUM_RE.finditer(result.replace(",", "")):
            if _to_number(m.group(0)) == a:
                return True
    return False


def resolve_scheme(task: str, schemes: dict[str, str] | None, default: str = "binary") -> str:
    """The reward scheme for one task family: an exact entry in `schemes` wins, then the family's group
    (`pytool_declared` -> `pytool`), then the run's single `reward_scheme`. Constraint tasks have no tool
    to use, so a run that trains them next to tool tasks needs both schemes at once."""
    if schemes:
        if task in schemes:
            return schemes[task]
        head = task.split("_", 1)[0]
        if head in schemes:
            return schemes[head]
    return default


def reward_from_verdict(v: Verdict, malformed: bool, scheme: str = "binary", from_tool: bool = False, n_calls: int = 0) -> float:
    """binary: 1/0. signed: +1/-1. shaped: 1 correct, 0 wrong-but-parsable, -0.5 malformed/unparsable.
    tool: 1 if correct AND the answer came out of a Python call, 0.5 if correct without one, 0 otherwise —
    the incentive to compute with the tool rather than in the head.
    fraction: the verdict's partial credit (constraints: the share of instructions satisfied), so an answer
    that obeys 2 of 3 rules is worth more than one that obeys none, long before any of them is perfect.
    plain: 1 if the answer is an ordinary chat answer (verify_plain) AND no tool was called, else 0 -- the
    anchor that keeps RL on math and tools from turning every reply into a tool call and a template.
    tool_strict: like tool, but correct-without-a-call earns 0.25 instead of 0.5. Run 3's chat anchor pulled
    tool use down on math too (algebra 0.96 -> 0.71): with the no-call credit at 0.5, a mental answer was
    worth half a tool answer and the anchor's "call less" pressure tipped the balance. A wider gap makes the
    sandbox worth reaching for again where it belongs, without softening the anchor where it works."""
    if scheme == "binary":
        return 1.0 if v.correct else 0.0
    if scheme == "fraction":
        if malformed:
            return 0.0
        return float(v.fraction if v.fraction is not None else (1.0 if v.correct else 0.0))
    if scheme in ("tool", "tool_strict"):
        if malformed:
            return 0.0  # includes tool calls outside the think span
        no_call = 0.5 if scheme == "tool" else 0.25
        return (1.0 if from_tool else no_call) if v.correct else 0.0
    if scheme == "plain":
        return 0.0 if (malformed or n_calls) else (1.0 if v.correct else 0.0)
    if scheme == "signed":
        return 1.0 if v.correct else -1.0
    if scheme == "shaped":
        if v.correct:
            return 1.0
        return -0.5 if (malformed or v.parsed is None) else 0.0
    raise ValueError(scheme)

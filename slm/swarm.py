"""Swarm inference: sample many, collapse by answer, verify in the sandbox, select with one more pass of the
same model.

    from slm.swarm import swarm_answer
    res = swarm_answer(model, tok, "A train travels 60 mph for 3 hours. How far?", k=16)
    res.final            # the selected answer, or None
    res.groups           # distinct answers with support counts and verification evidence

Why this shape (docs/results.md §14a, §16d). With greedy decoding the 336M model is right on 2-5% of GSM8K
and SVAMP problems, yet the right answer is somewhere among 32 samples for 47% / 73% of them: the pool has
the answer, so the whole problem is selection. Majority vote recovers almost none of it -- 22-24 distinct
answers per 32 samples means the right one is usually held by one or two candidates surrounded by twenty
confident wrong ones. Two things make selection tractable at this size:

  1. Mechanical work before the model judges. Candidates are grouped by their parsed final answer, and each
     group carries the evidence the sandbox already produced: how many attempts agreed, and whether the
     answer actually came out of a Python call that ran without error (`answer_from_tool`). A model that
     cannot pick one right answer out of sixty cannot be asked to; it can be asked to pick among a dozen
     distinct answers with support and provenance attached.
  2. The selector is the same model, asked in a prompt format it is trained on. `selector_messages` builds
     that prompt within a token budget (effective context is 3072); the selection SFT set and the RL `select`
     family teach the format and reward the right pick, so selection is a capability, not a prompt trick.

Every stage returns its intermediate state so the eval (scripts/swarm_eval.py) can report the ceiling at each
step -- pass@k, any-correct-after-verification, correct-group-in-prompt -- and the inference page can show
exactly what the swarm saw and chose.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

# The selector prompt. Kept as constants so the SFT builder, the RL family and inference all render the same
# thing; a model trained on one wording must see that wording.
SELECT_INTRO = ("Several attempts were made at this problem. Here are the distinct final answers they reached, how many "
                "attempts agreed on each, and whether the answer was computed by running Python code.")
SELECT_ASK = ("Decide which answer is correct. You may check with Python. Give the final answer on its own line as "
              "'#### <answer>'.")


@dataclass
class Candidate:
    idx: int
    think: str | None
    answer: str
    parsed: str | None      # the final answer as parsed from the answer span, or None
    key: str | None         # grouping key: numbers canonicalised, text lowercased
    terminated: bool
    n_calls: int
    n_errors: int
    calls: list[list[str]]  # [code, result] per call
    from_tool: bool         # the parsed answer came out of a real call (slm.rl.rewards.answer_from_tool)
    n_tokens: int

    @property
    def verified(self) -> bool:
        """Sandbox-backed: the answer was produced by a call that ran without error. This is the evidence a
        small model cannot fake in its head, and the thing majority vote ignores."""
        return self.from_tool and self.n_errors == 0


@dataclass
class Group:
    key: str
    answer: str
    support: int
    verified: int
    members: list[int]
    rationale: str


@dataclass
class SwarmResult:
    prompt: str
    k: int
    candidates: list[Candidate]
    groups: list[Group]                 # sorted: verified support first, then support
    majority: str | None                # most-supported answer
    verified_majority: str | None       # most-supported answer among verified candidates, else majority
    selector_messages: list[dict]
    selector_think: str | None
    selector_answer: str
    selector_calls: int
    final: str | None                   # the selector's parsed answer, or verified_majority when it gave none
    seconds: float
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------------------------------ pure parts
def answer_key(parsed: str | None) -> str | None:
    """Group key for a parsed answer: numbers by value (so 42, 42.0 and 42.00 agree), text by lowercase."""
    if parsed is None:
        return None
    from slm.rl.rewards import _to_number

    v = _to_number(parsed)
    if v is not None:
        return str(int(v)) if float(v).is_integer() else repr(float(v))
    t = " ".join(parsed.split()).strip(" .").lower()
    return t or None


def display_answer(parsed: str) -> str:
    from slm.rl.rewards import _to_number

    v = _to_number(parsed)
    if v is not None:
        return str(int(v)) if float(v).is_integer() else str(float(v))
    return " ".join(parsed.split()).strip(" .")


def collapse(cands: list[Candidate], max_rationale_chars: int = 240) -> list[Group]:
    """Distinct answers with support and evidence. The representative rationale is a verified member's think
    span when one exists (it carries the code that produced the answer), else the shortest member's."""
    by: dict[str, list[Candidate]] = {}
    for c in cands:
        if c.key is not None:
            by.setdefault(c.key, []).append(c)
    groups = []
    for key, members in by.items():
        ver = [c for c in members if c.verified]
        rep = min(ver or members, key=lambda c: len(c.think or ""))
        rationale = _trim((rep.think or "").strip(), max_rationale_chars)
        groups.append(Group(key=key, answer=display_answer(rep.parsed or key), support=len(members), verified=len(ver),
                            members=[c.idx for c in members], rationale=rationale))
    groups.sort(key=lambda g: (-g.verified, -g.support, g.key))
    return groups


def _trim(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def majority(groups: list[Group]) -> str | None:
    return max(groups, key=lambda g: (g.support, g.verified)).answer if groups else None


def verified_majority(groups: list[Group]) -> str | None:
    ver = [g for g in groups if g.verified]
    return max(ver, key=lambda g: (g.verified, g.support)).answer if ver else majority(groups)


def selector_messages(task_prompt: str, groups: list[Group], tok=None, budget_tokens: int = 2400, max_groups: int = 12) -> list[dict]:
    """The selection prompt: the task, then each distinct answer with its support and provenance and a short
    representative rationale. Fits a token budget by first dropping the least-supported groups, then
    shortening rationales -- the answers themselves are never dropped below `min_groups`, and the budget is
    measured with the real tokenizer when one is given."""
    gs = groups[:max_groups]
    chars = 240
    while True:
        lines = []
        for g in gs:
            how = f"agreed by {g.support} attempt{'s' if g.support != 1 else ''}"
            how += f"; computed with code in {g.verified}" if g.verified else "; not computed with code"
            lines.append(f"- Answer: {g.answer} ({how})" + (f"\n  Reasoning: {_trim(g.rationale, chars)}" if g.rationale else ""))
        content = f"{task_prompt.strip()}\n\n{SELECT_INTRO}\n" + "\n".join(lines) + f"\n\n{SELECT_ASK}"
        n = len(tok.encode(content)) if tok is not None else len(content) // 4
        if n <= budget_tokens or (len(gs) <= 3 and chars <= 60):
            return [{"role": "user", "content": content}]
        if chars > 60:
            chars = max(60, chars // 2)
        else:
            gs = gs[:-1]


# ------------------------------------------------------------------------------------------------ model parts
def _candidates_from_completions(tok, tcs) -> list[Candidate]:
    from slm.data.chat import parse_assistant
    from slm.rl.rewards import answer_from_tool, parse_final_span

    out = []
    for i, tc in enumerate(tcs):
        p = parse_assistant(tok, tc.ids)
        parsed = parse_final_span(p["answer"])
        calls = list(tc.calls)
        out.append(Candidate(idx=i, think=p["think"], answer=p["answer"], parsed=parsed, key=answer_key(parsed),
                             terminated=bool(p["terminated"]) and not p["malformed"], n_calls=tc.n_calls, n_errors=tc.n_errors,
                             calls=[list(c) for c in calls], from_tool=answer_from_tool(parsed, calls), n_tokens=len(tc.ids)))
    return out


def sample_candidates(model, tok, messages: list[dict], k: int, temperature: float = 0.8, top_p: float = 0.95,
                      max_new_tokens: int = 512, max_calls: int = 6, seed: int | None = None, functions=None) -> list[Candidate]:
    import torch

    from slm.data.chat import format_chat
    from slm.tools.loop import sample_with_tools

    prompt_ids = format_chat(tok, messages, add_generation_prompt=True, think_required=True, functions=functions).ids
    gen = torch.Generator(device=next(model.parameters()).device)
    if seed is not None:
        gen.manual_seed(seed)
    tcs = sample_with_tools(model, tok, [prompt_ids] * k, max_new_tokens, temperature, top_p, 0, gen, max_calls=max_calls, functions=functions)
    return _candidates_from_completions(tok, tcs)


def select(model, tok, messages: list[dict], max_new_tokens: int = 384, max_calls: int = 4):
    """One greedy pass (top_k=1) of the same model over the selection prompt, tool available so it can check."""
    import torch

    from slm.data.chat import format_chat, parse_assistant
    from slm.rl.rewards import parse_final_span
    from slm.tools.loop import sample_with_tools

    prompt_ids = format_chat(tok, messages, add_generation_prompt=True, think_required=True).ids
    gen = torch.Generator(device=next(model.parameters()).device)
    gen.manual_seed(0)
    tc = sample_with_tools(model, tok, [prompt_ids], max_new_tokens, 1.0, 1.0, 1, gen, max_calls=max_calls)[0]
    p = parse_assistant(tok, tc.ids)
    return p["think"], p["answer"], parse_final_span(p["answer"]), tc.n_calls


def swarm_answer(model, tok, task_prompt: str, k: int = 16, temperature: float = 0.8, top_p: float = 0.95,
                 max_new_tokens: int = 512, max_calls: int = 6, seed: int | None = None, budget_tokens: int = 2400,
                 max_groups: int = 12, answer_suffix: str | None = None) -> SwarmResult:
    """The whole pipeline for one task. `answer_suffix` is appended to the sampling prompt (the '#### <number>'
    instruction the verifiable tasks use); the selector prompt carries its own ask."""
    from slm.utils.sdpa import sdpa_context

    t0 = time.time()
    with sdpa_context("decode"):
        msgs = [{"role": "user", "content": task_prompt + (answer_suffix or "")}]
        cands = sample_candidates(model, tok, msgs, k, temperature, top_p, max_new_tokens, max_calls, seed)
        groups = collapse(cands)
        sel_msgs = selector_messages(task_prompt, groups, tok, budget_tokens, max_groups) if groups else []
        think, answer, parsed, n_calls = (None, "", None, 0)
        if groups:
            think, answer, parsed, n_calls = select(model, tok, sel_msgs)
    vm = verified_majority(groups)
    final = display_answer(parsed) if parsed is not None else vm
    return SwarmResult(prompt=task_prompt, k=k, candidates=cands, groups=groups, majority=majority(groups), verified_majority=vm,
                       selector_messages=sel_msgs, selector_think=think, selector_answer=answer, selector_calls=n_calls, final=final,
                       seconds=round(time.time() - t0, 2))

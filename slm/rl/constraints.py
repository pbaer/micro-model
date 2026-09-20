"""Instruction-following RL tasks: a writing prompt plus 1-3 machine-checkable constraints.

Everything else in `slm.rl` rewards a *right answer*; nothing rewards following an instruction that has no
answer at all ("exactly three sentences", "no commas", "end with this phrase"). These tasks do, and they
are the only RL family with no tool to use: the reward is the fraction of constraints satisfied, so a
partially obedient answer still gives gradient (`reward_scheme: fraction`).

The spec is the truth. A constraint is a small dict (`{"type": "sentences_exactly", "n": 3}`); the prompt
sentence is *composed from* the spec so the model reads plain English, and the checker evaluates the spec
and never parses the prompt. `Task.answer` is the JSON spec list, which makes the gold self-contained in
rollout dumps and in `checkpoints`-adjacent artifacts.

Constraints are drawn one per group, and pairs that cannot both hold (a postscript in an all-lowercase
answer, a title before a required first word) are excluded, so every task is satisfiable.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass

from slm.rl.tasks import Task, task_corpus

TOPICS = [
    "a public library", "the water cycle", "learning to cook", "a morning walk", "how bridges are built",
    "keeping a garden", "why maps are useful", "a long train journey", "the habits of crows", "repairing a bicycle",
    "why bread rises", "a village market", "how a lighthouse works", "taking notes while reading", "the sound of rain",
    "storing food for winter", "a school science fair", "why rivers meander", "learning an instrument", "a quiet harbour",
    "how paper is made", "the first week in a new town", "why clocks needed standard time", "planting trees on a slope",
    "an old stone wall", "how bees find flowers", "reading aloud to children", "a workshop full of tools",
]

# ------------------------------------------------------------------ text views (deterministic, no NLP)


def sentences(text: str) -> list[str]:
    """Sentences: every line is split on terminal punctuation, and a line without any is one sentence."""
    out = []
    for line in text.strip().splitlines():
        for s in re.split(r"(?<=[.!?])\s+", line.strip()):
            if s.strip():
                out.append(s.strip())
    return out


def words(text: str) -> list[str]:
    return text.split()


def paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def bullet_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip().startswith("- ")]


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![\w']){re.escape(word)}(?![\w'])", text, re.IGNORECASE) is not None


# ------------------------------------------------------------------ the constraint types


@dataclass(frozen=True)
class ConstraintType:
    name: str
    group: str  # at most one constraint per group in a task
    describe: Callable[[dict], str]
    check: Callable[[str, dict], bool]


def _t(name, group, describe, check) -> ConstraintType:
    return ConstraintType(name, group, describe, check)


TYPES: dict[str, ConstraintType] = {c.name: c for c in [
    _t("sentences_exactly", "sentences", lambda s: f"Write exactly {s['n']} sentences.",
       lambda t, s: len(sentences(t)) == s["n"]),
    _t("sentences_min", "sentences", lambda s: f"Write at least {s['n']} sentences.",
       lambda t, s: len(sentences(t)) >= s["n"]),
    _t("sentences_max", "sentences", lambda s: f"Write at most {s['n']} sentences.",
       lambda t, s: len(sentences(t)) <= s["n"]),
    _t("words_min", "words", lambda s: f"Use at least {s['n']} words in total.",
       lambda t, s: len(words(t)) >= s["n"]),
    _t("words_max", "words", lambda s: f"Use at most {s['n']} words in total.",
       lambda t, s: len(words(t)) <= s["n"]),
    _t("contains_word", "contains_word", lambda s: f"The word \"{s['word']}\" must appear somewhere in your answer.",
       lambda t, s: _has_word(t, s["word"])),
    _t("contains_phrase", "contains_phrase", lambda s: f"Include the phrase \"{s['phrase']}\" somewhere in your answer.",
       lambda t, s: s["phrase"].lower() in t.lower()),
    _t("forbid_word", "forbid_word", lambda s: f"Do not use the word \"{s['word']}\" anywhere.",
       lambda t, s: not _has_word(t, s["word"])),
    _t("bullets_exactly", "bullets", lambda s: f"Answer as exactly {s['n']} bullet points, each on its own line starting with \"- \".",
       lambda t, s: len(bullet_lines(t)) == s["n"]),
    _t("all_lowercase", "case", lambda s: "Write your entire answer in lowercase letters only; do not use any capital letters.",
       lambda t, s: t == t.lower()),
    _t("no_commas", "commas", lambda s: "Do not use any commas.",
       lambda t, s: "," not in t),
    _t("title_angles", "title", lambda s: "Give your answer a title wrapped in double angle brackets, like <<Title Here>>.",
       lambda t, s: re.search(r"<<[^<>\n]+>>", t) is not None),
    _t("end_with", "end", lambda s: f"Finish with exactly this phrase, and nothing after it: \"{s['phrase']}\"",
       lambda t, s: t.rstrip().lower().endswith(s["phrase"].lower())),
    _t("start_with", "start", lambda s: f"Your answer must begin with the word \"{s['word']}\".",
       lambda t, s: bool(words(t)) and words(t)[0].strip(".,;:!?\"'").lower() == s["word"].lower()),
    _t("postscript", "ps", lambda s: "At the very end, add a postscript on its own line starting with \"P.S.\"",
       lambda t, s: any(ln.strip().startswith("P.S.") for ln in t.splitlines())),
    _t("paragraphs_exactly", "paragraphs", lambda s: f"Write exactly {s['n']} paragraphs, separated by a blank line.",
       lambda t, s: len(paragraphs(t)) == s["n"]),
    _t("number_between", "number", lambda s: f"Mention a number between {s['lo']} and {s['hi']}.",
       lambda t, s: any(s["lo"] <= int(m) <= s["hi"] for m in re.findall(r"-?\d+", t))),
    _t("sentence_words_max", "sentence_len", lambda s: f"Keep every sentence under {s['n']} words.",
       lambda t, s: bool(sentences(t)) and all(len(words(x)) < s["n"] for x in sentences(t))),
]}

# pairs of groups that cannot both be satisfied (or are hopelessly ambiguous together)
CONFLICTS = {frozenset(p) for p in [
    ("case", "ps"), ("case", "title"), ("end", "ps"),
    ("bullets", "sentences"), ("bullets", "paragraphs"), ("bullets", "start"), ("bullets", "title"),
    ("start", "title"), ("ps", "paragraphs"), ("ps", "sentences"), ("title", "paragraphs"), ("title", "sentences"),
]}


def describe(spec: dict) -> str:
    return TYPES[spec["type"]].describe(spec)


def check_one(text: str, spec: dict) -> bool:
    if not text.strip():
        return False  # an empty answer satisfies nothing, not even "at most N words"
    try:
        return bool(TYPES[spec["type"]].check(text, spec))
    except (KeyError, ValueError, TypeError):
        return False


def check_constraints(text: str, specs: list[dict]) -> tuple[int, list[str]]:
    """(number satisfied, names of the ones that failed)."""
    failed = [s["type"] for s in specs if not check_one(text, s)]
    return len(specs) - len(failed), failed


def parse_specs(gold: str) -> list[dict]:
    """The spec list from a task's gold string (JSON). [] when it is not a constraint gold."""
    try:
        v = json.loads(gold)
    except (json.JSONDecodeError, TypeError):
        return []
    return [s for s in v if isinstance(s, dict) and s.get("type") in TYPES] if isinstance(v, list) else []


# ------------------------------------------------------------------ generation


def _draw(rng: random.Random, name: str, corpus, used_words: list[str]) -> dict | None:
    if name == "sentences_exactly":
        return {"type": name, "n": rng.randint(2, 5)}
    if name == "sentences_min":
        return {"type": name, "n": rng.randint(3, 6)}
    if name == "sentences_max":
        return {"type": name, "n": rng.randint(2, 4)}
    if name == "words_min":
        return {"type": name, "n": rng.choice([25, 30, 40, 50, 60])}
    if name == "words_max":
        return {"type": name, "n": rng.choice([40, 50, 60, 80, 100])}
    if name in ("contains_word", "forbid_word", "start_with"):
        pool = [w for w in corpus.words if 3 <= len(w) <= 9]
        for _ in range(20):
            w = rng.choice(pool)
            if all(w != u and w not in u and u not in w for u in used_words):
                used_words.append(w)
                return {"type": name, "word": w}
        return None
    if name == "contains_phrase":
        s = rng.choice(corpus.sentences).split()
        if len(s) < 4:
            return None
        i = rng.randint(0, len(s) - 3)
        phrase = " ".join(s[i : i + rng.choice([2, 3])])
        used_words.append(phrase)
        return {"type": name, "phrase": phrase}
    if name == "end_with":
        phrase = rng.choice(["that is the whole story", "and that is enough for now", "nothing more to add",
                             "so the work goes on", "which is why it matters", "and there it ends"])
        used_words.append(phrase)
        return {"type": name, "phrase": phrase}
    if name == "bullets_exactly":
        return {"type": name, "n": rng.randint(2, 5)}
    if name in ("all_lowercase", "no_commas", "title_angles", "postscript"):
        return {"type": name}
    if name == "paragraphs_exactly":
        return {"type": name, "n": rng.randint(2, 3)}
    if name == "number_between":
        lo = rng.choice([1, 5, 10, 20, 50])
        return {"type": name, "lo": lo, "hi": lo + rng.choice([5, 10, 20, 50])}
    if name == "sentence_words_max":
        return {"type": name, "n": rng.choice([10, 12, 15, 18])}
    raise ValueError(name)


def _n(specs: list[dict], name: str, default):
    return next((s["n"] for s in specs if s["type"] == name), default)


def _feasible(specs: list[dict]) -> bool:
    """Reject combinations that no text can satisfy (60 words in at most 2 sentences of under 10 words,
    a forbidden word that another constraint requires). Groups already keep out the direct contradictions."""
    max_sent = min(_n(specs, "sentences_exactly", 10**6), _n(specs, "sentences_max", 10**6))
    min_sent = max(_n(specs, "sentences_exactly", 0), _n(specs, "sentences_min", 0))
    per_sent = _n(specs, "sentence_words_max", 31) - 1
    if _n(specs, "words_min", 0) > max_sent * per_sent:
        return False
    w_max = _n(specs, "words_max", 10**6)
    if min_sent * 3 > w_max or _n(specs, "bullets_exactly", 0) * 3 > w_max or _n(specs, "paragraphs_exactly", 0) * 4 > w_max:
        return False
    forbidden = [s["word"] for s in specs if s["type"] == "forbid_word"]
    required = [s.get("word") or s.get("phrase", "") for s in specs if s["type"] in ("contains_word", "contains_phrase", "start_with", "end_with")]
    return not any(_has_word(r, w) for w in forbidden for r in required)


_INTROS = ["Write a short note about {t}.", "Write a few sentences about {t}.", "Write a short piece about {t}.",
           "Describe {t} for a reader who has never seen one.", "Explain {t} in plain English."]
_QUOTE_INTROS = ["Here is a line from a book: \"{s}\" Write a short comment on it.",
                 "Someone wrote: \"{s}\" Reply to that in your own words.",
                 "Take this sentence: \"{s}\" Write a short paragraph that follows on from it."]


def gen_constraints(rng: random.Random, n_constraints: int | None = None) -> Task | None:
    """A writing prompt plus 1-3 constraints stated in English and checkable from the spec."""
    corpus = task_corpus()
    if rng.random() < 0.25:
        intro = rng.choice(_QUOTE_INTROS).format(s=rng.choice(corpus.sentences))
    else:
        intro = rng.choice(_INTROS).format(t=rng.choice(TOPICS))
    k = n_constraints or rng.choices([1, 2, 3], weights=[0.25, 0.45, 0.30])[0]
    specs: list[dict] = []
    groups: set[str] = set()
    used_words: list[str] = []
    names = list(TYPES)
    rng.shuffle(names)
    for name in names:
        if len(specs) >= k:
            break
        g = TYPES[name].group
        if g in groups or any(frozenset((g, h)) in CONFLICTS for h in groups):
            continue
        spec = _draw(rng, name, corpus, used_words)
        if spec is None or not _feasible([*specs, spec]):
            continue
        specs.append(spec)
        groups.add(g)
    if not specs:
        return None
    rules = "\n".join(f"{i + 1}. {describe(s)}" for i, s in enumerate(specs))
    prompt = f"{intro}\nFollow these rules exactly:\n{rules}"
    gold = json.dumps(specs, sort_keys=True)
    return Task(id="", prompt=prompt, answer=gold, task="constraints",
                meta={"verifier": "constraints", "answer_style": "free", "constraints": specs, "numeric": False})

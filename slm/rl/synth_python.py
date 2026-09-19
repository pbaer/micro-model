"""Grammar-generated Python-tool conversations: the tool tags run *real* Python, not a calculator.

Every earlier tool set taught `<|python_call|>` as a one-line calculator (no loop, no `def`, no list, no
string method, no `math.` in 87K call spans). The sandbox (`slm/tools/pysandbox.py`) supports all of those,
state persists across calls and turns, declared functions are callable, and errors come back as readable
hints. This module writes an SFT set that teaches exactly that, from small grammars rather than templates:

    pipeline    produce data -> 1-3 transforms -> aggregate, several idioms per step
    strings     reverse / palindrome / counts / capitalize / replace / longest word / acronym
    numbers     gcd, lcm, primes, digit sums, factorials, powers, fibonacci, collatz, base conversion
    simulation  loops with state: doubling, interest, inventories, scoring rules
    multiturn   define a helper or a variable, reuse it across 2-4 turns (never recompute)
    runcode     "run this and tell me what it prints": the user's program, executed verbatim
    error       a call that really trips a sandbox hint, the hint read, then a corrected call
    declared    1-3 `FunctionDecl` capabilities (lookups, conversions, services); some are not needed
    arith       the plain word-problem families from slm.rl.synth, at a minority share

Correct by construction: every program is run in a `PySession` while generating (the gold answer is
cross-checked against an independently computed host value and the sample is dropped on any mismatch),
and `format_chat(tools=True)` runs it again at conversion time, so the recorded result span is exactly
what inference would insert.

Two whole families are written to the validation split only, so val measures generalisation to unseen
program shapes: `pipeline.dict` and `declared.distance` (see HOLDOUT_FAMILIES).

    python -m slm.rl.synth_python --tokenizer C:/slm-data/tokenizer/v1 --n 100000 --name synthetic-python-tools --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from slm.data.answers import apply_style
from slm.data.chat import format_chat
from slm.data.sft import SFT_DIR, SftShardWriter
from slm.data.sources import DATA_ROOT
from slm.data.tokenizer import SlmTokenizer
from slm.rl.tasks import _split_of
from slm.tools.functions import FunctionDecl, functions_env
from slm.tools.protocol import ToolSpan, encode_tool_span, run_tool, split_markup
from slm.tools.pysandbox import PySession
from slm.tools.pysandbox import _fmt as render_value  # the sandbox's own print formatting: gold strings must match it

FAMILY_SHARES = {"pipeline": 0.235, "strings": 0.12, "numbers": 0.12, "simulation": 0.08, "multiturn": 0.12,
                 "runcode": 0.08, "error": 0.05, "declared": 0.145, "arith": 0.05}
HOLDOUT_FAMILIES = ("pipeline.dict", "declared.distance")  # written to val only
DEFAULT_CORPUS = DATA_ROOT / "tokenized" / "v1" / "fineweb-edu-b" / "val"


# --------------------------------------------------------------------------------- real-text ingredients
_FALLBACK_SENTENCES = [
    "the river carried silt down from the mountains every spring",
    "scientists measured the temperature of the water at three depths",
    "a small library opened in the village hall last autumn",
    "children learn to read by hearing stories again and again",
    "the engine burns fuel and turns the wheels of the train",
    "farmers rotate crops to keep the soil rich and healthy",
    "a printing press made books cheap enough for ordinary people",
    "the moon pulls the ocean and creates the daily tides",
    "students practice writing short essays about what they read",
    "birds migrate south before the first hard frost arrives",
    "an old bridge crosses the narrow stream near the mill",
    "the teacher drew a simple diagram of the water cycle",
    "iron rusts when it is left out in the damp air",
    "bees carry pollen from one flower to the next one",
    "the town council voted to repair the public road",
    "light travels faster than sound through the open air",
    "a map shows rivers roads and the height of the land",
    "the baker starts work long before the sun comes up",
    "glaciers grind rock into fine powder as they move",
    "people once told the time by watching shadows move",
    "salt water freezes at a lower temperature than fresh water",
    "the museum keeps letters written more than a century ago",
    "a seed needs warmth water and light before it sprouts",
    "the harbour was busy with fishing boats every morning",
    "electric wires carry power from the dam to the city",
    "wind shapes the dunes along the northern coast",
    "a good question is often better than a quick answer",
    "the class measured the shadow of a pole at noon",
    "paper is made from wood fibre pressed into thin sheets",
    "mountains rise where two plates push against each other",
    "the doctor wrote careful notes about every patient",
    "young trees grow quickly in the gaps left by storms",
    "clocks in the station were checked against the telegraph",
    "rain soaks into the ground and feeds the hidden springs",
    "a lens bends light and brings the image into focus",
    "the miller ground grain into flour for the whole valley",
    "sailors used the stars to find their way at night",
    "cold air holds far less water than warm air does",
    "the road follows the valley floor for many miles",
    "a magnet attracts iron but it ignores copper and glass",
]
_GOODS = ["apples", "bolts", "cables", "crates", "lamps", "mugs", "nails", "pears", "plates", "ropes", "seeds", "tiles", "towels", "valves"]


@dataclass
class Corpus:
    sentences: list[str]
    words: list[str]


def _clean_sentences(text: str) -> list[str]:
    out, seen = [], set()
    for raw in re.split(r"(?<=[.!?])\s+", text):
        ws = [w.lower() for w in re.findall(r"[A-Za-z]+", raw)]
        if not (6 <= len(ws) <= 14) or any(len(w) > 12 for w in ws):
            continue
        s = " ".join(ws)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= 4000:
            break
    return out


def load_corpus(tok: SlmTokenizer | None = None, path: Path | None = None, n_tokens: int = 400_000) -> Corpus:
    """Sentences and words drawn from real text (a tokenized val split) so inputs look like prose. Falls
    back to a small built-in list when the corpus is not on this machine (tests, other checkouts)."""
    sents: list[str] = []
    if tok is not None and path is not None and Path(path).exists():
        try:
            from slm.data.loader import TokenStream

            ts = TokenStream(Path(path))
            sents = _clean_sentences(tok.decode([int(i) for i in ts.next_window(n_tokens)], skip_special=True))
        except Exception:  # noqa: BLE001 - ingredients are a nicety; never fail the build over them
            sents = []
    if len(sents) < 200:
        sents = list(_FALLBACK_SENTENCES)
    words = sorted({w for s in sents for w in s.split() if 3 <= len(w) <= 11})
    return Corpus(sents, words)


# --------------------------------------------------------------------------------- small helpers
def lit(v) -> str:
    """Python literal for a value, as the question and the program both show it."""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return '"' + v + '"'
    if isinstance(v, list):
        return "[" + ", ".join(lit(x) for x in v) + "]"
    if isinstance(v, tuple):
        return "(" + ", ".join(lit(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{lit(k)}: {lit(x)}" for k, x in v.items()) + "}"
    if isinstance(v, float):
        return repr(round(v, 4))
    return str(v)


class Names:
    """Varied, non-colliding variable names."""

    def __init__(self, rng: random.Random) -> None:
        self.rng, self.used = rng, set()

    def pick(self, pool: list[str]) -> str:
        free = [n for n in pool if n not in self.used]
        n = self.rng.choice(free) if free else f"{pool[0]}{len(self.used)}"
        self.used.add(n)
        return n


NUM_VARS = ["nums", "values", "data", "xs", "numbers", "readings", "scores", "items"]
OUT_VARS = ["kept", "picked", "sel", "chosen", "result", "out", "vals2", "step1"]
WORD_VARS = ["words", "tokens", "parts", "wlist", "terms"]
ACC_VARS = ["total", "acc", "answer", "s", "amount"]

_KEEP = set("""and as assert break class continue def del elif else except finally for from global if import in is lambda
not or pass raise return try while with yield True False None print len sum min max abs round sorted range list dict str
int float bool zip enumerate reversed tuple divmod pow math append sort index count pop extend insert remove reverse copy
upper lower strip lstrip rstrip split join replace startswith endswith isdigit isalpha find zfill title capitalize keys
values items get update sqrt floor ceil log log2 log10 exp gcd isqrt fabs trunc factorial pi e""".split())


def skeleton(code: str) -> str:
    """Program shape with numbers, strings and identifiers normalised: the diversity unit."""
    s = re.sub(r'"[^"\n]*"|\'[^\'\n]*\'', "S", code)
    s = re.sub(r"#[^\n]*", "", s)
    s = re.sub(r"\b\d+(?:\.\d+)?\b", "N", s)
    s = re.sub(r"\b[A-Za-z_]\w*\b", lambda m: m.group(0) if m.group(0) in _KEEP else "V", s)
    return re.sub(r"\s+", " ", s).strip()


_FEATURE_RES = {
    "loop": re.compile(r"\b(?:for|while)\b"),
    "def": re.compile(r"\bdef\b"),
    "math": re.compile(r"\bmath\."),
    "str_methods": re.compile(r"\.(?:split|join|upper|lower|strip|lstrip|rstrip|replace|startswith|endswith|isdigit|isalpha|find|zfill|title|capitalize)\("),
    "list_ops": re.compile(r"\.(?:append|sort|index|pop|extend|insert|remove|reverse)\(|\bsorted\(|\blist\(|\bzip\(|\benumerate\(|\breversed\(|\[[^\]\n]*\bfor\b[^\]\n]*\]|\[[^\]\n]*,"),
}


def features_of(codes: list[str], decl_names: list[str]) -> set[str]:
    blob = "\n".join(codes)
    feats = {k for k, r in _FEATURE_RES.items() if r.search(blob)}
    if any(re.search(rf"\b{re.escape(n)}\s*\(", blob) for n in decl_names):
        feats.add("decl_call")
    return feats


@dataclass
class Sample:
    """One conversation, ready for `encode_sample`."""

    family: str  # "pipeline.ints", "declared.catalogue", ...
    messages: list[dict]  # assistant turns carry "think" (markup) or "spans" (explicit call list)
    key: str  # dedup / split key
    decls: list[FunctionDecl] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)  # every program, for features and skeletons
    numeric: bool = True


def make_sample(family: str, msgs: list[dict], codes: list[str], decls: list[FunctionDecl] | None = None, numeric: bool = True) -> Sample:
    key = next(m["content"] for m in msgs if m["role"] == "user")
    return Sample(family, msgs, key, list(decls or []), codes, numeric)


# --------------------------------------------------------------------------------- encoding
def _turn_ids(tok: SlmTokenizer, spans: list[tuple[str, str]], content: str, session: PySession) -> list[int]:
    """Ids of one assistant turn whose think span is given as explicit ("text"|"call", body) spans.

    `format_chat`'s `<<...>>` markup path drops a *failing* call back to plain text (a dataset annotation
    the sandbox cannot reproduce must not become a call), so error-and-recover traces, which need the real
    `error: ...` hint inside a real call/result pair, are built here instead. `encode_sample` masks every
    result span afterwards, exactly as the markup path does.
    """
    ids = [tok.special("<|think|>")]
    for kind, body in spans:
        if kind == "text":
            ids += tok.encode(body)
        else:
            res, _ok = run_tool(body, session)
            call, result, _ = encode_tool_span(tok, ToolSpan("tool", code=body, result=res))
            ids += call + result
    return ids + [tok.special("<|/think|>"), *tok.encode(content), tok.end_id]


def encode_sample(tok: SlmTokenizer, s: Sample):
    """One conversation -> ChatEncoding. One `PySession` for the whole conversation, declared impls
    registered in it, and the invariant that no `<|python_result|>` token is ever a loss target."""
    session = PySession()
    if s.decls:
        session.register(functions_env(s.decls))
    msgs = []
    for m in s.messages:
        if m["role"] == "assistant" and "spans" in m:
            msgs.append({"role": "assistant", "ids": _turn_ids(tok, m["spans"], m["content"], session)})
        else:
            msgs.append(m)
    enc = format_chat(tok, msgs, think_required=True, tools=True, session=session, functions=s.decls or None)
    r_open, r_close = tok.special("<|python_result|>"), tok.special("<|/python_result|>")
    inside = False
    for i, t in enumerate(enc.ids):
        if t == r_open:
            inside = True
        if inside:
            enc.loss_mask[i] = 0
        if t == r_close:
            inside = False
    return enc


def verify(code: str, gold: str, session: PySession | None = None) -> bool:
    """The sandbox must reproduce the gold exactly; used while generating (drop on mismatch)."""
    out, ok = run_tool(code, session)
    return ok and out == gold


# --------------------------------------------------------------------------------- pipelines
def _maybe_comment(rng: random.Random, text: str) -> list[str]:
    return [f"# {text}"] if rng.random() < 0.22 else []


def src_ints(rng, nm, corpus):
    vals = [rng.randint(1, 60) for _ in range(rng.randint(8, 18))]
    v = nm.pick(NUM_VARS)
    intro = rng.choice([f"Here are some numbers: {lit(vals)}.", f"I have this list: {lit(vals)}.",
                        f"Take the list {lit(vals)}.", f"Given the numbers {lit(vals)}:"])
    return [f"{v} = {lit(vals)}"], v, vals, "nums", intro


def src_floats(rng, nm, corpus):
    vals = [round(rng.uniform(1.0, 60.0), 2) for _ in range(rng.randint(6, 12))]
    v = nm.pick(["prices", "costs", "amounts", "weights", "measures"])
    intro = rng.choice([f"These are the prices in dollars: {lit(vals)}.", f"I measured {lit(vals)}.",
                        f"Here is a list of amounts: {lit(vals)}."])
    return [f"{v} = {lit(vals)}"], v, vals, "nums", intro


def src_range(rng, nm, corpus):
    a = rng.randint(1, 30)
    b = a + rng.randint(8, 25)
    step = rng.choice([1, 1, 1, 2, 3])
    vals = list(range(a, b + 1, step))
    v = nm.pick(NUM_VARS)
    code = f"{v} = list(range({a}, {b + 1}{f', {step}' if step != 1 else ''}))"
    intro = (f"Consider the whole numbers from {a} to {b}." if step == 1
             else f"Consider every {step}rd number from {a} up to {b}." if step == 3 else f"Consider every second number from {a} up to {b}.")
    return [code], v, vals, "nums", intro


def src_words(rng, nm, corpus):
    vals = rng.sample(corpus.words, rng.randint(7, 14))
    v = nm.pick(WORD_VARS)
    intro = rng.choice([f"Here are some words: {lit(vals)}.", f"Take this word list: {lit(vals)}.",
                        f"I have the words {lit(vals)}."])
    return [f"{v} = {lit(vals)}"], v, vals, "words", intro


def src_text(rng, nm, corpus):
    s = rng.choice(corpus.sentences)
    v = nm.pick(["text", "sentence", "line", "phrase"])
    intro = rng.choice([f'Take this sentence: "{s}".', f'Here is a line of text: "{s}".', f'Consider the sentence "{s}".'])
    return [f"{v} = {lit(s)}"], v, s, "text", intro


def src_dict(rng, nm, corpus):
    keys = rng.sample(_GOODS, rng.randint(4, 7))
    d = {k: rng.randint(2, 90) for k in keys}
    v = nm.pick(["stock", "inventory", "counts", "shelf"])
    intro = rng.choice([f"My inventory is {lit(d)}.", f"The stock counts are {lit(d)}.", f"Here is the shelf: {lit(d)}."])
    return [f"{v} = {lit(d)}"], v, d, "dict", intro


SOURCES = {"ints": src_ints, "floats": src_floats, "range": src_range, "words": src_words, "text": src_text, "dict": src_dict}
SOURCE_WEIGHTS = {"ints": 0.24, "floats": 0.11, "range": 0.12, "words": 0.24, "text": 0.17, "dict": 0.12}


# ---- transforms: (rng, nm, src, val) -> (lines, new_val, kind, phrase) or None when degenerate
def _loop_filter(dst, src, cond):
    return [f"{dst} = []", f"for x in {src}:", f"    if {cond}:", f"        {dst}.append(x)"]


def tr_filter_even(rng, nm, src, val):
    odd = rng.random() < 0.4
    keep = [x for x in val if (x % 2 == 1) == odd]
    if len(keep) < 2:
        return None
    dst = nm.pick(OUT_VARS)
    cond = f"x % 2 == {1 if odd else 0}"
    lines = [f"{dst} = [x for x in {src} if {cond}]"] if rng.random() < 0.5 else _loop_filter(dst, src, cond)
    return lines, keep, "nums", f"keep only the {'odd' if odd else 'even'} ones"


def tr_filter_gt(rng, nm, src, val):
    k = sorted(val)[len(val) // 3]
    keep = [x for x in val if x > k]
    if len(keep) < 2 or len(keep) == len(val):
        return None
    dst = nm.pick(OUT_VARS)
    lines = [f"{dst} = [x for x in {src} if x > {lit(k)}]"] if rng.random() < 0.55 else _loop_filter(dst, src, f"x > {lit(k)}")
    return lines, keep, "nums", f"drop everything that is {lit(k)} or less"


def tr_filter_div(rng, nm, src, val):
    if any(isinstance(x, float) for x in val):
        return None
    d = rng.choice([3, 4, 5, 7])
    keep = [x for x in val if x % d == 0]
    if len(keep) < 2:
        return None
    dst = nm.pick(OUT_VARS)
    lines = [f"{dst} = [x for x in {src} if x % {d} == 0]"] if rng.random() < 0.5 else _loop_filter(dst, src, f"x % {d} == 0")
    return lines, keep, "nums", f"keep the multiples of {d}"


def tr_map_scale(rng, nm, src, val):
    k = rng.choice([2, 3, 4, 10])
    out = [x * k for x in val]
    dst = nm.pick(OUT_VARS)
    word = {2: "double", 3: "triple", 4: "quadruple", 10: "multiply by ten"}[k]
    if rng.random() < 0.5:
        lines = [f"{dst} = [x * {k} for x in {src}]"]
    else:
        lines = [f"{dst} = []", f"for x in {src}:", f"    {dst}.append(x * {k})"]
    return lines, out, "nums", f"{word} each value"


def tr_map_add(rng, nm, src, val):
    k = rng.randint(2, 25)
    out = [x + k for x in val]
    dst = nm.pick(OUT_VARS)
    if rng.random() < 0.5:
        lines = [f"{dst} = [x + {k} for x in {src}]"]
    else:
        lines = [f"{dst} = []", f"for x in {src}:", f"    {dst}.append(x + {k})"]
    return lines, out, "nums", f"add {k} to every value"


def tr_map_square(rng, nm, src, val):
    if any(abs(x) > 300 for x in val):
        return None
    out = [x * x for x in val]
    dst = nm.pick(OUT_VARS)
    lines = [f"{dst} = [x * x for x in {src}]"] if rng.random() < 0.6 else [f"{dst} = [x ** 2 for x in {src}]"]
    return lines, out, "nums", "square each of them"


def tr_map_round(rng, nm, src, val):
    if not any(isinstance(x, float) for x in val):
        return None
    out = [round(x) for x in val]
    dst = nm.pick(OUT_VARS)
    lines = [f"{dst} = [round(x) for x in {src}]"]
    return lines, out, "nums", "round each one to a whole number"


def tr_slice(rng, nm, src, val):
    if len(val) < 4:
        return None
    k = rng.randint(2, len(val) - 1)
    last = rng.random() < 0.4
    out = val[-k:] if last else val[:k]
    dst = nm.pick(OUT_VARS)
    lines = [f"{dst} = {src}[-{k}:]" if last else f"{dst} = {src}[:{k}]"]
    return lines, out, "nums" if not isinstance(val[0], str) else "words", f"take the {'last' if last else 'first'} {k}"


def tr_sort(rng, nm, src, val):
    desc = rng.random() < 0.45
    out = sorted(val)[::-1] if desc else sorted(val)
    dst = nm.pick(OUT_VARS)
    lines = [f"{dst} = sorted({src})[::-1]"] if desc else [f"{dst} = sorted({src})"]
    kind = "words" if isinstance(val[0], str) else "nums"
    return lines, out, kind, "sort them from " + ("largest to smallest" if desc else "smallest to largest") if kind == "nums" else ("sort them in reverse alphabetical order" if desc else "sort them alphabetically")


def tr_reverse(rng, nm, src, val):
    out = list(val)[::-1]
    dst = nm.pick(OUT_VARS)
    lines = [f"{dst} = {src}[::-1]"] if rng.random() < 0.6 else [f"{dst} = list(reversed({src}))"]
    return lines, out, "words" if isinstance(val[0], str) else "nums", "reverse the order"


def tr_unique(rng, nm, src, val):
    if len(val) == len(set(val)):
        return None
    out = []
    for x in val:
        if x not in out:
            out.append(x)
    dst = nm.pick(OUT_VARS)
    lines = [f"{dst} = []", f"for x in {src}:", f"    if x not in {dst}:", f"        {dst}.append(x)"]
    return lines, out, "words" if isinstance(val[0], str) else "nums", "remove the duplicates"


def tr_filter_len(rng, nm, src, val):
    k = rng.randint(3, 6)
    keep = [w for w in val if len(w) > k]
    if len(keep) < 2 or len(keep) == len(val):
        return None
    dst = nm.pick(OUT_VARS)
    lines = [f"{dst} = [w for w in {src} if len(w) > {k}]"] if rng.random() < 0.5 else [f"{dst} = []", f"for w in {src}:", f"    if len(w) > {k}:", f"        {dst}.append(w)"]
    return lines, keep, "words", f"keep only the words longer than {k} letters"


def tr_filter_startswith(rng, nm, src, val):
    counts = Counter(w[0] for w in val)
    good = [c for c, n in counts.items() if n >= 2]
    if not good:
        return None
    c = rng.choice(good)
    keep = [w for w in val if w.startswith(c)]
    dst = nm.pick(OUT_VARS)
    lines = [f'{dst} = [w for w in {src} if w.startswith("{c}")]'] if rng.random() < 0.5 else [f"{dst} = []", f"for w in {src}:", f'    if w.startswith("{c}"):', f"        {dst}.append(w)"]
    return lines, keep, "words", f'keep the words that start with "{c}"'


def tr_map_upper(rng, nm, src, val):
    out = [w.upper() for w in val]
    dst = nm.pick(OUT_VARS)
    lines = [f"{dst} = [w.upper() for w in {src}]"] if rng.random() < 0.6 else [f"{dst} = []", f"for w in {src}:", f"    {dst}.append(w.upper())"]
    return lines, out, "words", "put them in capitals"


def tr_map_capitalize(rng, nm, src, val):
    out = [w.capitalize() for w in val]
    dst = nm.pick(OUT_VARS)
    lines = [f"{dst} = [w.capitalize() for w in {src}]"] if rng.random() < 0.5 else [f"{dst} = [w[0].upper() + w[1:] for w in {src}]"]
    return lines, out, "words", "capitalise each word"


def tr_map_len(rng, nm, src, val):
    out = [len(w) for w in val]
    dst = nm.pick(NUM_VARS)
    lines = [f"{dst} = [len(w) for w in {src}]"] if rng.random() < 0.55 else [f"{dst} = []", f"for w in {src}:", f"    {dst}.append(len(w))"]
    return lines, out, "nums", "replace each word by its length"


def tr_split(rng, nm, src, val):
    out = val.split()
    dst = nm.pick(WORD_VARS)
    return [f"{dst} = {src}.split()"], out, "words", "split it into words"


def tr_replace(rng, nm, src, val):
    cand = [c for c in "aeiourstn" if c in val]
    if not cand:
        return None
    a = rng.choice(cand)
    b = rng.choice("xyz*-")
    out = val.replace(a, b)
    dst = nm.pick(["text2", "changed", "edited", "fixed"])
    return [f'{dst} = {src}.replace("{a}", "{b}")'], out, "text", f'replace every "{a}" with "{b}"'


def tr_upper_text(rng, nm, src, val):
    out = val.upper()
    dst = nm.pick(["shout", "caps", "loud"])
    return [f"{dst} = {src}.upper()"], out, "text", "put it in capitals"


def tr_dict_values(rng, nm, src, val):
    out = list(val.values())
    dst = nm.pick(NUM_VARS)
    lines = [f"{dst} = list({src}.values())"] if rng.random() < 0.55 else [f"{dst} = []", f"for k in {src}:", f"    {dst}.append({src}[k])"]
    return lines, out, "nums", "look at the counts"


def tr_dict_keys(rng, nm, src, val):
    out = list(val.keys())
    dst = nm.pick(WORD_VARS)
    lines = [f"{dst} = list({src}.keys())"] if rng.random() < 0.6 else [f"{dst} = []", f"for k in {src}:", f"    {dst}.append(k)"]
    return lines, out, "words", "look at the item names"


def tr_dict_filter(rng, nm, src, val):
    k = sorted(val.values())[len(val) // 2]
    out = [name for name in val if val[name] >= k]
    if len(out) < 2 or len(out) == len(val):
        return None
    dst = nm.pick(WORD_VARS)
    lines = [f"{dst} = []", f"for name in {src}:", f"    if {src}[name] >= {k}:", f"        {dst}.append(name)"]
    return lines, out, "words", f"keep the items with at least {k} in stock"


NUM_TRANSFORMS = [tr_filter_even, tr_filter_gt, tr_filter_div, tr_map_scale, tr_map_add, tr_map_square, tr_map_round, tr_slice, tr_sort, tr_reverse, tr_unique]
WORD_TRANSFORMS = [tr_filter_len, tr_filter_startswith, tr_map_upper, tr_map_capitalize, tr_map_len, tr_slice, tr_sort, tr_reverse, tr_unique]
TEXT_TRANSFORMS = [tr_split, tr_replace, tr_upper_text]
DICT_TRANSFORMS = [tr_dict_values, tr_dict_keys, tr_dict_filter]
TRANSFORMS = {"nums": NUM_TRANSFORMS, "words": WORD_TRANSFORMS, "text": TEXT_TRANSFORMS, "dict": DICT_TRANSFORMS}


# ---- aggregates: (rng, nm, src, val) -> (lines, gold_value, question) or None
def ag_sum(rng, nm, src, val):
    total = sum(val)
    is_float = isinstance(total, float)
    if is_float:
        total = round(total, 2)
    acc = nm.pick(ACC_VARS)
    tail = f"round({acc}, 2)" if is_float else acc
    r = rng.random()
    if r < 0.4:
        lines = [f"{acc} = sum({src})", tail]
    elif r < 0.8:
        lines = [f"{acc} = 0", f"for x in {src}:", f"    {acc} += x", tail]
    else:
        lines = [f"print(round(sum({src}), 2))" if is_float else f"print(sum({src}))"]
    return lines, total, rng.choice(["What is the total?", "What do they add up to?", "Give me the sum.", "How much is that altogether?"])


def ag_max(rng, nm, src, val):
    biggest = rng.random() < 0.6
    v = max(val) if biggest else min(val)
    acc = nm.pick(["best", "top", "peak", "winner"])
    if rng.random() < 0.5:
        lines = [f"{'max' if biggest else 'min'}({src})"]
    else:
        cmp = ">" if biggest else "<"
        lines = [f"{acc} = {src}[0]", f"for x in {src}:", f"    if x {cmp} {acc}:", f"        {acc} = x", acc]
    return lines, v, rng.choice([f"What is the {'largest' if biggest else 'smallest'} one?", f"Which is the {'biggest' if biggest else 'smallest'}?"])


def ag_len(rng, nm, src, val):
    n = len(val)
    lines = [f"len({src})"] if rng.random() < 0.6 else [f"print(len({src}))"]
    return lines, n, rng.choice(["How many are left?", "How many are there?", "Give me the count."])


def ag_mean(rng, nm, src, val):
    if not val:
        return None
    m = round(sum(val) / len(val), 2)
    acc = nm.pick(["mean", "average", "avg"])
    r = rng.random()
    if r < 0.25:
        m = math.floor(sum(val) / len(val))
        lines = [f"{acc} = math.floor(sum({src}) / len({src}))", acc]
        return lines, m, "What is the average, rounded down to a whole number?"
    if r < 0.6:
        lines = [f"{acc} = round(sum({src}) / len({src}), 2)", acc]
    else:
        lines = [f"{acc} = 0", f"for x in {src}:", f"    {acc} += x", f"round({acc} / len({src}), 2)"]
    return lines, m, rng.choice(["What is the average?", "What is the mean value?", "Give me the average, rounded to two decimals."])


def ag_count(rng, nm, src, val):
    if isinstance(val[0], str):
        k = rng.randint(3, 6)
        n = len([w for w in val if len(w) > k])
        if n == 0:
            return None
        acc = nm.pick(["n", "count", "hits"])
        if rng.random() < 0.5:
            lines = [f"len([w for w in {src} if len(w) > {k}])"]
        else:
            lines = [f"{acc} = 0", f"for w in {src}:", f"    if len(w) > {k}:", f"        {acc} += 1", acc]
        return lines, n, f"How many of them are longer than {k} letters?"
    k = sorted(val)[len(val) // 2]
    n = len([x for x in val if x > k])
    if n == 0:
        return None
    acc = nm.pick(["n", "count", "hits"])
    if rng.random() < 0.5:
        lines = [f"len([x for x in {src} if x > {lit(k)}])"]
    else:
        lines = [f"{acc} = 0", f"for x in {src}:", f"    if x > {lit(k)}:", f"        {acc} += 1", acc]
    return lines, n, f"How many of them are bigger than {lit(k)}?"


def ag_any(rng, nm, src, val):
    if isinstance(val[0], str):
        return None
    d = rng.choice([3, 4, 5, 6, 7])
    truth = any(isinstance(x, int) and x % d == 0 for x in val)
    if any(isinstance(x, float) for x in val):
        return None
    flag = nm.pick(["found", "seen", "ok", "hit"])
    lines = [f"{flag} = False", f"for x in {src}:", f"    if x % {d} == 0:", f"        {flag} = True", flag]
    return lines, truth, f"Is any of them a multiple of {d}?"


def ag_all(rng, nm, src, val):
    if isinstance(val[0], str) or any(isinstance(x, float) for x in val):
        return None
    k = min(val) - rng.randint(0, 3)
    truth = all(x >= k for x in val)
    flag = nm.pick(["ok", "allbig", "fine", "good"])
    lines = [f"{flag} = True", f"for x in {src}:", f"    if x < {k}:", f"        {flag} = False", flag]
    return lines, truth, f"Are they all at least {k}?"


def ag_first(rng, nm, src, val):
    if isinstance(val[0], str):
        return None
    k = sorted(val)[len(val) // 2]
    first = next((x for x in val if x > k), None)
    if first is None:
        return None
    acc = nm.pick(["first", "found", "hit"])
    lines = [f"{acc} = 0", f"for x in {src}:", f"    if x > {lit(k)}:", f"        {acc} = x", "        break", acc]
    return lines, first, f"What is the first value bigger than {lit(k)}?"


def ag_product(rng, nm, src, val):
    if isinstance(val[0], str) or any(isinstance(x, float) for x in val) or len(val) > 6:
        return None
    p = 1
    for x in val:
        p *= x
    if p > 10**12:
        return None
    acc = nm.pick(["prod", "product", "p"])
    lines = [f"{acc} = 1", f"for x in {src}:", f"    {acc} *= x", acc]
    return lines, p, "What is the product of all of them?"


def ag_longest(rng, nm, src, val):
    if not isinstance(val[0], str):
        return None
    best = max(val, key=len)
    if len([w for w in val if len(w) == len(best)]) > 1:
        return None
    acc = nm.pick(["longest", "best", "winner"])
    lines = [f"{acc} = {src}[0]", f"for w in {src}:", f"    if len(w) > len({acc}):", f"        {acc} = w", acc]
    return lines, best, rng.choice(["Which is the longest word?", "What is the longest one?"])


def ag_join(rng, nm, src, val):
    if not isinstance(val[0], str):
        return None
    sep = rng.choice([" ", "-", ", ", "|"])
    out = sep.join(val)
    lines = [f"{lit(sep)}.join({src})"]
    return lines, out, f'Join them with "{sep}" and show me the result.' if sep != " " else "Put them back together into one line."


def ag_alpha_first(rng, nm, src, val):
    if not isinstance(val[0], str):
        return None
    out = sorted(val)[0]
    if rng.random() < 0.5:
        lines = [f"sorted({src})[0]"]
    else:
        acc = nm.pick(["first", "top", "head"])
        lines = [f"{acc} = {src}[0]", f"for w in {src}:", f"    if w < {acc}:", f"        {acc} = w", acc]
    return lines, out, "Which one comes first alphabetically?"


def ag_text_len(rng, nm, src, val):
    n = len(val)
    return [f"len({src})"], n, "How many characters is that?"


def ag_word_count(rng, nm, src, val):
    n = len(val.split())
    if rng.random() < 0.6:
        lines = [f"len({src}.split())"]
    else:
        acc = nm.pick(["n", "count", "words_n"])
        lines = [f"{acc} = 0", f"for w in {src}.split():", f"    {acc} += 1", acc]
    return lines, n, rng.choice(["How many words is that?", "Count the words for me."])


def ag_letter_count(rng, nm, src, val):
    cand = [c for c in "aeiourstn" if val.count(c) >= 2]
    if not cand:
        return None
    c = rng.choice(cand)
    n = val.count(c)
    acc = nm.pick(["n", "count", "hits"])
    if rng.random() < 0.5:
        lines = [f'{src}.count("{c}")']
    else:
        lines = [f"{acc} = 0", f"for ch in {src}:", f'    if ch == "{c}":', f"        {acc} += 1", acc]
    return lines, n, f'How many times does the letter "{c}" appear?'


def ag_show(rng, nm, src, val):
    """Just show the transformed value (the pipeline itself is the answer)."""
    lines = [src] if rng.random() < 0.6 else [f"print({src})"]
    return lines, val, rng.choice(["What do you get?", "Show me the result.", "What is the list then?"])


def ag_rms(rng, nm, src, val):
    if any(abs(x) > 5000 for x in val):
        return None
    m = round(math.sqrt(sum(x * x for x in val) / len(val)), 2)
    acc = nm.pick(["sq", "squares", "ss"])
    if rng.random() < 0.5:
        lines = [f"{acc} = 0", f"for x in {src}:", f"    {acc} += x * x", f"round(math.sqrt({acc} / len({src})), 2)"]
    else:
        lines = [f"{acc} = sum([x * x for x in {src}])", f"round(math.sqrt({acc} / len({src})), 2)"]
    return lines, m, rng.choice(["What is the root mean square?", "Give me the quadratic mean, to two decimals."])


NUM_AGGS = [ag_sum, ag_max, ag_len, ag_mean, ag_count, ag_any, ag_all, ag_first, ag_product, ag_rms, ag_show]
WORD_AGGS = [ag_len, ag_longest, ag_join, ag_alpha_first, ag_count, ag_show]
TEXT_AGGS = [ag_text_len, ag_word_count, ag_letter_count]
AGGS = {"nums": NUM_AGGS, "words": WORD_AGGS, "text": TEXT_AGGS, "dict": []}

THINK_OPENERS = ["I'll work this out in Python.", "Let me run this.", "A short program does it.",
                 "Easiest with a little code.", "Let me build the list and compute it.", "I'll compute that step by step in code.",
                 "Quick program for this.", "Let me do the work in the sandbox."]
THINK_SECOND = ["Now the last step.", "Then the answer.", "And now the aggregate.",
                "With that in place:", "Now I can finish it.", "Then compute what was asked."]


def gen_pipeline(rng: random.Random, corpus: Corpus, kind: str | None = None) -> Sample | None:
    nm = Names(rng)
    if kind is None:
        kinds, weights = zip(*SOURCE_WEIGHTS.items())
        kind = rng.choices(kinds, weights=weights)[0]
    lines, var, val, vkind, intro = SOURCES[kind](rng, nm, corpus)
    phrases: list[str] = []
    n_tr = rng.choice([2, 2, 2, 3, 3, 4])
    if vkind == "dict":  # a dict must first become a list of counts or names
        n_tr = max(n_tr, 1)
    for _ in range(n_tr):
        pool = TRANSFORMS[vkind]
        if not pool:
            break
        res = None
        for fn in rng.sample(pool, len(pool)):
            res = fn(rng, nm, var, val)
            if res is not None:
                break
        if res is None:
            break
        new_lines, val, vkind, phrase = res
        m = re.match(r"^(\w+) =", new_lines[0])
        var = m.group(1) if m else var
        lines += _maybe_comment(rng, phrase) + new_lines
        phrases.append(phrase)
    if vkind == "dict" or not AGGS[vkind]:
        return None
    if (isinstance(val, list) and len(val) < 2) or (isinstance(val, str) and not val):
        return None
    agg = None
    for fn in rng.sample(AGGS[vkind], len(AGGS[vkind])):
        agg = fn(rng, nm, var, val)
        if agg is not None:
            break
    if agg is None:
        return None
    agg_lines, gold_val, question = agg
    lines += agg_lines
    gold = render_value(gold_val)
    question_text = intro + (" " + ", then ".join(phrases).capitalize() + "." if phrases else "") + " " + question
    # one call, or two (the first shows the intermediate value, the second finishes from the session state)
    split_at = None
    if len(phrases) >= 1 and len(lines) - len(agg_lines) >= 2 and rng.random() < 0.55:
        split_at = len(lines) - len(agg_lines)
    session = PySession()
    if split_at is None:
        code = "\n".join(lines)
        if not verify(code, gold, session):
            return None
        codes = [code]
        think = rng.choice(THINK_OPENERS) + f" <<<{code}>>>"
    else:
        code1 = "\n".join(lines[:split_at] + [var])
        code2 = "\n".join(lines[split_at:])
        out1, ok1 = run_tool(code1, session)
        if not ok1 or not verify(code2, gold, session):
            return None
        codes = [code1, code2]
        think = f"{rng.choice(THINK_OPENERS)} <<<{code1}>>> {rng.choice(THINK_SECOND)} <<<{code2}>>>"
    msgs = [{"role": "user", "content": question_text}, {"role": "assistant", "think": think, "content": "#### " + gold}]
    return make_sample(f"pipeline.{kind}", msgs, codes, numeric=_is_numeric(gold))


def _is_numeric(gold: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", gold))


# --------------------------------------------------------------------------------- strings
def gen_strings(rng: random.Random, corpus: Corpus) -> Sample | None:
    nm = Names(rng)
    kind = rng.choice(["reverse", "palindrome", "vowels", "letters", "words", "capitalize", "replace", "find", "longest", "acronym"])
    s = rng.choice(corpus.sentences)
    w = rng.choice([x for x in corpus.words if len(x) >= 4])
    v = nm.pick(["text", "sentence", "line", "s"])
    if kind == "reverse":
        target = w if rng.random() < 0.5 else s
        gold_val = target[::-1]
        if rng.random() < 0.55:
            code = f"{v} = {lit(target)}\n{v}[::-1]"
        else:
            out = nm.pick(["rev", "backwards", "flipped"])
            code = f"{v} = {lit(target)}\n{out} = \"\"\nfor ch in {v}:\n    {out} = ch + {out}\n{out}"
        q = rng.choice([f'Write "{target}" backwards for me.', f'What is "{target}" reversed?'])
    elif kind == "palindrome":
        cand = rng.choice(["level", "rotor", "radar", "stats", "civic", w, w, s.split()[0]])
        gold_val = cand == cand[::-1]
        if rng.random() < 0.5:
            code = f"{v} = {lit(cand)}\n{v} == {v}[::-1]"
        else:
            flag = nm.pick(["ok", "same", "pal"])
            code = f"{v} = {lit(cand)}\n{flag} = True\nfor i in range(len({v})):\n    if {v}[i] != {v}[len({v}) - 1 - i]:\n        {flag} = False\n{flag}"
        q = rng.choice([f'Is "{cand}" a palindrome?', f'Does "{cand}" read the same backwards?'])
    elif kind == "vowels":
        gold_val = sum(1 for c in s if c in "aeiou")
        acc = nm.pick(["n", "count", "vowels"])
        if rng.random() < 0.5:
            code = f'{v} = {lit(s)}\n{acc} = 0\nfor ch in {v}:\n    if ch in "aeiou":\n        {acc} += 1\n{acc}'
        else:
            code = f'{v} = {lit(s)}\nlen([ch for ch in {v} if ch in "aeiou"])'
        q = f'How many vowels are in "{s}"?'
    elif kind == "letters":
        cand = [c for c in "aeiourstnlm" if s.count(c) >= 2]
        if not cand:
            return None
        c = rng.choice(cand)
        gold_val = s.count(c)
        if rng.random() < 0.5:
            code = f'{v} = {lit(s)}\n{v}.count("{c}")'
        else:
            acc = nm.pick(["n", "hits", "count"])
            code = f'{v} = {lit(s)}\n{acc} = 0\nfor ch in {v}:\n    if ch == "{c}":\n        {acc} += 1\n{acc}'
        q = f'How many times does "{c}" appear in "{s}"?'
    elif kind == "words":
        gold_val = len(s.split())
        if rng.random() < 0.6:
            code = f"{v} = {lit(s)}\nlen({v}.split())"
        else:
            wv = nm.pick(WORD_VARS)
            code = f"{v} = {lit(s)}\n{wv} = {v}.split()\nlen({wv})"
        q = rng.choice([f'How many words are in "{s}"?', f'Count the words in "{s}".'])
    elif kind == "capitalize":
        gold_val = s.title()
        code = f"{v} = {lit(s)}\n{v}.title()" if rng.random() < 0.5 else f'{v} = {lit(s)}\n" ".join([w.capitalize() for w in {v}.split()])'
        q = f'Capitalise every word of "{s}".'
    elif kind == "replace":
        a = rng.choice([c for c in "aeiou" if c in s])
        b = rng.choice("xy*-")
        gold_val = s.replace(a, b)
        code = f'{v} = {lit(s)}\n{v}.replace("{a}", "{b}")'
        q = f'Replace every "{a}" in "{s}" with "{b}".'
    elif kind == "find":
        target = rng.choice(s.split()[1:]) if len(s.split()) > 2 else s.split()[0]
        gold_val = s.find(target)
        code = f'{v} = {lit(s)}\n{v}.find("{target}")'
        q = f'At which index does "{target}" start in "{s}"?'
    elif kind == "longest":
        ws = s.split()
        best = max(ws, key=len)
        if len([x for x in ws if len(x) == len(best)]) > 1:
            return None
        gold_val = best
        acc = nm.pick(["longest", "best"])
        code = f"{v} = {lit(s)}\n{acc} = \"\"\nfor w in {v}.split():\n    if len(w) > len({acc}):\n        {acc} = w\n{acc}"
        q = f'What is the longest word in "{s}"?'
    else:  # acronym
        ws = s.split()
        gold_val = "".join(x[0].upper() for x in ws)
        if rng.random() < 0.5:
            code = f'{v} = {lit(s)}\n"".join([w[0].upper() for w in {v}.split()])'
        else:
            acc = nm.pick(["acr", "initials", "letters"])
            code = f'{v} = {lit(s)}\n{acc} = ""\nfor w in {v}.split():\n    {acc} += w[0].upper()\n{acc}'
        q = f'Make an acronym from the first letters of "{s}".'
    gold = render_value(gold_val)
    if not verify(code, gold):
        return None
    think = rng.choice(THINK_OPENERS) + f" <<<{code}>>>"
    msgs = [{"role": "user", "content": q}, {"role": "assistant", "think": think, "content": "#### " + gold}]
    return make_sample(f"strings.{kind}", msgs, [code], numeric=_is_numeric(gold))


# --------------------------------------------------------------------------------- number theory & sequences
def gen_numbers(rng: random.Random, corpus: Corpus) -> Sample | None:
    nm = Names(rng)
    kind = rng.choice(["gcd", "lcm", "primes", "digitsum", "factorial", "power", "fib", "collatz", "base", "divisors", "hypot", "doublings"])
    if kind == "gcd":
        a, b = rng.randint(12, 900), rng.randint(12, 900)
        gold_val = math.gcd(a, b)
        r = rng.random()
        if r < 0.5:
            code = f"math.gcd({a}, {b})"
        elif r < 0.7:
            code = f"a = {a}\nb = {b}\nwhile b != 0:\n    a, b = b, a % b\na"
        else:
            code = f"def gcd(a, b):\n    while b != 0:\n        a, b = b, a % b\n    return a\n\ngcd({a}, {b})"
        q = rng.choice([f"What is the greatest common divisor of {a} and {b}?", f"Find gcd({a}, {b}).", f"What is the largest number that divides both {a} and {b}?"])
    elif kind == "lcm":
        a, b = rng.randint(4, 90), rng.randint(4, 90)
        g = math.gcd(a, b)
        gold_val = a * b // g
        if rng.random() < 0.6:
            code = f"a = {a}\nb = {b}\na * b // math.gcd(a, b)"
        else:
            code = f"def gcd(a, b):\n    while b != 0:\n        a, b = b, a % b\n    return a\n\n{a} * {b} // gcd({a}, {b})"
        q = rng.choice([f"What is the least common multiple of {a} and {b}?", f"Find the smallest number that both {a} and {b} divide."])
    elif kind == "primes":
        n = rng.randint(20, 160)
        primes = [x for x in range(2, n + 1) if all(x % d for d in range(2, int(x**0.5) + 1))]
        mode = rng.choice(["count", "sum", "largest"])
        gold_val = {"count": len(primes), "sum": sum(primes), "largest": primes[-1]}[mode]
        body = f"def is_prime(n):\n    if n < 2:\n        return False\n    d = 2\n    while d * d <= n:\n        if n % d == 0:\n            return False\n        d += 1\n    return True\n\nps = []\nfor x in range(2, {n + 1}):\n    if is_prime(x):\n        ps.append(x)\n"
        code = body + {"count": "len(ps)", "sum": "sum(ps)", "largest": "ps[-1]"}[mode]
        q = {"count": f"How many prime numbers are there up to {n}?", "sum": f"What is the sum of all primes up to {n}?", "largest": f"What is the largest prime not greater than {n}?"}[mode]
    elif kind == "digitsum":
        n = rng.randint(1000, 9_999_999)
        gold_val = sum(int(c) for c in str(n))
        r = rng.random()
        if r < 0.3:
            code = f"n = {n}\ntotal = 0\nfor ch in str(n):\n    total += int(ch)\ntotal"
        elif r < 0.6:
            code = f"n = {n}\ntotal = 0\nwhile n > 0:\n    total += n % 10\n    n = n // 10\ntotal"
        else:
            code = f"def digit_sum(n):\n    total = 0\n    for ch in str(n):\n        total += int(ch)\n    return total\n\ndigit_sum({n})"
        q = rng.choice([f"What is the sum of the digits of {n}?", f"Add up the digits of {n}."])
    elif kind == "factorial":
        n = rng.randint(5, 18)
        gold_val = math.factorial(n)
        if rng.random() < 0.55:
            code = f"math.factorial({n})"
        else:
            code = f"f = 1\nfor i in range(1, {n + 1}):\n    f *= i\nf"
        q = rng.choice([f"What is {n} factorial?", f"Compute {n}!.", f"How many ways can {n} different books be ordered on a shelf?"])
    elif kind == "power":
        b, e = rng.randint(2, 12), rng.randint(4, 14)
        mode = rng.choice(["value", "sum"])
        if mode == "value":
            gold_val = b**e
            code = f"{b} ** {e}" if rng.random() < 0.5 else f"p = 1\nfor i in range({e}):\n    p *= {b}\np"
            q = f"What is {b} to the power of {e}?"
        else:
            e = min(e, 9)
            gold_val = sum(b**i for i in range(1, e + 1))
            code = f"total = 0\nfor i in range(1, {e + 1}):\n    total += {b} ** i\ntotal"
            q = f"What is {b}^1 + {b}^2 + ... + {b}^{e}?"
    elif kind == "fib":
        a0, b0 = rng.choice([(1, 1), (1, 2), (2, 3), (3, 4), (0, 1)])
        n = rng.randint(10, 30)
        a, b = a0, b0
        for _ in range(n - 2):
            a, b = b, a + b
        gold_val = b
        if rng.random() < 0.45:
            code = f"a, b = {a0}, {b0}\nfor i in range({n - 2}):\n    a, b = b, a + b\nb"
        else:
            code = f"def seq(n):\n    a, b = {a0}, {b0}\n    for i in range(n - 2):\n        a, b = b, a + b\n    return b\n\nseq({n})"
        q = f"A sequence starts {a0}, {b0} and every later term is the sum of the two before it. What is term {n}?"
    elif kind == "collatz":
        n = rng.randint(7, 400)
        steps, x = 0, n
        while x != 1:
            x = x // 2 if x % 2 == 0 else 3 * x + 1
            steps += 1
        gold_val = steps
        code = f"n = {n}\nsteps = 0\nwhile n != 1:\n    if n % 2 == 0:\n        n = n // 2\n    else:\n        n = 3 * n + 1\n    steps += 1\nsteps"
        q = f"Start at {n}. Halve it when it is even, otherwise triple it and add one. How many steps until it reaches 1?"
    elif kind == "base":
        n = rng.randint(20, 4000)
        base = rng.choice([2, 2, 3, 8])
        digits, x = "", n
        while x > 0:
            digits = str(x % base) + digits
            x //= base
        gold_val = digits
        code = f'n = {n}\nout = ""\nwhile n > 0:\n    out = str(n % {base}) + out\n    n = n // {base}\nout'
        q = f"Write {n} in base {base}." if base != 2 else f"What is {n} in binary?"
    elif kind == "hypot":
        a, b = rng.randint(3, 60), rng.randint(3, 60)
        gold_val = round(math.sqrt(a * a + b * b), 2)
        if rng.random() < 0.6:
            code = f"a = {a}\nb = {b}\nround(math.sqrt(a * a + b * b), 2)"
        else:
            code = f"def hyp(a, b):\n    return round(math.sqrt(a ** 2 + b ** 2), 2)\n\nhyp({a}, {b})"
        q = rng.choice([f"A right triangle has legs of {a} and {b}. How long is the hypotenuse, to two decimals?",
                        f"What is the distance from the origin to the point ({a}, {b}), rounded to two decimals?"])
    elif kind == "doublings":
        start, target = rng.randint(2, 40), rng.randint(500, 900_000)
        k, x = 0, start
        while x < target:
            x *= 2
            k += 1
        gold_val = k
        if rng.random() < 0.5:
            code = f"math.ceil(math.log2({target} / {start}))"
            if math.ceil(math.log2(target / start)) != k:
                code = f"x = {start}\nk = 0\nwhile x < {target}:\n    x = x * 2\n    k += 1\nk"
        else:
            code = f"x = {start}\nk = 0\nwhile x < {target}:\n    x = x * 2\n    k += 1\nk"
        q = f"Starting from {start} and doubling each step, how many steps until the value reaches {target} or more?"
    else:  # divisors
        n = rng.randint(24, 600)
        divs = [d for d in range(1, n + 1) if n % d == 0]
        mode = rng.choice(["count", "sum", "list"])
        gold_val = {"count": len(divs), "sum": sum(divs), "list": divs}[mode]
        tail = {"count": "len(divs)", "sum": "sum(divs)", "list": "divs"}[mode]
        code = f"divs = []\nfor d in range(1, {n + 1}):\n    if {n} % d == 0:\n        divs.append(d)\n{tail}"
        q = {"count": f"How many divisors does {n} have?", "sum": f"What is the sum of all divisors of {n}?", "list": f"List every divisor of {n}."}[mode]
    gold = render_value(gold_val)
    if not verify(code, gold):
        return None
    think = rng.choice(THINK_OPENERS) + f" <<<{code}>>>"
    msgs = [{"role": "user", "content": q}, {"role": "assistant", "think": think, "content": "#### " + gold}]
    return make_sample(f"numbers.{kind}", msgs, [code], numeric=_is_numeric(gold))


# --------------------------------------------------------------------------------- simulations
def gen_simulation(rng: random.Random, corpus: Corpus) -> Sample | None:
    nm = Names(rng)
    kind = rng.choice(["doubling", "interest", "inventory", "scoring", "steps", "decay"])
    if kind == "doubling":
        start, n, k = rng.randint(3, 60), rng.randint(4, 14), rng.choice([2, 2, 3])
        gold_val = start * k**n
        var = nm.pick(["colony", "cells", "count", "population"])
        code = f"{var} = {start}\nfor day in range({n}):\n    {var} = {var} * {k}\n{var}"
        q = f"A colony starts with {start} cells and {'doubles' if k == 2 else 'triples'} every day. How many cells after {n} days?"
    elif kind == "interest":
        p, n = rng.randint(100, 5000), rng.randint(3, 20)
        rate = rng.choice([3, 4, 5, 6, 8, 10])
        bal = float(p)
        for _ in range(n):
            bal = round(bal * (1 + rate / 100), 2)
        gold_val = bal
        var = nm.pick(["balance", "amount", "savings"])
        code = f"{var} = {p}.0\nfor year in range({n}):\n    {var} = round({var} * {1 + rate / 100}, 2)\n{var}"
        q = f"I put {p} dollars in an account that pays {rate}% interest a year, rounded to cents each year. How much is there after {n} years?"
    elif kind == "inventory":
        keys = rng.sample(_GOODS, rng.randint(3, 5))
        stock = {k: rng.randint(5, 60) for k in keys}
        deliveries = [(rng.choice(keys), rng.randint(1, 25)) for _ in range(rng.randint(3, 6))]
        end = dict(stock)
        for k, q_ in deliveries:
            end[k] = end[k] + q_
        mode = rng.choice(["total", "one", "max"])
        gold_val = sum(end.values()) if mode == "total" else (end[keys[0]] if mode == "one" else max(end.values()))
        var = nm.pick(["stock", "shelf", "store"])
        tail = {"total": f"sum(list({var}.values()))", "one": f'{var}["{keys[0]}"]', "max": f"max(list({var}.values()))"}[mode]
        code = f"{var} = {lit(stock)}\ndeliveries = {lit(deliveries)}\nfor name, n in deliveries:\n    {var}[name] = {var}[name] + n\n{tail}"
        qtail = {"total": "How many items are in stock afterwards?", "one": f'How many {keys[0]} are there afterwards?', "max": "What is the largest stock count afterwards?"}[mode]
        q = f"My stock is {lit(stock)} and these deliveries arrive: {lit(deliveries)}. {qtail}"
    elif kind == "scoring":
        rounds = [rng.randint(-5, 20) for _ in range(rng.randint(5, 9))]
        bonus = rng.randint(3, 10)
        thr = rng.randint(8, 15)
        total = 0
        for r in rounds:
            total += r
            if r >= thr:
                total += bonus
        gold_val = total
        var = nm.pick(["score", "points", "total"])
        code = f"rounds = {lit(rounds)}\n{var} = 0\nfor r in rounds:\n    {var} += r\n    if r >= {thr}:\n        {var} += {bonus}\n{var}"
        q = f"A player scores {lit(rounds)} in successive rounds and gets a {bonus} point bonus for any round of {thr} or more. What is the final score?"
    elif kind == "steps":
        start, n = rng.randint(5, 80), rng.randint(5, 15)
        x = start
        for _ in range(n):
            x = x // 2 if x % 2 == 0 else x + 7
        gold_val = x
        var = nm.pick(["x", "value", "state"])
        if rng.random() < 0.4:
            code = f"def step(x):\n    if x % 2 == 0:\n        return x // 2\n    return x + 7\n\n{var} = {start}\nfor i in range({n}):\n    {var} = step({var})\n{var}"
        else:
            code = f"{var} = {start}\nfor i in range({n}):\n    if {var} % 2 == 0:\n        {var} = {var} // 2\n    else:\n        {var} = {var} + 7\n{var}"
        q = f"Start at {start}. Each step: halve it if it is even, otherwise add 7. What is the value after {n} steps?"
    else:  # decay
        start, n = rng.randint(400, 9000), rng.randint(3, 12)
        pct = rng.choice([10, 15, 20, 25, 30])
        x = float(start)
        for _ in range(n):
            x = round(x * (1 - pct / 100), 2)
        gold_val = x
        var = nm.pick(["value", "amount", "level"])
        if rng.random() < 0.35:
            x = start
            for _ in range(n):
                x = math.floor(x * (1 - pct / 100))
            gold_val = x
            code = f"{var} = {start}\nfor i in range({n}):\n    {var} = math.floor({var} * {1 - pct / 100})\n{var}"
        else:
            code = f"{var} = {start}.0\nfor i in range({n}):\n    {var} = round({var} * {1 - pct / 100}, 2)\n{var}"
        q = f"Something worth {start} loses {pct}% of its value every year, rounded to cents. What is it worth after {n} years?"
    gold = render_value(gold_val)
    if not verify(code, gold):
        return None
    think = rng.choice(THINK_OPENERS) + f" <<<{code}>>>"
    msgs = [{"role": "user", "content": q}, {"role": "assistant", "think": think, "content": "#### " + gold}]
    return make_sample(f"simulation.{kind}", msgs, [code], numeric=_is_numeric(gold))


# --------------------------------------------------------------------------------- multi-turn REPL
def gen_multiturn(rng: random.Random, corpus: Corpus) -> Sample | None:
    nm = Names(rng)
    kind = rng.choices(["helper", "data", "inventory"], weights=[0.56, 0.28, 0.16])[0]
    session = PySession()
    msgs: list[dict] = []
    codes: list[str] = []
    numeric = True

    asked: set[str] = set()

    def turn(q: str, code: str, gold_val) -> bool:
        """Add one turn if the session really produces `gold_val`; returns whether it was added."""
        nonlocal numeric
        gold = render_value(gold_val)
        if q in asked or not verify(code, gold, session):
            return False
        asked.add(q)
        codes.append(code)
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "think": rng.choice(THINK_OPENERS) + f" <<<{code}>>>", "content": "#### " + gold})
        numeric = numeric and _is_numeric(gold)
        return True

    if kind == "helper":
        which = rng.choice(["digit_sum", "score", "trim"])
        if which == "digit_sum":
            fname = "digit_sum"
            body = f"def {fname}(n):\n    total = 0\n    for ch in str(n):\n        total += int(ch)\n    return total"
            n1 = rng.randint(1000, 999_999)
            turn(f"Write a helper that adds up the digits of a number, and use it on {n1}.", f"{body}\n\n{fname}({n1})", sum(int(c) for c in str(n1)))
            for _ in range(rng.choice([2, 2, 3, 3, 4])):
                r = rng.random()
                if r < 0.45:
                    n2 = rng.randint(1000, 9_999_999)
                    turn(rng.choice([f"Now use it on {n2}.", f"And for {n2}?", f"Apply the same helper to {n2}."]), f"{fname}({n2})", sum(int(c) for c in str(n2)))
                elif r < 0.8:
                    ns = [rng.randint(100, 99999) for _ in range(rng.randint(3, 5))]
                    best = max(ns, key=lambda x: sum(int(c) for c in str(x)))
                    if len([x for x in ns if sum(int(c) for c in str(x)) == sum(int(c) for c in str(best))]) > 1:
                        continue
                    turn(f"Which of {lit(ns)} has the biggest digit sum?", f"nums = {lit(ns)}\nbest = nums[0]\nfor x in nums:\n    if {fname}(x) > {fname}(best):\n        best = x\nbest", best)
                else:
                    lo, hi = rng.randint(10, 60), 0
                    hi = lo + rng.randint(20, 60)
                    tot = sum(sum(int(c) for c in str(x)) for x in range(lo, hi + 1))
                    turn(f"What is the total digit sum of every number from {lo} to {hi}?", f"total = 0\nfor x in range({lo}, {hi + 1}):\n    total += {fname}(x)\ntotal", tot)
        elif which == "score":
            fname = "word_value"
            body = f"def {fname}(w):\n    total = 0\n    for ch in w:\n        if ch in \"aeiou\":\n            total += 3\n        else:\n            total += 1\n    return total"

            def wv(w):
                return sum(3 if c in "aeiou" else 1 for c in w)

            w1 = rng.choice(corpus.words)
            turn(f'Score a word by giving 3 points to each vowel and 1 to every other letter. Write that as a function and score "{w1}".', f"{body}\n\n{fname}({lit(w1)})", wv(w1))
            for _ in range(rng.choice([2, 2, 3, 3])):
                if rng.random() < 0.5:
                    w2 = rng.choice(corpus.words)
                    turn(f'What about "{w2}"?', f"{fname}({lit(w2)})", wv(w2))
                else:
                    ws = rng.sample(corpus.words, rng.randint(3, 5))
                    best = max(ws, key=wv)
                    if len([x for x in ws if wv(x) == wv(best)]) > 1:
                        continue
                    turn(f"Which of {lit(ws)} scores highest?", f"words = {lit(ws)}\nbest = words[0]\nfor w in words:\n    if {fname}(w) > {fname}(best):\n        best = w\nbest", best)
        else:
            fname = "shorten"
            body = f'def {fname}(w, k):\n    if len(w) <= k:\n        return w\n    return w[:k] + "."'
            w1 = rng.choice([x for x in corpus.words if len(x) >= 6])
            k = rng.randint(3, 5)
            turn(f'Write a helper that shortens a word to {k} letters followed by a dot, and try it on "{w1}".', f"{body}\n\n{fname}({lit(w1)}, {k})", w1[:k] + "." if len(w1) > k else w1)
            ws = rng.sample(corpus.words, rng.randint(4, 6))
            turn(f"Now shorten each of {lit(ws)} the same way.", f"words = {lit(ws)}\nout = []\nfor w in words:\n    out.append({fname}(w, {k}))\nout", [w[:k] + "." if len(w) > k else w for w in ws])
    elif kind == "data":
        vals = [rng.randint(3, 90) for _ in range(rng.randint(6, 10))]
        var = nm.pick(NUM_VARS)
        turn(rng.choice([f"Keep these numbers for me: {lit(vals)}. What do they add up to?", f"Here is my data: {lit(vals)}. Give me the total."]),
             f"{var} = {lit(vals)}\nsum({var})", sum(vals))
        cur = list(vals)
        for _ in range(rng.choice([2, 2, 3, 3, 4])):
            r = rng.random()
            if r < 0.3:
                turn(rng.choice(["And the average?", "What is the mean?"]), f"round(sum({var}) / len({var}), 2)", round(sum(cur) / len(cur), 2))
            elif r < 0.5:
                k = sorted(cur)[len(cur) // 2]
                n = len([x for x in cur if x > k])
                turn(f"How many of them are above {k}?", f"n = 0\nfor x in {var}:\n    if x > {k}:\n        n += 1\nn", n)
            elif r < 0.7:
                k = rng.randint(2, 20)
                nxt = [x + k for x in cur]
                if turn(f"Add {k} to every value and give me the new total.", f"{var} = [x + {k} for x in {var}]\nsum({var})", sum(nxt)):
                    cur = nxt
            elif r < 0.85:
                thr = sorted(cur)[len(cur) // 3]
                nxt = [x for x in cur if x > thr]
                if len(nxt) < 2:
                    break
                if turn(f"Drop anything at or below {thr} and tell me how many are left.", f"{var} = [x for x in {var} if x > {thr}]\nlen({var})", len(nxt)):
                    cur = nxt
            else:
                turn(rng.choice(["What is the biggest one now?", "And the maximum?"]), f"max({var})", max(cur))
    else:
        keys = rng.sample(_GOODS, rng.randint(3, 5))
        stock = {k: rng.randint(4, 70) for k in keys}
        var = nm.pick(["stock", "shelf", "store"])
        turn(f"Remember my stock: {lit(stock)}. How many items is that in total?", f"{var} = {lit(stock)}\nsum(list({var}.values()))", sum(stock.values()))
        cur = dict(stock)
        for _ in range(rng.choice([2, 2, 3, 3, 4])):
            r = rng.random()
            if r < 0.4:
                k, n = rng.choice(keys), rng.randint(2, 30)
                nxt = dict(cur)
                nxt[k] += n
                if turn(f"{n} more {k} arrived. What is the total now?", f'{var}["{k}"] = {var}["{k}"] + {n}\nsum(list({var}.values()))', sum(nxt.values())):
                    cur = nxt
            elif r < 0.7:
                best = max(cur, key=lambda x: cur[x])
                if len([x for x in cur if cur[x] == cur[best]]) > 1:
                    continue
                turn(rng.choice(["Which item do I have most of?", "What is my biggest stock item?"]),
                     f"best = list({var}.keys())[0]\nfor name in {var}:\n    if {var}[name] > {var}[best]:\n        best = name\nbest", best)
            else:
                thr = sorted(cur.values())[len(cur) // 2]
                n = len([x for x in cur if cur[x] >= thr])
                turn(f"How many items do I have at least {thr} of?", f"n = 0\nfor name in {var}:\n    if {var}[name] >= {thr}:\n        n += 1\nn", n)
    if len(msgs) < 4:
        return None
    return make_sample(f"multiturn.{kind}", msgs, codes, numeric=numeric)


# --------------------------------------------------------------------------------- run this code
RUNCODE_Q = ["What does this print?\n\n{c}", "Run this and tell me the output:\n\n{c}", "```python\n{c}\n```\nWhat is the output?",
             "I wrote this but I cannot run it right now. What does it give?\n\n{c}", "Execute this for me:\n\n{c}"]
RUNCODE_THINK = ["Let me run it exactly as written.", "I'll execute the program as given.", "Running it in the sandbox.",
                 "Let me just run the code."]


def _standalone_program(rng: random.Random, corpus: Corpus) -> tuple[str, str] | None:
    """A self-contained program from the same grammars, plus its single-line output."""
    for _ in range(6):
        gen = rng.choices([gen_pipeline, gen_numbers, gen_strings, gen_simulation], weights=[0.4, 0.25, 0.2, 0.15])[0]
        s = gen(rng, corpus)
        if s is None or len(s.codes) != 1:
            continue
        code = s.codes[0]
        out, ok = run_tool(code)
        if ok and out and "\n" not in out and len(code) < 600:
            return code, out
    return None


def gen_runcode(rng: random.Random, corpus: Corpus) -> Sample | None:
    got = _standalone_program(rng, corpus)
    if got is None:
        return None
    code, out = got
    q = rng.choice(RUNCODE_Q).format(c=code)
    think = rng.choice(RUNCODE_THINK) + f" <<<{code}>>>"
    msgs = [{"role": "user", "content": q}, {"role": "assistant", "think": think, "content": "#### " + out}]
    return make_sample("runcode.program", msgs, [code], numeric=_is_numeric(out))


# --------------------------------------------------------------------------------- error and recover
def _hint_core(hint: str) -> str:
    """The readable part of a sandbox error, for the model's own reading of it."""
    core = hint[len("error: "):] if hint.startswith("error: ") else hint
    core = re.sub(r"^line \d+: ", "", core)
    core = core.split("  |  ")[0].strip()
    return core[:110].rstrip(" ,;")


ERROR_REACTIONS = ["That is not available here ({h}), so I'll write it the plain way.",
                   "The sandbox refuses it: {h}. Let me redo it with basic Python.",
                   "Right, {h}. I'll do it manually.",
                   "It complains that {h}, so here is the supported version.",
                   "Not supported: {h}. Rewriting it."]


def _error_case(rng: random.Random, corpus: Corpus):
    """(intro, bad code, good code, gold value) where the bad code really trips a sandbox hint."""
    nums = [rng.randint(1, 80) for _ in range(rng.randint(5, 9))]
    words = rng.sample(corpus.words, rng.randint(4, 7))
    sent = rng.choice(corpus.sentences)
    kind = rng.choice(["import", "kwarg", "listmethod", "strmethod", "dictcomp", "set", "lambda", "module", "anybuiltin", "arity", "sqrt", "undefined", "try"])
    if kind == "import":
        return ("They want the mean.", f"data = {lit(nums)}\nimport statistics\nstatistics.mean(data)",
                "round(sum(data) / len(data), 2)", round(sum(nums) / len(nums), 2), f"What is the average of {lit(nums)}?")
    if kind == "kwarg":
        return ("Sort them the other way round.", f"nums = {lit(nums)}\nsorted(nums, reverse=True)", "sorted(nums)[::-1]",
                sorted(nums)[::-1], f"Sort {lit(nums)} from largest to smallest.")
    if kind == "listmethod":
        return ("Add them up.", f"nums = {lit(nums)}\nnums.sum()", "sum(nums)", sum(nums), f"What do {lit(nums)} add up to?")
    if kind == "strmethod":
        return ("Swap the case of every letter.", f"text = {lit(sent)}\ntext.swapcase()",
                'out = ""\nfor ch in text:\n    if ch == " ":\n        out += " "\n    else:\n        out += ch.upper()\nout',
                " ".join(w.upper() for w in sent.split()), f'Put "{sent}" in capitals.')
    if kind == "dictcomp":
        return ("Build a length table.", f"words = {lit(words)}\n{{w: len(w) for w in words}}",
                "sizes = {}\nfor w in words:\n    sizes[w] = len(w)\nsizes", {w: len(w) for w in words},
                f"Give me a table of each word in {lit(words)} and its length.")
    if kind == "set":
        dup = nums + [nums[0], nums[1]]
        rng.shuffle(dup)
        uniq = []
        for x in dup:
            if x not in uniq:
                uniq.append(x)
        return ("Remove the duplicates.", f"nums = {lit(dup)}\nlen({{x for x in nums}})",
                "uniq = []\nfor x in nums:\n    if x not in uniq:\n        uniq.append(x)\nlen(uniq)", len(uniq),
                f"How many different numbers are in {lit(dup)}?")
    if kind == "lambda":
        return ("Double each value.", f"nums = {lit(nums)}\ndouble = lambda x: x * 2\n[double(x) for x in nums]",
                "def double(x):\n    return x * 2\n\n[double(x) for x in nums]", [x * 2 for x in nums], f"Double every number in {lit(nums)}.")
    if kind == "module":
        return ("Take the mean.", f"nums = {lit(nums)}\nnp.mean(nums)", "round(sum(nums) / len(nums), 2)",
                round(sum(nums) / len(nums), 2), f"What is the mean of {lit(nums)}?")
    if kind == "anybuiltin":
        d = rng.choice([3, 4, 5, 7])
        return (f"Check for a multiple of {d}.", f"nums = {lit(nums)}\nany([x % {d} == 0 for x in nums])",
                f"found = False\nfor x in nums:\n    if x % {d} == 0:\n        found = True\nfound",
                any(x % d == 0 for x in nums), f"Is any number in {lit(nums)} a multiple of {d}?")
    if kind == "arity":
        w, h = rng.randint(3, 40), rng.randint(3, 40)
        return ("A small helper will do.", f"def area(w, h):\n    return w * h\n\narea({w})", f"area({w}, {h})", w * h,
                f"What is the area of a {w} by {h} rectangle?")
    if kind == "sqrt":
        n = rng.choice([144, 225, 400, 576, 1024, 1600, 2025])
        return ("Take the square root.", f"sqrt({n})", f"int(math.sqrt({n}))", int(n**0.5), f"What is the square root of {n}?")
    if kind == "undefined":
        return ("Add them up.", "total = 0\nfor x in values:\n    total += x\ntotal",
                f"values = {lit(nums)}\ntotal = 0\nfor x in values:\n    total += x\ntotal", sum(nums),
                f"Add up {lit(nums)} for me.")
    thr = sorted(nums)[len(nums) // 2]
    return ("Guard the division.", f"nums = {lit(nums)}\ntry:\n    total = sum(nums) / 0\nexcept:\n    total = 0\ntotal",
            f"big = [x for x in nums if x > {thr}]\nround(sum(big) / len(big), 2)",
            round(sum(x for x in nums if x > thr) / max(1, len([x for x in nums if x > thr])), 2),
            f"What is the average of the numbers in {lit(nums)} that are above {thr}?")


def gen_error(rng: random.Random, corpus: Corpus) -> Sample | None:
    intro, bad, good, gold_val, q = _error_case(rng, corpus)
    gold = render_value(gold_val)
    session = PySession()
    hint, ok = run_tool(bad, session)
    if ok:
        return None  # it must really fail
    if not verify(good, gold, session):
        return None
    reaction = rng.choice(ERROR_REACTIONS).format(h=_hint_core(hint))
    spans = [("text", intro + " "), ("call", bad), ("text", " " + reaction + " "), ("call", good)]
    msgs = [{"role": "user", "content": q}, {"role": "assistant", "spans": spans, "content": "#### " + gold}]
    return make_sample("error.recover", msgs, [bad, good], numeric=_is_numeric(gold))


# --------------------------------------------------------------------------------- declared functions
PRICES = {"pen": 2.5, "notebook": 4.0, "folder": 3.25, "stapler": 7.5, "ruler": 1.75, "eraser": 0.9, "marker": 2.2, "binder": 6.4, "tape": 1.6, "scissors": 5.8}
POPULATIONS = {"Aldershaw": 182_400, "Brinemouth": 61_250, "Calderport": 944_800, "Dunmarket": 12_900, "Elmsford": 305_700, "Farrowhill": 78_300, "Greyvale": 521_000, "Hollowbeck": 9_450}
CITY_XY = {"Aldershaw": (0, 0), "Brinemouth": (120, 45), "Calderport": (300, 20), "Dunmarket": (75, 260), "Elmsford": (410, 180), "Farrowhill": (95, 130), "Greyvale": (250, 330), "Hollowbeck": (5, 190)}
PARCEL_ZONES = {"local": 1.0, "regional": 2.4, "national": 4.1, "overseas": 9.75}


def _decl(name, sig, comment, impl):
    return FunctionDecl(name, sig, comment, impl)


def svc_catalogue():
    def unit_price(item):
        return PRICES[item]

    def catalogue():
        return sorted(PRICES.keys())

    return [_decl("unit_price", "def unit_price(item: str) -> float", "Catalogue price of one item in dollars. Use it instead of guessing a price.", unit_price),
            _decl("catalogue", "def catalogue() -> list", "The list of item names we sell, in alphabetical order.", catalogue)]


def svc_population():
    def city_population(city):
        return POPULATIONS[city]

    return [_decl("city_population", "def city_population(city: str) -> int", "Number of inhabitants of a city in our register.", city_population)]


def svc_distance():
    def distance_km(a, b):
        (x1, y1), (x2, y2) = CITY_XY[a], CITY_XY[b]
        return round(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)

    return [_decl("distance_km", "def distance_km(a: str, b: str) -> int", "Road distance in whole kilometres between two cities in our register.", distance_km)]


def svc_convert():
    def to_celsius(f):
        return round((f - 32) * 5 / 9, 1)

    return [_decl("to_celsius", "def to_celsius(f: float) -> float", "Convert a temperature in Fahrenheit to Celsius, rounded to one decimal.", to_celsius)]


def svc_shipping():
    def shipping_cost(weight_g, zone):
        return round(PARCEL_ZONES[zone] * (1 + weight_g / 1000), 2)

    return [_decl("shipping_cost", "def shipping_cost(weight_g: int, zone: str) -> float", "Postage in dollars for a parcel of that weight to that zone (local, regional, national, overseas).", shipping_cost)]


def svc_readings():
    def readings(day):
        return [(day * 7 + i * 13) % 40 + 5 for i in range(6)]

    return [_decl("readings", "def readings(day: int) -> list", "The six sensor readings recorded on a given day, as a list of integers.", readings)]


def svc_stock():
    def stock_level(item):
        return (len(item) * 17 + sum(ord(c) for c in item)) % 80 + 3

    return [_decl("stock_level", "def stock_level(item: str) -> int", "How many units of an item are in the warehouse right now.", stock_level)]


DECL_SERVICES = {"catalogue": svc_catalogue, "population": svc_population, "distance": svc_distance, "convert": svc_convert,
                 "shipping": svc_shipping, "readings": svc_readings, "stock": svc_stock}
DECL_WEIGHTS = {"catalogue": 0.22, "population": 0.14, "distance": 0.13, "convert": 0.12, "shipping": 0.13, "readings": 0.13, "stock": 0.13}


def gen_declared(rng: random.Random, corpus: Corpus) -> Sample | None:
    svc = rng.choices(list(DECL_WEIGHTS), weights=list(DECL_WEIGHTS.values()))[0]
    decls = DECL_SERVICES[svc]()
    impls = {d.name: d.impl for d in decls}
    extra = rng.random() < 0.3  # a second, unrelated declaration that is not needed
    if extra:
        other = rng.choice([s for s in DECL_SERVICES if s != svc])
        decls = decls + [DECL_SERVICES[other]()[0]]
    unneeded = rng.random() < 0.05  # the declarations do not help: plain Python is the answer
    recover = rng.random() < 0.12  # a call to a function that was never declared, then the right one

    if unneeded:
        s = gen_numbers(rng, corpus) if rng.random() < 0.5 else gen_pipeline(rng, corpus, kind=rng.choice(["ints", "range", "words"]))
        if s is None:
            return None
        s.decls = decls
        s.family = f"declared.unneeded.{svc}"
        return s

    if svc == "catalogue":
        items = rng.sample(list(PRICES), rng.randint(3, 5))
        mode = rng.choice(["basket", "cheapest", "total_all", "over"])
        if mode == "basket":
            basket = [(i, rng.randint(1, 6)) for i in items]
            gold_val = round(sum(PRICES[i] * q for i, q in basket), 2)
            code = f"basket = {lit(basket)}\ntotal = 0\nfor item, qty in basket:\n    total += unit_price(item) * qty\nround(total, 2)"
            q = "How much does this order cost: " + ", ".join(f"{qn} {i}" for i, qn in basket) + "?"
        elif mode == "cheapest":
            gold_val = min(impls["catalogue"](), key=lambda i: PRICES[i])
            code = "names = catalogue()\nbest = names[0]\nfor name in names:\n    if unit_price(name) < unit_price(best):\n        best = name\nbest"
            q = rng.choice(["Which item in the catalogue is the cheapest?", "What is the cheapest thing we sell?"])
        elif mode == "total_all":
            gold_val = round(sum(PRICES[i] for i in impls["catalogue"]()), 2)
            code = "total = 0\nfor name in catalogue():\n    total += unit_price(name)\nround(total, 2)"
            q = "What would one of every item in the catalogue cost?"
        else:
            thr = rng.choice([2.0, 3.0, 4.0, 5.0])
            gold_val = len([i for i in impls["catalogue"]() if PRICES[i] > thr])
            code = f"n = 0\nfor name in catalogue():\n    if unit_price(name) > {thr}:\n        n += 1\nn"
            q = f"How many catalogue items cost more than {thr} dollars?"
    elif svc == "population":
        cities = rng.sample(list(POPULATIONS), rng.randint(3, 5))
        mode = rng.choice(["sum", "largest", "above"])
        if mode == "sum":
            gold_val = sum(POPULATIONS[c] for c in cities)
            code = f"cities = {lit(cities)}\ntotal = 0\nfor c in cities:\n    total += city_population(c)\ntotal"
            q = f"What is the combined population of {', '.join(cities)}?"
        elif mode == "largest":
            gold_val = max(cities, key=lambda c: POPULATIONS[c])
            code = f"cities = {lit(cities)}\nbest = cities[0]\nfor c in cities:\n    if city_population(c) > city_population(best):\n        best = c\nbest"
            q = f"Which of {', '.join(cities)} has the most inhabitants?"
        else:
            thr = rng.choice([50_000, 100_000, 200_000])
            gold_val = len([c for c in cities if POPULATIONS[c] > thr])
            code = f"cities = {lit(cities)}\nn = 0\nfor c in cities:\n    if city_population(c) > {thr}:\n        n += 1\nn"
            q = f"How many of {', '.join(cities)} have more than {thr} inhabitants?"
    elif svc == "distance":
        route = rng.sample(list(CITY_XY), rng.randint(3, 5))
        mode = rng.choice(["route", "farthest"])
        if mode == "route":
            gold_val = sum(impls["distance_km"](route[i], route[i + 1]) for i in range(len(route) - 1))
            code = f"route = {lit(route)}\ntotal = 0\nfor i in range(len(route) - 1):\n    total += distance_km(route[i], route[i + 1])\ntotal"
            q = f"How long is the trip {' -> '.join(route)} in kilometres?"
        else:
            home = route[0]
            rest = route[1:]
            gold_val = max(rest, key=lambda c: impls["distance_km"](home, c))
            code = f"home = {lit(home)}\nothers = {lit(rest)}\nbest = others[0]\nfor c in others:\n    if distance_km(home, c) > distance_km(home, best):\n        best = c\nbest"
            q = f"Which of {', '.join(rest)} is farthest from {home}?"
    elif svc == "convert":
        temps = [rng.randint(-10, 110) for _ in range(rng.randint(3, 6))]
        mode = rng.choice(["list", "mean", "max"])
        if mode == "list":
            gold_val = [impls["to_celsius"](t) for t in temps]
            code = f"temps = {lit(temps)}\nout = []\nfor t in temps:\n    out.append(to_celsius(t))\nout"
            q = f"Convert {lit(temps)} degrees Fahrenheit to Celsius."
        elif mode == "mean":
            cs = [impls["to_celsius"](t) for t in temps]
            gold_val = round(sum(cs) / len(cs), 2)
            code = f"temps = {lit(temps)}\ncs = []\nfor t in temps:\n    cs.append(to_celsius(t))\nround(sum(cs) / len(cs), 2)"
            q = f"What is the average of {lit(temps)} Fahrenheit, in Celsius?"
        else:
            gold_val = max(impls["to_celsius"](t) for t in temps)
            code = f"temps = {lit(temps)}\ncs = []\nfor t in temps:\n    cs.append(to_celsius(t))\nmax(cs)"
            q = f"Of {lit(temps)} Fahrenheit, what is the warmest in Celsius?"
    elif svc == "shipping":
        parcels = [(rng.randint(100, 4000), rng.choice(list(PARCEL_ZONES))) for _ in range(rng.randint(2, 5))]
        mode = rng.choice(["total", "dearest"])
        if mode == "total":
            gold_val = round(sum(impls["shipping_cost"](w, z) for w, z in parcels), 2)
            code = f"parcels = {lit(parcels)}\ntotal = 0\nfor w, z in parcels:\n    total += shipping_cost(w, z)\nround(total, 2)"
            q = "What is the postage for these parcels: " + ", ".join(f"{w} g {z}" for w, z in parcels) + "?"
        else:
            costs = [impls["shipping_cost"](w, z) for w, z in parcels]
            gold_val = max(costs)
            code = f"parcels = {lit(parcels)}\ncosts = []\nfor w, z in parcels:\n    costs.append(shipping_cost(w, z))\nmax(costs)"
            q = "Which of these parcels costs most to send: " + ", ".join(f"{w} g {z}" for w, z in parcels) + "? Give me the price."
    elif svc == "readings":
        day = rng.randint(1, 60)
        mode = rng.choice(["sum", "mean", "max", "span"])
        vals = impls["readings"](day)
        if mode == "sum":
            gold_val = sum(vals)
            code = f"vals = readings({day})\nsum(vals)"
            q = f"What do the sensor readings from day {day} add up to?"
        elif mode == "mean":
            gold_val = round(sum(vals) / len(vals), 2)
            code = f"vals = readings({day})\nround(sum(vals) / len(vals), 2)"
            q = f"What is the average sensor reading on day {day}?"
        elif mode == "max":
            gold_val = max(vals)
            code = f"vals = readings({day})\nbest = vals[0]\nfor x in vals:\n    if x > best:\n        best = x\nbest"
            q = f"What was the highest reading on day {day}?"
        else:
            d2 = day + rng.randint(1, 5)
            gold_val = sum(impls["readings"](d)[0] for d in range(day, d2 + 1))
            code = f"total = 0\nfor d in range({day}, {d2 + 1}):\n    total += readings(d)[0]\ntotal"
            q = f"Add up the first reading of each day from day {day} to day {d2}."
    else:  # stock
        items = rng.sample(list(PRICES), rng.randint(3, 5))
        mode = rng.choice(["total", "low", "max"])
        if mode == "total":
            gold_val = sum(impls["stock_level"](i) for i in items)
            code = f"items = {lit(items)}\ntotal = 0\nfor it in items:\n    total += stock_level(it)\ntotal"
            q = f"How many units of {', '.join(items)} do we have altogether?"
        elif mode == "low":
            thr = rng.randint(20, 50)
            gold_val = [i for i in items if impls["stock_level"](i) < thr]
            code = f"items = {lit(items)}\nlow = []\nfor it in items:\n    if stock_level(it) < {thr}:\n        low.append(it)\nlow"
            q = f"Which of {', '.join(items)} have fewer than {thr} units in the warehouse?"
        else:
            gold_val = max(items, key=lambda i: impls["stock_level"](i))
            code = f"items = {lit(items)}\nbest = items[0]\nfor it in items:\n    if stock_level(it) > stock_level(best):\n        best = it\nbest"
            q = f"Which of {', '.join(items)} do we have most of?"

    gold = render_value(gold_val)
    session = PySession()
    session.register(functions_env(decls))
    if recover:
        wrong = rng.choice(["price_of", "get_price", "lookup", "population_of", "distance", "celsius", "postage", "sensor", "stock"])
        bad = f"{wrong}({lit(rng.choice(list(PRICES)))})"
        hint, ok = run_tool(bad, session)
        if ok:
            return None
        if not verify(code, gold, session):
            return None
        reaction = rng.choice(["No such function; the declared one is the right route.",
                               "That name does not exist here. The declaration gives the real one.",
                               "Wrong name. Let me use the function that was actually declared."])
        spans = [("text", "I'll use the declared function. "), ("call", bad), ("text", " " + reaction + " "), ("call", code)]
        msgs = [{"role": "user", "content": q}, {"role": "assistant", "spans": spans, "content": "#### " + gold}]
        return make_sample(f"declared.{svc}", msgs, [bad, code], decls=decls, numeric=_is_numeric(gold))
    if not verify(code, gold, session):
        return None
    think = rng.choice(["The declared function knows this, I don't have to guess.", "I'll call the declared function.",
                        "Straight through the declared helper.", "Let me use what was declared."]) + f" <<<{code}>>>"
    msgs = [{"role": "user", "content": q}, {"role": "assistant", "think": think, "content": "#### " + gold}]
    return make_sample(f"declared.{svc}", msgs, [code], decls=decls, numeric=_is_numeric(gold))


# --------------------------------------------------------------------------------- plain arithmetic (from slm.rl.synth)
def make_arith_pools(n: int, seed: int):
    from slm.rl.tasks import make_tasks

    return {"train": make_tasks(["arith2", "arith2mul", "arith_multi", "algebra", "word"], max(1, n), "train", seed),
            "heldout": make_tasks(["arith2", "arith2mul", "arith_multi", "algebra", "word"], max(50, n // 20), "heldout", seed + 1)}


def gen_arith(rng: random.Random, pools: dict, split: str) -> Sample | None:
    from slm.rl.synth import trace_for_tools

    pool = pools[split]
    if not pool:
        return None
    t = pool.pop()
    try:
        think = trace_for_tools(t)
    except ValueError:
        return None
    codes = [sp.code for sp in split_markup(think) if sp.kind == "tool"]
    msgs = [{"role": "user", "content": t.prompt}, {"role": "assistant", "think": think, "content": "#### " + t.answer}]
    s = make_sample(f"arith.{t.task}", msgs, codes, numeric=True)
    s.key = f"arith::{t.id}"
    return s


# --------------------------------------------------------------------------------- build
def generate_sample(family: str, rng: random.Random, corpus: Corpus, pools: dict | None = None) -> Sample | None:
    if family == "pipeline":
        return gen_pipeline(rng, corpus)
    if family == "strings":
        return gen_strings(rng, corpus)
    if family == "numbers":
        return gen_numbers(rng, corpus)
    if family == "simulation":
        return gen_simulation(rng, corpus)
    if family == "multiturn":
        return gen_multiturn(rng, corpus)
    if family == "runcode":
        return gen_runcode(rng, corpus)
    if family == "error":
        return gen_error(rng, corpus)
    if family == "declared":
        return gen_declared(rng, corpus)
    if family == "arith":
        return gen_arith(rng, pools, "train" if rng.random() < 0.95 else "heldout")
    raise ValueError(family)


def sample_stream(rng: random.Random, corpus: Corpus, n: int, pools: dict | None = None):
    """`n` distinct conversations drawn from the family mix."""
    fams, weights = list(FAMILY_SHARES), list(FAMILY_SHARES.values())
    seen: set[str] = set()
    made = attempts = 0
    while made < n and attempts < n * 40:
        attempts += 1
        fam = rng.choices(fams, weights=weights)[0]
        s = generate_sample(fam, rng, corpus, pools)
        if s is None or s.key in seen:
            continue
        seen.add(s.key)
        made += 1
        yield s


def split_for(s: Sample, val_permille: int = 20) -> str:
    if any(s.family.startswith(h) or s.family == h for h in HOLDOUT_FAMILIES):
        return "val"
    return "val" if _split_of(s.key, val_permille) == "heldout" else "train"


def build(tok: SlmTokenizer, n: int, out_root: Path, name: str = "synthetic-python-tools", seed: int = 0,
          corpus_path: Path | None = DEFAULT_CORPUS, marker_mix: float = 0.5) -> dict:
    rng = random.Random(seed)
    corpus = load_corpus(tok, corpus_path)
    pools = make_arith_pools(int(n * FAMILY_SHARES["arith"] * 1.3) + 50, seed)
    out = out_root / name
    writers = {"train": SftShardWriter(out / "train"), "val": SftShardWriter(out / "val")}
    fam_counts: Counter = Counter()
    split_counts: Counter = Counter()
    feat_counts: Counter = Counter()
    skels: dict[str, set] = {}
    n_calls = 0
    for s in sample_stream(rng, corpus, n, pools):
        apply_style(s.messages, rng, marker_mix if s.numeric else 0.0)
        enc = encode_sample(tok, s)
        split = split_for(s)
        writers[split].add(enc.ids, enc.loss_mask)
        top = s.family.split(".")[0]
        fam_counts[s.family] += 1
        split_counts[f"{top}:{split}"] += 1
        for f in features_of(s.codes, [d.name for d in s.decls]):
            feat_counts[f] += 1
        skels.setdefault(top, set()).update(skeleton(c) for c in s.codes)
        n_calls += len(s.codes)
    for w in writers.values():
        w.flush()
    total = sum(fam_counts.values())
    m = {"name": name, "tokenizer_sha256": tok.sha256, "think_required": True, "tools": True, "functions": True,
         "seed": seed, "requested": n, "conversations": total, "tool_calls": n_calls, "marker_mix": marker_mix,
         "corpus": str(corpus_path) if corpus_path else None, "corpus_sentences": len(corpus.sentences),
         "holdout_families": list(HOLDOUT_FAMILIES),
         "family_counts": dict(sorted(fam_counts.items())),
         "top_family_counts": {f: sum(v for k, v in fam_counts.items() if k.split(".")[0] == f) for f in FAMILY_SHARES},
         "split_counts": dict(sorted(split_counts.items())),
         "feature_coverage": {k: round(v / max(1, total), 4) for k, v in sorted(feat_counts.items())},
         "distinct_skeletons": {k: len(v) for k, v in sorted(skels.items())},
         "train_examples": writers["train"].total_examples, "train_tokens": writers["train"].total_tokens, "train_targets": writers["train"].total_targets,
         "val_examples": writers["val"].total_examples, "val_tokens": writers["val"].total_tokens, "val_targets": writers["val"].total_targets}
    (out / "manifest.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--name", default="synthetic-python-tools")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="tokenized val split used for real sentences and words")
    a = ap.parse_args()
    tok = SlmTokenizer.load(a.tokenizer)
    m = build(tok, a.n, SFT_DIR / Path(a.tokenizer).name, a.name, a.seed, Path(a.corpus) if a.corpus else None)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()

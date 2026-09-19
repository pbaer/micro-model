"""The grammar-generated Python-tool set (`slm/rl/synth_python.py`): every trace verifies in the sandbox,
declared functions resolve, error traces carry a real hint, the masks are right, the program shapes are
varied, the taught features are actually present, and the hold-out families never reach train.

CPU only, and the corpus falls back to the built-in sentences when C:\\slm-data is not present.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from slm.data.answers import apply_style
from slm.data.chat import format_chat, parse_assistant
from slm.data.tokenizer import SlmTokenizer, train_bpe
from slm.rl import synth_python as sp
from slm.tools import PySession, run_tool, split_markup
from slm.tools.functions import functions_env, parse_defs

FAMILIES = list(sp.FAMILY_SHARES)
# distinct normalised program skeletons required per top-level family in a 2000-conversation sample
SKELETON_FLOOR = {"pipeline": 200, "runcode": 40, "declared": 40, "simulation": 25, "multiturn": 25,
                  "error": 25, "numbers": 20, "strings": 15, "arith": 10}
FEATURE_FLOOR = {"loop": 0.25, "def": 0.10, "list_ops": 0.30, "str_methods": 0.15, "decl_call": 0.10, "math": 0.05}


@pytest.fixture(scope="module")
def tok():
    text = "print(12*35) = 420 the quick brown fox jumps over 7 lazy dogs total = sum(nums) for x in range(10): "
    return SlmTokenizer(train_bpe([text * 40, "he has 10 - 2 = 8 trees a = 5 def f(x): return x " * 40], vocab_size=400))


@pytest.fixture(scope="module")
def corpus():
    return sp.load_corpus()  # built-in fallback sentences: no data root needed


@pytest.fixture(scope="module")
def sample_set(corpus):
    """2000 conversations from the real family mix, generated once."""
    rng = random.Random(4)
    pools = sp.make_arith_pools(400, 7)
    return list(sp.sample_stream(rng, corpus, 2000, pools))


def _gold_of(msg: dict) -> str:
    c = msg["content"]
    return c[5:].strip() if c.startswith("#### ") else c


def test_every_family_generates(corpus):
    rng = random.Random(0)
    pools = sp.make_arith_pools(200, 1)
    for fam in FAMILIES:
        made = [s for s in (sp.generate_sample(fam, rng, corpus, pools) for _ in range(40)) if s is not None]
        assert len(made) >= 30, (fam, len(made))
        assert all(s.codes for s in made), fam


def test_traces_verify_in_the_sandbox(sample_set):
    """The gold answer of every turn is what the sandbox itself produces, in one session per conversation."""
    checked = Counter()
    for s in sample_set:
        session = PySession()
        if s.decls:
            session.register(functions_env(s.decls))
        for i, m in enumerate(s.messages):
            if m["role"] != "assistant":
                continue
            gold = _gold_of(m)
            if "spans" in m:  # error-and-recover: the last call is the correct one
                results = [run_tool(code, session) for kind, code in m["spans"] if kind == "call"]
                assert len(results) >= 2, s.family
                assert results[-1] == (gold, True), (s.family, results)
            else:
                spans = [x for x in split_markup(m["think"], session) if x.kind == "tool"]
                assert spans, (s.family, m["think"])
                assert spans[-1].result == gold, (s.family, m["think"], spans[-1].result, gold)
                assert all(x.code for x in spans)
                assert not m["think"].rstrip(".").endswith(">>" + gold), "a trace must not echo the tool result"
            checked[s.family.split(".")[0]] += 1
    assert set(checked) == set(FAMILIES), checked


def test_tool_markup_only_inside_think(sample_set):
    for s in sample_set:
        for m in s.messages:
            if m["role"] == "user":
                assert "<<" not in m["content"] or "python" in m["content"]
            else:
                assert "<<" not in m["content"] and ">>" not in m["content"]


def test_error_traces_read_a_real_hint(corpus):
    """The first call must really fail (the hint comes from the sandbox, never hand-written), and the
    second must produce the gold."""
    rng = random.Random(21)
    kinds = Counter()
    for _ in range(120):
        s = sp.gen_error(rng, corpus)
        if s is None:
            continue
        bad, good = s.codes
        session = PySession()
        hint, ok = run_tool(bad, session)
        assert not ok and hint.startswith("error: "), (bad, hint)
        core = sp._hint_core(hint)
        assert core and core in s.messages[1]["spans"][2][1], (hint, s.messages[1]["spans"][2][1])
        out, ok2 = run_tool(good, session)
        assert ok2 and out == _gold_of(s.messages[1]), (good, out)
        kinds[core[:18]] += 1
    assert len(kinds) >= 8, kinds  # several different sandbox refusals are exercised


def test_declared_functions_resolve_through_the_registered_impl(corpus, tok):
    rng = random.Random(33)
    seen_services, called = set(), 0
    for _ in range(200):
        s = sp.gen_declared(rng, corpus)
        if s is None:
            continue
        assert s.decls, s.family
        names = [d.name for d in s.decls]
        gold = _gold_of(s.messages[-1])
        session = PySession(functions=functions_env(s.decls))
        out = [run_tool(code, session) for code in s.codes][-1]
        assert out == (gold, True), (s.family, s.codes, out)
        if any(f"{n}(" in "\n".join(s.codes) for n in names):
            called += 1
            # the same programs in a session without the declarations cannot resolve the name
            plain = PySession()
            outs = [run_tool(code, plain) for code in s.codes]
            assert any("NameError" in o for o, ok in outs), s.codes
        seen_services.add(s.family)
    assert called > 150 and len(seen_services) >= 6, (called, seen_services)


def test_declared_blocks_are_masked_and_calls_are_targets(tok, corpus):
    """Round trip through format_chat: def blocks and results masked, calls a target."""
    rng = random.Random(5)
    s = None
    while s is None or not s.decls or "spans" in s.messages[-1]:
        s = sp.gen_declared(rng, corpus)
    apply_style(s.messages, rng, 0.5 if s.numeric else 0.0)
    enc = sp.encode_sample(tok, s)
    d_open, d_close = tok.special("<|python_def|>"), tok.special("<|/python_def|>")
    c_open, c_close = tok.special("<|python_call|>"), tok.special("<|/python_call|>")
    r_open, r_close = tok.special("<|python_result|>"), tok.special("<|/python_result|>")
    assert enc.ids[0] == tok.bos_id and enc.ids[1] == d_open
    last_def = len(enc.ids) - 1 - enc.ids[::-1].index(d_close)
    assert all(m == 0 for m in enc.loss_mask[1 : last_def + 1]), "declaration blocks are environment-written"
    assert [f.name for f in parse_defs(tok, enc.ids)] == [d.name for d in s.decls]
    i, j = enc.ids.index(c_open), enc.ids.index(c_close)
    assert all(enc.loss_mask[k] == 1 for k in range(i, j + 1)), "the call is a loss target"
    a, b = enc.ids.index(r_open), enc.ids.index(r_close)
    assert all(enc.loss_mask[k] == 0 for k in range(a, b + 1)), "the result is not"
    assert enc.loss_mask[enc.ids.index(tok.eos_id)] == 0


def test_masks_and_parse_over_every_family(sample_set, tok):
    """No result token is ever a loss target, every conversation ends well-formed, and the assistant turn
    parses back into think + answer with the tool spans inside the think."""
    rng = random.Random(9)
    r_open, r_close = tok.special("<|python_result|>"), tok.special("<|/python_result|>")
    a_tok = tok.special("<|assistant|>")
    for s in sample_set[:400]:
        apply_style(s.messages, rng, 0.5 if s.numeric else 0.0)
        enc = sp.encode_sample(tok, s)
        assert len(enc.ids) == len(enc.loss_mask) and enc.ids[-1] == tok.eos_id
        inside = False
        for i, t in enumerate(enc.ids):
            if t == r_open:
                inside = True
            if inside:
                assert enc.loss_mask[i] == 0, s.family
            if t == r_close:
                inside = False
        assert sum(enc.loss_mask) > 0
        start = len(enc.ids) - 1 - enc.ids[::-1].index(a_tok) + 1
        parsed = parse_assistant(tok, enc.ids[start:])
        assert not parsed["malformed"], s.family
        assert "<<" in parsed["think"] or "<<<" in parsed["think"], s.family


def test_multiturn_reuses_session_state(corpus):
    """A follow-up must fail in a fresh session (it depends on the earlier turn) but succeed in the
    conversation's own session."""
    rng = random.Random(12)
    checked = 0
    for _ in range(120):
        s = sp.gen_multiturn(rng, corpus)
        if s is None or len(s.codes) < 2:
            continue
        session = PySession()
        for code in s.codes:
            assert run_tool(code, session)[1], code
        later = s.codes[-1]
        if run_tool(later, PySession())[1]:
            continue  # a follow-up that happens to be self-contained
        checked += 1
    assert checked >= 60, checked


def test_runcode_executes_the_users_program_verbatim(corpus):
    rng = random.Random(77)
    n = 0
    for _ in range(60):
        s = sp.gen_runcode(rng, corpus)
        if s is None:
            continue
        code = s.codes[0]
        assert code in s.messages[0]["content"], "the program in the question is the one that is run"
        assert run_tool(code) == (_gold_of(s.messages[1]), True)
        n += 1
    assert n >= 40


def test_feature_coverage(sample_set):
    feats = Counter()
    for s in sample_set:
        for f in sp.features_of(s.codes, [d.name for d in s.decls]):
            feats[f] += 1
    share = {k: feats[k] / len(sample_set) for k in FEATURE_FLOOR}
    for k, floor in FEATURE_FLOOR.items():
        assert share[k] >= floor, (k, share)


def test_program_diversity(sample_set):
    skels: dict[str, set] = {}
    for s in sample_set:
        skels.setdefault(s.family.split(".")[0], set()).update(sp.skeleton(c) for c in s.codes)
    for fam, floor in SKELETON_FLOOR.items():
        assert len(skels[fam]) >= floor, (fam, len(skels[fam]))
    # normalisation really collapses "same shape, other numbers"
    assert sp.skeleton("nums = [1, 2, 3]\nsum(nums)") == sp.skeleton("xs = [9, 8, 7]\nsum(xs)")
    assert sp.skeleton("sum(xs)") != sp.skeleton("max(xs)")


def test_holdout_families_are_val_only(sample_set):
    assert len(sp.HOLDOUT_FAMILIES) == 2
    seen = Counter()
    for s in sample_set:
        split = sp.split_for(s)
        seen[(s.family, split)] += 1
        if any(s.family.startswith(h) for h in sp.HOLDOUT_FAMILIES):
            assert split == "val", s.family
    for h in sp.HOLDOUT_FAMILIES:
        assert sum(v for (f, sp_), v in seen.items() if f.startswith(h)) >= 20, h
        assert not any(f.startswith(h) and sp_ == "train" for (f, sp_) in seen)


def test_build_writes_shards_and_manifest(tok, tmp_path):
    m = sp.build(tok, 300, tmp_path, name="synthetic-python-tools", seed=2, corpus_path=None)
    assert m["think_required"] and m["tools"] and m["functions"]
    assert m["train_examples"] > 0 and m["val_examples"] > 0
    assert m["train_targets"] < m["train_tokens"]
    assert set(m["top_family_counts"]) == set(FAMILIES)
    assert m["holdout_families"] == list(sp.HOLDOUT_FAMILIES)
    assert (tmp_path / "synthetic-python-tools" / "train" / "tokens_00000.bin").exists()
    assert (tmp_path / "synthetic-python-tools" / "manifest.json").exists()

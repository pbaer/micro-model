"""The RL task families added for M9: the Python-tool grammars (`slm.rl.pytool`) and instruction
constraints (`slm.rl.constraints`), their verifiers, and per-family reward schemes.

CPU only (a training run owns the GPU): the rollout tests drive a tiny model with a scripted sampler.
"""

from __future__ import annotations

import json
import random

import pytest
import torch

from slm.config import ModelConfig
from slm.data.tokenizer import SlmTokenizer, train_bpe
from slm.model import Transformer
from slm.rl import constraints as C
from slm.rl.pytool import PYTOOL_FAMILIES, PYTOOL_WEIGHTS
from slm.rl.rewards import Verdict, exact_match, resolve_scheme, reward_from_verdict, verify_answer, verify_exact
from slm.rl.rollout import rollout_group
from slm.rl.tasks import GENERATORS, Task, make_tasks, prompt_messages
from slm.tools import PySession, run_tool
from slm.tools.functions import functions_env

PYTOOL = list(PYTOOL_FAMILIES)


@pytest.fixture(scope="module")
def tok():
    text = ("print(12*35) = 420 the quick brown fox jumps over 7 lazy dogs total = sum(nums) for x in range(10): "
            "#### answer write exactly two sentences about a public library - bullet P.S. <<Title>> ")
    return SlmTokenizer(train_bpe([text * 40, "he has 10 - 2 = 8 trees a = 5 def f(x): return x " * 40], vocab_size=420))


# ------------------------------------------------------------------ (A) Python-tool tasks


@pytest.mark.parametrize("family", PYTOOL)
def test_pytool_gold_is_what_the_sandbox_prints(family):
    """Correct by construction: re-running the reference program in a fresh session reproduces the gold."""
    tasks = make_tasks([family], 40, "train", seed=11)
    assert len(tasks) == 40
    for t in tasks:
        assert t.task == family and t.prompt and t.answer
        session = PySession()
        if t.meta.get("functions"):
            session.register(functions_env(t.meta["functions"]))
        out = ok = None
        for code in t.meta["codes"]:
            out, ok = run_tool(code, session)
        assert ok and out == t.answer, (family, t.meta["family"], t.meta["codes"], out, t.answer)
        # the verifier the rollout path will use accepts the gold written as the model must write it
        assert verify_answer(f"#### {t.answer}", t.answer).correct, (family, t.answer)


def test_pytool_umbrella_covers_the_families_and_is_deterministic():
    tasks = make_tasks(["pytool"], 300, "train", seed=3)
    tops = {t.meta["family"].split(".")[0] for t in tasks}
    assert tops == {PYTOOL_FAMILIES[f] for f in PYTOOL_WEIGHTS}, tops
    assert all(t.task in PYTOOL_FAMILIES for t in tasks)
    assert [t.prompt for t in tasks] == [t.prompt for t in make_tasks(["pytool"], 300, "train", seed=3)]
    assert all(name in GENERATORS for name in [*PYTOOL, "pytool", "constraints"])


def test_pytool_declared_tasks_carry_runnable_declarations():
    tasks = make_tasks(["pytool_declared"], 60, "train", seed=5)
    with_fns = [t for t in tasks if t.meta.get("functions")]
    assert len(with_fns) >= 55, len(with_fns)
    for t in with_fns:
        names = [d.name for d in t.meta["functions"]]
        assert all(d.impl is not None for d in t.meta["functions"]), names
        session = PySession(functions=functions_env(t.meta["functions"]))
        used = [n for n in names if any(f"{n}(" in c for c in t.meta["codes"])]
        assert used or t.meta["family"].startswith("declared.unneeded"), (t.meta["family"], names)
        out, ok = run_tool(t.meta["codes"][-1], session)
        assert ok and out == t.answer, (t.meta["family"], out, t.answer)
        # without the declarations the very same program cannot run: the impls are what make the task solvable
        if used:
            bad, ok2 = run_tool(t.meta["codes"][-1], PySession())
            assert not ok2 and "NameError" in bad, (t.meta["family"], bad)


@pytest.mark.parametrize("family", [*PYTOOL, "pytool", "constraints"])
def test_new_families_split_train_and_heldout_disjointly(family):
    tr = make_tasks([family], 120, "train", seed=2)
    ho = make_tasks([family], 40, "heldout", seed=2)
    assert len(tr) == 120 and len(ho) == 40
    assert not ({t.prompt for t in tr} & {t.prompt for t in ho})
    # the same seed on the other side of the split must not smuggle a prompt across either
    assert not ({t.prompt for t in make_tasks([family], 120, "train", seed=9)} & {t.prompt for t in ho})


def test_heldout_keeps_the_family_mix():
    """The held-out set selects best.pt, so it has to look like the training mix. gsm8k prompts come from a
    pre-filtered pool and always land in the split; generated ones only do about 1 time in 10."""
    names = ["gsm8k", "pytool", "pytool", "constraints", "constraints"] if _has_gsm8k() else ["arith2", "pytool", "pytool", "constraints", "constraints"]
    ho = make_tasks(names, 200, "heldout", seed=1)
    share = sum(t.task.startswith("pytool") or t.task == "constraints" for t in ho) / len(ho)
    assert 0.6 <= share <= 0.9, share


def _has_gsm8k() -> bool:
    try:
        from slm.rl.tasks import gsm8k_pool

        return bool(gsm8k_pool())
    except Exception:  # noqa: BLE001 - no data root on this machine
        return False


def test_prompt_suffix_matches_the_answer_shape():
    num = make_tasks(["arith2"], 1, "train", 0)[0]
    assert prompt_messages(num)[0]["content"].endswith("'#### <number>'.")
    words = [t for t in make_tasks(["pytool_strings", "pytool_declared"], 60, "train", 4) if not t.meta["numeric"]]
    assert words and all(prompt_messages(t)[0]["content"].endswith("'#### <answer>'.") for t in words)
    con = make_tasks(["constraints"], 1, "train", 0)[0]
    assert prompt_messages(con) == [{"role": "user", "content": con.prompt}]  # no marker instruction: the answer is the writing


# ------------------------------------------------------------------ verify_exact


@pytest.mark.parametrize("text,gold,ok", [
    ("#### mugs", "mugs", True),
    ("#### 'mugs'", "mugs", True),
    ("#### Mugs.", "mugs", True),
    ("#### mug", "mugs", False),
    ("#### mugs and tiles", "mugs", False),  # an extra token is a different answer
    ("#### True", "True", True),
    ("#### true", "True", True),
    ("#### False", "True", False),
    ("#### ['mugs', 'tiles']", "['mugs', 'tiles']", True),
    ("#### [mugs, tiles]", "['mugs', 'tiles']", True),
    ("#### mugs, tiles", "['mugs', 'tiles']", True),
    ("#### ['tiles', 'mugs']", "['mugs', 'tiles']", False),  # order is part of the answer
    ("#### ['mugs', 'bolts']", "['mugs', 'tiles']", False),  # one wrong item fails
    ("#### ['mugs', 'tiles', 'ropes']", "['mugs', 'tiles']", False),  # an extra item fails
    ("#### ['mugs']", "['mugs', 'tiles']", False),
    ("#### [3, 7]", "[3, 7]", True),
    ("#### [3.0, 7]", "[3, 7]", True),  # formatting is tolerated per item, as for a scalar
    ("#### []", "[]", True),
    ("#### nothing", "[]", False),
    ("the answer is mugs", "mugs", False),  # the marker is required
    ("#### mugs", "tiles", False),
])
def test_verify_exact(text, gold, ok):
    assert verify_exact(text, gold).correct is ok


def test_verify_exact_lenient_and_dispatch():
    assert verify_exact("the ones left are:\nmugs", "mugs", strict=False).correct  # lenient: the last line is the answer
    assert verify_answer("#### 7", "7").reason == "numeric compare"  # numeric gold keeps the numeric path
    assert verify_answer("#### 7.0", "7").correct and verify_answer("#### seven", "7").correct is False
    assert verify_answer("#### mugs", "mugs").reason == "exact compare"
    assert verify_answer("#### [3, 7]", "[3, 7]").reason == "exact compare"
    assert exact_match("3", "3.0") and not exact_match("3", "4")
    with pytest.raises(ValueError):
        verify_answer("#### 1", "1", "nonsense")


def test_non_numeric_answer_still_counts_as_tool_work():
    from slm.rl.rewards import answer_from_tool

    assert answer_from_tool("['mugs', 'tiles']", [("low = []\nfor it in items:\n    low.append(it)\nlow", "['mugs', 'tiles']")])
    assert not answer_from_tool("['mugs', 'tiles']", [("['mugs', 'tiles']", "['mugs', 'tiles']")])  # a bare literal launders nothing
    assert not answer_from_tool("mugs", [("sorted(x)", "['mugs', 'tiles']")])


# ------------------------------------------------------------------ (B) constraints

SATISFYING = {
    "sentences_exactly": ({"type": "sentences_exactly", "n": 2}, "One thing here. Two things there.", "Only one sentence here."),
    "sentences_min": ({"type": "sentences_min", "n": 3}, "A a. B b. C c.", "A a. B b."),
    "sentences_max": ({"type": "sentences_max", "n": 2}, "A a. B b.", "A a. B b. C c."),
    "words_min": ({"type": "words_min", "n": 5}, "one two three four five six", "one two three"),
    "words_max": ({"type": "words_max", "n": 4}, "one two three", "one two three four five"),
    "contains_word": ({"type": "contains_word", "word": "river"}, "the river was wide", "the stream was wide"),
    "contains_phrase": ({"type": "contains_phrase", "phrase": "carried silt"}, "it carried silt downstream", "it carried sand downstream"),
    "forbid_word": ({"type": "forbid_word", "word": "river"}, "the stream was wide", "the River was wide"),
    "bullets_exactly": ({"type": "bullets_exactly", "n": 2}, "- one\n- two", "- one\n- two\n- three"),
    "all_lowercase": ({"type": "all_lowercase"}, "all of it is lowercase.", "One Capital ruins it."),
    "no_commas": ({"type": "no_commas"}, "no commas at all here", "here, there are commas"),
    "title_angles": ({"type": "title_angles"}, "<<A Short Title>>\nthen the text", "A Short Title\nthen the text"),
    "end_with": ({"type": "end_with", "phrase": "nothing more to add"}, "I am done and nothing more to add\n", "nothing more to add, really"),
    "start_with": ({"type": "start_with", "word": "clearly"}, "Clearly, this works.", "This clearly works."),
    "postscript": ({"type": "postscript"}, "The note.\nP.S. one more thing", "The note.\nPost script: one more thing"),
    "paragraphs_exactly": ({"type": "paragraphs_exactly", "n": 2}, "first para\n\nsecond para", "first para\nsecond line"),
    "number_between": ({"type": "number_between", "lo": 10, "hi": 20}, "about 14 of them", "about 40 of them"),
    "sentence_words_max": ({"type": "sentence_words_max", "n": 5}, "one two. three four.", "one two three four five six."),
}


def test_every_constraint_type_has_a_case():
    assert set(SATISFYING) == set(C.TYPES) and len(C.TYPES) >= 15


@pytest.mark.parametrize("name", sorted(SATISFYING))
def test_constraint_checker_accepts_and_rejects(name):
    spec, good, bad = SATISFYING[name]
    assert C.check_one(good, spec), (name, good)
    assert not C.check_one(bad, spec), (name, bad)
    assert not C.check_one("   ", spec), name  # an empty answer satisfies nothing
    assert C.describe(spec) and isinstance(C.describe(spec), str)


def test_generated_constraint_tasks_are_well_formed():
    rng = random.Random(0)
    seen = set()
    for _ in range(400):
        t = C.gen_constraints(rng)
        assert t is not None
        specs = C.parse_specs(t.answer)
        assert 1 <= len(specs) <= 3 and specs == t.meta["constraints"]
        groups = [C.TYPES[s["type"]].group for s in specs]
        assert len(set(groups)) == len(groups)  # one per group
        for a in groups:
            for b in groups:
                assert a == b or frozenset((a, b)) not in C.CONFLICTS, (a, b)
        for s in specs:  # every constraint is stated in the prompt, in English, composed from the spec
            assert C.describe(s) in t.prompt
        seen.update(s["type"] for s in specs)
    assert seen == set(C.TYPES), set(C.TYPES) - seen


def test_constraint_verdict_and_fraction_scheme_is_monotone():
    specs = [{"type": "all_lowercase"}, {"type": "no_commas"}, {"type": "sentences_exactly", "n": 2}]
    gold = json.dumps(specs)
    texts = ["Nope, one sentence with a comma And capitals.",  # 0/3
             "no capitals, one sentence with a comma",  # 1/3
             "no capitals, two sentences. here is the second one",  # 2/3
             "no capitals. two clean sentences here"]  # 3/3
    rewards = []
    for i, text in enumerate(texts):
        v = verify_answer(text, gold, "constraints")
        assert v.fraction == pytest.approx(i / 3) and v.correct == (i == 3)
        rewards.append(reward_from_verdict(v, False, "fraction"))
    assert rewards == sorted(rewards) and rewards[0] == 0.0 and rewards[-1] == 1.0
    assert all(b > a for a, b in zip(rewards, rewards[1:]))
    # malformed scores nothing under every scheme, and a spec-less gold is simply wrong
    v = verify_answer(texts[-1], gold, "constraints")
    assert reward_from_verdict(v, True, "fraction") == 0.0 and reward_from_verdict(v, True, "tool") == 0.0
    assert not verify_answer("anything", "not json", "constraints").correct


def test_fraction_scheme_falls_back_to_the_binary_verdict():
    assert reward_from_verdict(Verdict(True, "7", "numeric compare"), False, "fraction") == 1.0
    assert reward_from_verdict(Verdict(False, "8", "numeric compare"), False, "fraction") == 0.0


# ------------------------------------------------------------------ per-family reward schemes


def test_reward_scheme_resolution():
    schemes = {"constraints": "fraction", "pytool": "tool", "pytool_runcode": "binary"}
    assert resolve_scheme("constraints", schemes, "tool") == "fraction"
    assert resolve_scheme("pytool_declared", schemes, "binary") == "tool"  # by group prefix
    assert resolve_scheme("pytool_runcode", schemes, "binary") == "binary"  # the exact name wins over the prefix
    assert resolve_scheme("gsm8k", schemes, "tool") == "tool"  # falls through to the run default
    assert resolve_scheme("arith1", {}, "shaped") == "shaped" and resolve_scheme("arith1", None, "binary") == "binary"


def test_existing_scheme_behaviour_is_unchanged():
    ok, bad = Verdict(True, "7", "numeric compare"), Verdict(False, "5", "numeric compare")
    assert [reward_from_verdict(ok, False, s) for s in ("binary", "signed", "shaped")] == [1.0, 1.0, 1.0]
    assert [reward_from_verdict(bad, False, s) for s in ("binary", "signed", "shaped")] == [0.0, -1.0, 0.0]
    assert reward_from_verdict(Verdict(False, None, "x"), True, "shaped") == -0.5
    assert reward_from_verdict(ok, False, "tool", True) == 1.0 and reward_from_verdict(ok, False, "tool", False) == 0.5
    with pytest.raises(ValueError):
        reward_from_verdict(ok, False, "nope")


# ------------------------------------------------------------------ end to end, scripted sampler


def _tiny_model(tok):
    cfg = ModelConfig(vocab_size=tok.vocab_size, n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, d_ff=128, max_seq_len=1024)
    torch.manual_seed(0)
    return Transformer(cfg)


def _script(monkeypatch, rounds: list[list[int]]):
    """Every row of the group gets the same scripted completion, one entry per decoding round."""
    state = {"i": 0}

    def fake_sample(model, tk, prompts, n, temperature, top_p=1.0, top_k=0, generator=None, stop=None):
        k = min(state["i"], len(rounds) - 1)
        state["i"] += 1
        return [rounds[k][:n] for _ in prompts]

    monkeypatch.setattr("slm.rl.rollout.sample_completions", fake_sample)


def test_rollout_of_a_declared_function_task(tok, monkeypatch):
    """A declared-function task end to end: the prompt carries the declaration blocks, the scripted call
    runs against the registered impl, and the tool scheme pays the full reward."""
    task = next(t for t in make_tasks(["pytool_declared"], 40, "train", 5)
                if t.meta.get("functions") and len(t.meta["codes"]) == 1 and "(" in t.meta["codes"][0])
    code = task.meta["codes"][0]
    assert tok.decode(tok.encode(code), skip_special=True) == code  # the tiny BPE round-trips the program
    call = [tok.special("<|python_call|>"), *tok.encode(code), tok.special("<|/python_call|>")]
    finish = [tok.special("<|/think|>"), *tok.encode(f"#### {task.answer}"), tok.end_id]
    _script(monkeypatch, [call, finish])
    m = _tiny_model(tok)
    g = rollout_group(m, tok, task, group_size=2, max_new_tokens=400, temperature=0.0, think_required=True,
                      tools=True, reward_scheme="binary", reward_schemes={"pytool": "tool"})
    d_open = tok.special("<|python_def|>")
    assert d_open in g[0].prompt_ids, "the declaration blocks belong to the prompt"
    for r in g:
        assert r.tool_calls == 1 and r.tool_errors == 0 and r.tool_results[0] == [code, task.answer]
        assert r.correct and not r.malformed and r.termination == "stop"
        assert r.answer_from_tool and r.reward == 1.0  # per-family scheme: pytool -> tool
        assert sum(r.gen_mask) < len(r.gen_mask) == r.n_tokens  # the inserted result is excluded


def test_rollout_without_the_declaration_fails_the_same_call(tok, monkeypatch):
    """The other half of the wiring: strip meta["functions"] and the identical program is a NameError."""
    task = next(t for t in make_tasks(["pytool_declared"], 40, "train", 5)
                if t.meta.get("functions") and len(t.meta["codes"]) == 1 and "(" in t.meta["codes"][0])
    names = [d.name for d in task.meta["functions"]]
    code = task.meta["codes"][0]
    if not any(f"{n}(" in code for n in names):
        pytest.skip("this draw does not call its declaration")
    bare = Task(id=task.id, prompt=task.prompt, answer=task.answer, task=task.task, meta={k: v for k, v in task.meta.items() if k != "functions"})
    call = [tok.special("<|python_call|>"), *tok.encode(code), tok.special("<|/python_call|>")]
    finish = [tok.special("<|/think|>"), *tok.encode(f"#### {task.answer}"), tok.end_id]
    _script(monkeypatch, [call, finish])
    g = rollout_group(_tiny_model(tok), tok, bare, group_size=1, max_new_tokens=400, temperature=0.0, tools=True)
    assert g[0].tool_errors == 1 and "NameError" in g[0].tool_results[0][1]
    assert tok.special("<|python_def|>") not in g[0].prompt_ids


def test_rollout_of_a_constraint_task(tok, monkeypatch):
    specs = [{"type": "all_lowercase"}, {"type": "no_commas"}, {"type": "sentences_exactly", "n": 2}]
    task = Task(id="c-1", prompt="write a short note about a public library", answer=json.dumps(specs), task="constraints",
                meta={"verifier": "constraints", "answer_style": "free", "constraints": specs})
    answer = "a quiet room full of books. anyone may walk in and read"  # lowercase, no commas, 2 sentences
    _script(monkeypatch, [[tok.special("<|/think|>"), *tok.encode(answer), tok.end_id]])
    m = _tiny_model(tok)
    g = rollout_group(m, tok, task, group_size=2, max_new_tokens=200, temperature=0.0, think_required=True,
                      reward_scheme="tool", reward_schemes={"constraints": "fraction"})
    for r in g:
        assert r.correct and r.fraction == 1.0 and r.reward == 1.0 and r.verifier == "constraints: all satisfied"
        assert r.gold == task.answer and r.task == "constraints"
    # two of three: the fraction scheme still pays, where `tool` (the run default) would have paid nothing
    half = "A quiet room full of books. anyone may walk in and read"
    _script(monkeypatch, [[tok.special("<|/think|>"), *tok.encode(half), tok.end_id]])
    g2 = rollout_group(m, tok, task, group_size=1, max_new_tokens=200, temperature=0.0, think_required=True,
                       reward_scheme="tool", reward_schemes={"constraints": "fraction"})
    assert not g2[0].correct and g2[0].fraction == pytest.approx(2 / 3) and g2[0].reward == pytest.approx(2 / 3)
    assert "all_lowercase" in g2[0].verifier
    # the prompt asks for no '####' marker
    assert tok.decode(g2[0].prompt_ids, skip_special=True).count("####") == 0

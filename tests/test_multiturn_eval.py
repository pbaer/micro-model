"""The scripted multi-turn eval (slm.eval.multiturn): held-out material, deterministic conversations, scorers that agree
with the RL verifiers, and the result-file shape the Evals tab reads. No model is loaded."""

import json
import random
import re

import pytest

import slm.eval.multiturn as M
import slm.rl.constraints as C
import slm.rl.synth_chat as SC
from slm.rl.rewards import verify_constraints, verify_recall


# ------------------------------------------------------------------------------------------------ held out from training
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower()).strip()


def _training_strings() -> tuple[set[str], set[str], set[str]]:
    """(table values, standing words / phrases, sentences) of the RL chat families and the constraints generator."""
    values = {v for t in (SC.PET_NAMES, SC.PEOPLE, SC.TOWNS, SC.JOBS, SC.DISHES, SC.FLOORS) for v in t}
    words, texts = set(), set()
    for stmt, q, _, subject, sibling in SC.RECALL_FACTS:
        texts |= {stmt, q, sibling}
        words.add(subject)
    texts |= {s.strip() for s in SC.REMEMBER if s.strip()} | set(SC.ACKS) | set(SC._REVISE_ASKS)
    texts |= {x for pair in SC._FALLBACK_PAIRS for x in pair}
    for seed in range(300):
        for rule in SC.RULES:
            spec, sentence, _ = rule(random.Random(seed))
            texts.add(sentence)
            words |= {spec[k] for k in ("phrase", "word") if k in spec}
        words.add(C._draw(random.Random(seed), "end_with", None, [])["phrase"])
    return values, words, texts


def _eval_strings() -> tuple[set[str], set[str], set[str]]:
    values = {v for t in (M.BOATS, M.GIVEN_NAMES, M.PLACES, M.INSTRUMENTS, M.HOBBIES, M.TORTOISE_NAMES) for v in t}
    words = set(M.EVAL_END_PHRASES) | set(M.SYS_END_PHRASES) | set(M.SYS_FORBID_WORDS)
    texts = {s for stmt, q, _ in M.ABSENT_FACTS for s in (stmt, q)} | {s.strip() for s in M.ABSENT_REMEMBER}
    texts |= {x for pair in M.PARAGRAPHS for x in pair} | set(M.REVISE_ASKS) | set(M.SYS_PERSONAS)
    probe = {"n": 2, "word": "zzz", "phrase": "zzz"}
    texts |= {f(probe) for f in M.SYS_SENTENCES.values()} | {f(probe) for f in M.REVISE_PHRASES.values()}
    return values, words, texts


def test_held_out_tables_share_no_string_with_the_training_families():
    tv, tw, tt = _training_strings()
    ev, ew, et = _eval_strings()
    train_all = {_norm(x) for x in tv | tw | tt}
    # table values: never equal, never one inside the other (verify_recall matches values as substrings)
    for e in ev:
        for t in tv | tw:
            a, b = _norm(e), _norm(t)
            assert a != b, (e, t)
            if min(len(a), len(b)) >= 3:
                assert a not in b and b not in a, (e, t)
    # standing phrases and banned words: disjoint as normalized strings and never nested
    for e in ew:
        assert _norm(e) not in train_all, e
        for t in tw:
            if min(len(_norm(e)), len(_norm(t))) >= 4:
                assert _norm(e) not in _norm(t) and _norm(t) not in _norm(e), (e, t)
    # sentences (statements, questions, paragraphs, system rules, rewrite asks): no equal or contained sentence
    for e in et:
        for t in tt:
            a, b = _norm(e), _norm(t)
            assert a != b, (e, t)
            if min(len(a), len(b)) >= 15:
                assert a not in b and b not in a, (e, t)
    # our instruction wording differs from the constraint describer the RL tasks use
    for name in M.REVISE_TYPES:
        spec = {"type": name, "n": 2, "word": "lamp", "phrase": "and that covers it."}
        assert _norm(M.REVISE_PHRASES[name](spec)) != _norm(C.describe(spec)), name


def test_paragraph_table_is_big_enough_and_well_formed():
    assert len(M.PARAGRAPHS) >= 30 and len({q for q, _ in M.PARAGRAPHS}) == len(M.PARAGRAPHS)
    for q, a in M.PARAGRAPHS:
        assert q.endswith(("?", ".")) and 2 <= len(C.sentences(a)) <= 4 and "," in a and a != a.lower(), q


# ------------------------------------------------------------------------------------------------ building
@pytest.mark.parametrize("kind", M.KINDS)
def test_each_kind_is_deterministic_per_seed(kind):
    a = M.build_conversations([kind], 24, seed=5)
    assert a == M.build_conversations([kind], 24, seed=5)
    assert a != M.build_conversations([kind], 24, seed=6)
    assert len(a) == 24 and all(c["kind"] == kind for c in a)


def test_recall_is_unchanged_and_kinds_do_not_disturb_each_other():
    old = M.make_conversations(40, seed=3)
    rec = M.build_conversations(["recall"], 40, seed=3)
    assert [{k: v for k, v in c.items() if k != "kind"} for c in rec] == old, "recall conversations are the pre-kinds ones"
    everything = M.build_conversations(list(M.KINDS), 40, seed=3)
    for k in M.KINDS:
        assert [c for c in everything if c["kind"] == k] == M.build_conversations([k], 40, seed=3), k
    with pytest.raises(ValueError):
        M.build_conversations(["recall", "nope"], 2, 0)


def test_recall_absent_asks_about_something_never_stated():
    convs = M.build_conversations(["recall_absent"], 60, seed=1)
    assert len({tuple(c["turns"]) for c in convs}) == 60
    for c in convs:
        assert c["stated"] in c["turns"][0] and c["stated"] in c["not_these"] and c["turns"][1] in M.DISTRACTORS
        assert not any(x.lower() in c["turns"][2].lower() for x in c["not_these"]), "the question must not name the answer"
        assert c["turns"][2] in {q for _, q, _ in M.ABSENT_FACTS} and c["given"] == []
    with pytest.raises(ValueError):
        M.make_absent(M.max_absent_conversations() + 1, 0)


def test_revise_constraints_are_well_formed_and_satisfiable_types():
    convs = M.build_conversations(["revise"], 80, seed=2)
    paragraphs = dict(M.PARAGRAPHS)
    for c in convs:
        q, ask = c["turns"]
        assert paragraphs[q] == c["given"][0], "the given answer is the eval's own paragraph for that question"
        specs = c["specs"]
        assert 1 <= len(specs) <= 3 and {s["type"] for s in specs} <= set(M.REVISE_TYPES)
        groups = [C.TYPES[s["type"]].group for s in specs]
        assert len(set(groups)) == len(groups) and not any(frozenset((g, h)) in C.CONFLICTS for g in groups for h in groups if g != h)
        for s in specs:
            assert M.REVISE_PHRASES[s["type"]](s) in ask
            if s["type"] in ("contains_word", "forbid_word"):
                assert C._has_word(c["given"][0], s["word"]), "keep / leave out a word the answer actually has"
        forbidden = [s["word"] for s in specs if s["type"] == "forbid_word"]
        required = [s.get("word") or s.get("phrase") for s in specs if s["type"] in ("contains_word", "end_with")]
        assert not any(C._has_word(r, w) for w in forbidden for r in required), specs
    assert {s["type"] for c in convs for s in c["specs"]} == set(M.REVISE_TYPES), "every type is drawn"


def test_sysrule_given_answer_keeps_the_rule_and_the_system_states_it():
    convs = M.build_conversations(["sysrule"], 80, seed=4)
    assert any(len(c["specs"]) == 2 for c in convs) and any(len(c["specs"]) == 1 for c in convs)
    for c in convs:
        ok, failed = C.check_constraints(c["given"][0], c["specs"])
        assert not failed, (c["specs"], c["given"][0])
        assert c["turns"][0] != c["turns"][1] and c["rule"] == "+".join(s["type"] for s in c["specs"])
        for s in c["specs"]:
            assert M.SYS_SENTENCES[s["type"]](s) in c["system"]
        assert tuple(sorted(s["type"] for s in c["specs"])) not in {("end_with", "sentences_max")}


def test_messages_carry_system_given_and_generated_turns():
    c = M.build_conversations(["sysrule"], 1, seed=0)[0]
    msgs = M.messages_for(c, 1, [])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == c["given"][0] and msgs[3]["content"] == c["turns"][1]
    folded = M.messages_for(c, 1, [], system_in_template=False)
    assert [m["role"] for m in folded] == ["user", "assistant", "user"] and folded[0]["content"].startswith(c["system"])
    r = M.build_conversations(["recall_absent"], 1, seed=0)[0]
    gen = [{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert [m.get("content") for m in M.messages_for(r, 2, gen)] == [r["turns"][0], "a", r["turns"][1], "b", r["turns"][2]]


def test_system_role_probe():
    class WithSystem:
        def render_chat(self, msgs):
            return "".join(f"<{m['role']}>{m['content']}" for m in msgs)

    class DropsSystem:
        def render_chat(self, msgs):
            return "".join(m["content"] for m in msgs if m["role"] != "system")

    class Raises:
        def render_chat(self, msgs):
            raise ValueError("System role not supported")

    assert M.system_role_supported(WithSystem())
    assert not M.system_role_supported(DropsSystem()) and not M.system_role_supported(Raises())


# ------------------------------------------------------------------------------------------------ scoring
ABSENT_REPLIES = ["You haven't told me what your canoe is called.", "Your canoe is called Kestrel.", "Kestrel, I think.", "Blue.",
                  "I don't know -- you never mentioned a canoe, only your sailboat Kestrel.", "#### none", ""]


def test_absent_scorer_is_the_reward_rule():
    table = M.BOATS
    gold = json.dumps({"absent": True, "not_these": table})
    for reply in ABSENT_REPLIES:
        assert M.score_absent(reply, table).correct == verify_recall(reply, gold).correct, reply
    assert M.score_absent(ABSENT_REPLIES[0], table).correct
    assert not M.score_absent(ABSENT_REPLIES[1], table).correct and not M.score_absent(ABSENT_REPLIES[4], table).correct


CONSTRAINT_REPLIES = ["- tides follow the moon\n- two bulges\n- two high tides a day", "Tides follow the Moon, mostly.",
                      "tides follow the moon and that covers it.", "One sentence. Two sentences. Three sentences. Four.", ""]


def test_constraint_scorer_is_the_reward_rule():
    for c in M.build_conversations(["revise", "sysrule"], 20, seed=9):
        for reply in CONSTRAINT_REPLIES:
            v = M.score_constraints(reply, c["specs"])
            ok, failed = C.check_constraints(reply, c["specs"])
            assert v.fraction == ok / len(c["specs"]) and v.correct == (not failed), (reply, c["specs"])
            assert v == verify_constraints(reply, json.dumps(c["specs"]))


def _fake_run(kinds, n=4, seed=0, tools=True):
    """Fill every generated turn with a hand-written reply and score, as `run` does after generation."""
    convs = M.build_conversations(kinds, n, seed)
    for i, c in enumerate(convs):
        k = c["kind"]
        finals = {"recall": c.get("fact", "") + " is what you told me." if i % 2 == 0 else "No idea.",
                  "recall_absent": "You never told me that." if i % 2 == 0 else M.ABSENT_FACTS[0][2][0],
                  "revise": "- one\n- two" if i % 2 == 0 else "Nothing, really.",
                  "sysrule": M.sysrule_transform("A plain reply, with a comma. And another sentence. And a third one.", c.get("specs") or [])}
        n_gen = len(c["turns"]) - len(c.get("given") or [])
        c["assistant"] = [{"answer": "ok", "think": "", "terminated": True, "tool_calls": 0, "n_tokens": 3} for _ in range(n_gen - 1)]
        c["assistant"].append({"answer": finals[k], "think": "", "terminated": i % 3 != 0, "tool_calls": int(i % 4 == 0), "n_tokens": 5})
    return convs, M._score(convs, n, seed, 0.0, tools=tools)


def test_scores_and_result_shape():
    convs, s = _fake_run(list(M.KINDS))
    old_keys = ["n", "seed", "recall", "format", "misfire", "templated", "mean_answer_tokens", "seconds"]
    assert list(s)[: len(old_keys)] == old_keys, "the pre-kinds summary fields come first, unchanged"
    assert s["kinds"] == list(M.KINDS) and s["counts"] == {k: 4 for k in M.KINDS}
    for k in ("recall_absent", "revise", "revise_all", "sysrule", "sysrule_all"):
        assert 0.0 <= s[k] <= 1.0, k
    assert s["sysrule_all"] == 1.0, "a reply transformed to keep the rule keeps it"
    assert set(s["by_kind"]) == set(M.KINDS) and all(set(v) >= {"n", "score", "format", "misfire", "templated"} for v in s["by_kind"].values())
    turns = sum(len(c["assistant"]) for c in convs)
    assert s["misfire"] == round(sum(c["misfires"] for c in convs) / turns, 3), "misfire is per generated turn, every kind"
    assert s["format"] == round(sum(c["format_ok"] for c in convs) / len(convs), 3)
    for c in convs:
        assert {"kind", "turns", "assistant", "format_ok", "misfires", "templated"} <= set(c)
        if c["kind"] in ("revise", "sysrule"):
            assert {"given", "specs", c["kind"], c["kind"] + "_all", "failed"} <= set(c)
        if c["kind"] == "recall_absent":
            assert {"stated", "not_these", "recall_absent", "reason"} <= set(c)
    res = {"checkpoint": "x.pt", "summary": s, "conversations": convs}
    assert json.loads(json.dumps(res)) == res


def test_recall_only_summary_matches_the_old_definition():
    convs, s = _fake_run(["recall"], n=6)
    assert s["recall"] == round(sum(c["fact"].lower() in c["assistant"][2]["answer"].lower() for c in convs) / 6, 3)
    assert s["misfire"] == round(sum(c["misfires"] for c in convs) / (3 * 6), 3)
    assert s["recall_absent"] is None and s["revise"] is None and s["sysrule"] is None and s["counts"] == {"recall": 6}
    _, ext = _fake_run(["recall", "sysrule"], tools=False)
    assert ext["misfire"] is None and all(v["misfire"] is None for v in ext["by_kind"].values())

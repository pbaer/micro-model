"""The chat-rehearsal RL family and the multi-turn eval's scaffolding (both CPU-only)."""

import random

import pytest

from slm.rl.rewards import Verdict, reward_from_verdict, verify_answer, verify_plain


def test_verify_plain_accepts_a_chat_answer_and_rejects_verifier_shapes():
    """The exact outputs M9 stage C produced on knowledge questions after RL saw only verifiable families."""
    assert verify_plain("The play Hamlet was written by William Shakespeare, around 1600.").correct
    assert verify_plain("It is a phenomenon called Rayleigh scattering, in which shorter wavelengths scatter more.").correct
    for bad in ("So the answer is 2.", "So the answer is True.", "That would be China.", "#### 42", "", "   "):
        v = verify_plain(bad)
        assert not v.correct, bad
    # dispatch by kind, and gold is irrelevant to it
    assert verify_answer("A nice long answer about cats and their habits.", "", "plain").correct
    assert not verify_answer("The answer is 7.", "", "plain").correct


def test_plain_scheme_pays_only_for_a_plain_answer_with_no_tool_call():
    ok = Verdict(True, "x", "plain chat answer")
    assert reward_from_verdict(ok, malformed=False, scheme="plain", n_calls=0) == 1.0
    assert reward_from_verdict(ok, malformed=False, scheme="plain", n_calls=1) == 0.0, "a tool call on a chat prompt is the misfire"
    assert reward_from_verdict(ok, malformed=True, scheme="plain", n_calls=0) == 0.0
    assert reward_from_verdict(Verdict(False, None, "empty answer"), malformed=False, scheme="plain") == 0.0
    # the argument is optional so every other scheme is unchanged
    assert reward_from_verdict(ok, malformed=False, scheme="binary") == 1.0


def test_tool_strict_widens_the_gap_without_touching_tool():
    """tool: 1.0 / 0.5 / 0. tool_strict: 1.0 / 0.25 / 0. Same answer, same verdict, only the no-call credit moves."""
    ok = Verdict(True, "42", "numeric compare")
    for scheme, no_call in (("tool", 0.5), ("tool_strict", 0.25)):
        assert reward_from_verdict(ok, False, scheme, from_tool=True) == 1.0
        assert reward_from_verdict(ok, False, scheme, from_tool=False) == no_call
        assert reward_from_verdict(Verdict(False, "41", "numeric compare"), False, scheme, from_tool=True) == 0.0
        assert reward_from_verdict(ok, True, scheme, from_tool=True) == 0.0


def test_chat_pool_and_pooled_families():
    """Pool entries are short non-computational user turns with the plain verifier and no answer suffix;
    make_tasks draws them through the same pooled path as gsm8k, split by prompt hash."""
    from slm.rl.tasks import POOLED, chat_pool, make_tasks, prompt_messages

    pool = chat_pool()
    if not pool:
        pytest.skip("SmolTalk raw parquet not present on this machine")
    assert "chat" in POOLED and "gsm8k" in POOLED
    for t in pool[:200]:
        assert t.task == "chat" and t.answer == ""
        assert t.meta["verifier"] == "plain" and t.meta["answer_style"] == "free"
        assert 15 <= len(t.prompt) <= 200 and "```" not in t.prompt
        assert prompt_messages(t) == [{"role": "user", "content": t.prompt}], "a chat prompt must not ask for ####"
    train = make_tasks(["chat", "arith1"], 40, "train", seed=1)
    held = make_tasks(["chat", "arith1"], 20, "heldout", seed=1)
    assert any(t.task == "chat" for t in train) and any(t.task == "chat" for t in held)
    assert not ({t.prompt for t in train} & {t.prompt for t in held}), "train/heldout leak"


def test_multiturn_conversations_are_deterministic_and_distinct():
    from slm.eval.multiturn import DISTRACTORS, FACTS, make_conversations

    a = make_conversations(48, seed=3)
    b = make_conversations(48, seed=3)
    assert a == b
    assert len({(c["turns"][0], c["turns"][1]) for c in a}) == 48, "distinct on (statement, distractor); a fact may recur with a new distractor"
    for c in a:
        assert len(c["turns"]) == 3 and c["fact"] in c["turns"][0] and c["turns"][1] in DISTRACTORS
        assert c["fact"] not in c["turns"][1] and c["fact"] not in c["turns"][2], "turn 3 must need the memory"
    # every fact template is reachable, and the cap is enforced rather than spun on
    from slm.eval.multiturn import max_conversations

    many = make_conversations(min(400, max_conversations()), seed=0)
    assert {stmt for stmt, _, _ in FACTS} <= {c["turns"][0].split(" Please")[0].replace(c["fact"], "{x}") for c in many}
    with pytest.raises(ValueError):
        make_conversations(max_conversations() + 1, seed=0)

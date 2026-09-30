"""The multi-turn chat families (slm.rl.synth_chat): history reaches the prompt, verifiers judge the final turn."""

import json
import random

import pytest

from slm.rl.rewards import reward_from_verdict, verify_answer
from slm.rl.synth_chat import RECALL_FACTS, gen_recall, gen_revise, gen_sysrule
from slm.rl.tasks import GENERATORS, make_tasks, prompt_messages


def test_recall_history_and_verifier():
    rng = random.Random(3)
    t = next(x for x in (gen_recall(rng) for _ in range(50)) if not x.meta["absent"])
    msgs = prompt_messages(t)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"], "two given exchanges, then the question"
    assert msgs[-1]["content"] == t.prompt and "####" not in msgs[-1]["content"], "free answer style: no marker asked for"
    fact = json.loads(t.answer)["fact"]
    assert fact in msgs[0]["content"]
    assert verify_answer(f"Your parrot is named {fact}!", t.answer, "recall").correct
    assert not verify_answer("I am not sure what you mean.", t.answer, "recall").correct
    assert not verify_answer(f"#### {fact}", t.answer, "recall").correct, "a verifier template is not chat"


def test_recall_absent_case_rewards_saying_so_and_punishes_a_guess():
    rng = random.Random(5)
    t = next(x for x in (gen_recall(rng) for _ in range(80)) if x.meta["absent"])
    g = json.loads(t.answer)
    assert g["absent"] and t.prompt not in [f[1] for f in RECALL_FACTS], "the sibling question is about something never stated"
    assert verify_answer("You haven't told me that yet, so I don't know.", t.answer, "recall").correct
    assert not verify_answer(f"It is {g['not_these'][0]}.", t.answer, "recall").correct, "guessing a table value is wrong"
    assert not verify_answer("Blue.", t.answer, "recall").correct, "an answer that neither names nor acknowledges gets nothing"


def test_revise_carries_the_previous_answer_and_checks_constraints():
    rng = random.Random(7)
    t = next(x for x in (gen_revise(rng) for _ in range(20)) if x is not None)
    msgs = prompt_messages(t)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    specs = json.loads(t.answer)
    assert any(s["type"] == "contains_word" for s in specs), "one content word of the original must survive"
    v = verify_answer("", t.answer, "constraints")
    assert not v.correct
    assert t.meta["verifier"] == "constraints"


def test_sysrule_system_prompt_and_obeying_history():
    from slm.rl.constraints import check_constraints

    rng = random.Random(11)
    for _ in range(10):
        t = gen_sysrule(rng)
        msgs = prompt_messages(t)
        assert msgs[0]["role"] == "system" and [m["role"] for m in msgs[1:]] == ["user", "assistant", "user"]
        specs = json.loads(t.answer)
        ok, failed = check_constraints(msgs[2]["content"], specs)
        assert ok == len(specs), f"the given assistant turn must already obey the rule: {failed} / {msgs[2]['content'][:80]}"


def test_families_are_registered_and_split():
    assert {"recall", "revise", "sysrule"} <= set(GENERATORS)
    tr = make_tasks(["recall", "revise", "sysrule"], 60, "train", seed=4)
    ho = make_tasks(["recall", "revise", "sysrule"], 30, "heldout", seed=4)
    assert len(tr) == 60 and len(ho) == 30 and not ({t.prompt + str(t.meta.get("history")) for t in tr} & {t.prompt + str(t.meta.get("history")) for t in ho})
    assert reward_from_verdict(verify_answer("You told me it is Waffles.", json.dumps({"fact": "Waffles"}), "recall"), False, "plain", False, n_calls=0) == 1.0
    assert reward_from_verdict(verify_answer("You told me it is Waffles.", json.dumps({"fact": "Waffles"}), "recall"), False, "plain", False, n_calls=1) == 0.0, "a tool call in chat is the misfire"


def test_format_chat_encodes_the_system_turn(tmp_path):
    pytest.importorskip("tokenizers")
    from pathlib import Path

    from slm.data.chat import format_chat
    from slm.data.tokenizer import SlmTokenizer

    tok_dir = Path(r"C:/slm-data/tokenizer/v1")
    if not tok_dir.exists():
        pytest.skip("tokenizer not on this machine")
    tok = SlmTokenizer.load(str(tok_dir))
    t = gen_sysrule(random.Random(2))
    enc = format_chat(tok, prompt_messages(t), add_generation_prompt=True, think_required=True)
    assert tok.special("<|system|>") in enc.ids and enc.ids[-1] == tok.special("<|think|>")
    assert enc.loss_mask[-1] == 0 and enc.loss_mask[0] == 0, "the generation prompt and the system turn are never targets"

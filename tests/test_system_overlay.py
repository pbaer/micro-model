"""`slm.data.chat.system_overlay`: the eval-only prompt overlay the prompt sweep (scripts/prompt_sweep.py) measures with."""

from pathlib import Path

import pytest

from slm.data.chat import apply_overlay, system_overlay


def test_overlay_is_a_no_op_outside_the_block_and_for_empty_text():
    msgs = [{"role": "user", "content": "hi"}]
    assert apply_overlay(msgs) is msgs
    with system_overlay(None):
        assert apply_overlay(msgs) is msgs
    with system_overlay(""):
        assert apply_overlay(msgs) is msgs


def test_overlay_inserts_or_prepends_a_system_turn_and_leaves_the_input_alone():
    msgs = [{"role": "user", "content": "hi"}]
    with system_overlay("Be careful."):
        out = apply_overlay(msgs)
    assert out == [{"role": "system", "content": "Be careful."}, {"role": "user", "content": "hi"}]
    assert msgs == [{"role": "user", "content": "hi"}], "the caller's messages are not mutated"
    with_sys = [{"role": "system", "content": "Only lowercase."}, {"role": "user", "content": "hi"}]
    with system_overlay("Be careful."):
        out = apply_overlay(with_sys)
    assert out[0] == {"role": "system", "content": "Be careful.\n\nOnly lowercase."} and out[1:] == with_sys[1:]
    assert with_sys[0]["content"] == "Only lowercase."
    assert apply_overlay(msgs) is msgs, "reset after the block"


def test_overlay_in_the_user_turn_touches_only_the_first_user_message():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]
    with system_overlay("Think.", where="user"):
        out = apply_overlay(msgs)
    assert [m["content"] for m in out] == ["s", "Think.\n\na", "b", "c"]
    with pytest.raises(ValueError):
        with system_overlay("x", where="assistant"):
            pass


def test_format_chat_applies_the_overlay():
    pytest.importorskip("tokenizers")
    from slm.data.chat import format_chat
    from slm.data.tokenizer import SlmTokenizer

    tok_dir = Path(r"C:/slm-data/tokenizer/v1")
    if not tok_dir.exists():
        pytest.skip("tokenizer not on this machine")
    tok = SlmTokenizer.load(str(tok_dir))
    msgs = [{"role": "user", "content": "What is 2 + 2?"}]
    plain = format_chat(tok, msgs, add_generation_prompt=True, think_required=True)
    with system_overlay("Be careful."):
        over = format_chat(tok, msgs, add_generation_prompt=True, think_required=True)
    assert tok.special("<|system|>") not in plain.ids and tok.special("<|system|>") in over.ids
    assert over.ids[-2:] == plain.ids[-2:], "the generation prompt is unchanged"
    assert format_chat(tok, msgs, add_generation_prompt=True, think_required=True).ids == plain.ids


def test_sweep_variants_are_well_formed():
    import importlib.util

    spec = importlib.util.spec_from_file_location("prompt_sweep", Path("scripts/prompt_sweep.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.VARIANTS["bare"] == ("system", None)
    assert all(where in ("system", "user") for where, _ in mod.VARIANTS.values())
    assert all(text for name, (_, text) in mod.VARIANTS.items() if name != "bare")

"""External comparison models (slm.eval.external): the registry, the offline switches, the needle's shared random draws,
and a CPU smoke test of the HF wrapper (skipped when the weights are not on disk)."""

import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest

import slm.eval.external as E
from slm.data.tokenizer import SlmTokenizer, train_bpe
from slm.eval.long_context import RealHaystack, _external_case, _make_case  # noqa: PLC2701

SPEC = {  # short name -> (hf id, chat model): what the comparison set was defined as
    "smollm2-135m": ("HuggingFaceTB/SmolLM2-135M", False),
    "smollm2-135m-instruct": ("HuggingFaceTB/SmolLM2-135M-Instruct", True),
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M", False),
    "smollm2-360m-instruct": ("HuggingFaceTB/SmolLM2-360M-Instruct", True),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", False),
    "qwen2.5-0.5b-instruct": ("Qwen/Qwen2.5-0.5B-Instruct", True),
    "gpt2-medium": ("openai-community/gpt2-medium", False),
}


def test_registry_entries():
    assert set(E.EXTERNAL_MODELS) == set(SPEC)
    for name, m in E.EXTERNAL_MODELS.items():
        assert (m.hf_id, m.is_chat) == SPEC[name]
        assert m.name == name and m.local_dir == E.MODELS_ROOT / name
        assert m.license in ("Apache-2.0", "MIT") and 100e6 < m.params < 600e6
        assert m.max_positions in (1024, 8192, 32768)
        b = m.block()
        assert {"hf_id", "params", "license", "is_chat", "train_tokens", "max_positions"} <= set(b)
    assert E.get("gpt2-medium").max_positions == 1024 and E.get("gpt2-medium").train_tokens is None
    assert E.checkpoint_label("qwen2.5-0.5b") == "external:qwen2.5-0.5b"
    assert E.run_dir("qwen2.5-0.5b", "runs") == Path("runs/ext_qwen2.5-0.5b")
    with pytest.raises(SystemExit):
        E.get("llama-7b")
    with pytest.raises(SystemExit, match="n/a"):
        E.chat_only("smollm2-135m", "multiturn")  # base models are n/a on chat evals, never given a template
    assert E.chat_only("smollm2-135m-instruct", "multiturn").is_chat


def test_model_json(tmp_path):
    import json

    p = E.write_model_json("smollm2-360m", tmp_path)
    assert p == tmp_path / "ext_smollm2-360m" / "model.json"
    assert json.loads(p.read_text(encoding="utf-8"))["hf_id"] == "HuggingFaceTB/SmolLM2-360M"


def test_offline_switches_are_forced():
    for k, v in E.OFFLINE_ENV.items():
        assert os.environ.get(k) == v, k
    import huggingface_hub.constants as hc

    hc.HF_HUB_OFFLINE = False  # as if imported before our module set the env
    E.ensure_offline()
    assert hc.is_offline_mode() is True
    if "datasets.config" in sys.modules:
        assert sys.modules["datasets.config"].HF_HUB_OFFLINE is True


# ------------------------------------------------------------------------------------------------ needle
@pytest.fixture(scope="module")
def tok():
    return SlmTokenizer(train_bpe(["The secret number is 123456. Question: What is the secret number? Answer: " * 30,
                                   "the village market opened early " * 30], vocab_size=320))


def _split(tmp_path: Path, tok: SlmTokenizer, n: int = 40000) -> Path:
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 200, size=n, dtype=np.uint16)
    ids[::50] = tok.eos_id
    ids[1::50] = tok.bos_id
    d = tmp_path / "src" / "val"
    d.mkdir(parents=True)
    ids.tofile(d / "shard_00000.bin")
    return d


class _CharModel:
    """An external model stand-in: one token per character (more tokens than our BPE: the haystack is cut short)."""

    def encode_plain(self, text):
        return [ord(c) for c in text]

    def prefix_ids(self):
        return []


class _DenseModel(_CharModel):
    """Four characters per token (fewer tokens than ours: the window has to be read further)."""

    def encode_plain(self, text):
        return [hash(text[i : i + 4]) & 0xFFFF for i in range(0, len(text), 4)]


@pytest.mark.parametrize("model", [_CharModel(), _DenseModel()])
def test_external_needle_shares_our_draws(tmp_path, tok, model):
    real = RealHaystack(tok, _split(tmp_path, tok))
    ours, ext = random.Random(7), random.Random(7)
    for L in (256, 512):
        for d in (0.0, 0.5, 1.0):
            ids_o, secret_o = _make_case(tok, L, d, ours, real, False)
            ids_e, secret_e = _external_case(model, tok, L, d, ext, real, f"x-{L}-{d}")
            assert secret_e == secret_o and ext.getstate() == ours.getstate()  # same secret, same window, rng in lockstep
            assert len(ids_e) == L - 16 == len(ids_o)
            needle = model.encode_plain(f" The secret number is {secret_e}. ")
            assert any(ids_e[i : i + len(needle)] == needle for i in range(len(ids_e) - len(needle) + 1))
            assert ids_e[-len(model.encode_plain("\n\nQuestion: What is the secret number?\nAnswer:")):] == model.encode_plain(
                "\n\nQuestion: What is the secret number?\nAnswer:")
            if d == 1.0 and type(model) is _CharModel:
                ours_text = tok.decode(ids_o[1:], skip_special=True)
                assert "".join(map(chr, ids_e[:40])) == ours_text[:40]  # the haystack starts where ours starts


# ------------------------------------------------------------------------------------------------ the wrapper, on CPU
_have = pytest.mark.skipif(not E.get("smollm2-135m-instruct").available(), reason="smollm2-135m-instruct weights not on disk")


@pytest.fixture(scope="module")
def instruct():
    import torch

    return E.load_external("smollm2-135m-instruct", device="cpu", dtype=torch.float32)


@_have
def test_cpu_smoke_generate_chat(instruct):
    msgs = [{"role": "system", "content": "You are a concise assistant."}, {"role": "user", "content": "What is the capital of France?"}]
    out = instruct.generate_chat(msgs, max_new_tokens=16, temperature=0.0)
    assert isinstance(out, str) and out.strip()
    assert instruct.generate_chat(msgs, max_new_tokens=16, top_k=1) == out  # greedy = temperature 0 = top_k 1
    assert instruct.render_chat(msgs).endswith("<|im_start|>assistant\n")  # the model's own template, generation prompt added


@_have
def test_cpu_left_padded_batch_equals_single_rows(instruct):
    convs = [[{"role": "user", "content": q}] for q in ("What is 2+2?", "Name a primary colour.", "Write one sentence about rain in the city.")]
    batch = [g.text for g in instruct.batch_generate_chat(convs, 12)]
    assert batch == [instruct.generate_chat(c, 12) for c in convs]  # exact in fp32: padding and positions are right
    a = instruct.batch_generate_chat([convs[0]] * 3, 12, temperature=0.8, top_p=0.95, seed=3)
    b = instruct.batch_generate_chat([convs[0]] * 3, 12, temperature=0.8, top_p=0.95, seed=3)
    assert [g.text for g in a] == [g.text for g in b]  # seeded sampling reproduces


@pytest.mark.skipif(not E.get("smollm2-135m").available(), reason="smollm2-135m weights not on disk")
def test_base_model_never_gets_a_chat_template():
    import torch

    base = E.load_external("smollm2-135m", device="cpu", dtype=torch.float32)
    assert not base.is_chat
    with pytest.raises(ValueError):
        base.render_chat([{"role": "user", "content": "hi"}])
    assert base.generate_text("The capital of France is", max_new_tokens=4).strip()

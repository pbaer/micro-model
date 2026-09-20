import pytest
import torch

from slm.config import ModelConfig
from slm.data.tokenizer import SlmTokenizer, train_bpe
from slm.model import Transformer
from slm.rl.advantages import group_advantages
from slm.rl.objectives import kl_penalty, policy_loss, sequence_logprobs
from slm.rl.rewards import parse_final_answer, reward_from_verdict, verify_numeric
from slm.rl.rollout import rollout_group, teacher_forced_logprobs
from slm.rl.tasks import GENERATORS, make_tasks


def test_tasks_disjoint_and_deterministic():
    tr = make_tasks(["arith1", "arith2", "algebra", "word"], 300, "train", seed=1)
    ho = make_tasks(["arith1", "arith2", "algebra", "word"], 100, "heldout", seed=1)
    # overlapping generators (arith1 is a subset of arith2) must not leak prompts across splits
    tr2 = make_tasks(["arith1", "arith2"], 4000, "train", seed=0)
    ho2 = make_tasks(["arith1", "arith2"], 100, "heldout", seed=0)
    assert not ({t.prompt for t in tr2} & {t.prompt for t in ho2})
    assert len(tr) == 300 and len(ho) == 100
    assert not ({t.prompt for t in tr} & {t.prompt for t in ho})
    assert [t.prompt for t in tr] == [t.prompt for t in make_tasks(["arith1", "arith2", "algebra", "word"], 300, "train", seed=1)]
    import random

    for name in ("arith1", "arith2", "arith2mul", "arith_multi", "algebra", "word"):
        t = GENERATORS[name](random.Random(3))
        assert t.answer.lstrip("-").isdigit(), (name, t.answer)
    # the grammar-based families (pytool_*, constraints) answer with whatever the sandbox printed or a
    # constraint spec, and may return None when a draw fails; see tests/test_rl_families.py
    for name, g in GENERATORS.items():
        t = g(random.Random(3))
        assert t is None or (t.answer and t.prompt and t.task), name


@pytest.mark.parametrize("text,gold,ok", [
    ("blah\n#### 42", "42", True),
    ("#### 41\nwait no\n#### 42", "42", True),  # last answer wins
    ("#### 1,234", "1234", True),
    ("#### $12.50", "12.5", True),
    ("#### 3/4", "0.75", True),
    ("#### -7.", "-7", True),
    ("So the answer is 42.", "42", False),  # natural style is wrong under the strict verifier...
    ("the answer is 42", "42", False),  # no marker
    ("#### 43", "42", False),
    ("####", "42", False),
])
def test_verifier(text, gold, ok):
    assert verify_numeric(text, gold).correct is ok


def test_parse_and_reward_schemes():
    assert parse_final_answer("x\n#### 12 apples") == "12"
    v = verify_numeric("no answer here", "1")
    assert reward_from_verdict(v, False, "binary") == 0 and reward_from_verdict(v, False, "signed") == -1 and reward_from_verdict(v, False, "shaped") == -0.5
    assert reward_from_verdict(verify_numeric("#### 1", "1"), False, "shaped") == 1.0


def test_group_advantages():
    r = torch.tensor([1.0, 0.0, 0.0, 1.0])
    a = group_advantages(r, normalize_std=True)
    assert torch.allclose(a, torch.tensor([1.0, -1.0, -1.0, 1.0]), atol=1e-4)
    assert torch.allclose(group_advantages(r, normalize_std=False), torch.tensor([0.5, -0.5, -0.5, 0.5]))
    assert torch.all(group_advantages(torch.zeros(4)) == 0)


def test_policy_loss_clipping_and_reinforce_equivalence():
    torch.manual_seed(0)
    logp = torch.randn(3, 5, requires_grad=True)
    old = logp.detach().clone()
    adv = torch.tensor([1.0, -1.0, 0.5])
    mask = torch.ones(3, 5)
    l_ratio, st = policy_loss(logp, old, adv, mask, 0.2, True)
    l_pg, _ = policy_loss(logp, old, adv, mask, 0.2, False)
    # at ratio == 1 the clipped objective's gradient equals REINFORCE's
    g1 = torch.autograd.grad(l_ratio, logp)[0]
    g2 = torch.autograd.grad(l_pg, logp)[0]
    assert torch.allclose(g1, g2, atol=1e-6) and st["clip_frac"] == 0.0
    # a big positive-advantage ratio move gets clipped: no gradient beyond 1+eps
    logp2 = (old + 1.0).requires_grad_(True)
    l, st2 = policy_loss(logp2, old, torch.ones(3), mask, 0.2, True)
    g = torch.autograd.grad(l, logp2)[0]
    assert st2["clip_frac"] == 1.0 and torch.all(g == 0)
    assert kl_penalty(old, old, mask).item() == pytest.approx(0.0) and kl_penalty(old, old + 0.3, mask, "k3").item() > 0


def test_rollout_group_cpu():
    torch.manual_seed(0)
    tok = SlmTokenizer(train_bpe(["What is 3 + 4? #### 7 " * 60], vocab_size=300))
    cfg = ModelConfig(vocab_size=tok.vocab_size, n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, d_ff=128, max_seq_len=256)
    m = Transformer(cfg)
    ref = Transformer(cfg)
    task = make_tasks(["arith1"], 1, "train", 0)[0]
    g = rollout_group(m, tok, task, group_size=4, max_new_tokens=12, temperature=1.0, seed=1, ref_model=ref)
    assert len(g) == 4 and all(r.n_tokens <= 12 and len(r.old_logprobs) == r.n_tokens == len(r.ref_logprobs) for r in g)
    # old logprobs match a fresh teacher-forced pass and every reward is a valid scheme value
    lp = teacher_forced_logprobs(m, g[0].prompt_ids, [r.completion_ids for r in g], tok.pad_id)
    assert all(abs(a - b) < 1e-4 for r, l in zip(g, lp) for a, b in zip(r.old_logprobs, l))
    assert all(r.reward in (0.0, 1.0) for r in g)
    # sequence_logprobs consistency
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    lp2 = sequence_logprobs(m(x[:, :-1]), x[:, 1:])
    assert lp2.shape == (1, 7) and torch.all(lp2 <= 0)


def test_synthetic_traces_are_correct_by_construction():
    import random

    from slm.rl.synth import trace_for

    rng = random.Random(0)
    for name in ["arith1", "arith2", "arith2mul", "arith_multi", "algebra", "word"]:
        for t in make_tasks([name], 30, "train", 7):
            tr = trace_for(t, rng)
            assert tr.strip().endswith(t.answer + ".") or t.answer in tr, (name, t.prompt, tr, t.answer)


def test_lenient_verifier_and_answer_styles():
    import random

    from slm.data.answers import SUFFIX, apply_style
    from slm.rl.rewards import verify_numeric

    assert verify_numeric("So the answer is 42.", "42", strict=False).correct and not verify_numeric("So the answer is 42.", "42").correct
    assert verify_numeric("#### 41\nno, 42", "42", strict=False).parsed == "41"  # the marker still wins when present
    seen = set()
    for seed in range(20):
        msgs = apply_style([{"role": "user", "content": "How many?"}, {"role": "assistant", "think": "t", "content": "#### 7"},
                            {"role": "user", "content": "And double?"}, {"role": "assistant", "think": "t", "content": "#### 14"}], random.Random(seed), 0.5)
        marker = msgs[1]["content"].startswith("#### ")
        seen.add(marker)
        assert marker == msgs[3]["content"].startswith("#### "), "one style per conversation"
        assert (SUFFIX.strip() in msgs[0]["content"]) == marker and (SUFFIX.strip() in msgs[2]["content"]) == marker
        if not marker:
            assert "7" in msgs[1]["content"] and "14" in msgs[3]["content"] and "####" not in msgs[1]["content"]
    assert seen == {True, False}

import math

import pytest
import torch
import torch.nn.functional as F

from slm.config import ModelConfig, RopeScaling, load_config
from slm.model import IGNORE_INDEX, KVCache, Transformer, chunked_cross_entropy
from slm.model.rope import apply_rope, rope_cos_sin, rope_inv_freq

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def tiny(**kw) -> ModelConfig:
    base = dict(vocab_size=256, n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, d_ff=128, max_seq_len=256)
    base.update(kw)
    return ModelConfig(**base)


def test_gqa_shapes():
    cfg = tiny()
    m = Transformer(cfg)
    assert m.blocks[0].attn.wqkv.weight.shape == (cfg.q_dim + 2 * cfg.kv_dim, cfg.d_model)
    x = torch.randint(0, cfg.vocab_size, (3, 17))
    logits = m(x)
    assert logits.shape == (3, 17, cfg.vocab_size)


def test_causal_masking():
    """Changing token t must not change logits at positions < t."""
    torch.manual_seed(0)
    cfg = tiny()
    m = Transformer(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 32))
    y = x.clone()
    y[0, 20] = (y[0, 20] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, b = m(x), m(y)
    assert torch.allclose(a[:, :20], b[:, :20], atol=1e-5)
    assert not torch.allclose(a[:, 20:], b[:, 20:])


def test_kv_head_expansion_matches_manual_repeat():
    """SDPA enable_gqa must equal explicitly repeating KV heads."""
    torch.manual_seed(0)
    B, Hq, Hkv, T, D = 2, 4, 2, 16, 8
    q = torch.randn(B, Hq, T, D)
    k = torch.randn(B, Hkv, T, D)
    v = torch.randn(B, Hkv, T, D)
    ref = F.scaled_dot_product_attention(q, k.repeat_interleave(Hq // Hkv, 1), v.repeat_interleave(Hq // Hkv, 1), is_causal=True)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    assert torch.allclose(ref, out, atol=1e-6)


def test_rope_relative_position_property():
    """<rope(q,m), rope(k,n)> depends only on m-n."""
    torch.manual_seed(0)
    D = 16
    cos, sin = rope_cos_sin(64, D, 10000.0)
    q = torch.randn(1, 1, 1, D)
    k = torch.randn(1, 1, 1, D)

    def score(m, n):
        qm = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
        kn = apply_rope(k, cos[n : n + 1], sin[n : n + 1])
        return (qm * kn).sum().item()

    assert math.isclose(score(5, 2), score(25, 22), rel_tol=1e-4, abs_tol=1e-5)
    assert math.isclose(score(10, 3), score(40, 33), rel_tol=1e-4, abs_tol=1e-5)
    assert not math.isclose(score(5, 2), score(5, 3), rel_tol=1e-3, abs_tol=1e-3)


def test_rope_scaling_variants():
    D = 16
    base, _ = rope_inv_freq(D, 10000.0)
    lin, ms = rope_inv_freq(D, 10000.0, RopeScaling(type="linear", factor=4.0))
    assert torch.allclose(lin, base / 4) and ms == 1.0
    ntk, _ = rope_inv_freq(D, 10000.0, RopeScaling(type="ntk", factor=4.0))
    assert ntk[0] == base[0]  # highest frequency untouched
    assert ntk[-1] < base[-1]  # low frequencies stretched
    yarn, ms = rope_inv_freq(D, 10000.0, RopeScaling(type="yarn", factor=4.0, original_max_seq_len=64))
    assert yarn.shape == base.shape and ms == pytest.approx(0.1 * math.log(4.0) + 1)
    assert torch.all(yarn <= base + 1e-8) and torch.all(yarn >= base / 4 - 1e-8)
    # linear interpolation: position 4p with factor 4 equals position p unscaled
    c0, s0 = rope_cos_sin(64, D, 10000.0)
    c1, s1 = rope_cos_sin(64, D, 10000.0, RopeScaling(type="linear", factor=4.0))
    assert torch.allclose(c1[8], c0[2], atol=1e-6) and torch.allclose(s1[8], s0[2], atol=1e-6)


def test_kv_cache_decode_matches_full_forward():
    torch.manual_seed(0)
    cfg = tiny()
    m = Transformer(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 24))
    with torch.no_grad():
        full = m(x)
        cache = KVCache(cfg, 2, 64, x.device, torch.float32)
        pre = m(x[:, :10], cache=cache)  # prefill
        steps = [pre[:, -1]]
        for t in range(10, 24):
            steps.append(m(x[:, t : t + 1], cache=cache)[:, -1])
        inc = torch.stack(steps, dim=1)  # predictions for positions 9..23
        # multi-token continuation with a non-empty cache (offset mask path)
        cache2 = KVCache(cfg, 2, 64, x.device, torch.float32)
        m(x[:, :10], cache=cache2)
        cont = m(x[:, 10:24], cache=cache2)
    assert torch.allclose(full[:, 9:], inc, atol=1e-4)
    assert torch.allclose(full[:, 10:], cont, atol=1e-4)


def test_generate_deterministic_under_seed():
    torch.manual_seed(0)
    cfg = tiny()
    m = Transformer(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 5))
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = m.generate(x, 12, temperature=0.9, top_p=0.9, generator=g1)
    b = m.generate(x, 12, temperature=0.9, top_p=0.9, generator=g2)
    assert torch.equal(a, b) and a.shape == (1, 17)
    greedy = m.generate(x, 12, temperature=0.0)
    assert torch.equal(greedy, m.generate(x, 12, temperature=0.0))


def test_chunked_ce_matches_full_and_respects_ignore_index():
    torch.manual_seed(0)
    h = torch.randn(2, 10, 8, requires_grad=True)
    w = torch.randn(50, 8, requires_grad=True)
    t = torch.randint(0, 50, (2, 10))
    t[0, :3] = IGNORE_INDEX
    full, n_full = chunked_cross_entropy(h, w, t, 0)
    chunked, n_ch = chunked_cross_entropy(h, w, t, 4)
    ref = F.cross_entropy(h.reshape(-1, 8) @ w.t(), t.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum")
    assert n_full.item() == n_ch.item() == 17
    assert torch.allclose(full, ref) and torch.allclose(chunked, ref, atol=1e-5)
    gf = torch.autograd.grad(full, h)[0]
    gc = torch.autograd.grad(chunked, h)[0]
    assert torch.allclose(gf, gc, atol=1e-5)
    assert torch.all(gf[0, :3] == 0)  # ignored positions get no gradient


def test_gradient_accumulation_equivalence():
    """sum-loss / global-token-count over micro-batches == single big batch, for grads."""
    torch.manual_seed(0)
    cfg = tiny()
    m = Transformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, 16))
    y = torch.randint(0, cfg.vocab_size, (4, 16))
    y[1, :5] = IGNORE_INDEX
    m.zero_grad()
    loss, n = m(x, y)
    (loss / n).backward()
    ref = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    n_total = (y != IGNORE_INDEX).sum()
    for i in range(0, 4, 1):
        l, _ = m(x[i : i + 1], y[i : i + 1])
        (l / n_total).backward()
    for a, b in zip(ref, [p.grad for p in m.parameters()]):
        assert torch.allclose(a, b, atol=1e-5)


def test_grad_checkpointing_matches():
    torch.manual_seed(0)
    cfg = tiny()
    m = Transformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    loss, n = m(x, x)
    g1 = torch.autograd.grad(loss / n, list(m.parameters()))
    m.cfg.grad_checkpointing = True
    loss2, _ = m(x, x)
    g2 = torch.autograd.grad(loss2 / n, list(m.parameters()))
    assert torch.allclose(loss, loss2)
    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, atol=1e-6)


def test_param_counts_of_shipped_configs():
    big = Transformer(load_config(ModelConfig, "configs/model/base_149m.yaml"))
    small = Transformer(load_config(ModelConfig, "configs/model/sanity_26m.yaml"))
    assert 148e6 < big.num_params() < 150e6
    assert 123e6 < big.num_params(non_embedding=True) < 125e6
    assert 25e6 < small.num_params() < 27e6


def test_tiny_overfit():
    """A tiny model must memorize a short repeated sequence (near-zero loss)."""
    torch.manual_seed(0)
    cfg = tiny(n_layers=2, d_model=64)
    m = Transformer(cfg).to(DEV)
    seq = torch.randint(0, cfg.vocab_size, (1, 64), device=DEV)
    x, y = seq[:, :-1], seq[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(300):
        loss, n = m(x, y)
        (loss / n).backward()
        opt.step()
        opt.zero_grad()
    final = (m(x, y)[0] / n).item()
    assert final < 0.05, f"final loss {final}"


def test_diagnostics_runs_on_tiny_model():
    from slm.eval.diagnostics import render_html, run_diagnostics

    torch.manual_seed(0)
    cfg = tiny()
    m = Transformer(cfg)
    batches = [(torch.randint(0, cfg.vocab_size, (2, 32)), torch.randint(0, cfg.vocab_size, (2, 32))) for _ in range(2)]
    d = run_diagnostics(m, batches, do_ablations=True, attn_max_len=32)
    assert len(d["residual"]["residual_rms"]) == cfg.n_layers + 1
    assert len(d["mlp"]["dead_frac"]) == cfg.n_layers
    assert len(d["attention"]["entropy"]) == cfg.n_layers and len(d["attention"]["entropy"][0]) == cfg.n_heads
    assert len(d["ablation"]["head_loss_delta"]) == cfg.n_layers and len(d["ablation"]["layer_loss_delta"]) == cfg.n_layers
    assert "blocks.0.attn.wqkv.weight" in d["spectra"]
    assert "<svg" in render_html(d)


def test_lm_eval_wrapper_scoring(tmp_path):
    """Context/continuation loglikelihood equals a direct computation, and greedy flag is consistent."""
    import types

    from slm.config import to_dict
    from slm.data.tokenizer import SlmTokenizer, train_bpe
    from slm.eval.lm_eval_wrapper import SlmLM
    from slm.utils.checkpoint import save_snapshot

    tok = SlmTokenizer(train_bpe(["the cat sat on the mat " * 60, "dogs run fast " * 60], vocab_size=300))
    tok.save(tmp_path / "tok")
    cfg = tiny(vocab_size=tok.vocab_size)
    m = Transformer(cfg)
    save_snapshot(tmp_path / "ck.pt", m, to_dict(cfg), {"tokenizer_sha256": tok.sha256})
    lm = SlmLM(str(tmp_path / "ck.pt"), str(tmp_path / "tok"), batch_size=4, device="cpu")
    reqs = [types.SimpleNamespace(args=("the cat", " sat on the mat")), types.SimpleNamespace(args=("dogs", " run fast")), types.SimpleNamespace(args=("", "the cat sat"))]
    res = lm.loglikelihood(reqs)
    assert len(res) == 3 and all(lp < 0 for lp, _ in res)
    # direct check for the first request
    ctx, cont = lm._encode_pair("the cat", " sat on the mat")
    ids = torch.tensor([ctx + cont])
    with torch.no_grad():
        logp = torch.log_softmax(lm.model(ids).float(), -1)[0]
    direct = sum(float(logp[len(ctx) - 1 + i, t]) for i, t in enumerate(cont))
    assert abs(direct - res[0][0]) < 1e-3
    assert lm.generate_until([types.SimpleNamespace(args=("the", {"until": ["\n"], "max_gen_toks": 5}))])[0] is not None

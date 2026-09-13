from pathlib import Path

import pytest

from slm.config import ModelConfig, load_config
from slm.model import Transformer
from slm.model.introspect import build_graph, flops_per_token, memory_budget
from slm.utils.profiling import flops_per_token as flops_ref

CONFIGS = sorted(Path("configs/model").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_graph_params_reconcile(path):
    cfg = load_config(ModelConfig, path)
    g = build_graph(cfg)
    m = Transformer(cfg)
    tot = g["totals"]
    assert tot["params"] == m.num_params()
    assert tot["params_non_embedding"] == m.num_params(non_embedding=True)
    assert sum(tot["by_family"].values()) == m.num_params()
    by_id = {n["id"]: n for n in g["nodes"]}
    root = by_id[g["root"]]
    assert set(root["children"]) == {"embed", "blocks", "final_norm", "lm_head", "loss"}
    blocks = [by_id[b] for b in by_id["blocks"]["children"]]
    assert len(blocks) == cfg.n_layers and all(b["params"] == blocks[0]["params"] for b in blocks)
    # block 0 is expanded and its children params sum to the block params
    b0 = blocks[0]
    assert sum(by_id[c]["params"] for c in b0["children"]) == b0["params"]
    attn = by_id["block0.attn"]
    assert sum(by_id[c]["params"] for c in attn["children"]) == attn["params"]
    # shapes are symbolic
    assert by_id["embed"]["shape_out"] == ["B", "T", cfg.d_model]
    assert by_id["block0.reshape"]["shape_out"][0] == ["B", cfg.n_heads, "T", cfg.head_dim]
    assert by_id["lm_head"]["shape_out"] == ["B", "T", cfg.vocab_size]


def test_flops_match_profiling():
    cfg = load_config(ModelConfig, "configs/model/base_149m.yaml")
    m = Transformer(cfg)
    n_ne = m.num_params(non_embedding=True)
    for T in (2048, 8192):
        assert flops_per_token(cfg, T, n_ne)["total"] == pytest.approx(flops_ref(cfg, T, n_ne))
        assert flops_per_token(cfg, T, n_ne, causal=True)["total"] == pytest.approx(flops_ref(cfg, T, n_ne, causal=True))
    g = build_graph(cfg)
    root = next(n for n in g["nodes"] if n["id"] == "model")
    # linear (non-attention-score) forward FLOPs from the graph ~ 2 * params (embedding lookup excluded, head included)
    expected = 2 * (n_ne + cfg.vocab_size * cfg.d_model)
    assert abs(root["flops_fwd"] - expected) / expected < 0.02


def test_memory_budget_sane():
    cfg = load_config(ModelConfig, "configs/model/base_149m.yaml")
    n = Transformer(cfg).num_params()
    b = memory_budget(cfg, n, microbatch=8, seq_len=2048)
    assert b["weights_fp32"] + b["grads_fp32"] + b["adam_states"] == n * 16
    assert b["activations"] > 0 and b["logits"] > 0 and b["total_train"] < 64 * 2**30
    b2 = memory_budget(cfg, n, microbatch=8, seq_len=2048, grad_checkpointing=True)
    assert b2["activations"] < b["activations"]

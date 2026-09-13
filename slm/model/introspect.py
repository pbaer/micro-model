"""Derive a teaching-grade module graph from the real model.

The graph is built by (1) walking the actual nn.Module tree for parameters, (2) running one
forward pass with shape hooks on a probe input (B=3, T=7, chosen so no dimension collides with the
config's sizes) and rewriting those dims to the symbols B and T, and (3) "expanders" that add the
non-module operations worth teaching (split, reshape, RoPE, SDPA with GQA groups, residual adds).
Per-node FLOPs and the memory budget are computed from the config; tests reconcile totals with
`Transformer.num_params()` and `profiling.flops_per_token`.
"""

from __future__ import annotations

from typing import Any

import torch

from slm.config import ModelConfig
from slm.model import Transformer

PROBE_B, PROBE_T = 3, 7


def _sym(shape: tuple[int, ...]) -> list[str | int]:
    return ["B" if d == PROBE_B else "T" if d == PROBE_T else int(d) for d in shape]


def _node(id_: str, kind: str, label: str, **kw) -> dict[str, Any]:
    n = {"id": id_, "kind": kind, "label": label, "params": 0, "shape_in": None, "shape_out": None, "flops_fwd": 0.0, "children": [], "attrs": {}, "note": ""}
    n.update(kw)
    return n


def build_graph(cfg: ModelConfig, expand_layers: int = 1) -> dict[str, Any]:
    """Return {nodes: [...], root: id, totals: {...}, memory: {...}, kv_cache: {...}} for this config.

    Only the first `expand_layers` blocks get their internal structure expanded (all blocks are
    identical); the rest are summarized. FLOPs are per token for the forward pass.
    """
    with torch.device("cpu"):
        model = Transformer(cfg).eval()
    shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    hooks = []
    for name, m in model.named_modules():
        def hook(mod, inp, out, name=name):
            i = inp[0].shape if inp and torch.is_tensor(inp[0]) else ()
            o = out.shape if torch.is_tensor(out) else ()
            shapes[name] = (tuple(i), tuple(o))
        hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        model(torch.randint(0, cfg.vocab_size, (PROBE_B, PROBE_T)))
    for h in hooks:
        h.remove()
    pcount = {name: sum(p.numel() for p in m.parameters(recurse=False)) for name, m in model.named_modules()}
    d, H, Hk, D, F, V, L = cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.d_ff, cfg.vocab_size, cfg.n_layers
    nodes: list[dict] = []

    def add(n: dict) -> str:
        nodes.append(n)
        return n["id"]

    def shp(name: str):
        i, o = shapes.get(name, ((), ()))
        return _sym(i), _sym(o)

    # ---- embedding
    si, so = shp("embed")
    emb = add(_node("embed", "Embedding", "token embedding", params=pcount["embed"], shape_in=["B", "T"], shape_out=so, flops_fwd=0.0,
                    attrs={"vocab": V, "d_model": d, "tied_with_lm_head": cfg.tie_embeddings}, note="lookup, no matmul; the same matrix is reused as the LM head when tied"))

    def block_nodes(li: int, expanded: bool) -> str:
        p = f"blocks.{li}"
        attn_p = pcount[f"{p}.attn.wqkv"] + pcount[f"{p}.attn.wo"] + pcount.get(f"{p}.attn.q_norm", 0) + pcount.get(f"{p}.attn.k_norm", 0)
        mlp_p = pcount[f"{p}.mlp.w13"] + pcount[f"{p}.mlp.w2"]
        norm_p = pcount[f"{p}.attn_norm"] + pcount[f"{p}.mlp_norm"]
        attn_flops = 2 * (d * (H * D + 2 * Hk * D)) + 2 * (H * D * d)  # projections per token
        attn_score_flops = 4 * H * D  # per token per key: qk and pv -> reported with T factor separately
        mlp_flops = 2 * d * 2 * F + 2 * F * d
        bid = f"block{li}"
        children = []
        if expanded:
            children.append(add(_node(f"{bid}.attn_norm", "RMSNorm", "attn_norm (RMSNorm)", params=pcount[f"{p}.attn_norm"], shape_in=["B", "T", d], shape_out=["B", "T", d], flops_fwd=4 * d,
                                      note="x / rms(x) * gain; no mean subtraction, no bias")))
            a_children = [
                add(_node(f"{bid}.wqkv", "Linear", "wqkv (fused q|k|v projection)", params=pcount[f"{p}.attn.wqkv"], shape_in=["B", "T", d], shape_out=["B", "T", H * D + 2 * Hk * D], flops_fwd=2 * d * (H * D + 2 * Hk * D),
                           attrs={"q_dim": H * D, "kv_dim": Hk * D})),
                add(_node(f"{bid}.split", "op", "split -> q, k, v", shape_in=["B", "T", H * D + 2 * Hk * D], shape_out=[["B", "T", H * D], ["B", "T", Hk * D], ["B", "T", Hk * D]])),
                add(_node(f"{bid}.reshape", "op", "view heads + transpose", shape_in=["B", "T", H * D], shape_out=[["B", H, "T", D], ["B", Hk, "T", D], ["B", Hk, "T", D]],
                           note=f"{H} query heads, {Hk} key/value heads, head_dim {D}")),
            ]
            if cfg.qk_norm:
                a_children.append(add(_node(f"{bid}.qknorm", "RMSNorm", "q_norm / k_norm (per-head RMSNorm)", params=pcount.get(f"{p}.attn.q_norm", 0) + pcount.get(f"{p}.attn.k_norm", 0),
                                            shape_in=["B", H, "T", D], shape_out=["B", H, "T", D], flops_fwd=4 * (H + Hk) * D, note="normalizes q and k over head_dim before RoPE; bounds attention logits for stability")))
            a_children.append(add(_node(f"{bid}.rope", "op", "RoPE (rotary position)", shape_in=["B", H, "T", D], shape_out=["B", H, "T", D], flops_fwd=6 * (H + Hk) * D,
                                        attrs={"theta": cfg.rope_theta, "scaling": cfg.rope_scaling.type, "factor": cfg.rope_scaling.factor}, note="rotates each (2i, 2i+1) pair of q and k by position-dependent angle; relative positions emerge in q.k")))
            a_children.append(add(_node(f"{bid}.sdpa", "SDPA", f"causal attention (GQA {H}q/{Hk}kv)", shape_in=[["B", H, "T", D], ["B", Hk, "T", D], ["B", Hk, "T", D]], shape_out=["B", H, "T", D],
                                        flops_fwd=attn_score_flops, attrs={"groups": H // Hk, "flops_note": "per token per attended key; total per token = 4*H*D*T (causal ~half)"},
                                        note=f"softmax(q k^T / sqrt({D})) v with causal mask; each KV head serves {H // Hk} query heads (enable_gqa)")))
            a_children.append(add(_node(f"{bid}.merge", "op", "transpose + merge heads", shape_in=["B", H, "T", D], shape_out=["B", "T", H * D])))
            a_children.append(add(_node(f"{bid}.wo", "Linear", "wo (output projection)", params=pcount[f"{p}.attn.wo"], shape_in=["B", "T", H * D], shape_out=["B", "T", d], flops_fwd=2 * H * D * d)))
            children.append(add(_node(f"{bid}.attn", "Attention", "attention", params=attn_p, shape_in=["B", "T", d], shape_out=["B", "T", d], flops_fwd=attn_flops, children=a_children)))
            children.append(add(_node(f"{bid}.add1", "op", "residual add (+)", shape_in=["B", "T", d], shape_out=["B", "T", d], flops_fwd=d, note="x = x + attn(norm(x))")))
            children.append(add(_node(f"{bid}.mlp_norm", "RMSNorm", "mlp_norm (RMSNorm)", params=pcount[f"{p}.mlp_norm"], shape_in=["B", "T", d], shape_out=["B", "T", d], flops_fwd=4 * d)))
            m_children = [
                add(_node(f"{bid}.w13", "Linear", "w13 (fused gate|up)", params=pcount[f"{p}.mlp.w13"], shape_in=["B", "T", d], shape_out=["B", "T", 2 * F], flops_fwd=2 * d * 2 * F)),
                add(_node(f"{bid}.chunk", "op", "chunk -> gate, up", shape_in=["B", "T", 2 * F], shape_out=[["B", "T", F], ["B", "T", F]])),
                add(_node(f"{bid}.silu", "op", "silu(gate) * up", shape_in=["B", "T", F], shape_out=["B", "T", F], flops_fwd=6 * F, note="SwiGLU gating: the 'gate' path decides which hidden units pass")),
                add(_node(f"{bid}.w2", "Linear", "w2 (down projection)", params=pcount[f"{p}.mlp.w2"], shape_in=["B", "T", F], shape_out=["B", "T", d], flops_fwd=2 * F * d)),
            ]
            children.append(add(_node(f"{bid}.mlp", "SwiGLU", "SwiGLU MLP", params=mlp_p, shape_in=["B", "T", d], shape_out=["B", "T", d], flops_fwd=mlp_flops, children=m_children)))
            children.append(add(_node(f"{bid}.add2", "op", "residual add (+)", shape_in=["B", "T", d], shape_out=["B", "T", d], flops_fwd=d, note="x = x + mlp(norm(x))")))
        return add(_node(bid, "Block", f"block {li}" + ("" if expanded else " (same structure)"), params=attn_p + mlp_p + norm_p, shape_in=["B", "T", d], shape_out=["B", "T", d],
                         flops_fwd=attn_flops + mlp_flops + 8 * d + 2 * d, children=children, attrs={"attention_score_flops_per_key": attn_score_flops, "expanded": expanded}))

    blocks = [block_nodes(i, i < expand_layers) for i in range(L)]
    stack = add(_node("blocks", "Stack", f"{L} transformer blocks (pre-norm, residual stream d={d})", params=sum(n["params"] for n in nodes if n["kind"] == "Block"),
                      shape_in=["B", "T", d], shape_out=["B", "T", d], flops_fwd=sum(n["flops_fwd"] for n in nodes if n["kind"] == "Block"), children=blocks))
    fnorm = add(_node("final_norm", "RMSNorm", "final_norm (RMSNorm)", params=pcount["final_norm"], shape_in=["B", "T", d], shape_out=["B", "T", d], flops_fwd=4 * d))
    head = add(_node("lm_head", "Linear", "LM head" + (" (tied to embedding)" if cfg.tie_embeddings else ""), params=0 if cfg.tie_embeddings else pcount["lm_head"],
                     shape_in=["B", "T", d], shape_out=["B", "T", V], flops_fwd=2 * d * V, attrs={"tied": cfg.tie_embeddings},
                     note="logits = h @ E^T; with chunked cross-entropy the [B*T, V] fp32 logits are never fully materialized in training"))
    loss = add(_node("loss", "op", "cross-entropy (next token)", shape_in=["B", "T", V], shape_out=[], flops_fwd=3 * V, note="-log p(target); sum over valid tokens, divided by the global token count of the update"))
    root = add(_node("model", "Transformer", "Transformer", params=model.num_params(), shape_in=["B", "T"], shape_out=["B", "T", V],
                     flops_fwd=0.0, children=[emb, stack, fnorm, head, loss]))
    root_node = nodes[-1]
    root_node["flops_fwd"] = sum(n["flops_fwd"] for n in nodes if n["id"] in root_node["children"])

    n_ne = model.num_params(non_embedding=True)
    totals = {
        "params": model.num_params(), "params_non_embedding": n_ne, "params_embedding": model.num_params() - n_ne,
        "by_family": {
            "embedding / tied head": pcount["embed"] + (0 if cfg.tie_embeddings else pcount["lm_head"]),
            "attention (wqkv, wo, qk-norm)": sum(n["params"] for n in nodes if n["kind"] == "Attention") if expand_layers else 0,
            "mlp (w13, w2)": sum(n["params"] for n in nodes if n["kind"] == "SwiGLU") if expand_layers else 0,
            "norms": sum(pcount[k] for k in pcount if k.endswith(("attn_norm", "mlp_norm", "final_norm"))),
        },
        "flops_fwd_per_token_linear": root_node["flops_fwd"],
        "flops_attention_per_token_per_key": 4 * H * D * L,
        "n_layers": L,
    }
    # families for all layers (expanded or not)
    totals["by_family"]["attention (wqkv, wo, qk-norm)"] = sum(pcount[f"blocks.{i}.attn.wqkv"] + pcount[f"blocks.{i}.attn.wo"] + pcount.get(f"blocks.{i}.attn.q_norm", 0) + pcount.get(f"blocks.{i}.attn.k_norm", 0) for i in range(L))
    totals["by_family"]["mlp (w13, w2)"] = sum(pcount[f"blocks.{i}.mlp.w13"] + pcount[f"blocks.{i}.mlp.w2"] for i in range(L))
    return {"root": root, "nodes": nodes, "totals": totals, "config": cfg_dict(cfg)}


def cfg_dict(cfg: ModelConfig) -> dict:
    from slm.config import to_dict

    return to_dict(cfg)


def flops_per_token(cfg: ModelConfig, seq_len: int, n_nonembed: int, causal: bool = False) -> dict:
    lin = 6 * n_nonembed
    head = 6 * cfg.vocab_size * cfg.d_model
    attn = 12 * cfg.n_layers * cfg.q_dim * seq_len * (0.5 if causal else 1.0)
    return {"linear": lin, "lm_head": head, "attention": attn, "total": lin + head + attn}


def memory_budget(cfg: ModelConfig, n_params: int, microbatch: int, seq_len: int, dtype_bytes: int = 2, grad_checkpointing: bool = False, loss_chunk: int = 0) -> dict:
    """Rough training memory (bytes) with fp32 master weights + AdamW and bf16 activations."""
    d, F, L, V, H, D = cfg.d_model, cfg.d_ff, cfg.n_layers, cfg.vocab_size, cfg.n_heads, cfg.head_dim
    tokens = microbatch * seq_len
    weights_fp32 = n_params * 4
    grads_fp32 = n_params * 4
    adam = n_params * 8
    # per token per layer activation bytes saved for backward (bf16), fused-attention (no T x T matrix):
    # inputs to norms (2d), qkv out (H*D+2Hk*D), attention out (H*D), wo in, w13 out (2F), silu*up (F), w2 in (F), residuals (2d)
    per_tok_layer = dtype_bytes * (2 * d + (H * D + 2 * cfg.n_kv_heads * D) + 2 * H * D + 2 * F + 2 * F + 2 * d)
    act = tokens * per_tok_layer * (1 if grad_checkpointing else L)
    if grad_checkpointing:
        act += tokens * d * dtype_bytes * L  # block inputs kept
    logits_tokens = tokens if loss_chunk <= 0 else min(tokens, loss_chunk)
    logits = logits_tokens * V * (4 + dtype_bytes + 4)  # fp32 logits + bf16 copy + fp32 grad
    kv_cache_per_token = 2 * L * cfg.n_kv_heads * D * dtype_bytes
    return {
        "weights_fp32": weights_fp32, "grads_fp32": grads_fp32, "adam_states": adam, "activations": act, "logits": logits,
        "total_train": weights_fp32 + grads_fp32 + adam + act + logits,
        "weights_bf16_inference": n_params * dtype_bytes, "kv_cache_per_token_bytes": kv_cache_per_token,
        "kv_cache_full_context": kv_cache_per_token * cfg.max_seq_len, "tokens": tokens,
        "note": "activation estimate assumes fused attention (no T x T scores stored); real peaks differ, compare with benchmark JSON",
    }


def kv_cache_shape(cfg: ModelConfig, batch: int, max_len: int) -> dict:
    return {"shape": [cfg.n_layers, batch, cfg.n_kv_heads, max_len, cfg.head_dim], "symbolic": ["L", "B", "n_kv_heads", "max_len", "head_dim"],
            "bytes_bf16": 2 * cfg.n_layers * batch * cfg.n_kv_heads * max_len * cfg.head_dim * 2}

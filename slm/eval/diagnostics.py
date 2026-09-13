"""Checkpoint diagnostics: is every part of the network doing something useful?

Runs on any Transformer + a handful of validation batches and reports
  - per-layer residual-stream RMS and per-block relative update size
  - SwiGLU hidden-unit usage (dead / rarely active units per layer)
  - per-head attention entropy, BOS-sink mass, mean attention distance, local (<=16) mass
  - head ablation (zero one head's output) and layer ablation (skip one block): val-loss deltas
  - weight spectra: effective rank, spectral norm, top singular value share per matrix
  - embedding norms and usage of tokens seen in the batches
Everything is returned as a JSON-serializable dict; `render_html` turns it into a standalone page.
"""

from __future__ import annotations

import html
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from slm.model import Transformer
from slm.model.rope import apply_rope


@torch.no_grad()
def _loss(model: Transformer, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    ls = torch.zeros((), device=batches[0][0].device)
    n = torch.zeros((), device=batches[0][0].device)
    for x, y in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            l, k = model(x, y)
        ls += l
        n += k
    return (ls / n).item()


@torch.no_grad()
def residual_stats(model: Transformer, batches) -> dict:
    L = model.cfg.n_layers
    rms = torch.zeros(L + 1)
    rel_update = torch.zeros(L)
    for x, _ in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            _, res = model.hidden_states(x, return_all=True)
        res = [r.float() for r in res]
        for i, r in enumerate(res):
            rms[i] += r.pow(2).mean().sqrt().item()
        for i in range(L):
            d = res[i + 1] - res[i]
            rel_update[i] += (d.norm(dim=-1) / res[i].norm(dim=-1).clamp_min(1e-6)).mean().item()
    return {"residual_rms": (rms / len(batches)).tolist(), "block_relative_update": (rel_update / len(batches)).tolist()}


@torch.no_grad()
def mlp_usage(model: Transformer, batches, active_thresh: float = 1e-2) -> dict:
    """Per layer: fraction of hidden units that are dead (<0.01% of tokens active) or rare (<1%)."""
    L, d_ff = model.cfg.n_layers, model.cfg.d_ff
    dev = next(model.parameters()).device
    active = torch.zeros(L, d_ff, device=dev)
    n_tok = 0
    stats = {}

    def make_hook(i):
        def hook(mod, inp, out):
            gate, up = out.float().chunk(2, dim=-1)
            act = F.silu(gate) * up
            scale = act.abs().mean().clamp_min(1e-9)
            active[i] += (act.abs() > active_thresh * scale).float().sum(dim=(0, 1))
            stats.setdefault("act_rms", torch.zeros(L, device=dev))[i] += act.pow(2).mean().sqrt()
        return hook

    hooks = [b.mlp.w13.register_forward_hook(make_hook(i)) for i, b in enumerate(model.blocks)]
    for x, _ in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            model.hidden_states(x)
        n_tok += x.numel()
    for h in hooks:
        h.remove()
    frac = active / n_tok
    return {
        "dead_frac": (frac < 1e-4).float().mean(dim=1).tolist(),
        "rare_frac": (frac < 1e-2).float().mean(dim=1).tolist(),
        "mean_active_frac": frac.mean(dim=1).tolist(),
        "act_rms": (stats["act_rms"] / len(batches)).tolist(),
    }


@torch.no_grad()
def attention_stats(model: Transformer, batches, max_len: int = 512) -> dict:
    """Recompute attention probabilities (math path) on the first `max_len` positions."""
    cfg = model.cfg
    L, H = cfg.n_layers, cfg.n_heads
    dev = next(model.parameters()).device
    ent = torch.zeros(L, H, device=dev)
    sink = torch.zeros(L, H, device=dev)
    dist = torch.zeros(L, H, device=dev)
    local = torch.zeros(L, H, device=dev)
    count = 0
    captured: dict[int, torch.Tensor] = {}

    def make_hook(i):
        def hook(mod, inp, out):
            captured[i] = inp[0]
        return hook

    hooks = [b.attn.register_forward_hook(make_hook(i)) for i, b in enumerate(model.blocks)]
    for x, _ in batches:
        x = x[:, :max_len]
        T = x.shape[1]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            model.hidden_states(x)
        cos, sin = model.rope_cos[:T], model.rope_sin[:T]
        pos = torch.arange(T, device=dev)
        rel = (pos[:, None] - pos[None, :]).clamp_min(0).float()  # query - key distance
        causal = torch.ones(T, T, device=dev, dtype=torch.bool).tril()
        for i, b in enumerate(model.blocks):
            a = b.attn
            h = captured[i].float()
            B = h.shape[0]
            q, k, v = a.wqkv(h).split([cfg.q_dim, cfg.kv_dim, cfg.kv_dim], dim=-1)
            q = q.view(B, T, H, cfg.head_dim).transpose(1, 2)
            k = k.view(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
            if a.q_norm is not None:
                q, k = a.q_norm(q), a.k_norm(k)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            k = k.repeat_interleave(H // cfg.n_kv_heads, dim=1)
            s = (q @ k.transpose(-1, -2)) / math.sqrt(cfg.head_dim)
            s = s.masked_fill(~causal, float("-inf"))
            p = s.softmax(-1)  # [B, H, T, T]
            ent[i] += (-(p * (p + 1e-12).log()).sum(-1)).mean(dim=(0, 2))
            sink[i] += p[..., 0].mean(dim=(0, 2))
            dist[i] += (p * rel).sum(-1).mean(dim=(0, 2))
            local[i] += (p * (rel <= 16)).sum(-1).mean(dim=(0, 2))
        count += 1
    for hk in hooks:
        hk.remove()
    return {
        "entropy": (ent / count).tolist(), "max_entropy": math.log(max_len / 2),
        "bos_sink": (sink / count).tolist(), "mean_distance": (dist / count).tolist(), "local16": (local / count).tolist(),
    }


@torch.no_grad()
def ablations(model: Transformer, batches, base_loss: float) -> dict:
    cfg = model.cfg
    L, H, D = cfg.n_layers, cfg.n_heads, cfg.head_dim
    head_delta = [[0.0] * H for _ in range(L)]
    for i, b in enumerate(model.blocks):
        for h in range(H):
            def pre(mod, inp, h=h):
                x = inp[0].clone()
                x[..., h * D : (h + 1) * D] = 0
                return (x,)
            hk = b.attn.wo.register_forward_pre_hook(pre)
            head_delta[i][h] = _loss(model, batches) - base_loss
            hk.remove()
    layer_delta = []
    for b in model.blocks:
        hk = b.register_forward_hook(lambda mod, inp, out: inp[0])
        layer_delta.append(_loss(model, batches) - base_loss)
        hk.remove()
    return {"head_loss_delta": head_delta, "layer_loss_delta": layer_delta}


@torch.no_grad()
def weight_spectra(model: Transformer) -> dict:
    out = {}
    for name, p in model.named_parameters():
        if p.dim() != 2:
            continue
        s = torch.linalg.svdvals(p.float())
        pn = s / s.sum()
        eff_rank = math.exp(-(pn * (pn + 1e-12).log()).sum().item())
        out[name] = {"shape": list(p.shape), "spectral_norm": s[0].item(), "eff_rank": eff_rank,
                     "top1_share": (s[0] ** 2 / (s**2).sum()).item(), "fro": p.float().norm().item()}
    return out


@torch.no_grad()
def embedding_stats(model: Transformer, batches, token_counts: torch.Tensor | None = None) -> dict:
    E = model.embed.weight.float()
    norms = E.norm(dim=-1)
    V = E.shape[0]
    counts = torch.zeros(V, device=E.device) if token_counts is None else token_counts.to(E.device).float()
    if token_counts is None:
        for x, _ in batches:
            counts += torch.bincount(x.flatten(), minlength=V).float()
    seen = counts > 0
    return {
        "vocab": V, "seen_frac": seen.float().mean().item(),
        "norm_mean_seen": norms[seen].mean().item() if seen.any() else 0.0,
        "norm_mean_unseen": norms[~seen].mean().item() if (~seen).any() else 0.0,
        "norm_quantiles": torch.quantile(norms, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], device=E.device)).tolist(),
    }


@torch.no_grad()
def run_diagnostics(model: Transformer, batches, do_ablations: bool = True, token_counts=None, attn_max_len: int = 512) -> dict:
    model.eval()
    t0 = time.time()
    base = _loss(model, batches)
    res = {"base_loss": base, "n_batches": len(batches), "tokens": sum(x.numel() for x, _ in batches)}
    res["residual"] = residual_stats(model, batches)
    res["mlp"] = mlp_usage(model, batches)
    res["attention"] = attention_stats(model, batches, attn_max_len)
    if do_ablations:
        res["ablation"] = ablations(model, batches, base)
    res["spectra"] = weight_spectra(model)
    res["embedding"] = embedding_stats(model, batches, token_counts)
    res["seconds"] = time.time() - t0
    return res


# ------------------------------------------------------------------------------ rendering
def _heatmap(rows: list[list[float]], title: str, row_label: str = "layer", col_label: str = "head", fmt: str = "{:.3f}", signed: bool = False) -> str:
    if not rows:
        return ""
    R, C = len(rows), len(rows[0])
    flat = [v for r in rows for v in r]
    lo, hi = min(flat), max(flat)
    cw, ch, left, top = 46, 18, 44, 22
    W, H = left + C * cw + 10, top + R * ch + 10
    cells = []
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            if signed:
                m = max(abs(lo), abs(hi), 1e-9)
                t = v / m
                color = f"rgb({int(255 - max(0, -t) * 200)},{int(255 - abs(t) * 120)},{int(255 - max(0, t) * 200)})" if t < 0 else f"rgb({int(255 - t * 200)},{int(255 - t * 120)},255)"
                color = f"rgb({int(255 - max(t, 0) * 180)},{int(255 - abs(t) * 160)},{int(255 - max(-t, 0) * 180)})"
            else:
                t = (v - lo) / (hi - lo) if hi > lo else 0.5
                color = f"rgb({int(255 - t * 200)},{int(255 - t * 120)},{int(255 - t * 30)})"
            cells.append(f'<rect x="{left + j * cw}" y="{top + i * ch}" width="{cw - 1}" height="{ch - 1}" fill="{color}"/>'
                         f'<text x="{left + j * cw + cw / 2}" y="{top + i * ch + 13}" font-size="9" text-anchor="middle">{fmt.format(v)}</text>')
    labels = "".join(f'<text x="{left - 4}" y="{top + i * ch + 13}" font-size="10" text-anchor="end">{row_label[0]}{i}</text>' for i in range(R))
    labels += "".join(f'<text x="{left + j * cw + cw / 2}" y="{top - 6}" font-size="10" text-anchor="middle">{col_label[0]}{j}</text>' for j in range(C))
    return f'<h3>{html.escape(title)}</h3><svg viewBox="0 0 {W} {H}" style="max-width:{W}px">{labels}{"".join(cells)}</svg>'


def render_html(d: dict, title: str = "diagnostics") -> str:
    L = len(d["residual"]["block_relative_update"])
    layer_rows = "".join(
        f"<tr><td>{i}</td><td>{d['residual']['residual_rms'][i + 1]:.3f}</td><td>{d['residual']['block_relative_update'][i]:.3f}</td>"
        f"<td>{d['mlp']['dead_frac'][i] * 100:.2f}%</td><td>{d['mlp']['rare_frac'][i] * 100:.1f}%</td><td>{d['mlp']['mean_active_frac'][i] * 100:.1f}%</td>"
        f"<td>{d.get('ablation', {}).get('layer_loss_delta', [float('nan')] * L)[i]:+.4f}</td></tr>"
        for i in range(L)
    )
    spectra_rows = "".join(
        f"<tr><td style='text-align:left'>{html.escape(n)}</td><td>{v['shape']}</td><td>{v['eff_rank']:.1f}</td><td>{v['spectral_norm']:.3f}</td><td>{v['top1_share'] * 100:.1f}%</td><td>{v['fro']:.2f}</td></tr>"
        for n, v in d["spectra"].items()
    )
    a = d["attention"]
    parts = [
        f"<h1>{html.escape(title)}</h1><p>base val loss {d['base_loss']:.4f} on {d['tokens']:,} tokens ({d['n_batches']} batches); {d['seconds']:.0f}s</p>",
        "<h2>Per-layer</h2><table><tr><th>layer</th><th>residual RMS (out)</th><th>block rel. update</th><th>dead MLP units</th><th>rare (&lt;1%)</th><th>mean active</th><th>skip-layer Δloss</th></tr>" + layer_rows + "</table>",
        _heatmap(a["entropy"], f"attention entropy per head (uniform over 256 = {a['max_entropy']:.2f})"),
        _heatmap(a["bos_sink"], "attention mass on <|bos|> (position 0)"),
        _heatmap(a["mean_distance"], "mean attention distance (tokens)", fmt="{:.0f}"),
        _heatmap(a["local16"], "attention mass within 16 tokens"),
    ]
    if "ablation" in d:
        parts.append(_heatmap(d["ablation"]["head_loss_delta"], "head ablation: Δ val loss when the head is zeroed", fmt="{:+.3f}", signed=True))
    e = d["embedding"]
    parts.append(f"<h2>Embeddings</h2><p>vocab {e['vocab']:,}; {e['seen_frac'] * 100:.1f}% of ids seen in sample; mean norm seen {e['norm_mean_seen']:.3f} vs unseen {e['norm_mean_unseen']:.3f}; norm quantiles (0/10/50/90/100%) {[round(x, 3) for x in e['norm_quantiles']]}</p>")
    parts.append("<h2>Weight spectra</h2><table><tr><th>parameter</th><th>shape</th><th>eff. rank</th><th>spectral norm</th><th>top-1 share</th><th>Frobenius</th></tr>" + spectra_rows + "</table>")
    css = "body{font-family:system-ui,Arial;padding:20px;max-width:1200px}table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #ddd;padding:3px 7px;text-align:right}th{background:#f3f4f6}h3{font-size:13px;margin:18px 0 4px}svg{display:block}"
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{css}</style></head><body>{''.join(parts)}</body></html>"


def save(d: dict, out_dir: Path, name: str = "diagnostics") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
    (out_dir / f"{name}.html").write_text(render_html(d, name), encoding="utf-8")

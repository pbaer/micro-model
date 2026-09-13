import { h } from "preact";
import { useEffect, useState } from "preact/hooks";
import htm from "htm";
import { api, fmtTok, fmtInt, fmtBytes, fmtNum, fmtSci } from "../components/util.js";
import { Chart } from "../components/chart.js";

const html = htm.bind(h);
const KIND_COLORS = { Transformer: "#111827", Embedding: "#7c3aed", Stack: "#1f2937", Block: "#2563eb", Attention: "#0891b2", SwiGLU: "#16a34a", RMSNorm: "#ea580c", Linear: "#334155", SDPA: "#0e7490", op: "#6b7280" };

function shapeStr(s, B, T) {
  if (s == null) return "";
  if (Array.isArray(s) && s.length && Array.isArray(s[0])) return s.map((x) => shapeStr(x, B, T)).join("  ");
  if (!Array.isArray(s)) return String(s);
  const sym = "[" + s.join(", ") + "]";
  const num = "[" + s.map((d) => (d === "B" ? B : d === "T" ? T : d)).join(", ") + "]";
  return sym === num ? sym : `${sym} = ${num}`;
}

function Node({ node, byId, B, T, depth, open, toggle, total }) {
  const has = node.children && node.children.length > 0;
  const isOpen = open.has(node.id);
  const color = KIND_COLORS[node.kind] || "#6b7280";
  return html`<div class="anode" style=${`margin-left:${depth * 18}px;border-left:4px solid ${color}`}>
    <div class="ahead" onClick=${has ? () => toggle(node.id) : null} style=${has ? "cursor:pointer" : ""}>
      <span class="akind" style=${"background:" + color}>${node.kind}</span>
      <b>${has ? (isOpen ? "▾ " : "▸ ") : ""}${node.label}</b>
      ${node.params ? html`<span class="muted"> · ${fmtInt(node.params)} params (${(node.params / total * 100).toFixed(1)}%)</span>` : ""}
      ${node.flops_fwd ? html`<span class="muted"> · ${(node.flops_fwd / 1e6).toFixed(2)} MFLOP/token fwd</span>` : ""}
    </div>
    <div class="ashape">${node.shape_in != null && node.shape_in.length ? html`<span>in ${shapeStr(node.shape_in, B, T)}</span>` : ""}${node.shape_out != null && node.shape_out.length ? html`<span> → out ${shapeStr(node.shape_out, B, T)}</span>` : ""}</div>
    ${node.note && html`<div class="anote">${node.note}</div>`}
    ${Object.keys(node.attrs || {}).length > 0 && html`<div class="anote muted">${Object.entries(node.attrs).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" · ")}</div>`}
    ${has && isOpen && node.children.map((c) => html`<${Node} node=${byId[c]} byId=${byId} B=${B} T=${T} depth=${depth + 1} open=${open} toggle=${toggle} total=${total} key=${c} />`)}
  </div>`;
}

function GqaSvg({ H, Hk, D }) {
  const g = H / Hk, w = 26, gap = 6, W = H * (w + gap) + 20, y1 = 20, y2 = 90;
  return html`<svg viewBox=${`0 0 ${W} 130`} style=${`max-width:${W}px;display:block`}>
    ${Array.from({ length: H }, (_, i) => html`<rect x=${10 + i * (w + gap)} y=${y1} width=${w} height="22" rx="4" fill="#0891b2" /><text x=${10 + i * (w + gap) + w / 2} y=${y1 + 15} font-size="10" fill="#fff" text-anchor="middle">q${i}</text>
      <line x1=${10 + i * (w + gap) + w / 2} y1=${y1 + 22} x2=${10 + (Math.floor(i / g) * g + (g - 1) / 2) * (w + gap) + w / 2} y2=${y2} stroke="#94a3b8" />`)}
    ${Array.from({ length: Hk }, (_, j) => { const cx = 10 + (j * g + (g - 1) / 2) * (w + gap) + w / 2; return html`<rect x=${cx - w} y=${y2} width=${2 * w} height="22" rx="4" fill="#7c3aed" /><text x=${cx} y=${y2 + 15} font-size="10" fill="#fff" text-anchor="middle">kv${j}</text>`; })}
    <text x="10" y="125" font-size="10" fill="#666">${H} query heads share ${Hk} key/value heads (${g} per group), head_dim ${D}. KV cache and k/v projections shrink by ${g}x vs. full multi-head attention.</text>
  </svg>`;
}

export function ArchPage() {
  const [cfgs, setCfgs] = useState(null);
  const [spec, setSpec] = useState("configs/model/base_149m.yaml");
  const [B, setB] = useState(8);
  const [T, setT] = useState(2048);
  const [gc, setGc] = useState(false);
  const [chunk, setChunk] = useState(0);
  const [g, setG] = useState(null);
  const [open, setOpen] = useState(new Set(["model", "blocks", "block0"]));
  const [train, setTrain] = useState(null);
  const [hp, setHp] = useState(null);
  const [bench, setBench] = useState(null);
  const [tab, setTab] = useState("graph");
  useEffect(() => { api("/api/arch/configs").then((c) => { setCfgs(c); if (c.trains.length) setTrain(c.trains[0]); }); }, []);
  useEffect(() => { api(`/api/arch/graph?config=${encodeURIComponent(spec)}&B=${B}&T=${T}&grad_checkpointing=${gc}&loss_chunk=${chunk}`).then(setG).catch((e) => setG({ error: String(e) })); }, [spec, B, T, gc, chunk]);
  useEffect(() => { if (train) api(`/api/arch/hparams?config=${encodeURIComponent(train)}`).then(setHp).catch(() => setHp(null)); }, [train]);
  useEffect(() => { if (cfgs && cfgs.benchmarks.length) api(`/api/arch/benchmark?path=${encodeURIComponent(cfgs.benchmarks[cfgs.benchmarks.length - 1])}`).then(setBench).catch(() => {}); }, [cfgs]);
  const toggle = (id) => setOpen((o) => { const n = new Set(o); n.has(id) ? n.delete(id) : n.add(id); return n; });
  if (!cfgs) return html`<div>loading…</div>`;
  const byId = g && g.nodes ? Object.fromEntries(g.nodes.map((n) => [n.id, n])) : {};
  const total = g && g.totals ? g.totals.params : 1;
  const c = g && g.config;
  const fam = g && g.totals ? Object.entries(g.totals.by_family) : [];
  const f = g && g.flops;
  const m = g && g.memory;
  return html`<div>
    <h1>Architecture</h1>
    <div class="row" style="margin:8px 0">
      <select value=${spec} onChange=${(e) => setSpec(e.target.value)}>
        <optgroup label="model configs">${cfgs.models.map((x) => html`<option value=${x}>${x}</option>`)}</optgroup>
        <optgroup label="train configs">${cfgs.trains.map((x) => html`<option value=${"train:" + x}>${x}</option>`)}</optgroup>
        <optgroup label="runs (stored model config)">${cfgs.runs.map((x) => html`<option value=${x}>${x}</option>`)}</optgroup>
      </select>
      <span class="muted">B</span><input type="number" value=${B} min="1" onChange=${(e) => setB(Math.max(1, Number(e.target.value)))} style="width:70px" />
      <span class="muted">T</span><select value=${T} onChange=${(e) => setT(Number(e.target.value))}>${[256, 512, 1024, 2048, 4096, 8192, 16384, 32768].map((n) => html`<option value=${n}>${n}</option>`)}</select>
      <button class=${gc ? "active" : ""} onClick=${() => setGc(!gc)}>grad checkpointing</button>
      <span class="muted">loss chunk</span><select value=${chunk} onChange=${(e) => setChunk(Number(e.target.value))}>${[0, 1024, 2048, 4096, 8192].map((n) => html`<option value=${n}>${n === 0 ? "full logits" : n}</option>`)}</select>
      ${["graph", "budget", "hparams"].map((x) => html`<button class=${tab === x ? "active" : ""} onClick=${() => setTab(x)}>${x}</button>`)}
    </div>
    ${g && g.error && html`<div class="empty-note">${g.error}</div>`}
    ${g && !g.error && c && html`<div class="sub">${g.label} · ${fmtInt(g.totals.params)} params (${fmtInt(g.totals.params_non_embedding)} non-embedding) · ${c.n_layers} layers · d_model ${c.d_model} · ${c.n_heads}q/${c.n_kv_heads}kv heads × ${c.head_dim} · d_ff ${c.d_ff} · vocab ${fmtInt(c.vocab_size)} · max ctx ${c.max_seq_len} · rope θ=${c.rope_theta} ${c.rope_scaling.type !== "none" ? `(${c.rope_scaling.type} ×${c.rope_scaling.factor})` : ""} · qk-norm ${c.qk_norm ? "on" : "off"} · tied ${c.tie_embeddings ? "yes" : "no"}</div>`}
    ${tab === "graph" && g && !g.error && html`<div class="two" style="grid-template-columns:3fr 2fr">
      <div>
        <div class="legend" style="margin-bottom:6px">Click a node to expand. Shapes are shown symbolically and for the chosen B and T. Only block 0 is expanded; all blocks are identical.</div>
        <${Node} node=${byId[g.root]} byId=${byId} B=${B} T=${T} depth=${0} open=${open} toggle=${toggle} total=${total} />
      </div>
      <div>
        <div class="panel"><h3>Grouped-query attention</h3><${GqaSvg} H=${c.n_heads} Hk=${c.n_kv_heads} D=${c.head_dim} /></div>
        <div class="panel" style="margin-top:10px"><h3>Parameters by family</h3><table>${fam.map(([k, v]) => html`<tr><td>${k}</td><td>${fmtInt(v)}</td><td>${(v / total * 100).toFixed(1)}%</td><td style="width:160px"><div class="bar" style="margin:0"><div style=${"width:" + (v / total * 100).toFixed(1) + "%"}></div></div></td></tr>`)}</table>
          <div class="legend" style="margin-top:6px">The embedding matrix is shared with the LM head (tied), so the 32K vocabulary costs ${(g.totals.params_embedding / total * 100).toFixed(0)}% of all parameters exactly once.</div></div>
        <div class="panel" style="margin-top:10px"><h3>KV cache (inference)</h3><div class="muted">shape ${JSON.stringify(g.kv_cache.symbolic)} = ${JSON.stringify(g.kv_cache.shape)} → ${fmtBytes(g.kv_cache.bytes_bf16)} in bf16 for one sequence at max context (${fmtBytes(m.kv_cache_per_token_bytes)} per token)</div></div>
      </div>
    </div>`}
    ${tab === "budget" && g && !g.error && html`<div>
      <h2>Training FLOPs per token at T=${T}</h2>
      <table><tr><th>term</th><th>FLOPs / token</th><th>share</th><th class="l">formula</th></tr>
        <tr><td>linear layers (fwd+bwd)</td><td>${(f.per_token_train.linear / 1e6).toFixed(0)}M</td><td>${(f.per_token_train.linear / f.per_token_train.total * 100).toFixed(0)}%</td><td class="l">6 × non-embedding params</td></tr>
        <tr><td>LM head</td><td>${(f.per_token_train.lm_head / 1e6).toFixed(0)}M</td><td>${(f.per_token_train.lm_head / f.per_token_train.total * 100).toFixed(0)}%</td><td class="l">6 × vocab × d_model</td></tr>
        <tr><td>attention scores (dense)</td><td>${(f.per_token_train.attention / 1e6).toFixed(0)}M</td><td>${(f.per_token_train.attention / f.per_token_train.total * 100).toFixed(0)}%</td><td class="l">12 × layers × (heads × head_dim) × T</td></tr>
        <tr><td><b>total (PaLM convention)</b></td><td><b>${(f.per_token_train.total / 1e9).toFixed(3)} GFLOP</b></td><td>100%</td><td class="l">causal kernels skip ~half the scores: ${(f.per_token_train_causal.total / 1e9).toFixed(3)} GFLOP executed</td></tr></table>
      <div class="legend" style="margin-top:6px">At 8K context the attention term alone rivals all the linear layers, which is why bulk pretraining runs at 2K and the 8K exposure is a final phase.</div>
      <h2>Training memory estimate (B=${B}, T=${T}, ${fmtTok(m.tokens)} tokens per micro-batch)</h2>
      <div class="tiles">
        <div class="tile"><div class="k">fp32 weights</div><div class="v">${fmtBytes(m.weights_fp32)}</div></div>
        <div class="tile"><div class="k">fp32 grads</div><div class="v">${fmtBytes(m.grads_fp32)}</div></div>
        <div class="tile"><div class="k">AdamW m, v</div><div class="v">${fmtBytes(m.adam_states)}</div></div>
        <div class="tile"><div class="k">activations (bf16)</div><div class="v">${fmtBytes(m.activations)}</div><div class="s">${gc ? "with block recompute" : "all layers saved"}</div></div>
        <div class="tile"><div class="k">logits</div><div class="v">${fmtBytes(m.logits)}</div><div class="s">${chunk ? `chunked (${chunk} tokens)` : "full [B·T, V] fp32 + grad"}</div></div>
        <div class="tile"><div class="k">total</div><div class="v">${fmtBytes(m.total_train)}</div><div class="s">RTX 4080 SUPER has 16 GiB</div></div>
        <div class="tile"><div class="k">bf16 weights (inference)</div><div class="v">${fmtBytes(m.weights_bf16_inference)}</div></div>
      </div>
      <div class="legend">${m.note}</div>
      ${bench && html`<h2>Measured (benchmark ${bench.config})</h2><table><tr><th>seq</th><th>microbatch</th><th>compile</th><th>tok/s</th><th>step ms</th><th>fwd</th><th>bwd</th><th>opt</th><th>peak VRAM</th><th>MFU</th><th>MFU (causal)</th></tr>
        ${bench.results.map((r) => html`<tr><td>${r.seq}</td><td>${r.microbatch}</td><td>${r.compile ? "yes" : "no"}</td><td>${fmtInt(r.tokens_per_sec)}</td><td>${r.step_ms.toFixed(0)}</td><td>${r.fwd_ms.toFixed(0)}</td><td>${r.bwd_ms.toFixed(0)}</td><td>${r.opt_ms.toFixed(0)}</td><td>${r.peak_vram_gib.toFixed(2)} GiB</td><td>${(r.mfu_palm * 100).toFixed(1)}%</td><td>${(r.mfu_causal * 100).toFixed(1)}%</td></tr>`)}</table>`}
    </div>`}
    ${tab === "hparams" && html`<div>
      <div class="row"><span class="muted">train config</span><select value=${train} onChange=${(e) => setTrain(e.target.value)}>${cfgs.trains.map((x) => html`<option value=${x}>${x}</option>`)}</select></div>
      ${hp && html`<div>
        <h2>Learning-rate schedule (${hp.lr.type})</h2>
        <div class="charts"><${Chart} title=${`lr vs tokens · peak ${fmtSci(hp.lr.peak)} · warmup ${fmtTok(hp.lr.warmup_tokens)} · min ${fmtSci(hp.lr.min_lr)}`} xmode="tokens" ymin=${0} series=${[{ label: "lr", x: hp.lr.tokens, y: hp.lr.lr }]} /></div>
        <div class="legend">Computed by the same lr_at() the trainer calls, indexed by tokens rather than steps, so a change of batch size does not silently change the schedule.</div>
        <h2>Batch and gradient accumulation</h2>
        <div class="panel"><div class="row" style="gap:6px;align-items:stretch">
          ${Array.from({ length: hp.batch.grad_accum }, (_, i) => html`<div class="mb"><b>micro-batch ${i + 1}</b><br/>${hp.batch.microbatch} × ${hp.batch.seq_len}<br/>= ${fmtTok(hp.batch.tokens_per_micro)} tokens<br/><span class="muted">fwd + bwd, grads accumulate</span></div>`)}
          <div class="mb" style="background:#dbeafe"><b>optimizer step</b><br/>${fmtTok(hp.batch.tokens_per_update)} tokens / update<br/>${fmtInt(hp.batch.updates_total)} updates total</div></div>
          <div class="legend" style="margin-top:6px">${hp.batch.note}</div></div>
        <h2>RoPE frequencies (head_dim ${hp.rope.head_dim}, θ=${hp.rope.theta})</h2>
        <div class="charts">
          <${Chart} title="wavelength (tokens per full rotation) per frequency pair" xmode="update" logy=${true} series=${[{ label: "wavelength", x: hp.rope.pairs, y: hp.rope.wavelength_tokens, points: true }, { label: "max context", x: hp.rope.pairs, y: hp.rope.pairs.map(() => hp.rope.max_seq_len), color: "#dc2626" }]} />
          <${Chart} title=${`inverse frequency per pair: none vs linear / ntk / yarn at ×${hp.rope.illustrated_factor}`} xmode="update" logy=${true} series=${[{ label: "none", x: hp.rope.pairs, y: hp.rope.inv_freq.none }, { label: "linear", x: hp.rope.pairs, y: hp.rope.inv_freq.linear }, { label: "ntk", x: hp.rope.pairs, y: hp.rope.inv_freq.ntk }, { label: "yarn", x: hp.rope.pairs, y: hp.rope.inv_freq.yarn }]} />
        </div>
        <div class="legend">${hp.rope.pairs_exceeding_context} of ${hp.rope.pairs.length} pairs never complete a rotation within the ${fmtTok(hp.rope.max_seq_len)}-token context (they act as slow "position counters"). Linear interpolation compresses every frequency; NTK stretches the base so high frequencies stay intact; YaRN interpolates only the low-frequency pairs and scales attention by mscale=${hp.rope.inv_freq.yarn_mscale.toFixed(3)}.</div>
        <h2>Cadence</h2>
        <table><tr><th>what</th><th>every</th><th>count over run</th></tr>
          <tr><td>validation eval</td><td>${fmtTok(hp.cadence.eval_every)} tokens</td><td>${hp.cadence.n_evals}</td></tr>
          <tr><td>sample generation</td><td>${fmtTok(hp.cadence.gen_every)} tokens</td><td>${Math.floor(hp.cadence.total_tokens / hp.cadence.gen_every)}</td></tr>
          <tr><td>milestone + snapshot + report</td><td>${fmtTok(hp.cadence.milestone_every)} tokens</td><td>${hp.cadence.n_milestones}</td></tr>
          <tr><td>full resumable checkpoint</td><td>${hp.cadence.ckpt_every_minutes} min</td><td>-</td></tr></table>
        <h2>Mixture</h2>
        <table><tr><th>source</th><th>weight</th></tr>${Object.entries(hp.mixture).map(([k, v]) => html`<tr><td>${k}</td><td>${(v * 100).toFixed(1)}%</td></tr>`)}</table>
      </div>`}
    </div>`}
  </div>`;
}

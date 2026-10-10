import { h } from "preact";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "preact/hooks";
import htm from "htm";
import { api, fmtTok, fmtDur, fmtNum, fmtInt } from "../components/util.js";
import { Info } from "../components/info.js";

const html = htm.bind(h);

// ----------------------------------------------------------------------------------------------- run lineage
// The Overview is a graph of runs: a node per run, an edge where a run started from another run's checkpoint
// (its init_from). Every run has at most one parent, so the lineage is a forest; it is drawn left to right, one
// column per generation, each tree's leaves on their own rows and a parent centred on its children.

export const STAGES = {
  base: { label: "pretraining", cls: "base" },
  ctx: { label: "context ext.", cls: "ctx" },
  sft: { label: "SFT", cls: "sft" },
  tool: { label: "tool SFT", cls: "tool" },
  reasoning: { label: "reasoning SFT", cls: "reasoning" },
  rl: { label: "RL", cls: "rl" },
};

export function stageOf(r) {
  if (r.stage === "grpo" || r.is_rl) return "rl";
  if (r.stage === "sft") {
    const mix = (r.mixture || []).join(" ");
    if (/tools?\b|-tools|tool-/.test(mix) || /_tool\d*_/.test(r.run_name)) return "tool";
    if (/reasoning|gsm8k|math/.test(mix)) return "reasoning";
    return "sft";
  }
  if (/ctx|16k|32k/.test(r.run_name)) return "ctx";
  return "base";
}

/** {run, file} from an init_from path ("runs/<run>/checkpoints/<file>", absolute or relative, either separator). */
export function parentOf(initFrom) {
  if (!initFrom) return null;
  const parts = String(initFrom).split(/[\\/]+/).filter(Boolean);
  const i = parts.lastIndexOf("checkpoints");
  if (i < 1) return { run: null, file: parts[parts.length - 1] || String(initFrom), raw: String(initFrom) };
  return { run: parts[i - 1], file: parts[i + 1] || "", raw: String(initFrom) };
}

export const GEOM = { nodeW: 176, nodeH: 58, colGap: 40, rowGap: 12, familyGap: 22, pad: 14 };

/** Layout of the run forest. Returns {nodes, edges, W, H}: nodes {id, run|null, x, y, depth, ghost}, edges
 *  {from, to, file, x1, y1, x2, y2}. A parent run that is not in the list (deleted, or another runs root) becomes a
 *  ghost node so the edge still says where the run came from. Trees are ordered newest first (by the most recent
 *  start in the tree), children oldest first, so each lineage reads top-down in the order it was trained. */
export function layoutLineage(runs, g = GEOM) {
  const byName = new Map(runs.map((r) => [r.run_name, r]));
  const parent = new Map(), kids = new Map(), ghosts = new Map();
  for (const r of runs) {
    const p = parentOf(r.init_from);
    if (!p) continue;
    const pid = p.run && byName.has(p.run) ? p.run : `ghost:${p.run || p.raw}`;
    if (!byName.has(pid) && !ghosts.has(pid)) ghosts.set(pid, { id: pid, label: p.run || p.raw });
    parent.set(r.run_name, { id: pid, file: p.file });
    (kids.get(pid) || kids.set(pid, []).get(pid)).push(r.run_name);
  }
  const ids = [...runs.map((r) => r.run_name), ...ghosts.keys()];
  const started = (id) => (byName.get(id) && byName.get(id).started) || "";
  for (const list of kids.values()) list.sort((a, b) => started(a).localeCompare(started(b)) || a.localeCompare(b));
  // the newest start anywhere in a tree orders the trees
  const latest = new Map();
  const newest = (id, seen = new Set()) => {
    if (latest.has(id)) return latest.get(id);
    if (seen.has(id)) return "";
    seen.add(id);
    let v = started(id);
    for (const k of kids.get(id) || []) { const t = newest(k, seen); if (t > v) v = t; }
    latest.set(id, v);
    return v;
  };
  const roots = ids.filter((id) => !parent.has(id) || parent.get(id).id === id);
  roots.sort((a, b) => newest(b).localeCompare(newest(a)) || a.localeCompare(b));
  const pos = new Map();
  let y = g.pad;
  const place = (id, depth, seen) => {  // returns the node's centre y
    if (seen.has(id)) return y;  // a cycle in bad data: draw the node once
    seen.add(id);
    const ch = kids.get(id) || [];
    let cy;
    if (!ch.length) { cy = y + g.nodeH / 2; y += g.nodeH + g.rowGap; }
    else { const ys = ch.map((k) => place(k, depth + 1, seen)); cy = (ys[0] + ys[ys.length - 1]) / 2; }
    pos.set(id, { x: g.pad + depth * (g.nodeW + g.colGap), y: cy - g.nodeH / 2, depth });
    return cy;
  };
  const seen = new Set();
  for (const r of roots) { place(r, 0, seen); y += g.familyGap; }
  for (const id of ids) if (!pos.has(id)) { place(id, 0, seen); y += g.familyGap; }  // anything a cycle cut off
  const nodes = ids.map((id) => ({ id, run: byName.get(id) || null, ghost: !byName.has(id), label: ghosts.has(id) ? ghosts.get(id).label : id, ...pos.get(id) }));
  const edges = [];
  for (const [child, p] of parent) {
    const a = pos.get(p.id), b = pos.get(child);
    if (!a || !b || p.id === child) continue;
    edges.push({ from: p.id, to: child, file: p.file, x1: a.x + g.nodeW, y1: a.y + g.nodeH / 2, x2: b.x, y2: b.y + g.nodeH / 2 });
  }
  const W = Math.max(...nodes.map((n) => n.x + g.nodeW), 0) + g.pad;
  const H = Math.max(y - g.familyGap - g.rowGap + g.pad, g.nodeH + 2 * g.pad);
  return { nodes, edges, W, H, parent, kids };
}

/** Every node on the path to the root and every descendant: the lineage a hover lights up. */
export function relatives(id, parent, kids) {
  const out = new Set([id]);
  let p = parent.get(id);
  while (p && !out.has(p.id)) { out.add(p.id); p = parent.get(p.id); }
  const stack = [...(kids.get(id) || [])];
  while (stack.length) { const k = stack.pop(); if (!out.has(k)) { out.add(k); stack.push(...(kids.get(k) || [])); } }
  return out;
}

// ----------------------------------------------------------------------------------------------- formatting
const paramsLabel = (n) => (n == null ? "" : n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : `${Math.round(n / 1e6)}M`);
const headline = (r) => r.quality ? { k: "judged", v: r.quality.overall.toFixed(2), tip: "judged quality, 1-5" }
  : r.heldout_acc != null ? { k: "held-out", v: `${(r.heldout_acc * 100).toFixed(0)}%`, tip: "held-out accuracy" }
  : r.best_val != null ? { k: "val", v: fmtNum(r.best_val, 3), tip: "best validation loss" } : null;
const amount = (r) => (r.total_steps ? `${fmtInt(r.update)} / ${fmtInt(r.total_steps)} steps` : fmtTok(r.tokens) + " tok");
const shortCkpt = (f) => String(f || "").replace(/\.pt$/, "");
/** An RL run lists one entry per draw of its task mix ("chat, chat, chat, constraints, ..."): show each task once with
 *  its share of the draws, most drawn first. */
export const taskMix = (tasks) => {
  const c = new Map();
  for (const t of tasks || []) c.set(t, (c.get(t) || 0) + 1);
  return [...c].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([t, n]) => (n > 1 ? `${t} ×${n}` : t)).join(", ");
};

function HoverCard({ node, at, parentInfo, nKids }) {
  const ref = useRef(null);
  const [xy, setXy] = useState(null);
  useLayoutEffect(() => {  // beside the pointer, clamped to the viewport
    const el = ref.current;
    if (!el || !at) return;
    const w = el.offsetWidth, hgt = el.offsetHeight, m = 10;
    let x = at.x + 16, y = at.y + 12;
    if (x + w + m > innerWidth) x = at.x - w - 16;
    if (y + hgt + m > innerHeight) y = innerHeight - hgt - m;
    setXy({ x: Math.max(m, x), y: Math.max(m, y) });
  }, [node && node.id, at && at.x, at && at.y]);
  if (!node) return null;
  const r = node.run;
  const style = xy ? `left:${xy.x}px;top:${xy.y}px` : "left:-9999px;top:0";
  if (!r) return html`<div class="lg-card" ref=${ref} style=${style}><div class="lg-card-t">${node.label}</div>
    <div class="muted">Not in this runs folder (deleted, or trained elsewhere). Runs below started from its checkpoints.</div></div>`;
  const st = STAGES[stageOf(r)];
  const rows = [
    ["status", html`<span class=${"status " + r.status}>${r.status}</span>${r.status === "running" ? ` · ${(r.progress * 100).toFixed(1)}% · ETA ${fmtDur(r.eta_s)}` : ""}`],
    ["stage", html`<span class=${"stage-badge " + st.cls}>${st.label}</span>${r.n_params ? ` · ${fmtInt(r.n_params)} params` : ""}`],
    ["started from", parentInfo ? html`<code>${parentInfo.run || parentInfo.raw}</code> › ${shortCkpt(parentInfo.file)}` : "random init"],
    ["trained", r.total_steps ? `${fmtInt(r.update)} of ${fmtInt(r.total_steps)} steps` : `${fmtTok(r.tokens)}${r.total_tokens ? ` of ${fmtTok(r.total_tokens)}` : ""} tokens`],
    r.cumulative_tokens > r.tokens && ["seen in total", `${fmtTok(r.cumulative_tokens)} tokens along its lineage`],
    r.is_rl && r.rl ? ["reward", `${fmtNum(r.rl.reward_mean, 3)} (last)`] : r.loss != null && ["loss", fmtNum(r.loss, 4)],
    r.best_val != null && !r.is_rl && ["best val loss", fmtNum(r.best_val, 3)],
    r.heldout_acc != null && ["held-out accuracy", `${(r.heldout_acc * 100).toFixed(1)}%`],
    r.quality && ["judged quality", `${r.quality.overall.toFixed(2)} at ${fmtTok(r.quality.tokens)} (${r.quality.n_judged} checkpoints judged)`],
    r.needle && r.needle.effective && ["needle retrieval", `holds to ${fmtInt(r.needle.effective)} tokens`],
    ["time", `${fmtDur(r.elapsed_s)}${r.tok_s_avg ? ` · ${fmtInt(r.tok_s_avg)} tok/s avg` : ""}`],
    ["started", `${r.started || "-"}${r.git_commit ? ` · git ${r.git_commit.slice(0, 8)}` : ""}`],
    (r.mixture || []).length > 0 && ["data", (r.mixture || []).slice(0, 6).join(", ") + ((r.mixture || []).length > 6 ? ` +${r.mixture.length - 6} more` : "")],
    (r.tasks || []).length > 0 && r.is_rl && ["RL tasks", taskMix(r.tasks)],
    nKids > 0 && ["continued by", `${nKids} run${nKids === 1 ? "" : "s"}`],
  ].filter(Boolean);
  return html`<div class="lg-card" ref=${ref} style=${style}>
    <div class="lg-card-t">${r.run_name}</div>
    <table class="lg-kv">${rows.map(([k, v]) => html`<tr key=${k}><th>${k}</th><td>${v}</td></tr>`)}</table>
    <div class="lg-card-foot muted">click to open the run page</div>
  </div>`;
}

function RunNode({ n, g, dim, hot, onEnter, onMove, onLeave }) {
  const style = `left:${n.x}px;top:${n.y}px;width:${g.nodeW}px;height:${g.nodeH}px`;
  if (n.ghost) return html`<div class=${"lg-node ghost" + (dim ? " dim" : "")} style=${style} onMouseEnter=${(e) => onEnter(n, e)} onMouseMove=${onMove} onMouseLeave=${onLeave}>
    <div class="lg-name">${n.label}</div><div class="lg-sub muted">not in this runs folder</div></div>`;
  const r = n.run, st = STAGES[stageOf(r)], hl = headline(r);
  return html`<a class=${`lg-node st-${st.cls} s-${r.status}` + (dim ? " dim" : "") + (hot ? " hot" : "")} style=${style} href=${"#/runs/" + encodeURIComponent(r.run_name)}
      onMouseEnter=${(e) => onEnter(n, e)} onMouseMove=${onMove} onMouseLeave=${onLeave} onFocus=${(e) => onEnter(n, e)} onBlur=${onLeave}
      aria-label=${`${r.run_name}, ${st.label}, ${r.status}`}>
    <div class="lg-name"><span class=${"lg-dot " + r.status} title=${r.status}></span>${r.run_name}</div>
    <div class="lg-sub"><span class="lg-stage">${st.label}</span><span>${paramsLabel(r.n_params)}</span><span>${amount(r)}</span></div>
    <div class="lg-sub">${hl ? html`<span title=${hl.tip}>${hl.k} <b>${hl.v}</b></span>` : html`<span class="muted">no eval yet</span>`}${r.status === "running" ? html`<span class="lg-prog"><span style=${`width:${(r.progress * 100).toFixed(1)}%`}></span></span>` : ""}</div>
  </a>`;
}

function LineageGraph({ runs }) {
  const g = GEOM;
  const L = useMemo(() => layoutLineage(runs, g), [runs]);
  const wrap = useRef(null);
  const [zoom, setZoom] = useState(null);  // null = fit the width (never below 75%)
  const [width, setWidth] = useState(1000);
  const [hover, setHover] = useState(null);  // {node, at}
  useLayoutEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);
  const fit = Math.max(0.75, Math.min(1, (width - 2) / L.W));
  const s = zoom == null ? fit : zoom;
  const lit = hover ? relatives(hover.node.id, L.parent, L.kids) : null;
  const onEnter = (n, e) => setHover({ node: n, at: e && e.clientX != null ? { x: e.clientX, y: e.clientY } : (() => { const b = e.target.getBoundingClientRect(); return { x: b.right, y: b.top }; })() });
  const onMove = (e) => setHover((h0) => (h0 ? { ...h0, at: { x: e.clientX, y: e.clientY } } : h0));
  const onLeave = () => setHover(null);
  const pinfo = hover && hover.node.run ? parentOf(hover.node.run.init_from) : null;
  return html`<div>
    <div class="row lg-tools">
      <span class="muted">zoom</span>
      <span class="seg"><button onClick=${() => setZoom(Math.max(0.5, +(s - 0.1).toFixed(2)))} title="zoom out">−</button>
        <button class=${zoom == null ? "active" : ""} onClick=${() => setZoom(null)} title="fit the width">fit</button>
        <button onClick=${() => setZoom(Math.min(1.5, +(s + 0.1).toFixed(2)))} title="zoom in">+</button></span>
      <span class="muted">${Math.round(s * 100)}%</span>
      <span class="lg-legend">${Object.values(STAGES).map((st) => html`<span key=${st.cls}><i class=${"lg-sw st-" + st.cls}></i>${st.label}</span>`)}
        <span><i class="lg-dot running"></i>running</span><span><i class="lg-dot stopped"></i>stopped</span><span class="muted">edge label = the checkpoint it started from, when not final.pt</span></span>
    </div>
    <div class="lg-wrap" ref=${wrap}>
      <div class="lg-canvas" style=${`width:${L.W * s}px;height:${L.H * s}px`}>
        <div class="lg-inner" style=${`width:${L.W}px;height:${L.H}px;transform:scale(${s})`}>
          <svg class="lg-edges" width=${L.W} height=${L.H} viewBox=${`0 0 ${L.W} ${L.H}`} aria-hidden="true">
            ${L.edges.map((e) => {
              const mx = (e.x1 + e.x2) / 2, on = lit && lit.has(e.from) && lit.has(e.to);
              const label = e.file && e.file !== "final.pt" ? shortCkpt(e.file) : null;
              return html`<g key=${e.from + ">" + e.to} class=${"lg-edge" + (lit ? (on ? " on" : " dim") : "")}>
                <path d=${`M${e.x1},${e.y1} C${mx},${e.y1} ${mx},${e.y2} ${e.x2 - 4},${e.y2}`} />
                <path class="lg-head" d=${`M${e.x2 - 7},${e.y2 - 4} L${e.x2},${e.y2} L${e.x2 - 7},${e.y2 + 4} Z`} />
                ${label && html`<text x=${e.x2 - 6} y=${e.y2 - 6} text-anchor="end">${label}</text>`}
              </g>`;
            })}
          </svg>
          ${L.nodes.map((n) => html`<${RunNode} key=${n.id} n=${n} g=${g} dim=${lit && !lit.has(n.id)} hot=${hover && hover.node.id === n.id} onEnter=${onEnter} onMove=${onMove} onLeave=${onLeave} />`)}
        </div>
      </div>
    </div>
    ${hover && html`<${HoverCard} node=${hover.node} at=${hover.at} parentInfo=${pinfo} nKids=${(L.kids.get(hover.node.id) || []).length} />`}
  </div>`;
}

// ----------------------------------------------------------------------------------------------- page
export function Home() {
  const [runs, setRuns] = useState(null);
  const [data, setData] = useState(null);
  const [gpu, setGpu] = useState(null);
  useEffect(() => {
    let alive = true;
    const tick = () => { api("/api/runs").then((r) => alive && setRuns(r)).catch(() => {}); api("/api/system/gpu").then((g) => alive && setGpu(g)).catch(() => {}); };
    tick();
    api("/api/data/sources").then(setData).catch(() => {});
    const id = setInterval(tick, 10000);
    return () => { alive = false; clearInterval(id); };
  }, []);
  const live = (runs || []).filter((r) => r.status === "running");
  const tags = data ? data.tags : [];
  return html`<div>
    <h1>Overview</h1>
    <div class="sub">${runs ? `${runs.length} runs · ${live.length} live` : "loading…"}${gpu && gpu.available ? ` · GPU ${gpu.used_gib.toFixed(1)}/${gpu.total_gib.toFixed(0)} GiB, ${gpu.util.toFixed(0)}% busy${gpu.temp_c != null ? `, ${gpu.temp_c.toFixed(0)} °C` : ""}${gpu.power_w != null ? `, ${gpu.power_w.toFixed(0)} W` : ""}${gpu.throttled ? " ⚠ throttling" : ""}` : ""}</div>
    ${live.map((r) => html`<a class="card" style="display:block;margin-bottom:10px" href=${"#/runs/" + encodeURIComponent(r.run_name)}>
      <div><b>${r.run_name}</b> <span class="status running">running</span> <span class="muted">· ${STAGES[stageOf(r)].label}</span></div>
      <div class="bar"><div style=${"width:" + (r.progress * 100).toFixed(1) + "%"}></div></div>
      <div class="muted">${r.total_steps ? `step ${fmtInt(r.update)} / ${fmtInt(r.total_steps)}` : `${fmtTok(r.tokens)} / ${fmtTok(r.total_tokens)}`} · ${r.is_rl ? `reward ${fmtNum(r.rl.reward_mean, 3)}` : `loss ${fmtNum(r.loss, 3)}`}${r.total_steps ? "" : ` · ${fmtInt(r.tok_s)} tok/s`} · ETA ${fmtDur(r.eta_s)}</div></a>`)}
    <h2>Runs and their lineage<${Info} k="run_graph" /></h2>
    ${!runs ? html`<div class="muted">loading…</div>` : runs.length === 0 ? html`<div class="empty-note">no runs yet</div>` : html`<${LineageGraph} runs=${runs} />`}
    <div class="legend" style="margin-top:6px">Hover a run for its details and to light up its lineage; click it for live charts, milestones, samples, checkpoints, events and config.</div>
    <h2>Data readiness<${Info} k="data_readiness" /></h2>
    ${!data ? html`<div class="muted">loading…</div>` : tags.length === 0 ? html`<div class="empty-note">no tokenized data yet</div>` : tags.map((t) => {
      const srcs = data.sources.filter((s) => s.tokenized[t]);
      const total = srcs.reduce((a, s) => a + (s.tokenized[t].train_tokens ?? 0), 0);  // a source with no count must not NaN the sum
      const sft = (data.sft && data.sft[t]) ? Object.keys(data.sft[t]) : [];
      return html`<div class="panel" style="margin-bottom:8px"><b>tokenizer ${t}</b> · ${fmtTok(total)} pretraining tokens across ${srcs.length} sources (${srcs.map((s) => `${s.name} ${s.tokenized[t].train_tokens == null ? "?" : fmtTok(s.tokenized[t].train_tokens)}`).join(", ")}) · ${sft.length} SFT/reasoning sets
        <div class="legend">details on the <a href="#/data">Data</a> page</div></div>`; })}
  </div>`;
}

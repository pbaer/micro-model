import { h } from "preact";
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import htm from "htm";
import { api, fmtTok, fmtTime, fmtBytes } from "../components/util.js";
import { Info } from "../components/info.js";

const html = htm.bind(h);

const SUPER = { public: "Public benchmarks", homebrew: "Homebrew evals" };
const PARAMS = (p) => p == null ? "unknown size" : p >= 1e9 ? (p / 1e9).toFixed(1) + "B" : Math.round(p / 1e6) + "M";
const enc = encodeURIComponent;
const EXTERNAL = "external models";
/** One line for an external comparison model's registry entry (runs/ext_<name>/model.json). */
const modelLine = (m) => `${m.hf_id} · ${PARAMS(m.params)} · ${m.license} · ${m.is_chat ? "chat" : "base"} model`
  + (m.train_tokens ? ` · ${fmtTok(m.train_tokens)} pretraining tokens (published)` : " · pretraining tokens not published");
export const detailHref = (run, ckpt, key) => `#/evals/${enc(run)}/${enc(ckpt)}/${enc(key)}`;

/** Card key for a column: one card per benchmark; the two swarm sets and the two pass@k sets share theirs. */
function cardKey(c) {
  if (c.key.startsWith("sw_")) return "ev_sw_" + c.key.replace(/^sw_[a-z0-9]+_/, "");
  if (c.key.startsWith("pk_")) return "ev_pass_at_k";
  return "ev_" + c.key;
}

/** Compact number in the style docs/results.md uses for that metric. */
export function fmtCell(v, fmt) {
  if (v == null || !isFinite(v)) return "n/a";
  if (fmt === "pct1") return (v * 100).toFixed(1);
  if (fmt === "pct") return (v * 100).toFixed(1) + "%";
  if (fmt === "int") return String(Math.round(v));
  if (fmt === "score") return v.toFixed(2);
  if (v === 0) return "0";
  let s = Number(v).toPrecision(3);  // frac: 3 significant digits, trailing zeros dropped but at least 2 decimals
  if (s.includes("e")) return s;
  if (s.includes(".")) s = s.replace(/0+$/, "").replace(/\.$/, "");
  const dec = s.includes(".") ? s.split(".")[1].length : 0;
  return dec >= 2 ? s : Number(v).toFixed(2);
}

/** Any value of a detail page: numbers by `fmt` (the table's formats plus num = up to 4 significant digits). */
function fmtVal(v, fmt) {
  if (v == null || v === "") return "–";
  if (typeof v === "boolean") return v ? "✓" : "✗";
  if (typeof v !== "number") return String(v);
  if (fmt === "int") return Math.round(v).toLocaleString();
  if (["pct1", "pct", "frac", "score"].includes(fmt)) return fmtCell(v, fmt);
  if (Number.isInteger(v)) return v.toLocaleString();
  return String(Number(v.toPrecision(4)));
}

/** Red (worst in its column, t = 0) through amber to green (best, t = 1): a linear walk along the hue. */
export const cellColour = (t) => t == null ? "var(--na-bg)" : `hsl(${Math.round(120 * t)}, 62%, 80%)`;

/** Rows sorted by one column: best first (dir "best") or worst first, respecting higher_is_better; n/a cells always
 *  last; ties keep the default chain order (Array.prototype.sort is stable). Returns [measured, unmeasured]. */
export function sortRows(rows, col, dir) {
  const has = rows.filter((r) => r.cells[col.key]), na = rows.filter((r) => !r.cells[col.key]);
  const sign = (col.higher_is_better ? -1 : 1) * (dir === "best" ? 1 : -1);
  return [has.slice().sort((a, b) => sign * (a.cells[col.key].value - b.cells[col.key].value)), na];
}

// ---------------------------------------------------------------------------------------------- hover card
// One card for the whole table, fed through a module-level setter so hovering never re-renders the table.
let setHover = null;

function HoverCard() {
  const [hv, setHv] = useState(null);
  const box = useRef(null);
  useEffect(() => { setHover = setHv; const off = () => setHv(null); addEventListener("scroll", off, true); return () => { setHover = null; removeEventListener("scroll", off, true); }; }, []);
  useLayoutEffect(() => {
    const c = box.current;
    if (!hv || !c) return;
    const a = hv.rect, m = 8, vw = document.documentElement.clientWidth, vh = window.innerHeight;
    let left = a.right + 6, top = a.top;
    if (left + c.offsetWidth > vw - m) left = Math.max(m, a.left - 6 - c.offsetWidth);
    if (top + c.offsetHeight > vh - m) top = Math.max(m, vh - m - c.offsetHeight);
    c.style.left = left + "px"; c.style.top = top + "px"; c.style.visibility = "visible";
  }, [hv]);
  if (!hv) return null;
  const { cell, col, row } = hv;
  return html`<div class="info-card ev-hover" ref=${box}>
    <div class="info-title">${col.label} <span class="muted">· ${col.group}</span></div>
    <div><b style="font-size:15px">${fmtCell(cell.value, col.fmt)}</b> <span class="muted">(${cell.value})</span> · ${row.run} / ${row.checkpoint}</div>
    ${row.model && html`<div class="ev-hsrc">external model: ${modelLine(row.model)}</div>`}
    <div class="ev-hsrc">${cell.source}</div>
    <ul>${cell.detail.split(" · ").filter(Boolean).map((x) => html`<li>${x}</li>`)}</ul>
    <div class="see">click for the full results: per-item outputs and grading where the eval saved them</div>
  </div>`;
}

function Cell({ cell, col, row }) {
  if (!cell) return html`<td class="ev-na">n/a</td>`;
  const enter = (e) => setHover && setHover({ cell, col, row, rect: e.currentTarget.getBoundingClientRect() });
  const leave = () => setHover && setHover(null);
  return html`<td style=${"background:" + cellColour(cell.t)} onMouseEnter=${enter} onMouseLeave=${leave}>
    <a class="ev-v" href=${detailHref(row.run, row.checkpoint, col.key)}>${fmtCell(cell.value, col.fmt)}</a></td>`;
}

// the view survives a visit to a detail page and back (module state, this tab only)
const view = { sort: null, hidden: new Set() };

export function EvalsPage() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  const [sort, setSortS] = useState(view.sort);  // {key, dir: "best" | "worst"} or null = default chain order
  const [hidden, setHiddenS] = useState(view.hidden);
  const setSort = (s) => { view.sort = s; setSortS(s); };
  const setHidden = (s) => { view.hidden = s; setHiddenS(s); };
  useEffect(() => { api("/api/evals").then(setD).catch((e) => setErr(String(e))); }, []);
  if (err) return html`<div><h1>Evals</h1><div class="panel warn">${err}</div></div>`;
  if (!d) return html`<div><h1>Evals</h1><div class="muted">loading…</div></div>`;

  const groups = [...new Set(d.columns.map((c) => c.group))];
  const cols = d.columns.filter((c) => !hidden.has(c.group));
  const spans = (key) => cols.reduce((a, c) => { const last = a[a.length - 1]; if (last && last.k === c[key]) last.n++; else a.push({ k: c[key], n: 1, c }); return a; }, []);
  const toggle = (g) => { const n = new Set(hidden); n.has(g) ? n.delete(g) : n.add(g); setHidden(n); };
  const filled = d.rows.reduce((a, r) => a + Object.keys(r.cells).length, 0);
  const sortCol = sort && d.columns.find((c) => c.key === sort.key);
  const clickCol = (c) => setSort(!sort || sort.key !== c.key ? { key: c.key, dir: "best" } : sort.dir === "best" ? { key: c.key, dir: "worst" } : null);
  const [measured, unmeasured] = sortCol ? sortRows(d.rows, sortCol, sort.dir) : [d.rows, []];
  const span = cols.length + 1;
  let lastParams;

  const rowHtml = (r) => {
    const alias = r.aliases.length ? ` = ${r.aliases.join(" = ")}` : "";
    return html`<tr>
      <td class="ev-first">${r.model ? html`<b title=${modelLine(r.model)}>${r.run}</b>` : html`<a href=${"#/runs/" + enc(r.run)}><b>${r.run}</b></a>`} <span class=${"stage-badge " + r.stage}>${r.stage}</span>
        <div class="legend">${sortCol ? html`<b>${PARAMS(r.params)}</b> · ` : ""}${r.model
          ? `${r.model.hf_id} · ${r.model.license} · ${r.model.is_chat ? "chat" : "base"}${r.tokens != null ? " · " + fmtTok(r.tokens) : ""}`
          : html`${r.checkpoint}${alias} · ${fmtTok(r.tokens)}${r.own_tokens != null && r.own_tokens !== r.tokens ? ` (${fmtTok(r.own_tokens)})` : ""}`}</div></td>
      ${cols.map((c) => html`<${Cell} cell=${r.cells[c.key]} col=${c} row=${r} />`)}
    </tr>`;
  };
  const sep = (text) => html`<tr class="ev-sep"><td colspan=${span}><span>${text}</span></td></tr>`;

  return html`<div class="evals">
    <h1>Evals<${Info} k="ev_table" /></h1>
    <div class="sub">${d.rows.length} evaluated checkpoints × ${d.columns.length} measures · ${filled} of ${d.rows.length * d.columns.length} cells measured · from the result files under <code>runs/</code></div>
    <div class="row" style="margin-bottom:8px">
      <span class="legend">groups:</span>
      ${groups.map((g) => html`<button class=${hidden.has(g) ? "" : "active"} onClick=${() => toggle(g)}>${g}</button>`)}
    </div>
    <div class="row legend ev-legend">
      <span><i class="ev-swatch ev-grad"></i> worst → best within a column<${Info} k="ev_colour" /></span>
      <span><i class="ev-swatch" style="background:var(--na-bg)"></i> n/a: never measured on that checkpoint</span>
      <span>↓ = lower is better</span>
      <span>click a column header to sort<${Info} k="ev_sort" /></span>
      <span>hover a score for its file and n, click it for the full results<${Info} k="ev_detail" /></span>
    </div>
    <div class="ev-wrap">
      <table class="ev-table">
        <thead>
          <tr><th class=${"ev-first ev-sortable" + (sortCol ? "" : " ev-sorted")} rowspan="3" onClick=${() => setSort(null)}
              title="the default order: size, then stage along the chain, then run and tokens">checkpoint<div class="legend">stage · tokens seen in total (this run)</div>
              <div class="ev-sort">${sortCol ? html`sorted by <b>${sortCol.label}</b> · click: default order` : "▾ chain order"}</div></th>
            ${spans("super").map((s) => html`<th class=${"ev-super " + s.k} colspan=${s.n}><span>${SUPER[s.k] || s.k}</span></th>`)}</tr>
          <tr>${spans("group").map((s) => html`<th class=${"ev-group " + s.c.super} colspan=${s.n}><span>${s.k}</span></th>`)}</tr>
          <tr>${cols.map((c) => {
            const on = sortCol && sortCol.key === c.key;
            return html`<th class=${"ev-col ev-sortable" + (on ? " ev-sorted" : "")} onClick=${() => clickCol(c)}
                aria-sort=${on ? (sort.dir === "best" === c.higher_is_better ? "descending" : "ascending") : "none"}
                title=${on ? (sort.dir === "best" ? "best first; click: worst first" : "worst first; click: default order") : "click: sort best first"}>
              ${c.label}${c.higher_is_better ? "" : " ↓"}<${Info} k=${cardKey(c)} />
              ${on && html`<div class="ev-sort">${sort.dir === "best" ? "▼ best first" : "▲ worst first"}</div>`}</th>`;
          })}</tr>
        </thead>
        <tbody>
          ${sortCol
            ? html`${measured.map(rowHtml)}${unmeasured.length ? sep(`n/a: ${sortCol.label} (${sortCol.group}) never measured on these ${unmeasured.length} checkpoints`) : ""}${unmeasured.map(rowHtml)}`
            : measured.map((r) => {
              const g = r.group === EXTERNAL ? EXTERNAL : r.params;
              const s = g !== lastParams;
              lastParams = g;
              const label = g === EXTERNAL ? "external models · open-weight comparison models, run locally through the same evals with their own tokenizer and chat template"
                : `${PARAMS(r.params)} parameters`;
              return html`${s ? sep(label) : ""}${rowHtml(r)}`;
            })}
        </tbody>
      </table>
    </div>
    <${HoverCard} />
    <div class="legend" style="margin-top:6px">Rows: every checkpoint with at least one result file, 336M chain first, each chain in stage order (base → sft → tool → rl), then by run name
      (sorted by a column, the size moves into each row's second line). External comparison models (<code>runs/ext_*</code>, <code>slm.eval.external</code>)
      come last in their own group; evals that need our tool protocol or think span are n/a for them, and base models are n/a on chat evals.
      Checkpoints of one run at the same token count are the same weights and share a row (<code>best.pt = step_00150.pt</code>).
      The numbers and what they led to are discussed in <code>docs/results.md</code>.</div>
  </div>`;
}

// ---------------------------------------------------------------------------------------------- detail page
function Val({ v, type }) {
  if (typeof v === "boolean" && (type === "flag" || type === "kv")) return v ? html`<b>yes</b>` : html`<span class="muted">no</span>`;
  if (typeof v === "boolean") return html`<span class=${v ? "evd-yes" : "evd-no"}>${v ? "✓" : "✗"}</span>`;
  if (type === "score" && typeof v === "number") return html`<span class=${v >= 4 ? "score-good" : v <= 2 ? "score-bad" : "score-mid"}>${Number.isInteger(v) ? v : v.toFixed(2)}</span>`;
  return fmtVal(v, type);
}

function Block({ b }) {
  let body;
  if (b.kind === "kv") body = html`<div class="evd-kv">${Object.entries(b.value).filter(([, v]) => v != null && v !== "").map(([k, v]) => html`<span><i>${k}</i> ${typeof v === "object" ? JSON.stringify(v) : html`<${Val} v=${v} type="kv" />`}</span>`)}</div>`;
  else if (b.kind === "tools") body = html`<div>${b.value.map((t) => html`<div class="toolcall"><pre class="code">${Array.isArray(t) ? t[0] : JSON.stringify(t)}</pre><span class="result">${Array.isArray(t) ? String(t[1]) : ""}</span></div>`)}</div>`;
  else if (b.kind === "think") body = html`<pre class="think">${b.value}</pre>`;
  else body = html`<pre class="evd-text">${typeof b.value === "string" ? b.value : JSON.stringify(b.value, null, 1)}</pre>`;
  return html`<div class="evd-blk"><div class="evd-lab">${b.label}</div>${body}</div>`;
}

function DataTable({ t }) {
  return html`<div class="evd-table"><h3>${t.title}</h3>
    <table><tr>${t.columns.map((c) => html`<th>${c.label}</th>`)}</tr>
      ${t.rows.map((r, i) => html`<tr class=${i === t.highlight ? "sel" : ""}>${r.map((v, j) => html`<td class=${t.columns[j].type === "text" ? "l" : ""}><${Val} v=${v} type=${t.columns[j].type} /></td>`)}</tr>`)}</table>
    ${t.note && html`<div class="legend">${t.note}</div>`}</div>`;
}

function Items({ it, filter, setFilter, qInput, setQInput, setOffset, open, toggleRow }) {
  if (!it.available) return html`<div class="panel evd-none"><b>No per-item data.</b> ${it.note}</div>`;
  const opts = [["all", "all", it.total], ["pass", it.labels.pass, it.counts.pass], ["fail", it.labels.fail, it.counts.fail], ["other", it.labels.other || "neither", it.counts.other]];
  const last = Math.min(it.offset + it.limit, it.n_filtered);
  return html`<div>
    <div class="row" style="margin-bottom:6px">
      ${opts.filter(([k, , n]) => k === "all" || n > 0).map(([k, lab, n]) => html`<button class=${filter === k ? "active" : ""} onClick=${() => setFilter(k)}>
        ${k === "pass" ? html`<span class="evd-yes">✓</span> ` : k === "fail" ? html`<span class="evd-no">✗</span> ` : ""}${lab} · ${n}</button>`)}
      <input type="text" placeholder="search prompt / output" value=${qInput} onInput=${(e) => setQInput(e.target.value)} style="width:220px" />
      <span class="muted" style="margin-left:auto">${it.n_filtered ? `rows ${it.offset + 1}–${last} of ${it.n_filtered}` : "no rows match"}${it.n_filtered !== it.total ? ` (of ${it.total})` : ""}</span>
      <button disabled=${it.offset === 0} onClick=${() => setOffset(Math.max(0, it.offset - it.limit))}>‹ prev</button>
      <button disabled=${last >= it.n_filtered} onClick=${() => setOffset(it.offset + it.limit)}>next ›</button>
    </div>
    <div class="legend" style="margin-bottom:6px">${it.note} ${it.source && html`<code>${it.source}</code>`} · click a row to expand it</div>
    <table class="evd-items">
      <tr><th>#</th><th title="verdict for this column">✓</th>${it.columns.map((c) => html`<th class=${c.type === "text" || c.type === "id" ? "l" : ""}>${c.label}</th>`)}</tr>
      ${it.rows.map((r) => html`
        <tr class=${"click" + (open.has(r.i) ? " sel" : "")} onClick=${() => toggleRow(r.i)}>
          <td class="muted">${r.i}</td>
          <td>${r.ok === true ? html`<span class="evd-yes">✓</span>` : r.ok === false ? html`<span class="evd-no">✗</span>` : html`<span class="muted">·</span>`}</td>
          ${it.columns.map((c) => c.type === "text" ? html`<td class="l"><div class="evd-clip">${fmtVal(r.c[c.key])}</div></td>` : c.type === "id" ? html`<td class="l evd-id">${fmtVal(r.c[c.key])}</td>` : html`<td><${Val} v=${r.c[c.key]} type=${c.type} /></td>`)}
        </tr>
        ${open.has(r.i) && html`<tr class="evd-x"><td colspan=${it.columns.length + 2}>${r.x.length ? r.x.map((b) => html`<${Block} b=${b} />`) : html`<span class="muted">nothing more stored for this row</span>`}</td></tr>`}`)}
    </table>
  </div>`;
}

export function EvalDetailPage({ run, checkpoint, colKey }) {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  const [file, setFile] = useState(null);
  const [filter, setFilterS] = useState("all");
  const [qInput, setQInput] = useState("");
  const [q, setQ] = useState("");
  const [offset, setOffset] = useState(0);
  const [open, setOpen] = useState(() => new Set());
  const setFilter = (f) => { setFilterS(f); setOffset(0); };
  useEffect(() => { const id = setTimeout(() => { setQ(qInput); setOffset(0); }, 250); return () => clearTimeout(id); }, [qInput]);
  useEffect(() => {
    let alive = true;
    const p = new URLSearchParams({ run, checkpoint, key: colKey, filter, q, offset: String(offset), limit: "100" });
    if (file) p.set("file", file);
    api("/api/evals/detail?" + p).then((x) => { if (alive) { setD(x); setErr(null); setOpen(new Set()); } }).catch((e) => alive && setErr(String(e)));
    return () => { alive = false; };
  }, [run, checkpoint, colKey, file, filter, q, offset]);
  const toggleRow = (i) => { const n = new Set(open); n.has(i) ? n.delete(i) : n.add(i); setOpen(n); };
  const back = html`<div class="legend" style="margin-bottom:6px"><a href="#/evals">← Evals</a> · <a href=${"#/runs/" + enc(run)}>run page: ${run}</a></div>`;
  if (err) return html`<div>${back}<h1>${colKey} · ${run} / ${checkpoint}</h1><div class="panel warn">${err}</div></div>`;
  if (!d) return html`<div>${back}<h1>${colKey} · ${run} / ${checkpoint}</h1><div class="muted">loading…</div></div>`;

  const c = d.column, ri = d.run_info || {};
  const ff = Object.entries(d.file_fields || {}).filter(([, v]) => v != null && v !== "");
  return html`<div class="evals evd">
    ${back}
    <h1>${c.label} <span class="muted" style="font-weight:400">· ${c.group}</span><${Info} k=${cardKey(c)} /></h1>
    <div class="sub"><b>${d.run}</b> <span class=${"stage-badge " + d.stage}>${d.stage}</span> ${d.checkpoint}${d.aliases.length ? " = " + d.aliases.join(" = ") : ""}${" · "}
      ${fmtTok(d.tokens)} tokens seen${d.own_tokens != null && d.own_tokens !== d.tokens ? ` (${fmtTok(d.own_tokens)} in this run)` : ""} · ${PARAMS(d.params)} parameters
      ${d.model ? html`<div class="legend">external model: ${modelLine(d.model)}</div>` : ""}</div>
    <div class="tiles">
      <div class="tile" style=${"background:" + cellColour(d.cell && d.source === d.cell.source ? d.cell.t : null)}>
        <div class="k">${c.label}${c.higher_is_better ? "" : " (lower is better)"}</div><div class="v">${fmtCell(d.value, c.fmt)}</div>
        <div class="s">column range ${fmtCell(c.min, c.fmt)} – ${fmtCell(c.max, c.fmt)}</div></div>
      ${d.summary.filter((s) => !(s.value === d.value && s.fmt && s.fmt !== "int")).map((s) => html`<div class="tile"><div class="k">${s.label}</div><div class="v evd-tv">${fmtVal(s.value, s.fmt)}</div></div>`)}
    </div>
    ${d.notes.map((n) => html`<div class="panel warn">${n}</div>`)}

    <h2>Source${d.sources.length > 1 ? "s" : ""}</h2>
    <table class="evd-src">
      <tr><th class="l">file</th><th>value</th><th class="l">details</th><th>size</th><th>written</th></tr>
      ${d.sources.map((s) => html`<tr class=${(s.shown ? "sel " : "") + (d.sources.length > 1 ? "click" : "")} onClick=${() => { if (d.sources.length > 1) { setFile(s.source); setOffset(0); } }}>
        <td class="l"><code>runs/${s.source}</code>${s.winner ? html` <span class="stage-badge">shown in the table</span>` : ""}</td><td>${fmtCell(s.value, c.fmt)}</td>
        <td class="l evd-det">${s.detail}</td><td>${s.size != null ? (s.size < 1 << 20 ? (s.size / 1024).toFixed(1) + " KiB" : fmtBytes(s.size)) : "–"}</td><td>${fmtTime(s.mtime)}</td></tr>`)}
    </table>
    ${d.sources.length > 1 && html`<div class="legend">Several files measured this cell; the table shows the top one (priority rules in the Evals card). Click a file to see its contents.</div>`}
    ${ff.length > 0 && html`<div class="evd-kv" style="margin-top:6px">${ff.map(([k, v]) => html`<span><i>${k}</i> ${String(v)}</span>`)}</div>`}

    ${d.tables.map((t) => html`<${DataTable} t=${t} />`)}

    <h2>Per-item results<${Info} k="ev_detail" /></h2>
    <${Items} it=${d.items} filter=${filter} setFilter=${setFilter} qInput=${qInput} setQInput=${setQInput} setOffset=${setOffset} open=${open} toggleRow=${toggleRow} />

    ${d.rubric_text && html`<details class="evd-more"><summary>judge instructions (the rubric text every packet carries)</summary><pre>${d.rubric_text}</pre></details>`}
    ${d.log && html`<details class="evd-more"><summary>log: <code>${d.log.file}</code> (last ${d.log.lines.length} lines, progress bars removed)</summary><pre>${d.log.lines.join("\n")}</pre></details>`}
    <details class="evd-more"><summary>run: ${d.run} · ${ri.stage || "?"}${ri.init_from ? ` · from ${ri.init_from}` : ""}${ri.started ? ` · started ${ri.started}` : ""}</summary>
      ${ri.notes && html`<p>${ri.notes}</p>`}
      <div class="legend">${ri.git_commit ? `git ${ri.git_commit.slice(0, 10)} · ` : ""}${ri.n_params ? PARAMS(ri.n_params) + " parameters" : ""}</div>
      ${ri.config ? html`<pre>${JSON.stringify(ri.config, null, 1)}</pre>` : html`<div class="muted">no run.json config</div>`}</details>
  </div>`;
}

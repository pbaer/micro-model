import { h } from "preact";
import { useEffect, useState } from "preact/hooks";
import htm from "htm";
import { api, fmtTok } from "../components/util.js";
import { Info } from "../components/info.js";

const html = htm.bind(h);

const SUPER = { public: "Public benchmarks", homebrew: "Homebrew evals" };
const PARAMS = (p) => p == null ? "unknown size" : p >= 1e9 ? (p / 1e9).toFixed(1) + "B" : Math.round(p / 1e6) + "M";

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

/** Red (worst in its column, t = 0) through amber to green (best, t = 1): a linear walk along the hue. */
export const cellColour = (t) => t == null ? "var(--na-bg)" : `hsl(${Math.round(120 * t)}, 62%, 80%)`;

function Cell({ cell, col, showSrc }) {
  if (!cell) return html`<td class="ev-na">n/a</td>`;
  const tip = `${col.label} (${col.group}) = ${cell.value}\n${cell.source}\n${cell.detail.split(" · ").join("\n")}`;
  return html`<td style=${"background:" + cellColour(cell.t)} title=${tip}>${fmtCell(cell.value, col.fmt)}${showSrc ? html`<div class="ev-src">${cell.source}</div>` : ""}</td>`;
}

export function EvalsPage() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  const [showSrc, setShowSrc] = useState(false);
  const [hidden, setHidden] = useState(() => new Set());
  useEffect(() => { api("/api/evals").then(setD).catch((e) => setErr(String(e))); }, []);
  if (err) return html`<div><h1>Evals</h1><div class="panel warn">${err}</div></div>`;
  if (!d) return html`<div><h1>Evals</h1><div class="muted">loading…</div></div>`;

  const groups = [...new Set(d.columns.map((c) => c.group))];
  const cols = d.columns.filter((c) => !hidden.has(c.group));
  const spans = (key) => cols.reduce((a, c) => { const last = a[a.length - 1]; if (last && last.k === c[key]) last.n++; else a.push({ k: c[key], n: 1, c }); return a; }, []);
  const toggle = (g) => setHidden((s) => { const n = new Set(s); n.has(g) ? n.delete(g) : n.add(g); return n; });
  const filled = d.rows.reduce((a, r) => a + Object.keys(r.cells).length, 0);
  let lastParams;

  return html`<div class="evals">
    <h1>Evals<${Info} k="ev_table" /></h1>
    <div class="sub">${d.rows.length} evaluated checkpoints × ${d.columns.length} measures · ${filled} of ${d.rows.length * d.columns.length} cells measured · from the result files under <code>runs/</code></div>
    <div class="row" style="margin-bottom:8px">
      <span class="legend">groups:</span>
      ${groups.map((g) => html`<button class=${hidden.has(g) ? "" : "active"} onClick=${() => toggle(g)}>${g}</button>`)}
      <button class=${showSrc ? "active" : ""} onClick=${() => setShowSrc(!showSrc)} style="margin-left:auto">${showSrc ? "hide" : "show"} sources</button>
    </div>
    <div class="row legend ev-legend">
      <span><i class="ev-swatch ev-grad"></i> worst → best within a column<${Info} k="ev_colour" /></span>
      <span><i class="ev-swatch" style="background:var(--na-bg)"></i> n/a: never measured on that checkpoint</span>
      <span>↓ = lower is better</span>
      <span>hover a cell for its file, n and details</span>
    </div>
    <div class="ev-wrap">
      <table class="ev-table">
        <thead>
          <tr><th class="ev-first" rowspan="3">checkpoint<div class="legend">stage · tokens seen in total (this run)</div></th>
            ${spans("super").map((s) => html`<th class=${"ev-super " + s.k} colspan=${s.n}><span>${SUPER[s.k] || s.k}</span></th>`)}</tr>
          <tr>${spans("group").map((s) => html`<th class=${"ev-group " + s.c.super} colspan=${s.n}><span>${s.k}</span></th>`)}</tr>
          <tr>${cols.map((c) => html`<th class="ev-col">${c.label}${c.higher_is_better ? "" : " ↓"}<${Info} k=${cardKey(c)} /></th>`)}</tr>
        </thead>
        <tbody>
          ${d.rows.map((r) => {
            const sep = r.params !== lastParams;
            lastParams = r.params;
            const alias = r.aliases.length ? ` = ${r.aliases.join(" = ")}` : "";
            return html`${sep ? html`<tr class="ev-sep"><td colspan=${cols.length + 1}><span>${PARAMS(r.params)} parameters</span></td></tr>` : ""}
              <tr>
                <td class="ev-first"><a href=${"#/runs/" + encodeURIComponent(r.run)}><b>${r.run}</b></a> <span class=${"stage-badge " + r.stage}>${r.stage}</span>
                  <div class="legend">${r.checkpoint}${alias} · ${fmtTok(r.tokens)}${r.own_tokens != null && r.own_tokens !== r.tokens ? ` (${fmtTok(r.own_tokens)})` : ""}</div></td>
                ${cols.map((c) => html`<${Cell} cell=${r.cells[c.key]} col=${c} showSrc=${showSrc} />`)}
              </tr>`;
          })}
        </tbody>
      </table>
    </div>
    <div class="legend" style="margin-top:6px">Rows: every checkpoint with at least one result file, 336M chain first, each chain in stage order (base → sft → tool → rl), then by run name.
      Checkpoints of one run at the same token count are the same weights and share a row (<code>best.pt = step_00150.pt</code>).
      The numbers and what they led to are discussed in <code>docs/results.md</code>.</div>
  </div>`;
}

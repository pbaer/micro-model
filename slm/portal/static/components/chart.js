import { h } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import htm from "htm";
import { fmtTok, fmtDur } from "./util.js";
import { Info } from "./info.js";

const html = htm.bind(h);
const COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"];
const NEAR_PX = 24;  // a series with no point exactly under the cursor shows its nearest point within this many pixels

const fmtVal = (v) => (v == null || !isFinite(v) ? "–" : Math.abs(v) >= 1e5 ? v.toPrecision(5) : Math.abs(v) < 0.001 && v !== 0 ? v.toExponential(3) : String(+Number(v).toPrecision(5)));

/** The value series `si` shows at cursor index `idx`: the value there, else the nearest non-null point whose x lies
 *  within NEAR_PX pixels (series sampled on different grids are joined, so most of a sparse series is null). */
export function valueNear(u, si, idx) {
  const ys = u.data[si], xs = u.data[0];
  if (ys[idx] != null) return { v: ys[idx], x: xs[idx], exact: true };
  const px0 = u.valToPos(xs[idx], "x");
  for (let d = 1; d < ys.length; d++) {
    let best = null;
    for (const j of [idx - d, idx + d]) {
      if (j < 0 || j >= ys.length || ys[j] == null) continue;
      if (Math.abs(u.valToPos(xs[j], "x") - px0) <= NEAR_PX) best = best == null || Math.abs(xs[j] - xs[idx]) < Math.abs(xs[best] - xs[idx]) ? j : best;
    }
    if (best != null) return { v: ys[best], x: xs[best], exact: false };
    const lo = idx - d >= 0 ? Math.abs(u.valToPos(xs[idx - d], "x") - px0) : Infinity;
    const hi = idx + d < xs.length ? Math.abs(u.valToPos(xs[idx + d], "x") - px0) : Infinity;
    if (lo > NEAR_PX && hi > NEAR_PX) break;  // past the pixel window on both sides
  }
  return null;
}

/** The hover card: one fixed element per chart, filled and moved by the uPlot cursor hook (no re-render per mouse
 *  move), shown only for the chart under the pointer (the synced charts move their cursor line but stay quiet). */
function hoverCardPlugin(xfmt, xlabel, colorOf, dashOf) {
  let card = null, over = null, inside = false;
  const hide = () => { if (card) card.style.display = "none"; };
  return {
    hooks: {
      init: (u) => {
        card = document.createElement("div");
        card.className = "chart-card";
        card.style.display = "none";
        document.body.appendChild(card);
        over = u.over;
        over.addEventListener("mouseenter", () => { inside = true; });
        over.addEventListener("mouseleave", () => { inside = false; hide(); });
      },
      setCursor: (u) => {
        const { idx, left, top } = u.cursor;
        if (!inside || idx == null || left == null || left < 0) return hide();
        const rows = [];
        for (let si = 1; si < u.series.length; si++) {
          const s = u.series[si];
          if (!s.show) continue;
          const hit = valueNear(u, si, idx);
          if (!hit) continue;
          rows.push({ v: hit.v, html: `<div class="cc-row"><i class="cc-sw" style="border-top:2px ${dashOf(si) ? "dashed" : "solid"} ${colorOf(si)}"></i><span class="cc-l">${escapeHtml(s.label)}</span><b>${fmtVal(hit.v)}</b>${hit.exact ? "" : `<span class="cc-at">at ${escapeHtml(xfmt(hit.x))}</span>`}</div>` });
        }
        if (!rows.length) return hide();
        rows.sort((a, b) => (isFinite(b.v) ? b.v : -Infinity) - (isFinite(a.v) ? a.v : -Infinity));  // highest value on top
        card.innerHTML = `<div class="cc-x">${escapeHtml(xlabel)} ${escapeHtml(xfmt(u.data[0][idx]))}</div>${rows.map((r) => r.html).join("")}`;
        card.style.display = "block";
        const b = over.getBoundingClientRect(), w = card.offsetWidth, hh = card.offsetHeight, m = 8;
        let x = b.left + left + 14, y = b.top + top + 14;
        if (x + w + m > innerWidth) x = b.left + left - w - 14;
        if (y + hh + m > innerHeight) y = b.top + top - hh - 14;
        card.style.left = `${Math.max(m, x)}px`;
        card.style.top = `${Math.max(m, y)}px`;
      },
      destroy: () => { if (card) card.remove(); card = null; },
    },
  };
}

const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

/** props: title, info (a cards.js key for the "?" beside the title), series=[{label, x:[], y:[], color?, points?, scale?, dash?}],
 *  xmode ('tokens'|'update'|'time'), logy, height, ymin, ymin2
 *  A series with scale:"y2" is drawn against a second axis on the right (e.g. temperature vs power).
 *  The title is rendered here rather than by uPlot so the info icon can sit inline after it. Hover values appear in a
 *  card beside the cursor, highest value on top; the legend below is static (name + line in the series' colour): hovering a name highlights
 *  that series, clicking it hides or shows it. The cursor draws only its vertical line. */
export function Chart({ title, info, series, xmode = "tokens", logy = false, height = 240, ymin, ymin2 }) {
  const ref = useRef(null);
  const plot = useRef(null);
  const [hidden, setHidden] = useState(() => new Set());
  const colorOf = (i) => series[i].color || COLORS[i % COLORS.length];
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const tables = series.map((s) => [s.x, s.y]);
    const data = tables.length === 1 ? tables[0] : uPlot.join(tables);
    const xfmt = xmode === "tokens" ? fmtTok : xmode === "time" ? (v) => fmtDur(v) : (v) => String(v);
    const yfmt = (u, vals) => vals.map((v) => (v == null || !isFinite(v) ? "" : Math.abs(v) >= 1000 ? v.toPrecision(4) : Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(1) : +v.toPrecision(4)));
    const hasY2 = series.some((s) => s.scale === "y2");
    const xlabel = xmode === "tokens" ? "at" : xmode === "time" ? "after" : xmode === "update" ? "step" : xmode;
    const opts = {
      width: el.clientWidth || 480,
      height,
      legend: { show: false },  // replaced by the static legend below and the hover card
      // y:false drops the horizontal crosshair line; the vertical one marks the position the hover card describes.
      // setSeries:false keeps a legend focus from propagating to the other synced charts (they share only the cursor).
      cursor: { y: false, sync: { key: "runs", setSeries: false }, focus: { prox: 0 } },
      focus: { alpha: 0.12 },
      plugins: [hoverCardPlugin(xfmt, xlabel, (si) => colorOf(si - 1), (si) => !!series[si - 1].dash)],
      scales: {
        x: { time: false },
        y: { distr: logy ? 3 : 1, range: ymin === undefined ? undefined : (u, min, max) => [Math.min(ymin, min), max] },
        ...(hasY2 ? { y2: { range: ymin2 === undefined ? undefined : (u, min, max) => [Math.min(ymin2, min), max] } } : {}),
      },
      axes: [
        { values: (u, vals) => vals.map(xfmt), stroke: "#666", grid: { stroke: "#eee" } },
        { stroke: "#666", grid: { stroke: "#eee" }, size: 60, values: yfmt },
        ...(hasY2 ? [{ scale: "y2", side: 1, stroke: "#666", grid: { show: false }, size: 60, values: yfmt }] : []),
      ],
      series: [
        { label: xmode },
        ...series.map((s, i) => ({ label: s.label, scale: s.scale === "y2" ? "y2" : "y", stroke: colorOf(i), width: s.width || 1.5, dash: s.dash ? [6, 4] : undefined,
          points: { show: !!s.points, size: 5 }, spanGaps: true, show: !hidden.has(s.label) })),
      ],
    };
    if (plot.current) plot.current.destroy();
    plot.current = new uPlot(opts, data, el);
    const ro = new ResizeObserver(() => plot.current && plot.current.setSize({ width: el.clientWidth, height }));
    ro.observe(el);
    return () => { ro.disconnect(); if (plot.current) { plot.current.destroy(); plot.current = null; } };
  }, [series, xmode, logy, height, ymin, ymin2]);
  const toggle = (i) => {
    const label = series[i].label, next = new Set(hidden);
    next.has(label) ? next.delete(label) : next.add(label);
    setHidden(next);
    if (plot.current) plot.current.setSeries(i + 1, { show: !next.has(label) });
  };
  const focus = (i) => { if (plot.current) plot.current.setSeries(i == null ? null : i + 1, { focus: true }); };
  return html`<div class="chart">${title && html`<div class="chart-title">${title}${info && html`<${Info} k=${info} />`}</div>`}<div ref=${ref}></div>
    ${series.length > 0 && html`<div class="chart-legend" onMouseLeave=${() => focus(null)}>${series.map((s, i) => html`<button key=${s.label + i} type="button"
        class=${"cl-item" + (hidden.has(s.label) ? " off" : "")} onMouseEnter=${() => !hidden.has(s.label) && focus(i)} onClick=${() => toggle(i)}
        title=${hidden.has(s.label) ? "hidden: click to show" : "click to hide; hover to highlight"}>
      <i class="cc-sw" style=${`border-top:2px ${s.dash ? "dashed" : "solid"} ${colorOf(i)}`}></i>${s.label}${s.scale === "y2" ? html`<span class="cl-axis"> (right axis)</span>` : ""}</button>`)}</div>`}
  </div>`;
}

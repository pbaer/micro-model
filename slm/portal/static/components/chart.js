import { h } from "preact";
import { useEffect, useRef } from "preact/hooks";
import htm from "htm";
import { fmtTok, fmtDur } from "./util.js";

const html = htm.bind(h);
const COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"];

/** props: title, series=[{label, x:[], y:[], color?, points?, scale?}], xmode ('tokens'|'update'|'time'), logy, height, ymin, ymin2
 *  A series with scale:"y2" is drawn against a second axis on the right (e.g. temperature vs power). */
export function Chart({ title, series, xmode = "tokens", logy = false, height = 240, ymin, ymin2 }) {
  const ref = useRef(null);
  const plot = useRef(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const tables = series.map((s) => [s.x, s.y]);
    const data = tables.length === 1 ? tables[0] : uPlot.join(tables);
    const xfmt = xmode === "tokens" ? fmtTok : xmode === "time" ? (v) => fmtDur(v) : (v) => String(v);
    const yfmt = (u, vals) => vals.map((v) => (v == null || !isFinite(v) ? "" : Math.abs(v) >= 1000 ? v.toPrecision(4) : Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(1) : +v.toPrecision(4)));
    const hasY2 = series.some((s) => s.scale === "y2");
    const opts = {
      width: el.clientWidth || 480,
      height,
      title,
      cursor: { sync: { key: "runs" } },
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
        { label: xmode, value: (u, v) => (v == null ? "-" : xfmt(v)) },
        ...series.map((s, i) => ({ label: s.label, scale: s.scale === "y2" ? "y2" : "y", stroke: s.color || COLORS[i % COLORS.length], width: s.width || 1.5, points: { show: !!s.points, size: 5 }, spanGaps: true, value: (u, v) => (v == null ? "-" : Number(v).toPrecision(5)) })),
      ],
    };
    if (plot.current) plot.current.destroy();
    plot.current = new uPlot(opts, data, el);
    const ro = new ResizeObserver(() => plot.current && plot.current.setSize({ width: el.clientWidth, height }));
    ro.observe(el);
    return () => { ro.disconnect(); if (plot.current) { plot.current.destroy(); plot.current = null; } };
  }, [series, xmode, logy, height, title, ymin, ymin2]);
  return html`<div class="chart"><div ref=${ref}></div></div>`;
}

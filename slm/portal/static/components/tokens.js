import { h } from "preact";
import htm from "htm";

const html = htm.bind(h);
const HUES = [210, 30, 120, 280, 60, 340, 170, 20];

/** Per-token labels from a ChatEncoding's segments ([[start, end, label], ...]); null where unlabelled. */
export function segmentLabels(segments, n) {
  const out = new Array(n).fill(null);
  (segments || []).forEach(([s, e, label]) => { for (let i = Math.max(0, s); i < Math.min(e, n); i++) out[i] = label; });
  return out;
}

/** pieces: [{id, piece, special, loss?}]; boundaries: set of indices that start a document;
 *  segments: chat segments, used to mark declared-function blocks (<|python_def|> ... <|/python_def|>) */
export function TokenChips({ pieces, boundaries, segments, showIds = false, lossMask = false, onHover }) {
  const b = boundaries ? new Set(boundaries) : null;
  const labels = segments ? segmentLabels(segments, pieces.length) : null;
  return html`<div class="chips">${pieces.map((p, i) => {
    const cls = ["chip", p.special ? "special" : "", b && b.has(i) ? "boundary" : "", lossMask ? (p.loss ? "target" : "masked") : "",
      labels && labels[i] === "python_def" ? "def" : ""].join(" ");
    const style = p.special || lossMask || (labels && labels[i]) ? "" : `background:hsl(${HUES[i % HUES.length]} 70% 92%);border-color:hsl(${HUES[i % HUES.length]} 50% 80%)`;
    const text = p.piece.replace(/ /g, "·").replace(/\n/g, "↵\n").replace(/\t/g, "→");
    return html`<span class=${cls} style=${style} title=${`id ${p.id}` + (labels && labels[i] ? ` · ${labels[i]}` : "") + (p.start != null ? ` · chars ${p.start}-${p.end}` : "")} onMouseEnter=${onHover ? () => onHover(p, i) : null}>${text}${showIds ? html`<sub>${p.id}</sub>` : ""}</span>`;
  })}</div>`;
}

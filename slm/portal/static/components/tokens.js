import { h } from "preact";
import htm from "htm";

const html = htm.bind(h);
const HUES = [210, 30, 120, 280, 60, 340, 170, 20];

/** pieces: [{id, piece, special, loss?}]; boundaries: set of indices that start a document */
export function TokenChips({ pieces, boundaries, showIds = false, lossMask = false, onHover }) {
  const b = boundaries ? new Set(boundaries) : null;
  return html`<div class="chips">${pieces.map((p, i) => {
    const cls = ["chip", p.special ? "special" : "", b && b.has(i) ? "boundary" : "", lossMask ? (p.loss ? "target" : "masked") : ""].join(" ");
    const style = p.special || lossMask ? "" : `background:hsl(${HUES[i % HUES.length]} 70% 92%);border-color:hsl(${HUES[i % HUES.length]} 50% 80%)`;
    const text = p.piece.replace(/ /g, "·").replace(/\n/g, "↵\n").replace(/\t/g, "→");
    return html`<span class=${cls} style=${style} title=${`id ${p.id}` + (p.start != null ? ` · chars ${p.start}-${p.end}` : "")} onMouseEnter=${onHover ? () => onHover(p, i) : null}>${text}${showIds ? html`<sub>${p.id}</sub>` : ""}</span>`;
  })}</div>`;
}

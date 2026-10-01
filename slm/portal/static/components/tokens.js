import { h } from "preact";
import { useEffect, useState } from "preact/hooks";
import htm from "htm";
import { api } from "./util.js";
import { splitSpecials } from "./specials.js";

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

// ---------------------------------------------------------------------------------- text view
// Every page that shows tokenized content opens in this text view; the tokens view (TokenChips) is a toggle.
// Reserved tokens (<|bos|>, <|user|>, <|think|>, <|python_call|>, ...) are never plain text here: they render
// as the same chip the tokens view uses, inline in the running text.

/** One reserved token as the tokens view draws it. */
export function SpecialChip({ name, cls = "", title, onMouseEnter }) {
  return html`<span class=${"chip special " + cls} title=${title || "reserved token"} onMouseEnter=${onMouseEnter}>${name}</span>`;
}

let SPECIALS = null;
let pending = null;
/** The registered special-token strings of the local tokenizers (from /api/tokenizers), loaded once per page. */
export function useSpecials() {
  const [s, setS] = useState(SPECIALS);
  useEffect(() => {
    if (SPECIALS) return;
    let alive = true;
    pending = pending || api("/api/tokenizers").then((ts) => (SPECIALS = [...new Set(ts.flatMap((t) => t.specials || []))])).catch(() => { pending = null; return []; });
    pending.then((x) => alive && setS(x));
    return () => { alive = false; };
  }, []);
  return s || [];
}

/** Decoded text with the reserved tokens as chips.
 *  runs: the API's exact runs ([{text} | {special, i}], each with `loss` where there is a mask), preferred when the
 *  ids are known; text: a plain string, split on the registered specials only (never a generic "<|...|>" match).
 *  lossMask colours text runs green (target) / grey (masked) like the tokens view; boundaries (token indices) put the
 *  red document-boundary outline on the specials there. */
export function TextWithSpecials({ text, runs, lossMask = false, boundaries, as = "pre", cls = "", style, title }) {
  const specials = useSpecials();
  const rs = runs || splitSpecials(text, specials);
  const b = boundaries ? new Set(boundaries) : null;
  const Tag = as;
  return html`<${Tag} class=${"spx " + cls} style=${style} title=${title}>${rs.map((r, k) => {
    const lc = lossMask ? (r.loss ? "target" : "masked") : "";
    if (r.special != null) return html`<${SpecialChip} key=${k} name=${r.special} cls=${lc + (b && b.has(r.i) ? " boundary" : "")} title=${r.i != null ? `token ${r.i}` : null} />`;
    return lc ? html`<span key=${k} class=${"t-" + lc}>${r.text}</span>` : r.text;
  })}<//>`;
}

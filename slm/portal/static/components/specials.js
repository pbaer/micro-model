// Pure helpers for the reserved-token text view (no imports, so node can load this module in the tests).

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const cache = new Map();

/** Split decoded text into runs at the registered special tokens:
 *  [{text: "..."} | {special: "<|bos|>"}, ...], the same shape the API's `runs` field has.
 *  Only strings in `specials` (the tokenizer's reserved block) split; any other "<|...|>" stays text. */
export function splitSpecials(text, specials) {
  const s = text == null ? "" : String(text);
  if (!s) return [];
  const list = [...new Set((specials || []).filter((x) => typeof x === "string" && x.length > 0))];
  if (!list.length || !list.some((x) => s.includes(x))) return [{ text: s }];
  const key = list.join("\u0000");
  let re = cache.get(key);
  if (!re) {
    re = new RegExp(list.sort((a, b) => b.length - a.length).map(escapeRe).join("|"), "g");  // longest first
    if (cache.size > 16) cache.clear();
    cache.set(key, re);
  }
  re.lastIndex = 0;
  const out = [];
  let last = 0, m;
  while ((m = re.exec(s))) {
    if (m.index > last) out.push({ text: s.slice(last, m.index) });
    out.push({ special: m[0] });
    last = m.index + m[0].length;
  }
  if (last < s.length) out.push({ text: s.slice(last) });
  return out;
}

/** The text back out of runs (inverse of splitSpecials). */
export const joinRuns = (runs) => (runs || []).map((r) => (r.special != null ? r.special : r.text)).join("");

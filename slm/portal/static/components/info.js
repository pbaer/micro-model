import { h } from "preact";
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import htm from "htm";
import { CARDS } from "./cards.js";

const html = htm.bind(h);
let nextId = 0;
let closeOther = null;  // one card open at a time

/** A small "?" that explains the concept behind the element it sits next to. `k` names a card in cards.js
 *  (`<${Info} k="val_loss" />`); a one-off card can pass `title` and children instead.
 *  Hover opens it, leaving closes it; a click or tap pins it until the next click, Escape or a click elsewhere;
 *  keyboard focus opens it too. The card is position:fixed and clamped to the viewport, so nothing around the
 *  icon moves, and it renders only while open. */
export function Info({ k, title, children }) {
  const card = k ? CARDS[k] : { t: title, b: children };
  const [open, setOpen] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [id] = useState(() => "info-" + ++nextId);
  const icon = useRef(null);
  const box = useRef(null);
  const timer = useRef(null);
  const later = (f, ms) => { clearTimeout(timer.current); timer.current = setTimeout(f, ms); };
  const show = () => { clearTimeout(timer.current); if (closeOther && closeOther.id !== id) closeOther.close(); closeOther = { id, close: hide }; setOpen(true); };
  const hide = () => { clearTimeout(timer.current); setOpen(false); setPinned(false); };

  useLayoutEffect(() => {  // place the card below the icon (above when there is no room), inside the viewport
    const c = box.current;
    if (!open || !c || !icon.current) return;
    const a = icon.current.getBoundingClientRect(), m = 8;
    const vw = document.documentElement.clientWidth, vh = window.innerHeight;
    const w = c.offsetWidth, ht = c.offsetHeight;
    let top = a.bottom + 6;
    if (top + ht > vh - m) top = a.top - 6 - ht >= m ? a.top - 6 - ht : Math.max(m, vh - m - ht);
    c.style.left = Math.max(m, Math.min(a.left - 14, vw - w - m)) + "px";
    c.style.top = top + "px";
    c.style.visibility = "visible";
  }, [open]);
  useEffect(() => {  // a scroll or resize elsewhere would leave the fixed card floating away from its icon
    if (!open) return;
    const onScroll = (e) => { if (!(box.current && box.current.contains(e.target))) hide(); };
    const onKey = (e) => { if (e.key === "Escape") hide(); };
    addEventListener("scroll", onScroll, true);
    addEventListener("resize", hide);
    addEventListener("keydown", onKey);
    return () => { removeEventListener("scroll", onScroll, true); removeEventListener("resize", hide); removeEventListener("keydown", onKey); };
  }, [open]);
  useEffect(() => () => { clearTimeout(timer.current); if (closeOther && closeOther.id === id) closeOther = null; }, []);
  if (!card) { console.warn(`Info: no card "${k}"`); return null; }

  return html`<span class="info-wrap" onClick=${(e) => e.stopPropagation()}
      onMouseEnter=${() => (open ? clearTimeout(timer.current) : later(show, 120))} onMouseLeave=${() => !pinned && later(hide, 180)}>
    <span class=${"info" + (open ? " on" : "")} ref=${icon} tabindex="0" aria-label=${"About: " + card.t} aria-describedby=${open ? id : undefined}
      onClick=${(e) => { e.preventDefault(); if (pinned) hide(); else { show(); setPinned(true); } }}
      onFocus=${(e) => e.target.matches(":focus-visible") && show()}
      onBlur=${(e) => { if (!(box.current && box.current.contains(e.relatedTarget))) hide(); }}>
      <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" stroke-width="1.3" />
        <path d="M6.1 6.2a1.95 1.95 0 1 1 2.6 1.85c-.45.17-.7.5-.7.95v.5" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" />
        <circle cx="8" cy="11.6" r=".85" fill="currentColor" /></svg></span>
    ${open && html`<div class="info-card" id=${id} role="tooltip" ref=${box} tabindex="-1"
        onBlur=${(e) => { if (!(e.currentTarget.parentNode.contains(e.relatedTarget))) hide(); }}>
      <div class="info-title">${card.t}</div>${card.b}</div>`}
  </span>`;
}

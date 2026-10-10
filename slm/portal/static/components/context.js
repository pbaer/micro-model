/** How much of the model's context window a conversation or a generation uses: a slim bar and "42% of 4096".
 *  `used` = tokens in the model's cache (prompt + everything generated, tool results included), `total` = the loaded
 *  model's context window (its max_seq_len; an external model's max positions). Green below 75%, amber to 90%,
 *  red beyond: near the end the worker shortens the reply to what still fits, and the turn may stop early. */
import { h } from "preact";
import htm from "htm";

const html = htm.bind(h);

export function contextLevel(used, total) {
  const f = total > 0 ? used / total : 0;
  return f >= 0.9 ? "full" : f >= 0.75 ? "high" : "ok";
}

export function ContextMeter({ used, total, live = false, note = "", compact = false }) {
  if (!total) return null;
  const u = Math.max(0, Math.min(used || 0, total));
  const pct = Math.round((u / total) * 100);
  const level = contextLevel(u, total);
  const left = total - u;
  const tip = `${u.toLocaleString()} of ${total.toLocaleString()} context tokens used (${pct}%) · ${left.toLocaleString()} left${note ? ` · ${note}` : ""}`;
  return html`<div class=${"ctx-meter " + level + (compact ? " compact" : "") + (live ? " live" : "")} title=${tip} role="meter"
      aria-label="context window used" aria-valuemin="0" aria-valuemax=${total} aria-valuenow=${u}>
    <div class="ctx-bar"><div class="ctx-fill" style=${`width:${Math.max(u > 0 ? 1 : 0, (u / total) * 100)}%`}></div></div>
    <span class="ctx-label"><b>${pct}%</b> of ${total.toLocaleString()}${compact ? "" : html` <span class="ctx-sub">· ${u.toLocaleString()} tokens${note ? ` · ${note}` : ""}</span>`}</span>
  </div>`;
}

import { h } from "preact";
import { useEffect, useState } from "preact/hooks";
import htm from "htm";
import { api, fmtTok, fmtDur, fmtNum } from "../components/util.js";

const html = htm.bind(h);

export function Home() {
  const [runs, setRuns] = useState(null);
  useEffect(() => {
    let alive = true;
    const tick = () => api("/api/runs").then((r) => alive && setRuns(r)).catch(() => {});
    tick();
    const id = setInterval(tick, 10000);
    return () => { alive = false; clearInterval(id); };
  }, []);
  const live = (runs || []).filter((r) => r.status === "running");
  const recent = (runs || []).slice(0, 6);
  return html`<div>
    <h1>Home</h1>
    <div class="sub">${runs ? `${runs.length} runs · ${live.length} live` : "loading…"}</div>
    <h2>Live runs</h2>
    ${live.length === 0 ? html`<div class="empty-note">No run is currently training.</div>` : html`<div class="cards">${live.map((r) => html`
      <a class="card" href=${"#/runs/" + encodeURIComponent(r.run_name)}>
        <div><b>${r.run_name}</b> <span class=${"status " + r.status}>${r.status}</span></div>
        <div class="bar"><div style=${"width:" + (r.progress * 100).toFixed(1) + "%"}></div></div>
        <div class="muted">${fmtTok(r.tokens)} / ${fmtTok(r.total_tokens)} · loss ${fmtNum(r.loss, 3)} · val ${fmtNum(r.val_loss, 3)} · ${Math.round(r.tok_s).toLocaleString()} tok/s · ETA ${fmtDur(r.eta_s)}</div>
      </a>`)}</div>`}
    <h2>Recent runs</h2>
    ${recent.length === 0 ? html`<div class="empty-note">No runs under the runs root yet.</div>` : html`<div class="cards">${recent.map((r) => html`
      <a class="card" href=${"#/runs/" + encodeURIComponent(r.run_name)}>
        <div><b>${r.run_name}</b> <span class=${"status " + r.status}>${r.status}</span></div>
        <div class="muted">${fmtTok(r.tokens)} tokens · best val ${fmtNum(r.best_val, 3)} · ${fmtDur(r.elapsed_s)} elapsed</div>
      </a>`)}</div>`}
  </div>`;
}

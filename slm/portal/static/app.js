import { h, render, Component } from "preact";
import { useEffect, useState } from "preact/hooks";
import htm from "htm";
import { api, fmtTok, fmtDur, fmtNum } from "./components/util.js";
import { Home } from "./pages/home.js";
import { RunDetail } from "./pages/runs.js";
import { DataPage } from "./pages/data.js";
import { TokenizerPage } from "./pages/tokenizer.js";
import { ArchPage } from "./pages/arch.js";
import { ModelPage } from "./pages/model.js";

const html = htm.bind(h);

function useHash() {
  const [hash, setHash] = useState(location.hash || "#/");
  useEffect(() => {
    const f = () => setHash(location.hash || "#/");
    addEventListener("hashchange", f);
    return () => removeEventListener("hashchange", f);
  }, []);
  return hash;
}

function GpuTile() {
  const [g, setG] = useState(null);
  useEffect(() => {
    let alive = true;
    const tick = () => api("/api/system/gpu").then((x) => alive && setG(x)).catch(() => {});
    tick();
    const id = setInterval(tick, 5000);
    return () => { alive = false; clearInterval(id); };
  }, []);
  if (!g) return html`<div class="gpu">GPU: …</div>`;
  if (!g.available) return html`<div class="gpu">GPU: unavailable</div>`;
  return html`<div class="gpu"><b>${g.name.replace("NVIDIA GeForce ", "")}</b><br/>
    ${g.used_gib.toFixed(1)} / ${g.total_gib.toFixed(1)} GiB · ${g.util.toFixed(0)}%<br/>
    ${g.power_w != null ? g.power_w.toFixed(0) + " W · " : ""}${g.temp_c != null ? g.temp_c.toFixed(0) + " °C" : ""}<br/>
    ${g.training_live.length ? html`<span style="color:#60a5fa">training: ${g.training_live.join(", ")}</span>` : html`<span>no live training</span>`}</div>`;
}

/** Catches render/effect exceptions of the page below it and shows them instead of a blank main area. */
class ErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { err: null }; }
  static getDerivedStateFromError(err) { return { err }; }
  componentDidCatch(err) { console.error(err); }
  componentDidUpdate(prev) { if (prev.route !== this.props.route && this.state.err) this.setState({ err: null }); }  // navigating away clears it
  render(props, state) {
    if (state.err) return html`<div class="boot error"><b>This page crashed.</b><pre>${String((state.err && state.err.stack) || state.err)}</pre><button onClick=${() => this.setState({ err: null })}>Retry</button></div>`;
    return props.children;
  }
}

function Placeholder({ name }) {
  return html`<div><h1>${name}</h1><p class="muted">Not built yet. See docs/command_center_plan.md.</p></div>`;
}

function App() {
  const hash = useHash();
  const [meta, setMeta] = useState(null);
  useEffect(() => { api("/api/meta").then(setMeta).catch(() => {}); }, []);
  const parts = hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const page = parts[0] || "home";
  let body;
  if (page === "home") body = html`<${Home} />`;
  else if (page === "runs" && parts[1]) body = html`<${RunDetail} run=${decodeURIComponent(parts[1])} key=${parts[1]} />`;
  else if (page === "runs") body = html`<${Home} />`;
  else if (page === "data") body = html`<${DataPage} />`;
  else if (page === "tokenizer") body = html`<${TokenizerPage} />`;
  else if (page === "arch") body = html`<${ArchPage} />`;
  else if (page === "inference" || page === "model") body = html`<${ModelPage} />`;
  else body = html`<${Placeholder} name=${page} />`;
  const pages = meta ? meta.pages : [{ id: "home", label: "Home" }, { id: "runs", label: "Runs" }];
  return html`<div class="layout">
    <nav>
      <div class="brand">slm command center</div>
      ${pages.map((p) => html`<a href=${"#/" + (p.id === "home" ? "" : p.id)} class=${page === p.id || (p.id === "home" && page === "runs") || (p.id === "inference" && page === "model") ? "active" : ""}>${p.label}</a>`)}
      <${GpuTile} />
    </nav>
    <main><${ErrorBoundary} route=${hash}>${body}</${ErrorBoundary}></main>
  </div>`;
}

try {
  render(html`<${App} />`, document.getElementById("app"));
  window.__appMounted = true;
} catch (e) {
  (window.__bootError || alert)(e && e.stack || e);
  throw e;
}

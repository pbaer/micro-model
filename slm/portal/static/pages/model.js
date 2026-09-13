import { h } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import htm from "htm";
import { api, fmtTok, fmtInt, fmtNum } from "../components/util.js";

const html = htm.bind(h);

const lpColor = (lp) => {
  if (lp == null) return "#f3f4f6";
  const p = Math.exp(lp);
  const hue = 120 * Math.min(1, Math.max(0, p));
  return `hsl(${hue} 70% ${88 - 18 * (1 - p)}%)`;
};

function SlotCard({ slot, info, ckpts, onLoad, onUnload, busy }) {
  const [sel, setSel] = useState("");
  const [dev, setDev] = useState("auto");
  const [force, setForce] = useState(false);
  const loaded = info && info.checkpoint;
  return html`<div class="panel">
    <div class="row"><b>slot ${slot}</b>
      ${loaded ? html`<span class="muted">${info.name} (${info.run || ""}) · ${info.device}/${info.dtype} · ${fmtInt(info.params)} params · trained ${fmtTok(info.tokens)} · val ${fmtNum(info.val_loss, 3)} · tokenizer ${info.tokenizer}${info.tokenizer_matched ? "" : " ⚠ sha mismatch"}</span>
        <button onClick=${() => onUnload(slot)} disabled=${busy}>unload</button>` : html`<span class="muted">empty</span>`}
    </div>
    <div class="row" style="margin-top:6px">
      <select value=${sel} onChange=${(e) => setSel(e.target.value)} style="max-width:420px">
        <option value="">choose checkpoint…</option>
        ${ckpts.map((c) => html`<option value=${c.path}>${c.run} / ${c.name} · ${fmtTok(c.tokens)} tok${c.val_loss != null ? ` · val ${c.val_loss.toFixed(3)}` : ""}</option>`)}
      </select>
      <select value=${dev} onChange=${(e) => setDev(e.target.value)}><option value="auto">auto</option><option value="cuda">cuda</option><option value="cpu">cpu</option></select>
      <label class="muted"><input type="checkbox" checked=${force} onChange=${(e) => setForce(e.target.checked)} /> force cuda</label>
      <button onClick=${() => sel && onLoad(slot, sel, dev, force)} disabled=${!sel || busy}>load</button>
    </div>
    ${loaded && info.device_reason && html`<div class="legend" style="margin-top:4px">device: ${info.device_reason}</div>`}
  </div>`;
}

function Stream({ tokens, prompt, hover, setHover }) {
  return html`<div class="chips" style="min-height:60px">
    ${prompt && prompt.map((p, i) => html`<span class="chip" style="background:#e5e7eb;color:#374151" key=${"p" + i}>${p.replace(/ /g, "·").replace(/\n/g, "↵\n")}</span>`)}
    ${tokens.map((t, i) => html`<span class=${"chip" + (t.special ? " special" : "")} style=${t.special ? "" : `background:${lpColor(t.logprob)}`} title=${`logprob ${t.logprob.toFixed(3)} · p=${Math.exp(t.logprob).toFixed(3)} · rank ${t.rank}`} onMouseEnter=${() => setHover(t)} key=${i}>${t.piece.replace(/ /g, "·").replace(/\n/g, "↵\n")}</span>`)}
  </div>`;
}

export function ModelPage() {
  const [status, setStatus] = useState(null);
  const [ckpts, setCkpts] = useState([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [mode, setMode] = useState("completion");
  const [text, setText] = useState("Once upon a time, there was a little girl named Lily. She loved to");
  const [messages, setMessages] = useState([{ role: "user", content: "What is 17 + 26?" }]);
  const [sampling, setSampling] = useState({ temperature: 0.8, top_p: 0.95, top_k: 0, max_new_tokens: 120, seed: 1234, logprobs_topk: 5 });
  const [useBoth, setUseBoth] = useState(false);
  const [out, setOut] = useState({ A: { prompt: null, tokens: [], done: null }, B: { prompt: null, tokens: [], done: null } });
  const [hover, setHover] = useState(null);
  const [streamId, setStreamId] = useState(null);
  const [score, setScore] = useState(null);
  const abortRef = useRef(null);

  const refresh = () => { api("/api/model/status").then(setStatus).catch(() => {}); api("/api/model/checkpoints").then(setCkpts).catch(() => {}); };
  useEffect(() => { refresh(); const id = setInterval(() => api("/api/model/status").then(setStatus).catch(() => {}), 10000); return () => clearInterval(id); }, []);

  const load = async (slot, path, device, force) => {
    setBusy(true); setErr(null);
    try { await api(`/api/model/slots/${slot}/load`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ checkpoint: path, device, force_cuda: force }) }); } catch (e) { setErr(String(e)); }
    setBusy(false); refresh();
  };
  const unload = async (slot) => { setBusy(true); await api(`/api/model/slots/${slot}/unload`, { method: "POST" }); setBusy(false); refresh(); };

  const generate = async () => {
    setErr(null); setScore(null);
    const slots = useBoth ? ["A", "B"] : ["A"];
    setOut({ A: { prompt: null, tokens: [], done: null }, B: { prompt: null, tokens: [], done: null } });
    setBusy(true);
    const ctrl = new AbortController(); abortRef.current = ctrl;
    try {
      const body = { slots, mode, text, messages, ...sampling, think_required: false };
      const r = await fetch("/api/model/generate", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal: ctrl.signal });
      const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = "";
      while (true) {
        const { value, done } = await reader.read(); if (done) break;
        buf += dec.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const chunk = buf.slice(0, idx); buf = buf.slice(idx + 2);
          const ev = /event: (\w+)/.exec(chunk)?.[1]; const dataLine = chunk.split("\n").find((l) => l.startsWith("data:"));
          if (!dataLine) continue; const data = JSON.parse(dataLine.slice(5));
          if (ev === "start") setStreamId(data.stream_id);
          else if (ev === "prompt") setOut((o) => ({ ...o, [data.slot]: { ...o[data.slot], prompt: data.pieces } }));
          else if (ev === "token") setOut((o) => ({ ...o, [data.slot]: { ...o[data.slot], tokens: [...o[data.slot].tokens, data] } }));
          else if (ev === "done") setOut((o) => ({ ...o, [data.slot]: { ...o[data.slot], done: data } }));
          else if (ev === "error") setErr(data.error);
        }
      }
    } catch (e) { if (e.name !== "AbortError") setErr(String(e)); }
    setBusy(false); setStreamId(null);
  };
  const cancel = () => { if (streamId) api(`/api/model/streams/${streamId}/cancel`, { method: "POST" }).catch(() => {}); if (abortRef.current) abortRef.current.abort(); };
  const doScore = async () => {
    setErr(null); setBusy(true);
    try { setScore(await api("/api/model/score", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ slot: "A", mode, text, messages }) })); } catch (e) { setErr(String(e)); }
    setBusy(false);
  };

  const slots = status ? status.slots : { A: {}, B: {} };
  const stat = (s) => { const o = out[s]; const t = o.tokens; if (!t.length) return ""; const mean = t.reduce((a, b) => a + b.logprob, 0) / t.length; return `${t.length} tokens · mean logprob ${mean.toFixed(3)} · mean entropy ${(t.reduce((a, b) => a + b.entropy, 0) / t.length).toFixed(2)}${o.done ? ` · ${o.done.tok_s.toFixed(1)} tok/s · ${o.done.reason}` : ""}`; };
  const divergence = useBoth && out.A.tokens.length && out.B.tokens.length ? out.A.tokens.findIndex((t, i) => !out.B.tokens[i] || out.B.tokens[i].id !== t.id) : -1;
  return html`<div>
    <h1>Model harness</h1>
    <div class="sub">${status ? (status.worker ? `worker alive · VRAM in worker ${fmtNum(status.vram_gib, 2)} GiB` : "worker idle (no VRAM held)") : "…"} ${status && status.live_runs && status.live_runs.length ? ` · ⚠ training live: ${status.live_runs.join(", ")} — loads default to CPU` : ""}
      ${status && status.worker && html` · <a href="#" onClick=${(e) => { e.preventDefault(); api("/api/model/worker/stop", { method: "POST" }).then(refresh); }}>release GPU (stop worker)</a>`}</div>
    ${err && html`<div class="panel" style="border-color:#fca5a5;color:#b91c1c">${err}</div>`}
    <div class="two">
      <${SlotCard} slot="A" info=${slots.A} ckpts=${ckpts} onLoad=${load} onUnload=${unload} busy=${busy} />
      <${SlotCard} slot="B" info=${slots.B} ckpts=${ckpts} onLoad=${load} onUnload=${unload} busy=${busy} />
    </div>
    <h2>Prompt</h2>
    <div class="row" style="margin-bottom:6px">
      ${["completion", "chat"].map((m) => html`<button class=${mode === m ? "active" : ""} onClick=${() => setMode(m)}>${m}</button>`)}
      <span class="muted">temp</span><input type="number" step="0.1" value=${sampling.temperature} onChange=${(e) => setSampling({ ...sampling, temperature: Number(e.target.value) })} style="width:60px" />
      <span class="muted">top-p</span><input type="number" step="0.05" value=${sampling.top_p} onChange=${(e) => setSampling({ ...sampling, top_p: Number(e.target.value) })} style="width:60px" />
      <span class="muted">top-k</span><input type="number" value=${sampling.top_k} onChange=${(e) => setSampling({ ...sampling, top_k: Number(e.target.value) })} style="width:60px" />
      <span class="muted">max new</span><input type="number" value=${sampling.max_new_tokens} onChange=${(e) => setSampling({ ...sampling, max_new_tokens: Number(e.target.value) })} style="width:70px" />
      <span class="muted">seed</span><input type="number" value=${sampling.seed} onChange=${(e) => setSampling({ ...sampling, seed: Number(e.target.value) })} style="width:80px" />
      <button onClick=${() => setSampling({ ...sampling, temperature: 0 })}>greedy</button>
      <label class="muted"><input type="checkbox" checked=${useBoth} onChange=${(e) => setUseBoth(e.target.checked)} /> A and B side by side</label>
    </div>
    ${mode === "completion" ? html`<textarea value=${text} onInput=${(e) => setText(e.target.value)}></textarea>` : html`<div class="panel">
      ${messages.map((m, i) => html`<div class="row" style="margin-bottom:6px;align-items:flex-start">
        <select value=${m.role} onChange=${(e) => setMessages(messages.map((x, j) => (j === i ? { ...x, role: e.target.value } : x)))}><option>system</option><option>user</option><option>assistant</option></select>
        <textarea style="min-height:40px;flex:1" value=${m.content} onInput=${(e) => setMessages(messages.map((x, j) => (j === i ? { ...x, content: e.target.value } : x)))}></textarea>
        <button onClick=${() => setMessages(messages.filter((_, j) => j !== i))}>✕</button></div>`)}
      <button onClick=${() => setMessages([...messages, { role: "user", content: "" }])}>+ message</button>
      <div class="legend" style="margin-top:4px">A base checkpoint has never seen the chat tokens; expect noise until SFT.</div></div>`}
    <div class="row" style="margin:8px 0">
      <button class="active" onClick=${generate} disabled=${busy || !slots.A || !slots.A.checkpoint}>generate</button>
      <button onClick=${cancel} disabled=${!streamId}>cancel</button>
      <button onClick=${doScore} disabled=${busy || !slots.A || !slots.A.checkpoint}>score prompt (teacher-forced)</button>
      ${hover && html`<span class="muted" style="font-family:var(--mono)">top-${hover.topk.length}: ${hover.topk.map((t) => `${JSON.stringify(t.piece)} ${Math.exp(t.logprob).toFixed(2)}`).join("  ")}</span>`}
    </div>
    <div class=${useBoth ? "two" : ""}>
      ${(useBoth ? ["A", "B"] : ["A"]).map((s) => html`<div key=${s}>
        <div class="muted" style="margin-bottom:4px"><b>${s}</b> ${stat(s)}${useBoth && divergence >= 0 && s === "A" ? ` · diverges at token ${divergence + 1}` : ""}</div>
        <${Stream} tokens=${out[s].tokens} prompt=${out[s].prompt} hover=${hover} setHover=${setHover} />
      </div>`)}
    </div>
    <div class="legend" style="margin-top:6px">chip color = probability the model assigned to the token it emitted (red = surprised, green = confident); hover a chip for the top-k alternatives at that step.</div>
    ${score && html`<h2>Teacher-forced scoring (slot A)</h2>
      <div class="muted">${score.n} tokens · mean logprob ${score.mean_logprob.toFixed(3)} · perplexity ${score.ppl ? score.ppl.toFixed(2) : "-"} (over loss-target tokens)</div>
      <div class="chips">${score.tokens.map((t, i) => html`<span class=${"chip" + (t.target ? "" : " masked")} style=${t.target ? `background:${lpColor(t.logprob)}` : ""} title=${t.logprob == null ? "first token" : `logprob ${t.logprob.toFixed(3)} · rank ${t.rank}`} key=${i}>${t.piece.replace(/ /g, "·").replace(/\n/g, "↵\n")}</span>`)}</div>`}
  </div>`;
}

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
      ${loaded ? html`<span class="muted"><span class=${"stage-badge " + (info.stage || "base")}>${info.stage || "base"}</span> ${info.name} (${info.run || ""}) · ${info.device}/${info.dtype} · ${fmtInt(info.params)} params · trained ${fmtTok(info.tokens)} · val ${fmtNum(info.val_loss, 3)} · tokenizer ${info.tokenizer}${info.tokenizer_matched ? "" : " ⚠ sha mismatch"}</span>
        <button onClick=${() => onUnload(slot)} disabled=${busy}>unload</button>` : html`<span class="muted">empty</span>`}
    </div>
    <div class="row" style="margin-top:6px">
      <select value=${sel} onChange=${(e) => setSel(e.target.value)} style="max-width:min(420px,100%);min-width:0">
        <option value="">choose checkpoint…</option>
        ${ckpts.map((c) => html`<option value=${c.path}>[${c.stage || "base"}] ${c.run} / ${c.name} · ${fmtTok(c.tokens)} tok${c.val_loss != null ? ` · val ${c.val_loss.toFixed(3)}` : ""}${c.heldout_acc != null ? ` · held-out ${(c.heldout_acc * 100).toFixed(0)}%` : ""}</option>`)}
      </select>
      <select value=${dev} onChange=${(e) => setDev(e.target.value)}><option value="auto">auto</option><option value="cuda">cuda</option><option value="cpu">cpu</option></select>
      <label class="muted"><input type="checkbox" checked=${force} onChange=${(e) => setForce(e.target.checked)} /> force cuda</label>
      <button onClick=${() => sel && onLoad(slot, sel, dev, force)} disabled=${!sel || busy}>load</button>
    </div>
    ${loaded && info.device_reason && html`<div class="legend" style="margin-top:4px">device: ${info.device_reason}</div>`}
  </div>`;
}

const showPiece = (p) => p.replace(/ /g, "·").replace(/\n/g, "↵\n");
const isSpecial = (p) => p.startsWith("<|") && p.endsWith("|>");

/** Generated output. mode "tokens": one chip per token colored by its probability; mode "text": the
 *  raw decoded text, with reserved tokens (<|bos|>, <|end|>, ...) still shown as highlighted markers. */
const tip = (t) => (t.inserted ? "inserted by the Python tool (no log-prob)" : `logprob ${t.logprob.toFixed(3)} · p=${Math.exp(t.logprob).toFixed(3)} · rank ${t.rank}`);

function Stream({ tokens, prompt, mode, setHover }) {
  if (mode === "text") {
    return html`<div class="rawout">
      ${prompt && prompt.map((p, i) => isSpecial(p) ? html`<span class="chip special" key=${"p" + i}>${p}</span>` : html`<span class="prompt-text" key=${"p" + i}>${p}</span>`)}
      ${tokens.map((t, i) => t.special ? html`<span class=${"chip special" + (t.inserted ? " inserted" : "")} title=${tip(t)} onMouseEnter=${() => !t.inserted && setHover(t)} key=${i}>${t.piece}</span>`
        : html`<span class=${t.inserted ? "inserted-text" : ""} title=${tip(t)} onMouseEnter=${() => !t.inserted && setHover(t)} key=${i}>${t.piece}</span>`)}
    </div>`;
  }
  return html`<div class="chips" style="min-height:60px">
    ${prompt && prompt.map((p, i) => html`<span class="chip" style="background:#e5e7eb;color:#374151" key=${"p" + i}>${showPiece(p)}</span>`)}
    ${tokens.map((t, i) => html`<span class=${"chip" + (t.special ? " special" : "") + (t.inserted ? " inserted" : "")} style=${t.special || t.inserted ? "" : `background:${lpColor(t.logprob)}`} title=${tip(t)} onMouseEnter=${() => !t.inserted && setHover(t)} key=${i}>${showPiece(t.piece)}</span>`)}
  </div>`;
}

const newSessionId = () => Math.random().toString(16).slice(2, 12);

export function ModelPage() {
  const [status, setStatus] = useState(null);
  const [ckpts, setCkpts] = useState([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [mode, setMode] = useState("completion");
  const [text, setText] = useState("Once upon a time, there was a little girl named Lily. She loved to");
  const [messages, setMessages] = useState([{ role: "user", content: "What is 17 + 26?" }]);
  const [sampling, setSampling] = useState({ temperature: 0.8, top_p: 0.95, top_k: 0, max_new_tokens: 300, seed: 1234, logprobs_topk: 5 });
  const [useBoth, setUseBoth] = useState(false);
  const [thinkReq, setThinkReq] = useState(true);
  const [tools, setTools] = useState(true);
  const [sessionId, setSessionId] = useState(newSessionId);
  const [calls, setCalls] = useState([]);
  const [view, setView] = useState("text");
  const [out, setOut] = useState({ A: { prompt: null, tokens: [], done: null }, B: { prompt: null, tokens: [], done: null } });
  const [hover, setHover] = useState(null);
  const [streamId, setStreamId] = useState(null);
  const [score, setScore] = useState(null);
  const abortRef = useRef(null);

  const refresh = () => { api("/api/model/status").then(setStatus).catch(() => {}); api("/api/model/checkpoints").then(setCkpts).catch(() => {}); };
  // the think-span switch follows the loaded checkpoint: reasoning / RL models were trained to open every answer with <|think|>, instruct models were not
  const stageA = status && status.slots && status.slots.A ? status.slots.A.stage : undefined;
  useEffect(() => { if (stageA) setThinkReq(stageA === "reasoning" || stageA === "rl"); }, [stageA]);
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
    setCalls([]);
    setBusy(true);
    const ctrl = new AbortController(); abortRef.current = ctrl;
    try {
      const body = { slots, mode, text, messages, ...sampling, think_required: mode === "chat" && thinkReq, tools: mode === "chat" && tools, session_id: sessionId, max_tool_calls: 8 };
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
          else if (ev === "tool") setCalls((c) => [...c, data]);
          else if (ev === "done") {
            setOut((o) => ({ ...o, [data.slot]: { ...o[data.slot], done: data } }));
            // a well-formed assistant turn (slot A, chat mode) becomes part of the conversation, followed by an empty user turn
            if (data.slot === "A" && mode === "chat" && data.assistant && data.assistant.well_formed) {
              setMessages((ms) => [...ms.filter((m, i) => !(i === ms.length - 1 && m.role === "user" && !m.content.trim() && false)),
                { role: "assistant", think: data.assistant.think, content: data.assistant.answer, ids: data.assistant.ids, n_calls: data.assistant.n_calls }, { role: "user", content: "" }]);
            }
          }
          else if (ev === "error") setErr(data.error);
        }
      }
    } catch (e) { if (e.name !== "AbortError") setErr(String(e)); }
    setBusy(false); setStreamId(null);
  };
  const cancel = () => { if (streamId) api(`/api/model/streams/${streamId}/cancel`, { method: "POST" }).catch(() => {}); if (abortRef.current) abortRef.current.abort(); };
  const newConversation = () => {
    api(`/api/model/sessions/${sessionId}/reset`, { method: "POST" }).catch(() => {});
    setSessionId(newSessionId()); setMessages([{ role: "user", content: "" }]); setCalls([]);
    setOut({ A: { prompt: null, tokens: [], done: null }, B: { prompt: null, tokens: [], done: null } });
  };
  const editMessage = (i, patch) => setMessages(messages.map((x, j) => (j === i ? { ...x, ...patch, ids: undefined } : x)));  // editing a generated turn re-encodes it
  const doScore = async () => {
    setErr(null); setBusy(true);
    try { setScore(await api("/api/model/score", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ slot: "A", mode, text, messages }) })); } catch (e) { setErr(String(e)); }
    setBusy(false);
  };

  const slots = status ? status.slots : { A: {}, B: {} };
  const stat = (s) => { const o = out[s]; const t = o.tokens; if (!t.length) return ""; const mean = t.reduce((a, b) => a + b.logprob, 0) / t.length; return `${t.length} tokens · mean logprob ${mean.toFixed(3)} · mean entropy ${(t.reduce((a, b) => a + b.entropy, 0) / t.length).toFixed(2)}${o.done ? ` · ${o.done.tok_s.toFixed(1)} tok/s · ${o.done.reason}` : ""}`; };
  const divergence = useBoth && out.A.tokens.length && out.B.tokens.length ? out.A.tokens.findIndex((t, i) => !out.B.tokens[i] || out.B.tokens[i].id !== t.id) : -1;
  return html`<div>
    <h1>Inference</h1>
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
      ${mode === "chat" && html`<label class="muted"><input type="checkbox" checked=${thinkReq} onChange=${(e) => setThinkReq(e.target.checked)} /> force ${"<|think|>"} (reasoning models)</label>
        <label class="muted"><input type="checkbox" checked=${tools} onChange=${(e) => setTools(e.target.checked)} /> python tool (REPL session ${sessionId})</label>
        <button onClick=${newConversation}>new conversation</button>`}
    </div>
    ${mode === "chat" && slots.A && slots.A.checkpoint && (slots.A.stage || "base") === "base" && html`<div class="panel warn"><b>Slot A holds a base checkpoint (${slots.A.run || slots.A.name}).</b> A base model has never seen the chat tokens: after ${"<|assistant|>"} it just continues web text, so chat output will be garbage. Use <b>completion</b> mode for it, or load an instruct / reasoning / RL checkpoint (m4_sft_149m, m5_reasoning_149m, m6_rl_gsm_tools_149m best.pt) for chat.</div>`}
    ${mode === "chat" && slots.A && slots.A.checkpoint && slots.A.stage === "sft" && html`<div class="legend">Instruct checkpoint: leave "force ${"<|think|>"}" off (it was not trained with think spans). Answers can run long; the reply is added to the conversation only if it closes with ${"<|end|>"} within max-new tokens.</div>`}
    ${mode === "chat" && slots.A && slots.A.checkpoint && (slots.A.stage === "reasoning" || slots.A.stage === "rl") && !thinkReq && html`<div class="legend">Reasoning / RL checkpoint: turn on "force ${"<|think|>"}" — it was trained to start every answer with a think span (and, for tool models, to call Python inside it).</div>`}
    ${mode === "completion" ? html`<textarea value=${text} onInput=${(e) => setText(e.target.value)}></textarea>` : html`<div class="panel">
      ${messages.map((m, i) => html`<div class="row" style="margin-bottom:6px;align-items:flex-start">
        <select value=${m.role} onChange=${(e) => editMessage(i, { role: e.target.value })}><option>system</option><option>user</option><option>assistant</option></select>
        <div style="flex:1;min-width:0">
          ${m.role === "assistant" && m.think != null && html`<pre class="think" title="think span (tool calls shown as <<code=result>> markup)">${m.think}</pre>`}
          <textarea style="min-height:40px;width:100%" placeholder=${m.role === "user" ? "type the next user message and press generate" : ""} value=${m.content} onInput=${(e) => editMessage(i, { content: e.target.value })}></textarea>
          ${m.ids && html`<div class="legend">generated turn: ${m.ids.length} tokens kept verbatim${m.n_calls ? ` · ${m.n_calls} python call${m.n_calls > 1 ? "s" : ""}` : ""} (editing re-encodes it)</div>`}
        </div>
        <button onClick=${() => setMessages(messages.filter((_, j) => j !== i))}>✕</button></div>`)}
      <button onClick=${() => setMessages([...messages, { role: "user", content: "" }])}>+ message</button>
      <div class="legend" style="margin-top:4px">Multi-turn: a well-formed assistant reply is appended here automatically with an empty user turn after it. Python calls inside the think span run in this conversation's session, so variables persist across turns. A base checkpoint has never seen the chat tokens; expect noise until SFT.</div></div>`}
    <div class="row" style="margin:8px 0">
      <button class="active" onClick=${generate} disabled=${busy || !slots.A || !slots.A.checkpoint}>generate</button>
      <button onClick=${cancel} disabled=${!streamId}>cancel</button>
      <button onClick=${doScore} disabled=${busy || !slots.A || !slots.A.checkpoint}>score prompt (teacher-forced)</button>
      <span class="muted" style="margin-left:10px">view</span>
      ${["text", "tokens"].map((v) => html`<button class=${view === v ? "active" : ""} onClick=${() => setView(v)}>${v}</button>`)}
      ${hover && html`<span class="muted" style="font-family:var(--mono)">top-${hover.topk.length}: ${hover.topk.map((t) => `${JSON.stringify(t.piece)} ${Math.exp(t.logprob).toFixed(2)}`).join("  ")}</span>`}
    </div>
    <div class=${useBoth ? "two" : ""}>
      ${(useBoth ? ["A", "B"] : ["A"]).map((s) => html`<div key=${s}>
        <div class="muted" style="margin-bottom:4px"><b>${s}</b> ${stat(s)}${useBoth && divergence >= 0 && s === "A" ? ` · diverges at token ${divergence + 1}` : ""}</div>
        <${Stream} tokens=${out[s].tokens} prompt=${out[s].prompt} mode=${view} setHover=${setHover} />
      </div>`)}
    </div>
    ${calls.length > 0 && html`<div class="panel" style="margin-top:6px"><b>python calls</b>
      ${calls.map((c, i) => html`<div class="toolcall" key=${i}><pre class="code">${c.code}</pre><span class=${c.ok ? "result ok" : "result err"}>${c.result}</span></div>`)}</div>`}
    ${out.A.done && out.A.done.assistant && html`<div class="legend" style="margin-top:4px">assistant turn: ${out.A.done.assistant.well_formed ? "well-formed, added to the conversation" : `not well-formed (${out.A.done.reason}${out.A.done.assistant.malformed ? ", malformed" : ""}) — not added`}</div>`}
    <div class="legend" style="margin-top:6px">${view === "tokens" ? "chip color = probability the model assigned to the token it emitted (red = surprised, green = confident); hover a chip for the top-k alternatives at that step." : "raw decoded text; reserved tokens are shown as markers. Hover any word for its log-prob and the top-k alternatives."}</div>
    ${score && html`<h2>Teacher-forced scoring (slot A)</h2>
      <div class="muted">${score.n} tokens · mean logprob ${score.mean_logprob.toFixed(3)} · perplexity ${score.ppl ? score.ppl.toFixed(2) : "-"} (over loss-target tokens)</div>
      ${view === "tokens" ? html`<div class="chips">${score.tokens.map((t, i) => html`<span class=${"chip" + (t.target ? "" : " masked")} style=${t.target ? `background:${lpColor(t.logprob)}` : ""} title=${t.logprob == null ? "first token" : `logprob ${t.logprob.toFixed(3)} · rank ${t.rank}`} key=${i}>${showPiece(t.piece)}</span>`)}</div>`
        : html`<div class="rawout">${score.tokens.map((t, i) => isSpecial(t.piece) ? html`<span class="chip special" key=${i}>${t.piece}</span>` : html`<span class=${t.target ? "" : "prompt-text"} title=${t.logprob == null ? "" : `logprob ${t.logprob.toFixed(3)}`} key=${i}>${t.piece}</span>`)}</div>`}`}
  </div>`;
}

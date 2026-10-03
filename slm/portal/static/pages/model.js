import { h } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import htm from "htm";
import { api, fmtTok, fmtInt, fmtNum, readSSE } from "../components/util.js";
import { segmentLabels, SpecialChip, TextWithSpecials, useSpecials } from "../components/tokens.js";
import { Info } from "../components/info.js";
import { SwarmPanel } from "../components/swarm.js";

const html = htm.bind(h);

const lpColor = (lp) => {
  if (lp == null) return "#f3f4f6";
  const p = Math.exp(lp);
  const hue = 120 * Math.min(1, Math.max(0, p));
  return `hsl(${hue} 70% ${88 - 18 * (1 - p)}%)`;
};

const fmtParams = (n) => (n == null ? "?" : n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : `${Math.round(n / 1e6)}M`);
const ctxLabel = (n) => (n == null ? "?" : n >= 1024 ? `${Math.round(n / 1024)}K` : String(n));
/** One line for an external comparison model (a slot's info or a /checkpoints entry). */
const extLine = (m) => `${m.hf_id} · ${fmtParams(m.params)} params · ${m.license} · ${m.is_chat ? "chat" : "base"} · context ${ctxLabel(m.context)}`;

export function SlotCard({ slot, info, ckpts, onLoad, onUnload, busy }) {
  const [sel, setSel] = useState("");
  const [dev, setDev] = useState("auto");
  const [force, setForce] = useState(false);
  const loaded = info && info.checkpoint;
  const ours = ckpts.filter((c) => !c.external), ext = ckpts.filter((c) => c.external);
  const picked = ext.find((c) => c.path === sel);
  return html`<div class="panel">
    <div class="row"><b>slot ${slot}<${Info} k="slots" /></b>
      ${loaded && info.external ? html`<span class="muted"><span class="stage-badge external">external</span><${Info} k="external_slot" /> <b>${info.name}</b> · ${extLine(info)} · ${info.device}/${info.dtype} · own tokenizer and chat template</span>
        <button onClick=${() => onUnload(slot)} disabled=${busy}>unload</button>`
      : loaded ? html`<span class="muted"><span class=${"stage-badge " + (info.stage || "base")}>${info.stage || "base"}</span> ${info.name} (${info.run || ""}) · ${info.device}/${info.dtype} · ${fmtInt(info.params)} params · trained ${fmtTok(info.tokens)} · val ${fmtNum(info.val_loss, 3)} · tokenizer ${info.tokenizer}${info.tokenizer_matched ? "" : " ⚠ sha mismatch"}</span>
        <button onClick=${() => onUnload(slot)} disabled=${busy}>unload</button>` : html`<span class="muted">empty</span>`}
    </div>
    ${loaded && info.external && html`<div class="legend" style="margin-top:4px">${info.is_chat ? "completion, chat (its own template) and swarm" : "completion only: a base model gets no chat template"} · no Python tool, declared functions or forced think span · no teacher-forced scoring · swarm verification n/a</div>`}
    <div class="row" style="margin-top:6px">
      <select value=${sel} onChange=${(e) => setSel(e.target.value)} style="max-width:min(420px,100%);min-width:0">
        <option value="">choose checkpoint…</option>
        <optgroup label="our checkpoints">
        ${ours.map((c) => html`<option value=${c.path}>[${c.stage || "base"}] ${c.run} / ${c.name}${c.stage === "rl" && !["best", "final"].includes(c.kind) ? " (training snapshot; prefer best.pt)" : ""} · ${fmtTok(c.tokens)} tok${c.val_loss != null ? ` · val ${c.val_loss.toFixed(3)}` : ""}${c.heldout_acc != null ? ` · held-out ${(c.heldout_acc * 100).toFixed(0)}%` : ""}</option>`)}
        </optgroup>
        ${ext.length > 0 && html`<optgroup label="external models (open-weight, run locally)">
          ${ext.map((c) => html`<option value=${c.path} disabled=${!c.available}>[external] ${c.name} · ${fmtParams(c.params)} · ${c.is_chat ? "chat" : "base"} · ctx ${ctxLabel(c.context)} · ${c.license}${c.available ? "" : " · weights not on disk"}</option>`)}
        </optgroup>`}
      </select>
      <select value=${dev} onChange=${(e) => setDev(e.target.value)}><option value="auto">auto</option><option value="cuda">cuda</option><option value="cpu">cpu</option></select>
      <label class="muted"><input type="checkbox" checked=${force} onChange=${(e) => setForce(e.target.checked)} /> force cuda</label>
      <button onClick=${() => sel && onLoad(slot, sel, dev, force)} disabled=${!sel || busy}>load</button>
    </div>
    ${picked && html`<div class="legend" style="margin-top:4px">${picked.name}: ${extLine(picked)} · ${picked.notes}${picked.available ? "" : ` · weights missing: python -m slm.eval.external download ${picked.name}`}</div>`}
    ${loaded && info.device_reason && html`<div class="legend" style="margin-top:4px">device: ${info.device_reason}</div>`}
  </div>`;
}

const showPiece = (p) => p.replace(/ /g, "·").replace(/\n/g, "↵\n");

/** Generated output. mode "tokens": one chip per token colored by its probability; mode "text": the
 *  raw decoded text, with reserved tokens (<|bos|>, <|end|>, ...) still shown as highlighted markers. */
const tip = (t) => (t.inserted ? "inserted by the Python tool (no log-prob)" : `logprob ${t.logprob.toFixed(3)} · p=${Math.exp(t.logprob).toFixed(3)} · rank ${t.rank}`);

function Stream({ tokens, prompt, promptSpecial, segments, mode, setHover }) {
  const isDef = segmentLabels(segments, prompt ? prompt.length : 0).map((l) => l === "python_def");  // declared-function blocks in the prompt
  // which prompt pieces are reserved tokens: the worker's per-id flags (exact, also for an external model's own specials);
  // without them, only the registered special strings of our tokenizer count, never a generic "<|...|>" match
  const specials = useSpecials();
  const isSp = (p, i) => (promptSpecial ? !!promptSpecial[i] : specials.includes(p));
  if (mode === "text") {
    return html`<div class="rawout spx">
      ${prompt && prompt.map((p, i) => isSp(p, i) ? html`<${SpecialChip} key=${"p" + i} name=${p} cls=${isDef[i] ? "def" : ""} title=${isDef[i] ? "declared function (masked)" : "reserved token (prompt)"} />`
        : isDef[i] ? html`<span class="chip def" title="declared function (masked)" key=${"p" + i}>${p}</span>`
        : html`<span class="prompt-text" key=${"p" + i}>${p}</span>`)}
      ${tokens.map((t, i) => t.special ? html`<${SpecialChip} key=${i} name=${t.piece} cls=${t.inserted ? "inserted" : ""} title=${tip(t)} onMouseEnter=${() => !t.inserted && setHover(t)} />`
        : html`<span class=${t.inserted ? "inserted-text" : ""} title=${tip(t)} onMouseEnter=${() => !t.inserted && setHover(t)} key=${i}>${t.piece}</span>`)}
    </div>`;
  }
  return html`<div class="chips" style="min-height:60px">
    ${prompt && prompt.map((p, i) => html`<span class=${"chip" + (isSp(p, i) ? " special" : "") + (isDef[i] ? " def" : "")} style=${isDef[i] || isSp(p, i) ? "" : "background:#e5e7eb;color:#374151"} title=${isDef[i] ? "declared function (masked)" : ""} key=${"p" + i}>${isSp(p, i) ? p : showPiece(p)}</span>`)}
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
  const [funcs, setFuncs] = useState("");  // declared functions, JSON: [{name, signature, comment}]
  const [calls, setCalls] = useState([]);
  const [view, setView] = useState("text");
  const [out, setOut] = useState({ A: { prompt: null, tokens: [], done: null }, B: { prompt: null, tokens: [], done: null } });
  const [hover, setHover] = useState(null);
  const [streamId, setStreamId] = useState(null);
  const [score, setScore] = useState(null);
  const abortRef = useRef(null);

  const slots = status ? status.slots : { A: {}, B: {} };
  const runSlots = useBoth ? ["A", "B"] : ["A"];
  const extInRun = runSlots.some((s) => slots[s] && slots[s].external);
  const refresh = () => { api("/api/model/status").then(setStatus).catch(() => {}); api("/api/model/checkpoints").then(setCkpts).catch(() => {}); };
  // the think-span switch follows the loaded checkpoint: reasoning / RL models were trained to open every answer with <|think|>, instruct models were not
  const stageA = status && status.slots && status.slots.A ? status.slots.A.stage : undefined;
  useEffect(() => {
    if (!stageA) return;
    const reasoning = stageA === "reasoning" || stageA === "rl";
    setThinkReq(reasoning);
    // small reasoning/RL policies are fragile under sampling (their RL rollouts at 0.8 were 30-40% malformed): default to greedy
    setSampling((sp) => ({ ...sp, temperature: reasoning ? 0 : 0.8 }));
  }, [stageA]);
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
      let decls = null;
      if (mode === "chat" && funcs.trim()) {
        try { decls = JSON.parse(funcs); } catch (e) { setErr(`declared functions: not valid JSON (${e.message})`); setBusy(false); return; }
      }
      // an external slot takes none of our protocol (the worker refuses it): with one in the run, all three are off
      const proto = mode === "chat" && !extInRun;
      const body = { slots, mode, text, messages, ...sampling, think_required: proto && thinkReq, tools: proto && tools, session_id: sessionId, max_tool_calls: 8, functions: proto ? decls : null };
      const r = await fetch("/api/model/generate", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal: ctrl.signal });
      await readSSE(r, (ev, data) => {
        if (ev === "start") setStreamId(data.stream_id);
        else if (ev === "prompt") setOut((o) => ({ ...o, [data.slot]: { ...o[data.slot], prompt: data.pieces, promptSpecial: data.special, segments: data.segments } }));
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
      });
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

  const stat = (s) => { const o = out[s]; const t = o.tokens; if (!t.length) return ""; const mean = t.reduce((a, b) => a + b.logprob, 0) / t.length; return `${t.length} tokens · mean logprob ${mean.toFixed(3)} · mean entropy ${(t.reduce((a, b) => a + b.entropy, 0) / t.length).toFixed(2)}${o.done ? ` · ${o.done.tok_s.toFixed(1)} tok/s · ${o.done.reason}` : ""}${slots[s] && slots[s].external ? ` · ${slots[s].name}'s own tokens` : ""}`; };
  const extNames = runSlots.filter((s) => slots[s] && slots[s].external).map((s) => `${s} (${slots[s].name})`);
  const extBase = runSlots.filter((s) => slots[s] && slots[s].external && !slots[s].is_chat);
  const chatBlocked = mode === "chat" && extBase.length > 0;  // a base external model has no chat template: n/a, never invented
  const divergence = useBoth && out.A.tokens.length && out.B.tokens.length ? out.A.tokens.findIndex((t, i) => !out.B.tokens[i] || out.B.tokens[i].id !== t.id) : -1;
  return html`<div>
    <h1>Inference</h1>
    <div class="sub">${status ? (status.worker ? `worker alive · VRAM in worker ${fmtNum(status.vram_gib, 2)} GiB` : "worker idle (no VRAM held)") : "…"}<${Info} k="worker" /> ${status && status.live_runs && status.live_runs.length ? ` · ⚠ training live: ${status.live_runs.join(", ")} — loads default to CPU` : ""}
      ${status && status.worker && html` · <a href="#" onClick=${(e) => { e.preventDefault(); api("/api/model/worker/stop", { method: "POST" }).then(refresh); }}>release GPU (stop worker)</a>`}</div>
    ${err && html`<div class="panel" style="border-color:#fca5a5;color:#b91c1c">${err}</div>`}
    <div class="two">
      <${SlotCard} slot="A" info=${slots.A} ckpts=${ckpts} onLoad=${load} onUnload=${unload} busy=${busy} />
      <${SlotCard} slot="B" info=${slots.B} ckpts=${ckpts} onLoad=${load} onUnload=${unload} busy=${busy} />
    </div>
    <h2>Prompt</h2>
    <div class="row" style="margin-bottom:6px">
      ${["completion", "chat", "swarm"].map((m) => html`<button class=${mode === m ? "active" : ""} onClick=${() => setMode(m)}>${m}</button>`)}
      ${mode !== "swarm" && html`
      <span class="muted">temp</span><input type="number" step="0.1" value=${sampling.temperature} onChange=${(e) => setSampling({ ...sampling, temperature: Number(e.target.value) })} style="width:60px" />
      <span class="muted">top-p</span><input type="number" step="0.05" value=${sampling.top_p} onChange=${(e) => setSampling({ ...sampling, top_p: Number(e.target.value) })} style="width:60px" />
      <span class="muted">top-k</span><input type="number" value=${sampling.top_k} onChange=${(e) => setSampling({ ...sampling, top_k: Number(e.target.value) })} style="width:60px" />
      <span class="muted">max new</span><input type="number" value=${sampling.max_new_tokens} onChange=${(e) => setSampling({ ...sampling, max_new_tokens: Number(e.target.value) })} style="width:70px" />
      <span class="muted">seed</span><input type="number" value=${sampling.seed} onChange=${(e) => setSampling({ ...sampling, seed: Number(e.target.value) })} style="width:80px" />
      <button onClick=${() => setSampling({ ...sampling, temperature: 0 })}>greedy</button><${Info} k="sampling" />
      <label class="muted"><input type="checkbox" checked=${useBoth} onChange=${(e) => setUseBoth(e.target.checked)} /> A and B side by side</label>
      ${mode === "chat" && html`<label class="muted" title=${extInRun ? "our protocol: n/a for an external slot" : ""}><input type="checkbox" checked=${thinkReq && !extInRun} disabled=${extInRun} onChange=${(e) => setThinkReq(e.target.checked)} /> force ${"<|think|>"} (reasoning models)</label><${Info} k="force_think" />
        <label class="muted" title=${extInRun ? "our protocol: n/a for an external slot" : ""}><input type="checkbox" checked=${tools && !extInRun} disabled=${extInRun} onChange=${(e) => setTools(e.target.checked)} /> python tool (REPL session ${sessionId})</label><${Info} k="python_tool" />
        <button onClick=${newConversation}>new conversation</button>`}`}
    </div>
    ${mode === "swarm" ? html`<${SwarmPanel} slots=${slots} onError=${setErr} busy=${busy} setBusy=${setBusy} />` : html`
    ${chatBlocked && html`<div class="panel warn"><b>Chat mode is n/a for ${extBase.map((s) => `slot ${s} (${slots[s].name})`).join(" and ")}: an external base model.</b> A base model is never given a chat template, even when its tokenizer ships one, and none is invented for it. Use <b>completion</b> mode, or load an instruct model.<${Info} k="external_slot" /></div>`}
    ${!chatBlocked && extNames.length > 0 && html`<div class="legend">External slot ${extNames.join(", ")}: ${mode === "chat" ? "the conversation is rendered with the model's own chat template; " : "the text is encoded with the model's own tokenizer (no BOS); "}the Python tool, declared functions and forced ${"<|think|>"} are ours and are off for this run. Log-probs and top-k are the model's own, over its own vocabulary: per token, they are not comparable with ours.<${Info} k="external_slot" /></div>`}
    ${mode === "chat" && slots.A && slots.A.checkpoint && !slots.A.external && (slots.A.stage || "base") === "base" && html`<div class="panel warn"><b>Slot A holds a base checkpoint (${slots.A.run || slots.A.name}).</b> A base model has never seen the chat tokens: after ${"<|assistant|>"} it just continues web text, so chat output will be garbage. Use <b>completion</b> mode for it, or load an instruct / reasoning / RL checkpoint (m4_sft_149m, m5_reasoning_149m, m6_rl_gsm_tools_149m best.pt) for chat.</div>`}
    ${mode === "chat" && slots.A && slots.A.checkpoint && slots.A.stage === "sft" && html`<div class="legend">Instruct checkpoint: leave "force ${"<|think|>"}" off (it was not trained with think spans). Answers can run long; the reply is added to the conversation only if it closes with ${"<|end|>"} within max-new tokens.</div>`}
    ${mode === "chat" && slots.A && slots.A.checkpoint && (slots.A.stage === "reasoning" || slots.A.stage === "rl") && html`<div class="legend">Reasoning / RL checkpoint: "force ${"<|think|>"}" is on (it opens every answer with a think span; tool models call Python inside it) and decoding defaults to greedy — at temperature 0.8 these small policies often fail to close the turn.</div>`}
    ${mode === "completion" ? html`<textarea value=${text} onInput=${(e) => setText(e.target.value)}></textarea>` : html`<div class="panel">
      ${messages.map((m, i) => html`<div class="row" style="margin-bottom:6px;align-items:flex-start">
        <select value=${m.role} onChange=${(e) => editMessage(i, { role: e.target.value })}><option>system</option><option>user</option><option>assistant</option></select>
        <div style="flex:1;min-width:0">
          ${m.role === "assistant" && m.think != null && html`<${TextWithSpecials} text=${m.think} cls="think" title="think span (tool calls shown as <<code=result>> markup)" />`}
          <textarea style="min-height:40px;width:100%" placeholder=${m.role === "user" ? "type the next user message and press generate" : ""} value=${m.content} onInput=${(e) => editMessage(i, { content: e.target.value })}></textarea>
          ${m.ids && html`<div class="legend">generated turn: ${m.ids.length} tokens kept verbatim${m.n_calls ? ` · ${m.n_calls} python call${m.n_calls > 1 ? "s" : ""}` : ""} (editing re-encodes it)</div>`}
        </div>
        <button onClick=${() => setMessages(messages.filter((_, j) => j !== i))}>✕</button></div>`)}
      <button onClick=${() => setMessages([...messages, { role: "user", content: "" }])}>+ message</button>
      <details style="margin-top:6px"><summary class="muted">declared functions (${extInRun ? "n/a for an external slot" : funcs.trim() ? "set" : "none"})</summary>
        <textarea disabled=${extInRun} style="min-height:60px;width:100%;font-family:var(--mono)" placeholder=${'[{"name": "unit_price", "signature": "def unit_price(item: str) -> float", "comment": "Catalogue price of an item in dollars."}]'}
          value=${funcs} onInput=${(e) => setFuncs(e.target.value)}></textarea>
        <div class="legend">JSON list. Each declaration is emitted as a masked ${"<|python_def|>"}signature${"<|python_comment|>"}comment${"<|/python_def|>"} block right after ${"<|bos|>"} (highlighted in the views below). The portal has no implementations, so calling one reports that it is declared but not available here.<${Info} k="declared_functions" /></div>
      </details>
      <div class="legend" style="margin-top:4px">Multi-turn: a well-formed assistant reply is appended here automatically with an empty user turn after it. Python calls inside the think span run in this conversation's session, so variables persist across turns. A base checkpoint has never seen the chat tokens; expect noise until SFT.</div></div>`}
    <div class="row" style="margin:8px 0">
      <button class="active" onClick=${generate} disabled=${busy || !slots.A || !slots.A.checkpoint || chatBlocked}>generate</button>
      <button onClick=${cancel} disabled=${!streamId}>cancel</button>
      <button onClick=${doScore} disabled=${busy || !slots.A || !slots.A.checkpoint || !!slots.A.external} title=${slots.A && slots.A.external ? "n/a for an external slot" : ""}>score prompt (teacher-forced)</button><${Info} k="score" />
      ${slots.A && slots.A.external && html`<span class="muted">scoring: n/a (external)</span>`}
      <span class="muted" style="margin-left:10px">view</span>
      ${["text", "tokens"].map((v) => html`<button class=${view === v ? "active" : ""} onClick=${() => setView(v)}>${v}</button>`)}<${Info} k="token_view" />
      ${hover && html`<span class="muted" style="font-family:var(--mono)">top-${hover.topk.length}: ${hover.topk.map((t) => `${JSON.stringify(t.piece)} ${Math.exp(t.logprob).toFixed(2)}`).join("  ")}</span>`}
    </div>
    <div class=${useBoth ? "two" : ""}>
      ${(useBoth ? ["A", "B"] : ["A"]).map((s) => html`<div key=${s}>
        <div class="muted" style="margin-bottom:4px"><b>${s}</b> ${stat(s)}${useBoth && divergence >= 0 && s === "A" ? ` · diverges at token ${divergence + 1}` : ""}</div>
        <${Stream} tokens=${out[s].tokens} prompt=${out[s].prompt} promptSpecial=${out[s].promptSpecial} segments=${out[s].segments} mode=${view} setHover=${setHover} />
      </div>`)}
    </div>
    ${calls.length > 0 && html`<div class="panel" style="margin-top:6px"><b>python calls</b>
      ${calls.map((c, i) => html`<div class="toolcall" key=${i}><pre class="code">${c.code}</pre><span class=${c.ok ? "result ok" : "result err"}>${c.result}</span></div>`)}</div>`}
    ${out.A.done && out.A.done.assistant && html`<div class="legend" style="margin-top:4px">assistant turn: ${out.A.done.assistant.well_formed ? "well-formed, added to the conversation" : `not well-formed (${out.A.done.reason}${out.A.done.assistant.malformed ? ", malformed" : ""}) — not added`}</div>`}
    <div class="legend" style="margin-top:6px">${view === "tokens" ? "chip color = probability the model assigned to the token it emitted (red = surprised, green = confident); hover a chip for the top-k alternatives at that step." : "decoded text (the default view); reserved tokens (<|user|>, <|think|>, <|python_call|>, ...) stay visible as the same chips the tokens view uses. Hover any word for its log-prob and the top-k alternatives."}</div>
    ${score && html`<h2>Teacher-forced scoring (slot A)<${Info} k="score" /></h2>
      <div class="muted">${score.n} tokens · mean logprob ${score.mean_logprob.toFixed(3)} · perplexity ${score.ppl ? score.ppl.toFixed(2) : "-"} (over loss-target tokens)</div>
      ${view === "tokens" ? html`<div class="chips">${score.tokens.map((t, i) => html`<span class=${"chip" + (t.target ? "" : " masked")} style=${t.target ? `background:${lpColor(t.logprob)}` : ""} title=${t.logprob == null ? "first token" : `logprob ${t.logprob.toFixed(3)} · rank ${t.rank}`} key=${i}>${showPiece(t.piece)}</span>`)}</div>`
        : html`<div class="rawout spx">${score.tokens.map((t, i) => t.special ? html`<${SpecialChip} key=${i} name=${t.piece} />` : html`<span class=${t.target ? "" : "prompt-text"} title=${t.logprob == null ? "" : `logprob ${t.logprob.toFixed(3)}`} key=${i}>${t.piece}</span>`)}</div>`}`}`}
  </div>`;
}

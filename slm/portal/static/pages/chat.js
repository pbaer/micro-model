/** Inference tab, chat mode (the default): a plain chat with the best all-round chat checkpoint, sensible sampling
 *  defaults and a fresh random seed per reply, the think span folded away behind a "reasoning" toggle, several
 *  conversations kept in this browser (localStorage), and the few settings that matter behind a gear. Everything
 *  else the worker can do (slots A/B, completion mode, swarm, token views, scoring, declared functions) is the
 *  advanced mode: pages/model.js at #/inference/advanced. */
import { h } from "preact";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "preact/hooks";
import htm from "htm";
import { api, readSSE } from "../components/util.js";
import { Info } from "../components/info.js";

const html = htm.bind(h);

// measured on m10_rl final.pt (scripts/chat_sampling_defaults.py, docs/results.md §27): see the "chat_settings" card
export const DEFAULTS = { model: null, temperature: 0.8, top_p: 0.95, top_k: 0, max_new_tokens: 1536, tools: true };
// v2 stores only the settings the user changed, so a new default reaches every browser (v1 stored a full snapshot on
// first visit, which froze whatever the defaults were then: max_new_tokens 512 for everyone who had opened the page)
const CHATS_KEY = "slm.chat.v1", SETTINGS_KEY = "slm.chat.settings.v2", OLD_SETTINGS_KEYS = ["slm.chat.settings.v1"];
const overrides = (s) => Object.fromEntries(Object.entries(s).filter(([k, v]) => k in DEFAULTS && DEFAULTS[k] !== v));
const SUGGESTIONS = [
  "Tell me a short story about a lighthouse keeper.",
  "Explain how rain forms, for a ten-year-old.",
  "What is 17 × 23? Show your working.",
  "Write a Python function that checks whether a word is a palindrome.",
];

const uid = () => Math.random().toString(36).slice(2, 10);
const newSeed = () => Math.floor(Math.random() * 2 ** 31);
const load = (k, fallback) => { try { const v = JSON.parse(localStorage.getItem(k)); return v == null ? fallback : v; } catch { return fallback; } };
const save = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage full or blocked: the session still works */ } };
const newChat = () => ({ id: uid(), title: "", created: Date.now(), updated: Date.now(), session: uid(), messages: [] });
const titleOf = (c) => c.title || (c.messages.find((m) => m.role === "user") || {}).content?.split("\n")[0].slice(0, 48) || "New chat";
const ago = (t) => { const s = (Date.now() - t) / 1000; return s < 60 ? "just now" : s < 3600 ? `${Math.floor(s / 60)} min ago` : s < 86400 ? `${Math.floor(s / 3600)} h ago` : `${Math.floor(s / 86400)} d ago`; };
const modelLabel = (c) => (c.external ? c.name : `${c.run} / ${c.name.replace(/\.pt$/, "")}`);

// ------------------------------------------------------------------------------------------------ rendering
/** The think text of a finished turn carries tool calls as <<code=result>> (one line) or <<<code>>>=result markup
 *  (slm/tools/protocol.py); split it into text and call segments for display. */
export function splitThink(text) {
  const out = [];
  const re = /<<<([\s\S]+?)>>>=?([^\n<]*)|<<([^<>\n]+?)=([^<>\n]*?)>>/g;
  let last = 0, m;
  while ((m = re.exec(text || ""))) {
    if (m.index > last) out.push({ text: text.slice(last, m.index) });
    out.push(m[1] != null ? { code: m[1].trim(), result: m[2].trim() } : { code: m[3].trim(), result: m[4].trim() });
    last = m.index + m[0].length;
  }
  if (last < (text || "").length) out.push({ text: text.slice(last) });
  return out;
}

/** Inline markdown: `code`, **bold**, *italic*. Builds nodes, never HTML strings. */
function inline(s) {
  const out = [];
  const re = /(`[^`\n]+`)|(\*\*[^*\n]+\*\*)|(\*[^*\n]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(s))) {
    if (m.index > last) out.push(s.slice(last, m.index));
    if (m[1]) out.push(html`<code key=${m.index}>${m[1].slice(1, -1)}</code>`);
    else if (m[2]) out.push(html`<b key=${m.index}>${m[2].slice(2, -2)}</b>`);
    else out.push(html`<i key=${m.index}>${m[3].slice(1, -1)}</i>`);
    last = m.index + m[0].length;
  }
  if (last < s.length) out.push(s.slice(last));
  return out;
}

/** A small, safe markdown renderer for answers: fenced code, headings, bullet and numbered lists, paragraphs. */
export function Markdown({ text }) {
  const lines = String(text || "").replace(/\r/g, "").split("\n");
  const blocks = [];
  let i = 0, key = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (/^```/.test(l)) {
      const lang = l.slice(3).trim(); const buf = []; i++;
      while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]);
      i++;
      blocks.push(html`<pre key=${key++} class="chat-code" data-lang=${lang}>${buf.join("\n")}</pre>`);
      continue;
    }
    if (/^\s*([-*•])\s+/.test(l)) {
      const items = [];
      while (i < lines.length && /^\s*([-*•])\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*[-*•]\s+/, ""));
      blocks.push(html`<ul key=${key++}>${items.map((t, j) => html`<li key=${j}>${inline(t)}</li>`)}</ul>`);
      continue;
    }
    if (/^\s*\d+[.)]\s+/.test(l)) {
      const items = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*\d+[.)]\s+/, ""));
      blocks.push(html`<ol key=${key++}>${items.map((t, j) => html`<li key=${j}>${inline(t)}</li>`)}</ol>`);
      continue;
    }
    if (/^#{1,6}\s+/.test(l)) { blocks.push(html`<p key=${key++} class="chat-h">${inline(l.replace(/^#+\s+/, ""))}</p>`); i++; continue; }
    if (!l.trim()) { i++; continue; }
    const buf = [];
    while (i < lines.length && lines[i].trim() && !/^```|^\s*[-*•]\s+|^\s*\d+[.)]\s+|^#{1,6}\s+/.test(lines[i])) buf.push(lines[i++]);
    blocks.push(html`<p key=${key++}>${buf.flatMap((t, j) => (j ? [html`<br key=${"b" + j} />`, ...inline(t)] : inline(t)))}</p>`);
  }
  return html`<div class="chat-md">${blocks}</div>`;
}

function Reasoning({ think, calls, open, setOpen, live }) {
  const segs = useMemo(() => splitThink(think), [think]);
  const nCalls = calls ? calls.length : segs.filter((s) => s.code != null).length;
  const words = (think || "").split(/\s+/).filter(Boolean).length;
  const empty = !words && !nCalls && !live;
  if (empty) return null;
  return html`<div class=${"chat-think" + (open ? " open" : "")}>
    <button class="chat-think-toggle" onClick=${() => setOpen(!open)} aria-expanded=${open}>
      <span class="chev">${open ? "▾" : "▸"}</span>${live ? html`<span class="chat-live">thinking…</span>` : `reasoning`}
      <span class="muted"> · ${words} word${words === 1 ? "" : "s"}${nCalls ? ` · ${nCalls} Python call${nCalls === 1 ? "" : "s"}` : ""}</span>
    </button>
    ${open && html`<div class="chat-think-body">
      ${calls != null ? html`<div class="chat-think-text">${think}</div>${(calls || []).map((c, i) => html`<div key=${i} class="chat-call"><pre>${c.code}</pre><div class=${"res" + (c.ok ? "" : " err")}>${c.result}</div></div>`)}`
        : segs.map((s, i) => s.code != null ? html`<div key=${i} class="chat-call"><pre>${s.code}</pre><div class="res">${s.result}</div></div>` : html`<div key=${i} class="chat-think-text">${s.text}</div>`)}
    </div>`}
  </div>`;
}

function Bubble({ m, live, onRegenerate, last }) {
  const [open, setOpen] = useState(false);
  if (m.role === "user") return html`<div class="chat-msg user"><div class="chat-bubble">${m.content}</div></div>`;
  return html`<div class="chat-msg assistant">
    <div class="chat-avatar" title=${m.model || ""}>S</div>
    <div class="chat-body">
      <${Reasoning} think=${m.think} calls=${live ? m.calls : null} open=${open} setOpen=${setOpen} live=${live && m.phase === "think"} />
      ${m.content ? html`<${Markdown} text=${m.content} />` : live ? (m.phase === "think" ? null : html`<div class="chat-caret"></div>`) : html`<div class="muted">(no answer)</div>`}
      ${live && m.content && html`<span class="chat-caret inline"></span>`}
      ${!live && html`<div class="chat-meta">
        ${m.truncated && html`<span class="chat-cut" title=${m.reason || ""}>stopped early</span>`}
        ${m.tok_s != null && html`<span>${m.tok_s.toFixed(0)} tok/s</span>`}
        ${m.seed != null && html`<span title="the sampling seed of this reply (a new one is drawn for every reply)">seed ${m.seed}</span>`}
        ${last && onRegenerate && html`<button class="chat-link" onClick=${onRegenerate}>regenerate</button>`}
      </div>`}
    </div>
  </div>`;
}

// ------------------------------------------------------------------------------------------------ page
export function ChatPage() {
  const [status, setStatus] = useState(null);
  const [ckpts, setCkpts] = useState([]);
  const [chats, setChats] = useState(() => { const c = load(CHATS_KEY, []); return Array.isArray(c) && c.length ? c : [newChat()]; });
  const [cur, setCur] = useState(() => load(CHATS_KEY + ".cur", null));
  const [settings, setSettings] = useState(() => {
    OLD_SETTINGS_KEYS.forEach((k) => { try { localStorage.removeItem(k); } catch { /* blocked storage */ } });
    return { ...DEFAULTS, ...load(SETTINGS_KEY, {}) };
  });
  const [showOpts, setShowOpts] = useState(false);
  const [draft, setDraft] = useState("");
  const [live, setLive] = useState(null);  // the reply being streamed: {think, content, phase, calls}
  const [streamId, setStreamId] = useState(null);
  const [loading, setLoading] = useState(null);  // "loading <model>…" while the worker loads a checkpoint
  const [err, setErr] = useState(null);
  const [editing, setEditing] = useState(null);  // {id, text}: the conversation whose title is being edited in place
  const [confirmDel, setConfirmDel] = useState(null);  // id of the conversation asking "delete?" in its row
  const abortRef = useRef(null);
  const scrollRef = useRef(null);
  const taRef = useRef(null);
  const liveRef = useRef(null);

  const chat = chats.find((c) => c.id === cur) || chats[0];
  useEffect(() => { save(CHATS_KEY, chats); }, [chats]);
  useEffect(() => { if (chat && chat.id !== cur) setCur(chat.id); save(CHATS_KEY + ".cur", chat ? chat.id : null); }, [chat && chat.id]);
  useEffect(() => { save(SETTINGS_KEY, overrides(settings)); }, [settings]);
  const refresh = () => { api("/api/model/status").then(setStatus).catch(() => {}); api("/api/model/checkpoints").then(setCkpts).catch(() => {}); };
  useEffect(() => { refresh(); const id = setInterval(() => api("/api/model/status").then(setStatus).catch(() => {}), 10000); return () => clearInterval(id); }, []);

  // the model: slot A. Chosen in the settings (path), else whatever chat-capable checkpoint slot A already holds, else the default.
  const slotA = status && status.slots ? status.slots.A : null;
  const chatCks = ckpts.filter((c) => c.chat_ok && (!c.external || c.available));
  const defaultCk = ckpts.find((c) => c.default_chat) || null;
  const loadedOk = slotA && slotA.checkpoint && chatCks.find((c) => c.path === slotA.checkpoint);
  const wanted = settings.model ? chatCks.find((c) => c.path === settings.model) || null : loadedOk || defaultCk;
  const ready = !!(slotA && slotA.checkpoint && wanted && slotA.checkpoint === wanted.path);
  const proto = ready && !slotA.external;  // our protocol (think span, Python tool); an external chat model takes neither
  const thinkRequired = proto && (slotA.stage === "rl" || slotA.stage === "reasoning");
  const busy = !!live || !!loading;

  const ensureModel = async () => {
    if (ready) return true;
    if (!wanted) { setErr("No chat-capable checkpoint is available (an SFT, reasoning or RL run with a final.pt, or a downloaded external chat model)."); return false; }
    setLoading(`loading ${modelLabel(wanted)}…`); setErr(null);
    try {
      await api("/api/model/slots/A/load", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ checkpoint: wanted.path, device: "auto" }) });
      const st = await api("/api/model/status"); setStatus(st);
      return st.slots && st.slots.A && st.slots.A.checkpoint === wanted.path;
    } catch (e) { setErr(`could not load ${modelLabel(wanted)}: ${e.message || e}`); return false; } finally { setLoading(null); }
  };
  useEffect(() => { if (ckpts.length && status && !ready && !loading && wanted && !(slotA && slotA.checkpoint)) ensureModel(); }, [ckpts.length, status && status.worker, wanted && wanted.path]);  // an empty slot A: load on arrival, so the first message answers right away

  const update = (id, f) => setChats((cs) => cs.map((c) => (c.id === id ? { ...f(c), updated: Date.now() } : c)));
  const scrollDown = () => { const el = scrollRef.current; if (el) el.scrollTop = el.scrollHeight; };
  useLayoutEffect(scrollDown, [chat && chat.messages.length, live && (live.content.length + live.think.length)]);

  const generate = async (c) => {
    if (!(await ensureModel())) return;
    const seed = newSeed();
    const msgs = c.messages.map((m) => (m.role === "user" ? { role: "user", content: m.content } : { role: "assistant", content: m.content, think: m.think, ids: m.ids, n_calls: m.n_calls }));
    const body = { slots: ["A"], mode: "chat", messages: msgs, temperature: settings.temperature, top_p: settings.top_p, top_k: settings.top_k, max_new_tokens: settings.max_new_tokens, seed,
                   logprobs_topk: 1, think_required: thinkRequired, tools: proto && settings.tools, session_id: c.session, max_tool_calls: 8, functions: null };
    const st = { think: "", content: "", phase: thinkRequired ? "think" : "answer", calls: [], code: null };
    liveRef.current = st; setLive({ ...st }); setErr(null);
    const ctrl = new AbortController(); abortRef.current = ctrl;
    let done = null;
    try {
      const r = await fetch("/api/model/generate", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal: ctrl.signal });
      await readSSE(r, (ev, d) => {
        if (ev === "start") setStreamId(d.stream_id);
        else if (ev === "token") {
          const s = liveRef.current;
          if (d.special) {
            if (d.piece === "<|think|>") s.phase = "think";
            else if (d.piece === "<|/think|>") s.phase = "answer";
            else if (d.piece === "<|python_call|>") s.code = "";
            else if (d.piece === "<|/python_call|>") s.code = null;
          } else if (d.inserted) { /* a tool result: shown from the tool event */ }
          else if (s.code != null) s.code += d.piece;
          else if (s.phase === "think") s.think += d.piece;
          else s.content += d.piece;
          setLive({ ...s });
        } else if (ev === "tool") { const s = liveRef.current; s.calls = [...s.calls, d]; setLive({ ...s }); }
        else if (ev === "done" && d.slot === "A") done = d;
        else if (ev === "error") setErr(d.error);
      });
    } catch (e) { if (e.name !== "AbortError") setErr(String(e.message || e)); }
    const s = liveRef.current;
    const a = done && done.assistant;
    const content = (a && a.answer != null ? a.answer : s.content).trim();
    const reply = { role: "assistant", content, think: a ? a.think : s.think, ids: a && a.well_formed ? a.ids : undefined, n_calls: a ? a.n_calls : s.calls.length,
                    truncated: !(a && a.well_formed), reason: done ? done.reason : "cancelled", seed, tok_s: done ? done.tok_s : null, model: wanted ? modelLabel(wanted) : null, at: Date.now() };
    if (content || reply.think || s.calls.length) update(c.id, (x) => ({ ...x, messages: [...x.messages, reply] }));
    liveRef.current = null; setLive(null); setStreamId(null); abortRef.current = null;
  };

  const send = async (text) => {
    const t = (text != null ? text : draft).trim();
    if (!t || busy || !chat) return;
    setDraft("");
    if (taRef.current) taRef.current.style.height = "";
    const c = { ...chat, messages: [...chat.messages, { role: "user", content: t, at: Date.now() }] };
    update(chat.id, () => c);
    await generate(c);
  };
  const stop = () => { if (streamId) api(`/api/model/streams/${streamId}/cancel`, { method: "POST" }).catch(() => {}); if (abortRef.current) abortRef.current.abort(); };
  const regenerate = async () => {
    if (!chat || busy) return;
    const ms = chat.messages.slice();
    while (ms.length && ms[ms.length - 1].role === "assistant") ms.pop();
    if (!ms.length) return;
    const c = { ...chat, messages: ms };
    update(chat.id, () => c);
    await generate(c);
  };
  const create = () => { const c = newChat(); setChats((cs) => [c, ...cs]); setCur(c.id); setDraft(""); setTimeout(() => taRef.current && taRef.current.focus(), 0); };
  // Rename and delete are inline (an input in the row, a confirm strip in the row), never window.prompt/confirm:
  // embedded browsers such as the Claude desktop app's pane block native dialogs (prompt throws, confirm returns false).
  const remove = (id) => {
    const c = chats.find((x) => x.id === id);
    if (!c) return;
    setConfirmDel(null);
    if (c.session) api(`/api/model/sessions/${c.session}/reset`, { method: "POST" }).catch(() => {});
    setChats((cs) => { const rest = cs.filter((x) => x.id !== id); return rest.length ? rest : [newChat()]; });
    if (cur === id) setCur(null);
  };
  const askRemove = (id) => {  // an empty conversation goes at once; one with messages asks in its row
    const c = chats.find((x) => x.id === id);
    setEditing(null);
    if (c && c.messages.length) setConfirmDel(id); else remove(id);
  };
  const startRename = (id) => { const c = chats.find((x) => x.id === id); setConfirmDel(null); if (c) setEditing({ id, text: titleOf(c) }); };
  const commitRename = () => {
    if (!editing) return;
    const c = chats.find((x) => x.id === editing.id);
    const t = editing.text.trim();
    // an emptied title falls back to the first user line; an unchanged one stays automatic
    if (c && t !== titleOf(c)) update(editing.id, (x) => ({ ...x, title: t }));
    setEditing(null);
  };
  const focusOnce = (el) => { if (el && !el.dataset.focused) { el.dataset.focused = "1"; el.focus(); el.select(); } };
  const onKey = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } };
  const grow = (e) => { const el = e.target; el.style.height = ""; el.style.height = Math.min(200, el.scrollHeight) + "px"; };
  const setS = (k, v) => setSettings((s) => ({ ...s, [k]: v }));
  const pickModel = (path) => { setS("model", path || null); };
  const trainingLive = status && status.live_runs && status.live_runs.length > 0;

  const modelPill = loading ? html`<span class="chat-pill busy">${loading}</span>`
    : ready ? html`<span class="chat-pill ok" title=${`slot A · ${slotA.device}/${slotA.dtype}`}>${modelLabel(wanted)}${wanted.default_chat ? "" : ""} · ${slotA.device}</span>`
    : wanted ? html`<span class="chat-pill" title="loads when you send the first message">${modelLabel(wanted)} · not loaded</span>`
    : html`<span class="chat-pill warn">no chat model</span>`;

  return html`<div class="chat">
    <aside class="chat-side">
      <button class="chat-new" onClick=${create}>+ New chat</button>
      <div class="chat-list">
        ${chats.map((c) => {
          const isEdit = editing && editing.id === c.id, isDel = confirmDel === c.id;
          return html`<div key=${c.id} class=${"chat-item" + (chat && c.id === chat.id ? " sel" : "") + (isEdit ? " editing" : "") + (isDel ? " deleting" : "")}
              onClick=${() => { if (!isEdit) setCur(c.id); }} title=${isEdit || isDel ? "" : titleOf(c)}>
            ${isEdit ? html`<input class="chat-item-input" ref=${focusOnce} value=${editing.text} aria-label="conversation title"
                onClick=${(e) => e.stopPropagation()} onInput=${(e) => setEditing({ id: c.id, text: e.target.value })}
                onKeyDown=${(e) => { if (e.key === "Enter") { e.preventDefault(); commitRename(); } else if (e.key === "Escape") { e.preventDefault(); setEditing(null); } }}
                onBlur=${commitRename} />`
              : html`<div class="t" onDblClick=${(e) => { e.stopPropagation(); startRename(c.id); }}>${titleOf(c)}</div>`}
            ${isDel ? html`<div class="chat-item-confirm" onClick=${(e) => e.stopPropagation()}>
                <span>Delete this chat?</span>
                <button class="danger" onClick=${() => remove(c.id)}>Delete</button>
                <button onClick=${() => setConfirmDel(null)}>Cancel</button>
              </div>`
              : html`<div class="s muted">${c.messages.length ? `${c.messages.length} messages · ${ago(c.updated)}` : "empty"}</div>`}
            ${!isEdit && !isDel && html`<div class="chat-item-acts">
              <button title="rename" aria-label="rename" onClick=${(e) => { e.stopPropagation(); startRename(c.id); }}>✎</button>
              <button title="delete" aria-label="delete" onClick=${(e) => { e.stopPropagation(); askRemove(c.id); }}>✕</button>
            </div>`}
          </div>`;
        })}
      </div>
      <a class="chat-adv" href="#/inference/advanced" title="slots A/B, completion and swarm modes, token views, scoring, declared functions">Advanced mode →</a>
    </aside>
    <section class="chat-main">
      <header class="chat-head">
        <div class="chat-title">${chat ? titleOf(chat) : "Chat"}<${Info} k="chat_mode" /></div>
        <div class="chat-head-r">
          ${modelPill}
          ${trainingLive && html`<span class="chat-pill warn" title=${`training live: ${status.live_runs.join(", ")} — the model loads on the CPU`}>training live</span>`}
          <button class=${"chat-gear" + (showOpts ? " on" : "")} onClick=${() => setShowOpts(!showOpts)} title="settings" aria-label="settings">⚙</button>
        </div>
      </header>
      ${showOpts && html`<div class="chat-opts">
        <label><span>model<${Info} k="chat_model" /></span>
          <select value=${wanted ? wanted.path : ""} onChange=${(e) => pickModel(e.target.value)}>
            ${!chatCks.length && html`<option value="">none available</option>`}
            ${chatCks.map((c) => html`<option key=${c.path} value=${c.path}>${modelLabel(c)}${c.default_chat ? " (default)" : ""}${c.external ? " (external)" : ""}</option>`)}
          </select></label>
        <label><span>temperature <b>${settings.temperature.toFixed(2)}</b></span><input type="range" min="0" max="1.2" step="0.05" value=${settings.temperature} onInput=${(e) => setS("temperature", Number(e.target.value))} /></label>
        <label><span>top-p <b>${settings.top_p.toFixed(2)}</b></span><input type="range" min="0.5" max="1" step="0.05" value=${settings.top_p} onInput=${(e) => setS("top_p", Number(e.target.value))} /></label>
        <label><span>top-k <b>${settings.top_k || "off"}</b></span><input type="range" min="0" max="100" step="5" value=${settings.top_k} onInput=${(e) => setS("top_k", Number(e.target.value))} /></label>
        <label><span>max reply tokens <b>${settings.max_new_tokens}</b></span><input type="range" min="64" max="2048" step="64" value=${settings.max_new_tokens} onInput=${(e) => setS("max_new_tokens", Number(e.target.value))} /></label>
        <label class="chat-check"><input type="checkbox" checked=${settings.tools} disabled=${!proto && ready} onChange=${(e) => setS("tools", e.target.checked)} /> Python tool${ready && !proto ? " (n/a for an external model)" : ""}</label>
        <div class="chat-opts-foot"><button onClick=${() => setSettings({ ...DEFAULTS })}>reset to defaults</button><${Info} k="chat_settings" /><span class="muted">a new random seed is drawn for every reply</span></div>
      </div>`}
      ${err && html`<div class="chat-err">${err} <button class="chat-link" onClick=${() => setErr(null)}>dismiss</button></div>`}
      <div class="chat-scroll" ref=${scrollRef}>
        ${chat && !chat.messages.length && !live ? html`<div class="chat-welcome">
          <div class="chat-welcome-t">What would you like to talk about?</div>
          <div class="chat-welcome-s muted">A ${wanted && wanted.params ? "" : "336M-parameter "}language model trained from scratch in this repo. It is small: short, concrete questions work best, and it can use a Python tool for arithmetic.</div>
          <div class="chat-sugg">${SUGGESTIONS.map((s) => html`<button key=${s} onClick=${() => send(s)} disabled=${busy}>${s}</button>`)}</div>
        </div>` : null}
        ${chat && chat.messages.map((m, i) => html`<${Bubble} key=${(m.at || i) + ":" + i} m=${m} last=${i === chat.messages.length - 1 && m.role === "assistant"} onRegenerate=${!busy ? regenerate : null} />`)}
        ${live && html`<${Bubble} m=${{ role: "assistant", ...live }} live=${true} />`}
      </div>
      <footer class="chat-compose">
        <textarea ref=${taRef} rows="1" placeholder=${ready || wanted ? "Message… (Enter to send, Shift+Enter for a new line)" : "no chat model available"} value=${draft}
          onInput=${(e) => { setDraft(e.target.value); grow(e); }} onKeyDown=${onKey} disabled=${!wanted && !ready}></textarea>
        ${live ? html`<button class="chat-send stop" onClick=${stop} title="stop generating">■</button>`
          : html`<button class="chat-send" onClick=${() => send()} disabled=${busy || !draft.trim() || (!wanted && !ready)} title="send">➤</button>`}
      </footer>
    </section>
  </div>`;
}

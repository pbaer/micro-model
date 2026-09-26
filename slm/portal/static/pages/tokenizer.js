import { h } from "preact";
import { useEffect, useState } from "preact/hooks";
import htm from "htm";
import { api } from "../components/util.js";
import { TokenChips } from "../components/tokens.js";
import { Info } from "../components/info.js";

const html = htm.bind(h);
const SAMPLE = `The quick brown fox jumps over the lazy dog. In 1492, Columbus sailed with 3 ships.
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
Let $f(x) = \\frac{x^2 + 1}{\\sqrt{x}}$. <|user|> is not special here.`;

export function TokenizerPage() {
  const [tags, setTags] = useState([]);
  const [tag, setTag] = useState(null);
  const [text, setText] = useState(SAMPLE);
  const [mode, setMode] = useState("raw");
  const [enc, setEnc] = useState(null);
  const [showIds, setShowIds] = useState(false);
  const [hover, setHover] = useState(null);
  const [q, setQ] = useState("");
  const [vocab, setVocab] = useState([]);
  const [messages, setMessages] = useState([
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "What is 17 + 26?" },
    { role: "assistant", think: "17 + 26: 17 + 20 = 37, 37 + 6 = 43.", content: "#### 43" },
  ]);
  useEffect(() => { api("/api/tokenizers").then((t) => { setTags(t); if (t.length && !tag) setTag(t[t.length - 1].tag); }); }, []);
  useEffect(() => {
    if (!tag) return;
    const body = mode === "chat" ? { mode, messages } : { mode, text };
    api(`/api/tokenizers/${tag}/encode`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }).then(setEnc).catch(() => setEnc(null));
  }, [tag, text, mode, messages]);
  useEffect(() => { if (tag) api(`/api/tokenizers/${tag}/vocab?q=${encodeURIComponent(q)}&limit=60`).then(setVocab).catch(() => {}); }, [tag, q]);
  const t = tags.find((x) => x.tag === tag);
  return html`<div>
    <h1>Tokenizer</h1>
    <div class="sub">${t ? `${t.tag} · vocab ${t.vocab_size.toLocaleString()} (${t.n_special} reserved specials) · sha ${t.sha256.slice(0, 12)}` : "no tokenizer found under data root"}<${Info} k="tokenizer" /></div>
    <div class="row" style="margin-bottom:8px">
      <select value=${tag} onChange=${(e) => setTag(e.target.value)}>${tags.map((x) => html`<option value=${x.tag}>${x.tag}</option>`)}</select>
      ${["raw", "document", "chat"].map((m) => html`<button class=${mode === m ? "active" : ""} onClick=${() => setMode(m)}>${m}</button>`)}<${Info} k="tok_modes" />
      <button class=${showIds ? "active" : ""} onClick=${() => setShowIds(!showIds)}>ids</button>
      ${enc && html`<span class="muted">${enc.n_tokens} tokens${enc.n_chars != null ? html` · ${enc.n_chars} chars · ${enc.chars_per_token.toFixed(2)} chars/token<${Info} k="chars_per_token" />` : ""}${enc.n_target != null ? ` · ${enc.n_target} loss targets` : ""}</span>`}
      ${hover && html`<span class="muted" style="font-family:var(--mono)">hover: id ${hover.id} ${JSON.stringify(hover.piece)}</span>`}
    </div>
    ${mode !== "chat" ? html`<textarea value=${text} onInput=${(e) => setText(e.target.value)}></textarea>` : html`<div class="panel">
      ${messages.map((m, i) => html`<div class="row" style="margin-bottom:6px;align-items:flex-start">
        <select value=${m.role} onChange=${(e) => setMessages(messages.map((x, j) => (j === i ? { ...x, role: e.target.value } : x)))}><option>system</option><option>user</option><option>assistant</option></select>
        ${m.role === "assistant" && html`<textarea style="min-height:40px;flex:1" placeholder="think (optional)" value=${m.think || ""} onInput=${(e) => setMessages(messages.map((x, j) => (j === i ? { ...x, think: e.target.value || undefined } : x)))}></textarea>`}
        <textarea style="min-height:40px;flex:2" value=${m.content} onInput=${(e) => setMessages(messages.map((x, j) => (j === i ? { ...x, content: e.target.value } : x)))}></textarea>
        <button onClick=${() => setMessages(messages.filter((_, j) => j !== i))}>✕</button></div>`)}
      <button onClick=${() => setMessages([...messages, { role: "user", content: "" }])}>+ message</button>
      <div class="legend" style="margin-top:6px"><span><b style="color:#15803d">green</b> = loss target (assistant content + ${"<|end|>"})</span><span><b>grey</b> = masked (prompt tokens, ${"<|eos|>"})</span></div>
    </div>`}
    <div style="margin-top:10px">${enc ? html`<${TokenChips} pieces=${enc.pieces} showIds=${showIds} lossMask=${mode === "chat"} onHover=${setHover} />` : html`<div class="empty-note">…</div>`}</div>
    <h2>Vocabulary<${Info} k="vocab" /></h2>
    <div class="row"><input type="text" placeholder="search pieces" value=${q} onInput=${(e) => setQ(e.target.value)} /><span class="muted">${vocab.length} shown</span></div>
    <div class="chips" style="margin-top:8px;max-height:200px">${vocab.map((v) => html`<span class="chip" title=${"id " + v.id}>${v.piece.replace(/ /g, "·")}<sub>${v.id}</sub></span>`)}</div>
  </div>`;
}

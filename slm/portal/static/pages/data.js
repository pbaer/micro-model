import { h } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import htm from "htm";
import { api, fmtTok, fmtInt, fmtBytes, fmtNum } from "../components/util.js";
import { TokenChips } from "../components/tokens.js";

const html = htm.bind(h);

// Routes (the hash parts after "#/data"): [] = recipe list · ["recipe", <encoded id>] = one recipe ·
// ["catalog"] / ["documents"] = the source-centric views. Recipe ids carry ":" and "/", so they travel encoded.
export const dataHref = (...parts) => "#/data" + parts.map((p) => "/" + encodeURIComponent(p)).join("");

export function DataPage({ parts = [] }) {
  const route = parts[0] || "recipes";
  const tabs = [["recipes", "recipes", dataHref()], ["catalog", "sources", dataHref("catalog")], ["documents", "documents", dataHref("documents")]];
  return html`<div>
    <h1>Data</h1>
    <div class="row" style="margin:8px 0">${tabs.map(([id, label, href]) => html`<a href=${href}><button class=${route === id ? "active" : ""}>${label}</button></a>`)}</div>
    ${route === "recipes" && (parts[1] ? html`<${Recipe} id=${decodeURIComponent(parts[1])} key=${parts[1]} />` : html`<${RecipeList} />`)}
    ${route === "recipe" && html`<${Recipe} id=${decodeURIComponent(parts[1] || "")} key=${parts[1]} />`}
    ${route === "catalog" && html`<${Sources} />`}
    ${route === "documents" && html`<${Documents} />`}
  </div>`;
}

const STAGE = { pretrain: "pretrain", sft: "sft", rl: "rl" };

function RecipeList() {
  const [rs, setRs] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => { api("/api/data/recipes").then(setRs).catch((e) => setErr(String(e))); }, []);
  if (err) return html`<div class="panel" style="border-color:#fca5a5;color:#b91c1c">${err}</div>`;
  if (!rs) return html`<div>loading…</div>`;
  return html`<div>
    <div class="sub">Every training config and run, by stage. A recipe is the data section of one plan (config) or of one run (what it actually started with).</div>
    <table><tr><th>stage</th><th class="l">recipe</th><th>status</th><th>sources</th><th>seq</th><th>tokens</th><th class="l">init_from</th><th class="l">plan</th></tr>
    ${rs.map((r) => html`<tr class="click" onClick=${() => { location.hash = dataHref("recipes", r.id).slice(1); }}>
      <td><span class=${"stage-badge " + (r.stage === "sft" ? "sft" : r.stage === "rl" ? "rl" : "")}>${STAGE[r.stage] || r.stage}</span></td>
      <td class="l">${r.run_name}${r.kind === "config" ? html` <span class="muted">(plan only)</span>` : ""}</td>
      <td>${r.status}</td>
      <td>${r.stage === "rl" ? (r.rl ? r.rl.tasks.join(", ") : "-") : r.rows.length}</td>
      <td>${r.seq_len || "-"}</td>
      <td>${r.stage === "rl" ? (r.rl ? r.rl.total_steps + " steps" : "-") : html`${fmtTok(r.tokens)}${r.total_tokens ? " / " + fmtTok(r.total_tokens) : ""}`}</td>
      <td class="l"><span class="muted">${(r.init_from || "random init").replace("runs/", "").replace("/checkpoints", "")}</span></td>
      <td class="l">${r.plan_differs === true ? html`<b style="color:#b45309">yaml differs from run</b>` : r.plan_differs === false ? html`<span class="muted">matches yaml</span>` : ""}</td></tr>`)}
    </table>
  </div>`;
}

/** weight / planned / available / epochs table shared by the mixture and the extra_val block */
function MixTable({ rows, showPlanned, streams }) {
  return html`<table><tr><th class="l">source</th><th>weight</th>${showPlanned ? html`<th>planned</th>` : ""}<th>available (train)</th><th>epochs</th>${streams ? html`<th>consumed</th>` : ""}<th>val tokens</th><th class="l">prepared from</th></tr>
    ${rows.map((r) => { const st = streams && streams[r.source]; return html`<tr>
      <td class="l">${r.source}${r.missing ? html` <b style="color:#b91c1c">not on disk</b>` : ""}</td>
      <td>${(r.weight * 100).toFixed(1)}%</td>
      ${showPlanned ? html`<td>${fmtTok(r.planned_tokens)}</td>` : ""}
      <td>${fmtTok(r.available_tokens)}</td>
      <td style=${r.epochs > 1.5 ? "color:#b91c1c;font-weight:600" : ""}>${r.epochs == null ? (showPlanned ? "no data" : "-") : r.epochs.toFixed(2)}</td>
      ${streams ? html`<td>${st ? html`${fmtTok(st.tokens)} <span class="muted">(${st.epoch.toFixed(2)} ep)</span>` : html`<span class="muted">-</span>`}</td>` : ""}
      <td>${fmtTok(r.val_tokens)}</td>
      <td class="l"><span class="legend">${r.provenance || "-"}</span></td></tr>`; })}
  </table>`;
}

function RlPanel({ rl }) {
  const rows = [["tasks", rl.tasks.map((t) => `${t} ${(rl.task_weights[t] * 100).toFixed(0)}%`).join(" · ")],
    ["prompts", `${fmtInt(rl.n_train_prompts)} train / ${fmtInt(rl.n_heldout_prompts)} held out (split by prompt-text hash, 10% held out)`],
    ["rollouts", `${rl.prompts_per_step} prompts/step x ${rl.group_size} samples, max ${rl.max_new_tokens} new tokens, T=${rl.temperature} top_p=${rl.top_p}`],
    ["format", `think span ${rl.think_required ? "mandatory" : "optional"}${rl.tools ? ` · Python tool, max ${rl.max_tool_calls} calls` : " · no tools"}`],
    ["reward", `${rl.reward_scheme}: ${rl.reward_rule}`],
    ["objective", `clip ${rl.clip_eps} · KL ${rl.kl_coef} (${rl.kl_kind}) · lr ${rl.lr}`],
    ["collapse guards", `entropy_stop ${rl.entropy_stop || "off"} · kl_stop ${rl.kl_stop || "off"}`]];
  return html`<div><h2>Prompts and reward</h2>
    <table>${rows.map(([k, v]) => html`<tr><td>${k}</td><td class="l">${v}</td></tr>`)}</table>
    <div class="legend" style="margin-top:6px">RL has no token mixture: the model trains on its own samples. Prompts are generated deterministically from the task list and the seed.</div></div>`;
}

function Recipe({ id }) {
  const [r, setR] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => { setR(null); setErr(null); api(`/api/data/recipe?id=${encodeURIComponent(id)}`).then(setR).catch((e) => setErr(String(e))); }, [id]);
  if (err) return html`<div class="panel" style="border-color:#fca5a5;color:#b91c1c">${err}</div>`;
  if (!r) return html`<div>loading…</div>`;
  return html`<div>
    <div class="row" style="margin-bottom:6px"><a href=${dataHref()}><button>‹ all recipes</button></a>
      <b>${r.run_name}</b><span class="stage-badge ${r.stage === "sft" ? "sft" : r.stage === "rl" ? "rl" : ""}">${r.stage}</span>
      <span class="muted">${r.kind === "run" ? "run (what it started with)" : "plan (yaml)"} · ${r.status}</span>
      ${r.kind === "run" && r.config_path && html`<a href=${dataHref("recipes", "config:" + r.config_path)}><button>see the plan</button></a>`}
      ${r.kind === "run" && html`<a href=${"#/runs/" + encodeURIComponent(r.run_name)}><button>run page</button></a>`}</div>
    <div class="sub">tokenizer ${r.tokenizer_tag} · ${r.seq_len ? `seq_len ${r.seq_len} · ` : ""}${r.total_tokens ? fmtTok(r.total_tokens) + " tokens" : ""}${r.total_note ? ` (${r.total_note})` : ""}
      ${r.init_from ? ` · init_from ${r.init_from}` : " · random init"}${r.root ? ` · ${r.root}` : ""}</div>
    ${r.plan_differs === true && html`<div class="panel" style="border-color:#fcd34d">The yaml <code>${r.config_path}</code> no longer matches what this run started with (configs get edited between phases). This page shows the run.</div>`}
    ${r.rl ? html`<${RlPanel} rl=${r.rl} />` : html`<div>
      <${MixTable} rows=${r.rows} showPlanned=${true} streams=${r.stream_sources} />
      <div class="legend" style="margin-top:6px">epochs = planned tokens / available tokens; above 1.5 (red) the source is repeated.${r.stream_sources ? " 'consumed' is the loader's own per-source count at the last checkpoint." : ""}</div>
      ${r.extra_val.length > 0 && html`<h2>extra_val_mixture (drift set, val splits only)</h2>
        <${MixTable} rows=${r.extra_val} showPlanned=${false} />
        <div class="legend">Pretraining validation tracked alongside the stage's own val loss, so base-model drift is visible; ${fmtTok(r.extra_val_tokens)} tokens per evaluation.</div>`}
    </div>`}
  </div>`;
}

function Sources() {
  const [ov, setOv] = useState(null);
  useEffect(() => { api("/api/data/sources").then(setOv).catch(() => {}); }, []);
  if (!ov) return html`<div>loading…</div>`;
  return html`<div>
    <table><tr><th>source</th><th>kind</th><th>license</th><th>raw files</th><th>raw size</th><th>raw rows</th>${ov.tags.map((t) => html`<th>train tokens (${t})</th><th>val tokens (${t})</th><th>docs (${t})</th><th>dropped (${t})</th>`)}<th class="l">notes</th></tr>
    ${ov.sources.map((s) => html`<tr><td>${s.name}</td><td>${s.kind}</td><td>${s.license}</td><td>${s.raw_files}</td><td>${fmtBytes(s.raw_bytes)}</td><td>${fmtInt(s.raw_rows)}</td>
      ${ov.tags.map((t) => { const m = s.tokenized[t]; return m ? html`<td>${fmtTok(m.train_tokens)}</td><td>${fmtTok(m.val_tokens)}</td><td>${fmtInt(m.train_docs)}</td><td>${(m.docs_dropped / Math.max(1, m.docs_seen) * 100).toFixed(2)}%</td>` : html`<td colspan="4" class="muted">${(ov.sft && ov.sft[t] && Object.keys(ov.sft[t]).some((n) => n.startsWith(s.name) || (s.name === "gsm8k" && n.startsWith("gsm8k")))) ? "chat-formatted → see SFT table below" : "not tokenized"}</td>`; })}
      <td class="l">${s.notes}</td></tr>`)}
    </table>
    ${Object.keys(ov.sft || {}).length > 0 && html`<h2>SFT / reasoning sets (chat-formatted, with loss masks)</h2>
      ${Object.entries(ov.sft).map(([tag, sets]) => html`<div class="sub">tokenizer ${tag}</div>
      <table><tr><th>set</th><th>train examples</th><th>train tokens</th><th>loss targets</th><th>val examples</th><th>max len</th><th>think span</th><th>dropped / too long</th></tr>
      ${Object.entries(sets).map(([name, m]) => html`<tr><td>${name}</td><td>${fmtInt(m.train_examples)}</td><td>${fmtTok(m.train_tokens)}</td><td>${m.train_tokens ? (m.train_targets / m.train_tokens * 100).toFixed(0) + "%" : "-"}</td><td>${fmtInt(m.val_examples)}</td><td>${m.max_len || "-"}</td><td>${m.think_required ? "mandatory" : "no"}</td><td>${fmtInt(m.dropped || 0)} / ${fmtInt(m.too_long || 0)}</td></tr>`)}</table>`)}`}
  </div>`;
}

function Hist({ edges, counts, label }) {
  const max = Math.max(1, ...counts);
  return html`<div><h3>${label}</h3><div class="hist" style="margin-bottom:18px">${counts.map((c, i) => html`<div style=${"height:" + (c / max * 100).toFixed(1) + "%"} title=${`${fmtTok(edges[i])}+: ${fmtInt(c)}`}><span>${fmtTok(edges[i])}</span></div>`)}</div></div>`;
}

function Documents() {
  const [ov, setOv] = useState(null);
  const [tag, setTag] = useState(null);
  const [source, setSource] = useState(null);
  const [split, setSplit] = useState("train");
  const [shards, setShards] = useState([]);
  const [shard, setShard] = useState(0);
  const [docs, setDocs] = useState(null);
  const [offset, setOffset] = useState(0);
  const [doc, setDoc] = useState(null);
  const [stats, setStats] = useState(null);
  const [view, setView] = useState("doc");
  const [mode, setMode] = useState("text");
  const [win, setWin] = useState(null);
  const [winStart, setWinStart] = useState(0);
  const [winLen, setWinLen] = useState(1024);
  // raw-parquet state (sources without tokenized shards, e.g. chat sets)
  const [files, setFiles] = useState([]);
  const [file, setFile] = useState(0);
  const [rg, setRg] = useState(0);
  const [page, setPage] = useState(null);
  const [rawTokens, setRawTokens] = useState(null);

  useEffect(() => { api("/api/data/sources").then((o) => { setOv(o); if (o.tags.length) setTag(o.tags[o.tags.length - 1]); }); }, []);
  const tokenized = ov && tag ? ov.sources.filter((s) => s.tokenized[tag]).map((s) => s.name) : [];
  const rawOnly = ov && tag ? ov.sources.filter((s) => !s.tokenized[tag] && s.raw_files > 0).map((s) => s.name) : [];
  const isTok = tokenized.includes(source);
  useEffect(() => { if (ov && !source && tokenized.length) setSource(tokenized[0]); }, [ov, tag]);
  const base = isTok ? `/api/data/tokenized/${tag}/${source}/${split}` : null;
  useEffect(() => {
    setDoc(null); setWin(null); setStats(null); setDocs(null); setPage(null); setRawTokens(null); setShard(0); setOffset(0); setView("doc");
    if (!source) return;
    if (isTok) { api(`${base}/shards`).then(setShards).catch(() => setShards([])); api(`${base}/stats`).then(setStats).catch(() => setStats(null)); }
    else { setFile(0); setRg(0); api(`/api/data/raw/${source}/files`).then(setFiles).catch(() => setFiles([])); }
  }, [source, split, tag]);
  useEffect(() => { if (base) api(`${base}/docs?shard=${shard}&offset=${offset}&limit=200`).then(setDocs).catch(() => setDocs(null)); }, [base, shard, offset]);
  useEffect(() => { if (base && view === "window") api(`${base}/window?shard=${shard}&start=${winStart}&length=${winLen}`).then(setWin).catch(() => setWin(null)); }, [base, shard, winStart, winLen, view]);
  useEffect(() => { if (!isTok && source && files.length) api(`/api/data/raw/${source}/docs?file=${file}&rg=${rg}&limit=100`).then(setPage).catch(() => setPage(null)); }, [source, files, file, rg]);
  const [tokErr, setTokErr] = useState(null);
  useEffect(() => {  // raw docs: tokenize on demand when the tokens view is chosen (chat rows go through the SFT chat formatter)
    if (!doc || isTok || mode === "text" || !tag) { setRawTokens(null); setTokErr(null); return; }
    const body = doc.messages ? { mode: "chat", messages: doc.messages } : { mode: "document", text: doc.text };
    api(`/api/tokenizers/${tag}/encode`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }).then((r) => { setRawTokens(r); setTokErr(null); }).catch((e) => { setRawTokens(null); setTokErr(String(e)); });
  }, [doc, mode, isTok, tag]);

  const fillRef = useRef(null);
  useEffect(() => {  // size the list/preview pair to the space left below the toolbar, and keep it sized as the window changes
    const el = fillRef.current;
    if (!el) return;
    const fit = () => { const top = el.getBoundingClientRect().top + window.scrollY; el.style.height = Math.max(360, window.innerHeight - top - 26) + "px"; };
    fit();
    addEventListener("resize", fit);
    return () => removeEventListener("resize", fit);
  });
  const openTok = (d) => api(`${base}/doc?shard=${shard}&doc=${d}`).then((x) => { setDoc(x); setView("doc"); });
  const openRaw = (f, r, row) => api(`/api/data/raw/${source}/doc?file=${f}&rg=${r}&row=${row}`).then((x) => { setDoc(x); setView("doc"); });
  const random = () => {
    if (isTok) { if (docs) openTok(Math.floor(Math.random() * docs.n_docs)); }
    else api(`/api/data/raw/${source}/sample?n=1&seed=${Math.floor(Math.random() * 1e6)}`).then((r) => r.length && openRaw(r[0].file, r[0].rg, r[0].row));
  };
  if (!ov) return html`<div>loading…</div>`;
  const f = files[file];
  const pieces = doc ? (isTok ? doc.pieces : rawTokens && rawTokens.pieces) : null;
  return html`<div>
    <div class="row" style="margin-bottom:8px">
      <select value=${tag} onChange=${(e) => { setTag(e.target.value); setSource(null); }}>${ov.tags.map((t) => html`<option value=${t}>${t}</option>`)}</select>
      <select value=${source} onChange=${(e) => setSource(e.target.value)}>
        <optgroup label="tokenized (pretraining shards)">${tokenized.map((s) => html`<option value=${s}>${s}</option>`)}</optgroup>
        <optgroup label="raw parquet only">${rawOnly.map((s) => html`<option value=${s}>${s}</option>`)}</optgroup>
      </select>
      ${isTok ? html`<select value=${split} onChange=${(e) => setSplit(e.target.value)}><option>train</option><option>val</option></select>
        <select value=${shard} onChange=${(e) => { setShard(Number(e.target.value)); setOffset(0); }}>${shards.map((s) => html`<option value=${s.shard}>shard ${s.shard} · ${fmtTok(s.tokens)} tok · ${fmtInt(s.docs)} docs</option>`)}</select>`
      : html`<select value=${file} onChange=${(e) => { setFile(Number(e.target.value)); setRg(0); }}>${files.map((x) => html`<option value=${x.index}>${x.name} · ${fmtInt(x.rows)} rows · ${x.row_groups} row groups</option>`)}</select>
        ${f && html`<span class="muted">row group</span><input type="number" min="0" max=${f.row_groups - 1} value=${rg} onChange=${(e) => setRg(Math.max(0, Math.min(f.row_groups - 1, Number(e.target.value))))} style="width:80px" />`}`}
      <button onClick=${random}>random doc</button>
      ${isTok && ["doc", "window", "stats"].map((v) => html`<button class=${view === v ? "active" : ""} onClick=${() => setView(v)}>${v}</button>`)}
      <span class="muted" style="margin-left:10px">show as</span>
      ${["text", "tokens", "ids"].map((m) => html`<button class=${mode === m ? "active" : ""} onClick=${() => setMode(m)}>${m}</button>`)}
    </div>
    <div class="two fill" ref=${fillRef}>
      <div class="col">
        ${isTok && docs && html`<div class="row" style="margin-bottom:4px"><span class="muted">${fmtInt(docs.n_docs)} docs in shard; showing ${offset}–${Math.min(offset + 200, docs.n_docs)}</span>
          <button onClick=${() => setOffset(Math.max(0, offset - 200))}>‹</button><button onClick=${() => setOffset(Math.min(docs.n_docs - 1, offset + 200))}>›</button></div>
          <table><tr><th>doc</th><th>start</th><th>tokens</th></tr>
          ${docs.docs.map((d) => html`<tr class=${"click" + (doc && doc.doc === d.doc ? " sel" : "")} onClick=${() => openTok(d.doc)}><td>${d.doc}</td><td>${fmtInt(d.start)}</td><td>${fmtInt(d.length)}</td></tr>`)}</table>`}
        ${!isTok && page && html`<table><tr><th>row</th><th>chars</th><th class="l">preview</th></tr>
          ${page.docs.map((d) => html`<tr class="click" onClick=${() => openRaw(file, rg, d.row)}><td>${d.row}</td><td>${fmtInt(d.chars)}${d.turns ? html`<div class="legend">${d.turns} turns</div>` : ""}</td><td class="l">${d.preview.slice(0, 140)}</td></tr>`)}</table>`}
        ${!isTok && !page && html`<div class="empty-note">no files</div>`}
      </div>
      <div class="col">
        ${view === "doc" && (!doc ? html`<div class="empty-note">pick a document (or "random doc")</div>` : html`<div class="grow">
          <div class="sub">${isTok ? `doc ${doc.doc} · starts at token ${fmtInt(doc.start)} · ${fmtInt(doc.length)} tokens incl. bos/eos · ${fmtInt(doc.text.length)} chars` : `file ${doc.file} · row group ${doc.rg} · row ${doc.row} · ${fmtInt(doc.text.length)} chars${rawTokens ? ` · ${fmtInt(rawTokens.n_tokens)} tokens` : ""}`}</div>
          ${!isTok && doc.meta && html`<table style="margin-bottom:8px">${Object.entries(doc.meta).map(([k, v]) => html`<tr><td>${k}</td><td class="l">${String(v).slice(0, 200)}</td></tr>`)}</table>`}
          ${mode === "text" ? html`<pre class="grow">${doc.text}</pre>` : pieces ? html`<div class="grow">
              <${TokenChips} pieces=${pieces} showIds=${mode === "ids"} lossMask=${!isTok && !!doc.messages} />
              ${!isTok && doc.messages && html`<div class="legend" style="margin-top:6px">exactly what SFT trains on: chat format with reserved tokens; <b style="color:#15803d">green</b> = loss target (assistant turns + ${"<|end|>"}), grey = masked · ${rawTokens.n_tokens} tokens, ${rawTokens.n_target} targets</div>`}
            </div>` : tokErr ? html`<div class="panel" style="border-color:#fca5a5;color:#b91c1c">tokenization failed: ${tokErr}</div>` : html`<div class="empty-note">tokenizing…</div>`}
        </div>`)}
        ${view === "window" && html`<div class="grow">
          <div class="row" style="margin-bottom:6px"><span class="muted">window start</span><input type="number" value=${winStart} step=${winLen} min="0" onChange=${(e) => setWinStart(Math.max(0, Number(e.target.value)))} style="width:140px" />
            <span class="muted">length</span><select value=${winLen} onChange=${(e) => setWinLen(Number(e.target.value))}>${[256, 1024, 2048, 4096, 8192].map((n) => html`<option value=${n}>${n}</option>`)}</select>
            <button onClick=${() => setWinStart(Math.max(0, winStart - winLen))}>‹ prev</button><button onClick=${() => setWinStart(winStart + winLen)}>next ›</button>
            ${win && html`<span class="muted">${win.doc_starts.length} document boundaries (red) · shard has ${fmtTok(win.shard_tokens)} tokens</span>`}</div>
          ${win ? (mode === "text"
              ? html`<pre class="grow">${win.pieces.map((p) => (p.special ? html`<b class="boundary-mark">${p.piece}</b>` : p.piece))}</pre>`
              : html`<${TokenChips} pieces=${win.pieces} boundaries=${win.doc_starts} showIds=${mode === "ids"} />`) : html`<div class="empty-note">…</div>`}
          <div class="legend" style="margin-top:6px">This is exactly what one training row of this length looks like: a contiguous slice of the token stream, which may start mid-document; ${"<|bos|>"}/${"<|eos|>"} mark boundaries (red in both views).</div></div>`}
        ${view === "stats" && (stats ? html`<div>
          <div class="tiles">
            <div class="tile"><div class="k">docs</div><div class="v">${fmtInt(stats.docs)}</div></div>
            <div class="tile"><div class="k">tokens</div><div class="v">${fmtTok(stats.tokens)}</div></div>
            <div class="tile"><div class="k">mean tokens/doc</div><div class="v">${fmtNum(stats.mean_len, 0)}</div><div class="s">p10 ${fmtNum(stats.p10, 0)} · p50 ${fmtNum(stats.p50, 0)} · p90 ${fmtNum(stats.p90, 0)} · p99 ${fmtNum(stats.p99, 0)}</div></div>
            <div class="tile"><div class="k">docs ≥ 4K tokens</div><div class="v">${fmtInt(stats.docs_over_4k)}</div><div class="s">${fmtTok(stats.tokens_over_4k)} tokens (${(stats.tokens_over_4k / Math.max(1, stats.tokens) * 100).toFixed(1)}%)</div></div>
            <div class="tile"><div class="k">docs ≥ 8K tokens</div><div class="v">${fmtInt(stats.docs_over_8k)}</div><div class="s">${fmtTok(stats.tokens_over_8k)} tokens (${(stats.tokens_over_8k / Math.max(1, stats.tokens) * 100).toFixed(1)}%)</div></div>
          </div>
          <${Hist} edges=${stats.hist_edges} counts=${stats.hist} label="documents by length (tokens, log bins)" />
          <${Hist} edges=${stats.hist_edges} counts=${stats.tokens_by_bin} label="tokens by document-length bin" />
        </div>` : html`<div class="empty-note">computing stats…</div>`)}
      </div>
    </div>
  </div>`;
}

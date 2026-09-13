import { h } from "preact";
import { useEffect, useState } from "preact/hooks";
import htm from "htm";
import { api, fmtTok, fmtInt, fmtBytes, fmtNum } from "../components/util.js";
import { TokenChips } from "../components/tokens.js";

const html = htm.bind(h);

export function DataPage() {
  const [tab, setTab] = useState("sources");
  return html`<div>
    <h1>Data</h1>
    <div class="row" style="margin:8px 0">${["sources", "mixture", "tokenized", "raw"].map((x) => html`<button class=${tab === x ? "active" : ""} onClick=${() => setTab(x)}>${x}</button>`)}</div>
    ${tab === "sources" && html`<${Sources} />`}
    ${tab === "mixture" && html`<${Mixture} />`}
    ${tab === "tokenized" && html`<${Tokenized} />`}
    ${tab === "raw" && html`<${Raw} />`}
  </div>`;
}

function Sources() {
  const [ov, setOv] = useState(null);
  useEffect(() => { api("/api/data/sources").then(setOv).catch(() => {}); }, []);
  if (!ov) return html`<div>loading…</div>`;
  return html`<div>
    <table><tr><th>source</th><th>kind</th><th>license</th><th>raw files</th><th>raw size</th><th>raw rows</th>${ov.tags.map((t) => html`<th>train tokens (${t})</th><th>val tokens (${t})</th><th>docs (${t})</th><th>dropped (${t})</th>`)}<th class="l">notes</th></tr>
    ${ov.sources.map((s) => html`<tr><td>${s.name}</td><td>${s.kind}</td><td>${s.license}</td><td>${s.raw_files}</td><td>${fmtBytes(s.raw_bytes)}</td><td>${fmtInt(s.raw_rows)}</td>
      ${ov.tags.map((t) => { const m = s.tokenized[t]; return m ? html`<td>${fmtTok(m.train_tokens)}</td><td>${fmtTok(m.val_tokens)}</td><td>${fmtInt(m.train_docs)}</td><td>${(m.docs_dropped / Math.max(1, m.docs_seen) * 100).toFixed(2)}%</td>` : html`<td colspan="4" class="muted">not tokenized</td>`; })}
      <td class="l">${s.notes}</td></tr>`)}
    </table>
  </div>`;
}

function Mixture() {
  const [configs, setConfigs] = useState([]);
  const [cfg, setCfg] = useState(null);
  const [mix, setMix] = useState(null);
  useEffect(() => { api("/api/data/configs").then((c) => { setConfigs(c); if (c.length) setCfg(c[0]); }); }, []);
  useEffect(() => { if (cfg) api(`/api/data/mixture?config=${encodeURIComponent(cfg)}`).then(setMix).catch(() => setMix(null)); }, [cfg]);
  return html`<div>
    <div class="row"><span class="muted">train config:</span><select value=${cfg} onChange=${(e) => setCfg(e.target.value)}>${configs.map((c) => html`<option value=${c}>${c}</option>`)}</select></div>
    ${mix && html`<div style="margin-top:10px"><div class="sub">tokenizer ${mix.tag} · ${fmtTok(mix.total_tokens)} planned tokens · seq_len ${mix.seq_len}</div>
      <table><tr><th>source</th><th>weight</th><th>planned tokens</th><th>available (train)</th><th>epochs</th><th>val tokens</th></tr>
      ${mix.rows.map((r) => html`<tr><td>${r.source}</td><td>${(r.weight * 100).toFixed(1)}%</td><td>${fmtTok(r.planned_tokens)}</td><td>${fmtTok(r.available_tokens)}</td>
        <td style=${r.epochs > 1.5 ? "color:#b91c1c;font-weight:600" : ""}>${r.epochs == null ? "no data" : r.epochs.toFixed(2)}</td><td>${fmtTok(r.val_tokens)}</td></tr>`)}</table>
      <div class="legend" style="margin-top:6px">epochs &gt; 1.5 (red) means that source will be repeated; consider downloading more of it.</div></div>`}
  </div>`;
}

function Hist({ edges, counts, label }) {
  const max = Math.max(1, ...counts);
  return html`<div><h3>${label}</h3><div class="hist" style="margin-bottom:18px">${counts.map((c, i) => html`<div style=${"height:" + (c / max * 100).toFixed(1) + "%"} title=${`${fmtTok(edges[i])}+: ${fmtInt(c)}`}><span>${fmtTok(edges[i])}</span></div>`)}</div></div>`;
}

function Tokenized() {
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
  const [win, setWin] = useState(null);
  const [winStart, setWinStart] = useState(0);
  const [winLen, setWinLen] = useState(1024);
  const [showIds, setShowIds] = useState(false);
  useEffect(() => { api("/api/data/sources").then((o) => { setOv(o); if (o.tags.length) { setTag(o.tags[o.tags.length - 1]); } }); }, []);
  const sources = ov && tag ? ov.sources.filter((s) => s.tokenized[tag]).map((s) => s.name) : [];
  useEffect(() => { if (sources.length && !source) setSource(sources[0]); }, [ov, tag]);
  const base = tag && source ? `/api/data/tokenized/${tag}/${source}/${split}` : null;
  useEffect(() => { if (!base) return; setShard(0); setOffset(0); setDoc(null); setWin(null); api(`${base}/shards`).then(setShards).catch(() => setShards([])); api(`${base}/stats`).then(setStats).catch(() => setStats(null)); }, [base]);
  useEffect(() => { if (base) api(`${base}/docs?shard=${shard}&offset=${offset}&limit=200`).then(setDocs).catch(() => setDocs(null)); }, [base, shard, offset]);
  useEffect(() => { if (base && view === "window") api(`${base}/window?shard=${shard}&start=${winStart}&length=${winLen}`).then(setWin).catch(() => setWin(null)); }, [base, shard, winStart, winLen, view]);
  const openDoc = (d) => api(`${base}/doc?shard=${shard}&doc=${d}`).then((x) => { setDoc(x); setView("doc"); });
  const randomDoc = () => { if (docs) openDoc(Math.floor(Math.random() * docs.n_docs)); };
  if (!ov) return html`<div>loading…</div>`;
  return html`<div>
    <div class="row" style="margin-bottom:8px">
      <select value=${tag} onChange=${(e) => { setTag(e.target.value); setSource(null); }}>${ov.tags.map((t) => html`<option value=${t}>${t}</option>`)}</select>
      <select value=${source} onChange=${(e) => setSource(e.target.value)}>${sources.map((s) => html`<option value=${s}>${s}</option>`)}</select>
      <select value=${split} onChange=${(e) => setSplit(e.target.value)}><option>train</option><option>val</option></select>
      <select value=${shard} onChange=${(e) => { setShard(Number(e.target.value)); setOffset(0); }}>${shards.map((s) => html`<option value=${s.shard}>shard ${s.shard} · ${fmtTok(s.tokens)} tok · ${fmtInt(s.docs)} docs</option>`)}</select>
      <button onClick=${randomDoc}>random doc</button>
      ${["doc", "window", "stats"].map((v) => html`<button class=${view === v ? "active" : ""} onClick=${() => setView(v)}>${v}</button>`)}
      <button class=${showIds ? "active" : ""} onClick=${() => setShowIds(!showIds)}>ids</button>
    </div>
    <div class="two">
      <div>
        ${docs && html`<div class="row" style="margin-bottom:4px"><span class="muted">${fmtInt(docs.n_docs)} docs in shard; showing ${offset}–${Math.min(offset + 200, docs.n_docs)}</span>
          <button onClick=${() => setOffset(Math.max(0, offset - 200))}>‹</button><button onClick=${() => setOffset(Math.min(docs.n_docs - 1, offset + 200))}>›</button></div>
          <div style="max-height:560px;overflow:auto"><table><tr><th>doc</th><th>start</th><th>tokens</th></tr>
          ${docs.docs.map((d) => html`<tr class="click" onClick=${() => openDoc(d.doc)}><td>${d.doc}</td><td>${fmtInt(d.start)}</td><td>${fmtInt(d.length)}</td></tr>`)}</table></div>`}
      </div>
      <div>
        ${view === "doc" && (doc ? html`<div class="sub">doc ${doc.doc} · starts at token ${fmtInt(doc.start)} · ${fmtInt(doc.length)} tokens (incl. bos/eos)</div><${TokenChips} pieces=${doc.pieces} showIds=${showIds} />` : html`<div class="empty-note">pick a document</div>`)}
        ${view === "window" && html`<div>
          <div class="row" style="margin-bottom:6px"><span class="muted">window start</span><input type="number" value=${winStart} step=${winLen} min="0" onChange=${(e) => setWinStart(Math.max(0, Number(e.target.value)))} style="width:140px" />
            <span class="muted">length</span><select value=${winLen} onChange=${(e) => setWinLen(Number(e.target.value))}>${[256, 1024, 2048, 4096, 8192].map((n) => html`<option value=${n}>${n}</option>`)}</select>
            <button onClick=${() => setWinStart(Math.max(0, winStart - winLen))}>‹ prev</button><button onClick=${() => setWinStart(winStart + winLen)}>next ›</button>
            ${win && html`<span class="muted">${win.doc_starts.length} document boundaries (red) · shard has ${fmtTok(win.shard_tokens)} tokens</span>`}</div>
          ${win ? html`<${TokenChips} pieces=${win.pieces} boundaries=${win.doc_starts} showIds=${showIds} />` : html`<div class="empty-note">…</div>`}
          <div class="legend" style="margin-top:6px">This is exactly what one training row of this length looks like: a contiguous slice of the token stream, which may start mid-document; <|bos|>/<|eos|> mark boundaries.</div></div>`}
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

function Raw() {
  const [ov, setOv] = useState(null);
  const [source, setSource] = useState(null);
  const [files, setFiles] = useState([]);
  const [file, setFile] = useState(0);
  const [rg, setRg] = useState(0);
  const [page, setPage] = useState(null);
  const [doc, setDoc] = useState(null);
  const [sample, setSample] = useState(null);
  useEffect(() => { api("/api/data/sources").then((o) => { setOv(o); const withRaw = o.sources.filter((s) => s.raw_files > 0); if (withRaw.length) setSource(withRaw[0].name); }); }, []);
  useEffect(() => { if (source) { setFile(0); setRg(0); setDoc(null); setSample(null); api(`/api/data/raw/${source}/files`).then(setFiles).catch(() => setFiles([])); } }, [source]);
  useEffect(() => { if (source && files.length) api(`/api/data/raw/${source}/docs?file=${file}&rg=${rg}&limit=100`).then(setPage).catch(() => setPage(null)); }, [source, files, file, rg]);
  const open = (f, r, row) => api(`/api/data/raw/${source}/doc?file=${f}&rg=${r}&row=${row}`).then(setDoc);
  if (!ov) return html`<div>loading…</div>`;
  const f = files[file];
  return html`<div>
    <div class="row" style="margin-bottom:8px">
      <select value=${source} onChange=${(e) => setSource(e.target.value)}>${ov.sources.filter((s) => s.raw_files > 0).map((s) => html`<option value=${s.name}>${s.name}</option>`)}</select>
      <select value=${file} onChange=${(e) => { setFile(Number(e.target.value)); setRg(0); }}>${files.map((x) => html`<option value=${x.index}>${x.name} · ${fmtInt(x.rows)} rows · ${x.row_groups} row groups${x.id_only ? " (ids only)" : ""}</option>`)}</select>
      ${f && html`<span class="muted">row group</span><input type="number" min="0" max=${f.row_groups - 1} value=${rg} onChange=${(e) => setRg(Math.max(0, Math.min(f.row_groups - 1, Number(e.target.value))))} style="width:90px" />`}
      <button onClick=${() => api(`/api/data/raw/${source}/sample?n=12&seed=${Math.floor(Math.random() * 1e6)}`).then(setSample)}>random sample</button>
    </div>
    <div class="two">
      <div style="max-height:600px;overflow:auto">
        ${sample ? html`<div class="sub">random sample across files and row groups</div><table><tr><th>file</th><th>rg</th><th>row</th><th class="l">preview</th></tr>
          ${sample.map((d) => html`<tr class="click" onClick=${() => open(d.file, d.rg, d.row)}><td>${d.file}</td><td>${d.rg}</td><td>${d.row}</td><td class="l">${d.preview.slice(0, 160)}</td></tr>`)}</table>`
        : page ? html`<table><tr><th>row</th><th>chars</th><th class="l">preview</th></tr>
          ${page.docs.map((d) => html`<tr class="click" onClick=${() => open(file, rg, d.row)}><td>${d.row}</td><td>${fmtInt(d.chars)}</td><td class="l">${d.preview.slice(0, 160)}</td></tr>`)}</table>` : html`<div class="empty-note">no files</div>`}
      </div>
      <div>${doc ? html`<div class="sub">file ${doc.file} · row group ${doc.rg} · row ${doc.row} · ${fmtInt(doc.text.length)} chars</div>
        <table style="margin-bottom:8px">${Object.entries(doc.meta).map(([k, v]) => html`<tr><td>${k}</td><td class="l">${String(v).slice(0, 200)}</td></tr>`)}</table>
        <pre style="max-height:600px">${doc.text}</pre>` : html`<div class="empty-note">pick a document</div>`}</div>
    </div>
  </div>`;
}

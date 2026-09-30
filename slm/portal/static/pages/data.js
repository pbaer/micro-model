import { h } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import htm from "htm";
import { api, fmtTok, fmtInt, fmtBytes, fmtNum } from "../components/util.js";
import { TokenChips } from "../components/tokens.js";
import { Info } from "../components/info.js";

const html = htm.bind(h);

// Routes (the hash parts after "#/data"): [] = recipe list · ["recipe"|"recipes", <encoded id>] = one recipe ·
// ["compare", <id a>, <id b>] · ["chain", "run:<name>"] · ["catalog"] = the source list ·
// ["source", <name>] = one source. Recipe ids carry ":" and "/", so they travel encoded.
export const dataHref = (...parts) => "#/data" + parts.map((p) => "/" + encodeURIComponent(p)).join("");

export function DataPage({ parts = [] }) {
  const route = parts[0] || "recipes";
  const tabs = [["recipes", "recipes", dataHref()], ["catalog", "catalog", dataHref("catalog")]];
  return html`<div>
    <h1>Data</h1>
    <div class="row" style="margin:8px 0">${tabs.map(([id, label, href]) => html`<a href=${href}><button class=${route === id ? "active" : ""}>${label}</button></a>`)}</div>
    ${route === "recipes" && (parts[1] ? html`<${Recipe} id=${decodeURIComponent(parts[1])} key=${parts[1]} />` : html`<${RecipeList} />`)}
    ${route === "recipe" && html`<${Recipe} id=${decodeURIComponent(parts[1] || "")} key=${parts[1]} />`}
    ${route === "compare" && html`<${Compare} a=${decodeURIComponent(parts[1] || "")} b=${decodeURIComponent(parts[2] || "")} key=${parts.join("|")} />`}
    ${route === "chain" && html`<${ChainView} id=${decodeURIComponent(parts[1] || "")} key=${parts[1]} />`}
    ${route === "catalog" && html`<${Catalog} />`}
    ${route === "source" && html`<${SourcePage} name=${decodeURIComponent(parts[1] || "")} key=${parts[1]} />`}
  </div>`;
}

/** run name out of an init_from path ("runs/m3_base_8k_149m/checkpoints/final.pt" -> "m3_base_8k_149m") */
export function parentRun(initFrom) {
  const p = String(initFrom || "").replace(/\\/g, "/").split("/");
  const i = p.lastIndexOf("checkpoints");
  return i > 0 ? p[i - 1] : null;
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
    <table><tr><th>stage</th><th class="l">recipe<${Info} k="recipe" /></th><th>status<${Info} k="run_status" /></th><th>sources</th><th>seq<${Info} k="seq_len" /></th><th>tokens</th><th class="l">init_from<${Info} k="init_from" /></th><th class="l">plan<${Info} k="plan_differs" /></th></tr>
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
function MixTable({ rows, showPlanned, streams, sel, onPick }) {
  return html`<table><tr><th class="l">source</th><th>weight<${Info} k="mix_weight" /></th>${showPlanned ? html`<th>planned<${Info} k="planned" /></th>` : ""}<th>available (train)<${Info} k="available" /></th><th>epochs<${Info} k="epochs" /></th>${streams ? html`<th>consumed<${Info} k="consumed" /></th>` : ""}<th>val tokens<${Info} k="val_tokens" /></th><th class="l">prepared from<${Info} k="provenance" /></th></tr>
    ${rows.map((r) => { const st = streams && streams[r.source]; return html`<tr class=${(onPick && !r.missing ? "click" : "") + (sel === r.source ? " sel" : "")} onClick=${onPick && !r.missing ? () => onPick(r.source) : null}>
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

/** Sizes a .two.fill pair to the space left below it, and keeps it sized as the window changes. */
function useFill() {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const fit = () => { const top = el.getBoundingClientRect().top + window.scrollY; el.style.height = Math.max(360, window.innerHeight - top - 26) + "px"; };
    fit();
    addEventListener("resize", fit);
    return () => removeEventListener("resize", fit);
  });
  return ref;
}

/** raw record -> prepared record -> training row, for one source of one recipe. */
function Inspector({ row, tag, seqLen }) {
  const isSft = row.kind === "sft";
  const [split, setSplit] = useState("train");
  const [shards, setShards] = useState([]);
  const [shard, setShard] = useState(0);
  const [list, setList] = useState(null);
  const [offset, setOffset] = useState(0);
  const [item, setItem] = useState(null);
  const [sub, setSub] = useState("prepared");
  const [mode, setMode] = useState("tokens");
  const [win, setWin] = useState(null);
  const [winStart, setWinStart] = useState(0);
  const [raw, setRaw] = useState(null);
  const [trace, setTrace] = useState(null);
  const base = `/api/data/${isSft ? "sft" : "tokenized"}/${tag}/${row.source}/${split}`;
  const len = seqLen || 2048;
  useEffect(() => { setShard(0); setOffset(0); setItem(null); setWin(null); setWinStart(0); setTrace(null); api(`${base}/shards`).then(setShards).catch(() => setShards([])); }, [base]);
  useEffect(() => { api(`${base}/${isSft ? "examples" : "docs"}?shard=${shard}&offset=${offset}&limit=200`).then(setList).catch(() => setList(null)); }, [base, shard, offset]);
  useEffect(() => { if (sub === "row") api(`${base}/window?shard=${shard}&start=${winStart}&length=${len + 1}`).then(setWin).catch(() => setWin(null)); }, [base, shard, winStart, sub, len]);
  useEffect(() => {  // the raw column needs a parquet row; sample one from the source this set was prepared from
    if (sub !== "raw" || !row.raw) { return; }
    if (raw) return;
    api(`/api/data/raw/${row.raw.source}/sample?n=1&seed=${Math.floor(Math.random() * 1e6)}`).then((s) => s.length && api(`/api/data/raw/${row.raw.source}/doc?file=${s[0].file}&rg=${s[0].rg}&row=${s[0].row}`).then(setRaw)).catch(() => {});
  }, [sub, row.raw, raw]);
  const open = (i) => api(`${base}/${isSft ? "example" : "doc"}?shard=${shard}&${isSft ? "ex" : "doc"}=${i}`).then((x) => { setItem(x); setSub("prepared"); });
  const doTrace = () => {
    if (!raw) return;
    setTrace("…");
    api(`/api/data/raw/${row.raw.source}/trace`, { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ file: raw.file, rg: raw.rg, row: raw.row, tag, sft_set: isSft ? row.source : null }) }).then(setTrace).catch((e) => setTrace({ reason: String(e) }));
  };
  const fillRef = useFill();
  const items = list ? (isSft ? list.examples : list.docs) : [];
  const n = list ? (isSft ? list.n_examples : list.n_docs) : 0;
  const pieces = item && item.pieces;
  return html`<div>
    <div class="row" style="margin:8px 0 6px">
      <b>${row.source}</b>
      <select value=${split} onChange=${(e) => setSplit(e.target.value)}><option>train</option><option>val</option></select>
      <select value=${shard} onChange=${(e) => { setShard(Number(e.target.value)); setOffset(0); setItem(null); }}>
        ${shards.map((s) => html`<option value=${s.shard}>shard ${s.shard} · ${fmtTok(s.tokens)} tok · ${fmtInt(isSft ? s.examples : s.docs)} ${isSft ? "examples" : "docs"}</option>`)}</select>
      ${["raw", "prepared", "row"].map((v) => html`<button class=${sub === v ? "active" : ""} onClick=${() => setSub(v)}>${v}</button>`)}<${Info} k="inspector_views" />
      <span class="muted" style="margin-left:8px">show as<${Info} k="show_as" /></span>
      ${["text", "tokens", "ids"].map((m) => html`<button class=${mode === m ? "active" : ""} onClick=${() => setMode(m)}>${m}</button>`)}
    </div>
    <div class="two fill" ref=${fillRef}>
      <div class="col">
        ${list && html`<div class="row" style="margin-bottom:4px"><span class="muted">${fmtInt(n)} ${isSft ? "examples" : "docs"}; showing ${offset}–${Math.min(offset + 200, n)}</span>
          <button onClick=${() => { setOffset(Math.max(0, offset - 200)); }}>‹</button><button onClick=${() => { setOffset(Math.min(Math.max(0, n - 1), offset + 200)); }}>›</button>
          <button onClick=${() => n && open(Math.floor(Math.random() * n))}>random</button></div>
          <table><tr><th>${isSft ? "example" : "doc"}</th><th>start</th><th>tokens</th></tr>
          ${items.map((d) => { const i = isSft ? d.ex : d.doc; return html`<tr class=${"click" + (item && (isSft ? item.ex : item.doc) === i ? " sel" : "")} onClick=${() => open(i)}><td>${i}</td><td>${fmtInt(d.start)}</td><td>${fmtInt(d.length)}</td></tr>`; })}</table>`}
      </div>
      <div class="col">
        ${sub === "prepared" && (!item ? html`<div class="empty-note">pick ${isSft ? "an example" : "a document"} on the left (or "random")</div>` : html`<div class="grow">
          <div class="sub">${isSft ? `example ${item.ex}` : `doc ${item.doc}`} · starts at token ${fmtInt(item.start)} · ${fmtInt(item.length)} tokens${isSft ? ` · ${fmtInt(item.n_target)} loss targets (${(item.n_target / Math.max(1, item.length) * 100).toFixed(0)}%)` : " incl. bos/eos"}</div>
          ${mode === "text" ? html`<pre class="grow">${item.text}</pre>` : html`<div class="grow"><${TokenChips} pieces=${pieces} showIds=${mode === "ids"} lossMask=${isSft} /></div>`}
          <div class="legend" style="margin-top:6px">${isSft
            ? html`The stored SFT example, exactly as the trainer reads it: <b style="color:#15803d">green</b> = loss target (assistant content and ${"<|end|>"}), grey = masked.`
            : "The stored pretraining document, bos/eos included."}</div></div>`)}
        ${sub === "row" && html`<div class="grow">
          <div class="row" style="margin-bottom:6px"><span class="muted">window start</span>
            <input type="number" value=${winStart} step=${len} min="0" onChange=${(e) => setWinStart(Math.max(0, Number(e.target.value)))} style="width:140px" />
            <button onClick=${() => setWinStart(Math.max(0, winStart - len))}>‹ prev</button><button onClick=${() => setWinStart(winStart + len)}>next ›</button>
            <button onClick=${() => win && setWinStart(Math.floor(Math.random() * Math.max(1, win.shard_tokens - len)))}>random</button>
            ${win && html`<span class="muted">${(isSft ? win.example_starts : win.doc_starts).length} ${isSft ? "example" : "document"} boundaries (red)${isSft ? ` · ${fmtInt(win.n_target)} loss targets` : ""} · shard has ${fmtTok(win.shard_tokens)} tokens</span>`}</div>
          ${win ? (mode === "text"
            ? html`<pre class="grow">${win.pieces.map((p) => (p.special ? html`<b class="boundary-mark">${p.piece}</b>` : p.piece))}</pre>`
            : html`<${TokenChips} pieces=${win.pieces} boundaries=${isSft ? win.example_starts : win.doc_starts} showIds=${mode === "ids"} lossMask=${isSft} />`) : html`<div class="empty-note">…</div>`}
          <div class="legend" style="margin-top:6px">One training row of ${fmtInt(len)} + 1 tokens, exactly as the loader cuts it: a contiguous slice that may start mid-${isSft ? "example" : "document"}.
            ${isSft ? html` The loss denominator is the number of green positions.` : html` Every token is a target; this source contributes ${(row.weight * 100).toFixed(1)}% of rows.`}</div></div>`}
        ${sub === "raw" && html`<div class="grow">
          ${!row.raw ? html`<div class="panel"><b>No raw parquet for this source.</b><div class="legend" style="margin-top:6px">${row.provenance}</div></div>`
            : !raw ? html`<div class="empty-note">sampling a row from ${row.raw.source}…</div>` : html`<div class="grow">
            <div class="sub">${row.raw.source} · file ${raw.file} · row group ${raw.rg} · row ${raw.row} · ${fmtInt(raw.text.length)} chars
              <button style="margin-left:8px" onClick=${() => { setRaw(null); setTrace(null); }}>another row</button>
              <button onClick=${doTrace}>trace through preparation</button></div>
            ${trace && trace !== "…" && html`<div class="panel" style=${"margin-bottom:6px;border-color:" + (trace.kept ? "#86efac" : "#fca5a5")}>${trace.reason}</div>`}
            ${trace && trace.pieces && mode !== "text" ? html`<div class="grow"><${TokenChips} pieces=${trace.pieces} showIds=${mode === "ids"} lossMask=${isSft} /></div>`
              : html`<pre class="grow">${raw.text}</pre>`}
            <div class="legend" style="margin-top:6px">There is no stored row-to-document mapping, so "trace" re-derives the result with the same tokenizer and the same filters rather than looking it up.${isSft ? " For SFT the stored example on the left is the ground truth; the marker/natural style is drawn from a per-set RNG that cannot be replayed for one row." : ""}</div>
          </div>`}
        </div>`}
      </div>
    </div>
  </div>`;
}

/** The deterministic prompt list, rendered as the exact generation prompt the trainer builds. */
function RlPrompts({ id, thinkRequired }) {
  const [split, setSplit] = useState("train");
  const [p, setP] = useState(null);
  const [offset, setOffset] = useState(0);
  const [sel, setSel] = useState(0);
  const [mode, setMode] = useState("tokens");
  useEffect(() => { setP(null); setSel(0); api(`/api/data/rl/prompts?id=${encodeURIComponent(id)}&split=${split}&offset=${offset}&limit=20`).then(setP).catch(() => setP({ prompts: [], n: 0 })); }, [id, split, offset]);
  const cur = p && p.prompts[sel];
  return html`<div><h2>Prompt sample<${Info} k="rl_prompt_sample" /></h2>
    <div class="row" style="margin-bottom:6px">
      ${["train", "heldout"].map((s) => html`<button class=${split === s ? "active" : ""} onClick=${() => { setSplit(s); setOffset(0); }}>${s}</button>`)}
      <button onClick=${() => setOffset(Math.max(0, offset - 20))}>‹</button><button onClick=${() => setOffset(offset + 20)}>›</button>
      ${p && html`<span class="muted">${fmtInt(p.n)} ${split} prompts; showing ${offset}–${Math.min(offset + 20, p.n)}</span>`}
      <span class="muted" style="margin-left:8px">show as<${Info} k="show_as" /></span>
      ${["text", "tokens", "ids"].map((m) => html`<button class=${mode === m ? "active" : ""} onClick=${() => setMode(m)}>${m}</button>`)}
    </div>
    ${!p ? html`<div class="empty-note">…</div>` : !p.prompts.length ? html`<div class="empty-note">no prompts for this split</div>` : html`<div class="two fill" style="height:420px">
      <div class="col"><table><tr><th>task</th><th class="l">prompt</th><th>gold</th></tr>
        ${p.prompts.map((x, i) => html`<tr class=${"click" + (i === sel ? " sel" : "")} onClick=${() => setSel(i)}><td>${x.task}</td><td class="l">${x.prompt.slice(0, 110)}</td><td>${x.gold}</td></tr>`)}</table></div>
      <div class="col">${cur && html`<div class="grow">
        <div class="sub">${cur.prompt_id} · ${cur.task} · ${fmtInt(cur.n_tokens)} tokens · gold <b>${cur.gold}</b></div>
        ${mode === "text" ? html`<pre class="grow">${cur.prompt}</pre>` : html`<div class="grow"><${TokenChips} pieces=${cur.pieces} showIds=${mode === "ids"} /></div>`}
        <div class="legend" style="margin-top:6px">Exactly the generation prompt the rollouts start from: <code>${"<|bos|><|user|>"}</code>question + answer-format suffix<code>${"<|end|><|assistant|>"}</code>${thinkRequired ? html`<code>${"<|think|>"}</code>` : ""}. The list is deterministic (seeded <code>make_tasks</code>), so these are the trainer's own prompts; the held-out split is a hash partition of the prompt text.</div>
      </div>`}</div>
    </div>`}
  </div>`;
}

/** What the model actually saw during RL: its own samples, with the reward each earned. */
function Rollouts({ run }) {
  const [d, setD] = useState(null);
  const [step, setStep] = useState(null);
  const [sel, setSel] = useState(0);
  const [mode, setMode] = useState("text");
  useEffect(() => { setSel(0); api(`/api/runs/${encodeURIComponent(run)}/rollouts?limit=40${step == null ? "" : "&step=" + step}`).then((x) => { setD(x); if (step == null) setStep(x.step); }).catch(() => setD(null)); }, [run, step]);
  if (!d) return html`<div><h2>Rollouts<${Info} k="rl_rollouts_view" /></h2><div class="empty-note">no rollouts stored for this run</div></div>`;
  const cur = d.rollouts[sel];
  return html`<div><h2>Rollouts<${Info} k="rl_rollouts_view" /></h2>
    <div class="row" style="margin-bottom:6px"><span class="muted">step</span>
      <select value=${d.step} onChange=${(e) => setStep(Number(e.target.value))}>${d.steps.map((s) => html`<option value=${s}>${s}</option>`)}</select>
      <span class="muted">${fmtInt(d.n)} rollouts in this step</span>
      <span class="muted" style="margin-left:8px">show as<${Info} k="show_as" /></span>
      ${["text", "tokens", "ids"].map((m) => html`<button class=${mode === m ? "active" : ""} onClick=${() => setMode(m)}>${m}</button>`)}</div>
    <div class="two fill" style="height:460px">
      <div class="col"><table><tr><th>task</th><th class="l">prompt</th><th>reward<${Info} k="rl_reward_scheme" /></th><th>parsed<${Info} k="parsed" /></th><th>len</th></tr>
        ${d.rollouts.map((r, i) => html`<tr class=${"click" + (i === sel ? " sel" : "")} onClick=${() => setSel(i)}>
          <td>${r.task}</td><td class="l">${String(r.prompt).slice(0, 80)}</td>
          <td style=${"font-weight:600;color:" + (r.reward > 0 ? "#15803d" : "#b91c1c")}>${fmtNum(r.reward, 2)}</td>
          <td>${r.malformed ? html`<b style="color:#b45309">malformed</b>` : r.parsed == null ? html`<span class="muted">-</span>` : String(r.parsed).slice(0, 12)}</td>
          <td>${fmtInt(r.n_tokens)}</td></tr>`)}</table></div>
      <div class="col">${cur && html`<div class="grow">
        <div class="sub">${cur.prompt_id} · ${cur.task} · gold <b>${cur.gold}</b> · parsed <b>${cur.parsed == null ? "-" : String(cur.parsed)}</b> · reward <b>${fmtNum(cur.reward, 3)}</b> · ${cur.verifier || "?"} · ${cur.termination}${cur.malformed ? " · malformed" : ""}</div>
        ${mode === "text" || !cur.pieces
          ? html`<pre class="grow">${cur.prompt + "\n\n--- completion ---\n" + cur.text}</pre>`
          : html`<div class="grow"><${TokenChips} pieces=${cur.pieces.map((p, i) => ({ ...p, loss: i >= cur.prompt_len }))} boundaries=${[cur.prompt_len]} showIds=${mode === "ids"} lossMask=${true} /></div>`}
        <div class="legend" style="margin-top:6px">The prompt plus the model's own completion. Only completion tokens are policy targets; a tool result inside the think span is never one. The reward is what the verifier returned for this sample.</div>
      </div>`}</div>
    </div>
  </div>`;
}

const RL_CARDS = { tasks: "rl_tasks", prompts: "rl_prompts", rollouts: "rl_rollouts", format: "rl_format", reward: "rl_reward_scheme", objective: "rl_objective", "collapse guards": "rl_guards" };

function RlPanel({ rl, id, runName }) {
  const rows = [["tasks", rl.tasks.map((t) => `${t} ${(rl.task_weights[t] * 100).toFixed(0)}%`).join(" · ")],
    ["prompts", `${fmtInt(rl.n_train_prompts)} train / ${fmtInt(rl.n_heldout_prompts)} held out (split by prompt-text hash, 10% held out)`],
    ["rollouts", `${rl.prompts_per_step} prompts/step x ${rl.group_size} samples, max ${rl.max_new_tokens} new tokens, T=${rl.temperature} top_p=${rl.top_p}`],
    ["format", `think span ${rl.think_required ? "mandatory" : "optional"}${rl.tools ? ` · Python tool, max ${rl.max_tool_calls} calls` : " · no tools"}`],
    ["reward", `${rl.reward_scheme}: ${rl.reward_rule}`],
    ["objective", `clip ${rl.clip_eps} · KL ${rl.kl_coef} (${rl.kl_kind}) · lr ${rl.lr}`],
    ["collapse guards", `entropy_stop ${rl.entropy_stop || "off"} · kl_stop ${rl.kl_stop || "off"}`]];
  return html`<div><h2>Prompts and reward<${Info} k="rl_recipe" /></h2>
    <table>${rows.map(([k, v]) => html`<tr><td>${k}<${Info} k=${RL_CARDS[k]} /></td><td class="l">${v}</td></tr>`)}</table>
    <div class="legend" style="margin-top:6px">RL has no token mixture: the model trains on its own samples. Prompts are generated deterministically from the task list and the seed.</div>
    <${RlPrompts} id=${id} thinkRequired=${rl.think_required} />
    ${runName && html`<${Rollouts} run=${runName} key=${runName} />`}</div>`;
}

function Recipe({ id }) {
  const [r, setR] = useState(null);
  const [err, setErr] = useState(null);
  const [pick, setPick] = useState(null);
  useEffect(() => { setR(null); setErr(null); setPick(null); api(`/api/data/recipe?id=${encodeURIComponent(id)}`).then(setR).catch((e) => setErr(String(e))); }, [id]);
  if (err) return html`<div class="panel" style="border-color:#fca5a5;color:#b91c1c">${err}</div>`;
  if (!r) return html`<div>loading…</div>`;
  const picked = pick && r.rows.find((x) => x.source === pick);
  const parent = parentRun(r.init_from);  // the default pairing for a run: the phase it continues
  return html`<div>
    <div class="row" style="margin-bottom:6px"><a href=${dataHref()}><button>‹ all recipes</button></a>
      <b>${r.run_name}</b><span class="stage-badge ${r.stage === "sft" ? "sft" : r.stage === "rl" ? "rl" : ""}">${r.stage}</span>
      <span class="muted">${r.kind === "run" ? "run (what it started with)" : "plan (yaml)"} · ${r.status}<${Info} k="recipe" /></span>
      ${r.kind === "run" && r.config_path && html`<a href=${dataHref("recipes", "config:" + r.config_path)}><button>see the plan</button></a>`}
      ${r.kind === "run" && html`<a href=${"#/runs/" + encodeURIComponent(r.run_name)}><button>run page</button></a>`}
      ${parent && html`<a href=${dataHref("compare", "run:" + parent, r.id)}><button>compare with ${parent}</button></a>`}
      <a href=${dataHref("compare", r.id, "")}><button>compare with…</button></a>
      ${r.kind === "run" && html`<a href=${dataHref("chain", "run:" + r.run_name)}><button>chain</button></a>`}</div>
    <div class="sub">tokenizer ${r.tokenizer_tag} · ${r.seq_len ? `seq_len ${r.seq_len} · ` : ""}${r.total_tokens ? fmtTok(r.total_tokens) + " tokens" : ""}${r.total_note ? ` (${r.total_note})` : ""}
      ${r.init_from ? ` · init_from ${r.init_from}` : " · random init"}${r.root ? ` · ${r.root}` : ""}</div>
    ${r.plan_differs === true && html`<div class="panel" style="border-color:#fcd34d">The yaml <code>${r.config_path}</code> no longer matches what this run started with (configs get edited between phases). This page shows the run.</div>`}
    ${r.rl ? html`<${RlPanel} rl=${r.rl} id=${r.id} runName=${r.kind === "run" ? r.run_name : null} />` : html`<div>
      <${MixTable} rows=${r.rows} showPlanned=${true} streams=${r.stream_sources} sel=${pick} onPick=${(s) => setPick(s === pick ? null : s)} />
      <div class="legend" style="margin-top:6px">epochs = planned tokens / available tokens; above 1.5 (red) the source is repeated.${r.stream_sources ? " 'consumed' is the loader's own per-source count at the last checkpoint." : ""} Click a row to inspect what the model sees from it.</div>
      ${picked && html`<${Inspector} row=${picked} tag=${picked.kind === "sft" ? r.tag : r.tokenizer_tag} seqLen=${r.seq_len} key=${picked.source + r.id} />`}
      ${!picked && r.extra_val.length > 0 && html`<h2>extra_val_mixture (drift set, val splits only)<${Info} k="extra_val" /></h2>
        <${MixTable} rows=${r.extra_val} showPlanned=${false} />
        <div class="legend">Pretraining validation tracked alongside the stage's own val loss, so base-model drift is visible; ${fmtTok(r.extra_val_tokens)} tokens per evaluation.</div>`}
    </div>`}
  </div>`;
}

// ------------------------------------------------------------------ compare two recipes
const recipeLabel = (r) => (r ? `${r.run_name}${r.kind === "config" ? " (plan)" : ""}` : "…");

/** Two recipes side by side, computed entirely from two /api/data/recipe payloads. */
function Compare({ a, b }) {
  const [ra, setRa] = useState(null);
  const [rb, setRb] = useState(null);
  const [err, setErr] = useState(null);
  const [all, setAll] = useState([]);
  useEffect(() => { api("/api/data/recipes").then(setAll).catch(() => {}); }, []);
  useEffect(() => { setRa(null); setErr(null); if (a) api(`/api/data/recipe?id=${encodeURIComponent(a)}`).then(setRa).catch((e) => setErr(String(e))); }, [a]);
  useEffect(() => { setRb(null); if (b) api(`/api/data/recipe?id=${encodeURIComponent(b)}`).then(setRb).catch((e) => setErr(String(e))); }, [b]);
  const pick = (which, id) => { location.hash = dataHref("compare", which === "a" ? id : a, which === "a" ? b : id).slice(1); };
  const sel = (which, cur) => html`<select value=${cur} onChange=${(e) => pick(which, e.target.value)}>
    <option value="">…</option>${all.map((r) => html`<option value=${r.id}>${r.stage} · ${r.run_name}${r.kind === "config" ? " (plan)" : ""}</option>`)}</select>`;
  if (err) return html`<div class="panel" style="border-color:#fca5a5;color:#b91c1c">${err}</div>`;
  const rows = [];
  if (ra && rb) {
    const names = [...ra.rows.map((r) => r.source), ...rb.rows.map((r) => r.source).filter((n) => !ra.rows.some((x) => x.source === n))];
    for (const n of names) rows.push([n, ra.rows.find((x) => x.source === n) || null, rb.rows.find((x) => x.source === n) || null]);
    rows.sort((x, y) => (y[1] ? y[1].weight : 0) + (y[2] ? y[2].weight : 0) - ((x[1] ? x[1].weight : 0) + (x[2] ? x[2].weight : 0)));
  }
  const HDR_CARDS = { seq_len: "seq_len", init_from: "init_from", extra_val_mixture: "extra_val" };
  const hdr = [["seq_len", (r) => r.seq_len || "-"], ["total tokens", (r) => fmtTok(r.total_tokens)], ["stage", (r) => r.stage],
    ["tokenized / sft root", (r) => r.root || "-"], ["init_from", (r) => r.init_from || "random init"],
    ["extra_val_mixture", (r) => (r.extra_val.length ? r.extra_val.map((x) => `${x.source} ${(x.weight * 100).toFixed(0)}`).join(" · ") : "none")]];
  return html`<div>
    <div class="row" style="margin-bottom:6px"><a href=${dataHref()}><button>‹ all recipes</button></a>
      <b>compare</b>${sel("a", a)}<span class="muted">vs</span>${sel("b", b)}
      <button onClick=${() => pick("a", b) || (location.hash = dataHref("compare", b, a).slice(1))}>swap</button></div>
    ${!ra || !rb ? html`<div class="empty-note">pick two recipes</div>` : html`<div>
      <table><tr><th class="l">header</th><th class="l">${recipeLabel(ra)}</th><th class="l">${recipeLabel(rb)}</th></tr>
        ${hdr.map(([k, f]) => { const x = String(f(ra)), y = String(f(rb)); return html`<tr><td class="l">${k}${HDR_CARDS[k] && html`<${Info} k=${HDR_CARDS[k]} />`}</td>
          <td class="l">${x}</td><td class=${"l" + (x === y ? "" : " diff")} style=${x === y ? "" : "color:#b45309;font-weight:600"}>${y}</td></tr>`; })}</table>
      <h2>mixture</h2>
      <table><tr><th class="l">source</th><th>weight A<${Info} k="mix_weight" /></th><th>weight B</th><th>Δ (pp)</th><th>planned A</th><th>planned B</th><th>epochs A<${Info} k="epochs" /></th><th>epochs B</th></tr>
        ${rows.map(([n, x, y]) => { const d = ((y ? y.weight : 0) - (x ? x.weight : 0)) * 100; const only = !x || !y;
          return html`<tr style=${only ? "background:#fef3c7" : ""}>
            <td class="l">${n}${only ? html` <span class="muted">only in ${x ? "A" : "B"}</span>` : ""}</td>
            <td>${x ? (x.weight * 100).toFixed(1) + "%" : "-"}</td><td>${y ? (y.weight * 100).toFixed(1) + "%" : "-"}</td>
            <td style=${Math.abs(d) < 0.05 ? "" : "font-weight:600;color:" + (d > 0 ? "#15803d" : "#b91c1c")}>${(d > 0 ? "+" : "") + d.toFixed(1)}</td>
            <td>${x ? fmtTok(x.planned_tokens) : "-"}</td><td>${y ? fmtTok(y.planned_tokens) : "-"}</td>
            <td>${x && x.epochs != null ? x.epochs.toFixed(2) : "-"}</td><td>${y && y.epochs != null ? y.epochs.toFixed(2) : "-"}</td></tr>`; })}
      </table>
      <div class="legend" style="margin-top:6px">Δ is B minus A in percentage points of the normalized mixture. Rows in only one recipe are highlighted; planned tokens are that recipe's own total x weight, so they are comparable only when the totals are.</div>
    </div>`}
  </div>`;
}

// ------------------------------------------------------------------ the init_from chain
const CHAIN_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#ca8a04", "#db2777", "#4b5563", "#65a30d"];

/** Cumulative exposure per source along the init_from chain, one row per stage plus a stacked bar. */
function ChainView({ id }) {
  const run = id.startsWith("run:") ? id.slice(4) : id;
  const [c, setC] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => { setC(null); setErr(null); api(`/api/data/chain?run=${encodeURIComponent(run)}`).then(setC).catch((e) => setErr(String(e))); }, [run]);
  if (err) return html`<div class="panel" style="border-color:#fca5a5;color:#b91c1c">${err}</div>`;
  if (!c) return html`<div>loading…</div>`;
  const color = (n) => CHAIN_COLORS[c.sources.indexOf(n) % CHAIN_COLORS.length];
  const widest = Math.max(1, ...c.runs.map((r) => r.tokens_used));
  return html`<div>
    <div class="row" style="margin-bottom:6px"><a href=${dataHref()}><button>‹ all recipes</button></a>
      <b>${run}</b><span class="muted">data seen by these weights, back through init_from<${Info} k="chain" /></span></div>
    <div class="sub">${c.runs.length} stages · ${fmtTok(c.cumulative_tokens)} cumulative tokens</div>
    <div style="margin:10px 0 16px">
      ${c.runs.map((r) => html`<div style="margin-bottom:6px">
        <div class="legend">${r.run} <span class="muted">· ${r.stage} · ${fmtTok(r.tokens_used)}${r.checkpoint ? ` up to ${r.checkpoint}` : ""}${r.actual ? "" : " (expected)"}</span></div>
        <div style=${"display:flex;height:16px;width:" + (r.tokens_used / widest * 100).toFixed(1) + "%;min-width:2px;border-radius:3px;overflow:hidden"}>
          ${c.sources.filter((n) => r.per_source[n]).map((n) => html`<div title=${`${n}: ${fmtTok(r.per_source[n])}`}
            style=${"background:" + color(n) + ";width:" + (r.per_source[n] / Math.max(1, r.tokens_used) * 100).toFixed(2) + "%"}></div>`)}
          ${!c.sources.some((n) => r.per_source[n]) && html`<div style="background:#cbd5e1;width:100%"></div>`}
        </div></div>`)}
      <div class="row" style="flex-wrap:wrap;gap:8px;margin-top:8px">${c.sources.map((n) => html`<span class="legend"><span style=${"display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;background:" + color(n)}></span>${n}</span>`)}</div>
    </div>
    <div style="overflow-x:auto"><table><tr><th class="l">run</th><th>stage</th><th>seq</th><th>tokens used</th><th>source</th>${c.sources.map((n) => html`<th>${n}</th>`)}</tr>
      ${c.runs.map((r) => html`<tr><td class="l"><a href=${dataHref("recipes", "run:" + r.run)}>${r.run}</a>${r.checkpoint ? html` <span class="muted">→ ${r.checkpoint}</span>` : ""}</td>
        <td><span class=${"stage-badge " + (r.stage === "sft" ? "sft" : r.stage === "rl" ? "rl" : "")}>${r.stage}</span></td>
        <td>${r.seq_len || "-"}</td><td>${fmtTok(r.tokens_used)}</td>
        <td>${r.rl ? html`<span class="muted">own samples</span>` : r.actual ? html`<b style="color:#15803d">actual</b>` : html`<span class="muted">expected</span>`}</td>
        ${c.sources.map((n) => html`<td>${r.per_source[n] ? fmtTok(r.per_source[n]) : html`<span class="muted">-</span>`}</td>`)}</tr>`)}
      <tr style="font-weight:600;border-top:2px solid #94a3b8"><td class="l">total</td><td></td><td></td><td>${fmtTok(c.cumulative_tokens)}</td><td></td>
        ${c.sources.map((n) => html`<td>${fmtTok(c.totals[n])}</td>`)}</tr>
    </table></div>
    <div class="legend" style="margin-top:6px">"actual" rows come from the loader's own per-source counters in the last <code>checkpoint</code> record of that run (scaled when the child loaded a mid-run checkpoint); "expected" rows are tokens used x normalized mixture weight, because runs started before the trainers logged that block carry no counters.${c.any_expected ? "" : " Every row here is actual."} RL stages train on the model's own samples, so they have no source mixture.</div>
  </div>`;
}

function Catalog() {
  const [ov, setOv] = useState(null);
  const [recipes, setRecipes] = useState([]);
  const [showUnused, setShowUnused] = useState(false);
  useEffect(() => { api("/api/data/sources").then(setOv).catch(() => {}); api("/api/data/recipes").then(setRecipes).catch(() => {}); }, []);
  if (!ov) return html`<div>loading…</div>`;
  const used = new Set();
  for (const r of recipes) for (const x of r.rows || []) used.add(x.source);
  const nUses = (n) => recipes.filter((r) => (r.rows || []).some((x) => x.source === n)).length;
  const keep = (n) => showUnused || (used.has(n) && !/-v1$/.test(n));
  const shown = ov.sources.filter((s) => keep(s.name));
  const sets = [];
  for (const [tag, m] of Object.entries(ov.sft || {})) for (const [n, x] of Object.entries(m)) if (keep(n)) sets.push([tag, n, x]);
  const nHidden = ov.sources.length - shown.length + Object.values(ov.sft || {}).reduce((a, m) => a + Object.keys(m).length, 0) - sets.length;
  return html`<div>
    <div class="row" style="margin-bottom:6px"><span class="sub" style="margin:0">Every raw, tokenized and chat-formatted data set. A set is "used" when some recipe's mixture names it.</span>
      <button class=${showUnused ? "active" : ""} onClick=${() => setShowUnused(!showUnused)}>show unused</button>
      ${!showUnused && html`<span class="muted">${nHidden} unused or *-v1 sets hidden</span>`}</div>
    <table><tr><th class="l">source</th><th>kind<${Info} k="source_kind" /></th><th>raw files</th><th>raw size</th><th>raw rows</th>${ov.tags.map((t) => html`<th>train tokens (${t})<${Info} k="tokenizer_tag" /></th><th>val tokens (${t})</th><th>docs (${t})</th>`)}<th>used by</th></tr>
    ${shown.map((s) => html`<tr class="click" onClick=${() => { location.hash = dataHref("source", s.name).slice(1); }}>
      <td class="l">${s.name}${used.has(s.name) ? "" : html` <span class="muted">unused</span>`}${Object.values(s.prepared || {}).some((p) => p.manifest && p.manifest.filter) ? html` <span class="muted" title="a filtered selection of the raw books; the source page shows the rules">· filtered</span>` : ""}</td><td>${s.kind}</td><td>${s.raw_files}</td><td>${fmtBytes(s.raw_bytes)}</td><td>${fmtInt(s.raw_rows)}</td>
      ${ov.tags.map((t) => { const p = (s.prepared || {})[t]; return p ? html`<td>${fmtTok(p.train_tokens)}</td><td>${fmtTok(p.val_tokens)}</td><td>${fmtInt(p.train_docs)}</td>` : html`<td colspan="3" class="muted">not tokenized</td>`; })}
      <td>${nUses(s.name)}</td></tr>`)}
    </table>
    ${sets.length > 0 && html`<h2>SFT / reasoning sets (chat-formatted, with loss masks)</h2>
      <table><tr><th>tag</th><th class="l">set</th><th>train examples</th><th>train tokens</th><th>loss targets<${Info} k="loss_targets" /></th><th>val examples</th><th>max len<${Info} k="max_len" /></th><th>think span<${Info} k="think_span" /></th><th>dropped / too long<${Info} k="dropped" /></th><th>used by</th></tr>
      ${sets.map(([tag, n, m]) => html`<tr class="click" onClick=${() => { location.hash = dataHref("source", n).slice(1); }}>
        <td>${tag}</td><td class="l">${n}</td><td>${fmtInt(m.train_examples)}</td><td>${fmtTok(m.train_tokens)}</td><td>${m.train_tokens ? (m.train_targets / m.train_tokens * 100).toFixed(0) + "%" : "-"}</td>
        <td>${fmtInt(m.val_examples)}</td><td>${m.max_len || "-"}</td><td>${m.think_required ? "mandatory" : "no"}</td><td>${fmtInt(m.dropped || 0)} / ${fmtInt(m.too_long || 0)}</td><td>${nUses(n)}</td></tr>`)}</table>`}
    <div class="legend" style="margin-top:6px">Click a row for its provenance chain, prepared artifacts, the recipes that use it, and the document browser.</div>
  </div>`;
}

/** One catalog entry: raw files, prepared artifacts per tag, provenance, children, used_by, and a browser. */
function SourcePage({ name }) {
  const [s, setS] = useState(null);
  const [err, setErr] = useState(null);
  const [browse, setBrowse] = useState(false);
  useEffect(() => { setS(null); setErr(null); setBrowse(false); api(`/api/data/source/${encodeURIComponent(name)}`).then(setS).catch((e) => setErr(String(e))); }, [name]);
  if (err) return html`<div class="panel" style="border-color:#fca5a5;color:#b91c1c">${err}</div>`;
  if (!s) return html`<div>loading…</div>`;
  return html`<div>
    <div class="row" style="margin-bottom:6px"><a href=${dataHref("catalog")}><button>‹ catalog</button></a><b>${s.name}</b>
      <span class="muted">${s.kind}${s.license ? " · " + s.license : ""}${s.repo ? " · " + s.repo : ""}</span>
      <button class=${browse ? "active" : ""} onClick=${() => setBrowse(!browse)}>browse</button></div>
    ${s.notes && html`<div class="sub">${s.notes}</div>`}
    ${s.raw_files.length > 0 && html`<div><h2>raw files</h2>
      <table><tr><th>#</th><th class="l">file</th><th>rows</th><th>row groups</th><th>size</th></tr>
      ${s.raw_files.map((f) => html`<tr><td>${f.index}</td><td class="l">${f.name}</td><td>${fmtInt(f.rows)}</td><td>${f.row_groups}</td><td>${fmtBytes(f.bytes)}</td></tr>`)}</table></div>`}
    <h2>prepared artifacts<${Info} k="prepared_artifacts" /></h2>
    ${Object.keys(s.prepared).length === 0 ? html`<div class="empty-note">nothing prepared from this source</div>`
      : html`<table><tr><th>tag<${Info} k="tokenizer_tag" /></th><th>kind</th><th>made by<${Info} k="provenance" /></th><th>train tokens</th><th>train docs</th><th>val tokens</th><th>shards</th><th>loss targets<${Info} k="loss_targets" /></th><th class="l">provenance</th></tr>
      ${Object.entries(s.prepared).map(([k, p]) => html`<tr><td>${p.tag}</td><td>${p.kind}</td><td>${p.made_by}</td><td>${fmtTok(p.train_tokens)}</td><td>${fmtInt(p.train_docs)}</td><td>${fmtTok(p.val_tokens)}</td><td>${p.train_shards}</td>
        <td>${p.targets ? (p.targets / Math.max(1, p.train_tokens) * 100).toFixed(0) + "%" : "-"}</td><td class="l"><span class="legend">${p.provenance}</span></td></tr>`)}</table>`}
    ${Object.values(s.prepared).filter((p) => p.manifest && p.manifest.filter).map((p) => html`<${SelectionFilter} p=${p} rawBytes=${s.raw_files.reduce((a, f) => a + f.bytes, 0)} />`)}
    ${s.parents.length > 0 && html`<div class="sub">derived from ${s.parents.map((p) => html`<a href=${dataHref("source", p)}>${p}</a> `)}</div>`}
    ${s.children.length > 0 && html`<div class="sub">feeds ${s.children.map((p) => html`<a href=${dataHref("source", p)}>${p}</a> `)}</div>`}
    <h2>used by</h2>
    ${s.used_by.length === 0 ? html`<div class="empty-note">no recipe's mixture names this source</div>`
      : html`<table><tr><th>stage</th><th class="l">recipe</th><th>weight</th></tr>
      ${s.used_by.map((u) => html`<tr class="click" onClick=${() => { location.hash = dataHref("recipes", u.recipe_id).slice(1); }}>
        <td><span class=${"stage-badge " + (u.stage === "sft" ? "sft" : u.stage === "rl" ? "rl" : "")}>${u.stage}</span></td>
        <td class="l">${u.run_name}${u.kind === "config" ? html` <span class="muted">(plan)</span>` : ""}</td><td>${(u.weight * 100).toFixed(1)}%</td></tr>`)}</table>`}
    ${browse && html`<div><h2>browse</h2><${Documents} initial=${s.name} /></div>`}
  </div>`;
}

/** The `filter` block a selecting preparer (slm/data/gutenberg.py) writes into its manifest: the funnel from the
 *  books in the release to the books on disk, one row per rule. */
const RULE_OP = { date: "≥", min_words: "≥", english: "≥", caps: "≤", verse: "≤", archaic: "≤", dialogue: "≥", not_selected: "≥" };
function SelectionFilter({ p, rawBytes }) {
  const m = p.manifest, f = m.filter, b = m.books || {}, sel = f.selected || {};
  const nIn = Object.values(f.books_in || {}).reduce((a, x) => a + x, 0);
  const thr = (r) => (r.threshold == null ? "-" : `${RULE_OP[r.rule] || ""} ${typeof r.threshold === "number" && r.threshold < 10 && r.rule !== "date" ? fmtNum(r.threshold, 3).replace(/0+$/, "").replace(/\.$/, "") : r.threshold}`);
  return html`<div>
    <h2>selection filter (${p.tag})<${Info} k="gutenberg" /></h2>
    <div class="tiles">
      <div class="tile"><div class="k">books in the release</div><div class="v">${fmtInt(nIn)}</div><div class="s">${Object.entries(f.books_in || {}).map(([k, v]) => `${k} ${fmtInt(v)}`).join(" · ")}</div></div>
      <div class="tile"><div class="k">books kept</div><div class="v">${fmtInt(sel.train)} <span class="muted" style="font-size:13px">+ ${fmtInt(sel.val)} val</span></div><div class="s">${fmtInt(f.candidates)} passed every rule</div></div>
      <div class="tile"><div class="k">dialogue cutoff</div><div class="v">${f.dialogue_cutoff == null ? "-" : fmtNum(f.dialogue_cutoff, 3)}</div><div class="s">densest first until ${fmtTok((f.config || {}).target_tokens)} tokens</div></div>
      <div class="tile"><div class="k">mean tokens / book</div><div class="v">${fmtTok(b.mean_tokens_per_book)}</div><div class="s">${fmtInt(m.train_docs)} train docs of ≤ ${fmtTok(b.segment_tokens)}</div></div>
      <div class="tile"><div class="k">on disk</div><div class="v">${fmtBytes(rawBytes)}</div><div class="s">raw parquet · ${fmtBytes((m.train_tokens + m.val_tokens) * 2)} tokenized</div></div>
    </div>
    <table><tr><th class="l">rule</th><th>threshold</th><th>removed (train)</th><th>tokens removed</th><th>removed (val)</th><th>fail it at all</th><th class="l">what it checks · examples removed</th></tr>
      ${f.rules.map((r) => html`<tr><td class="l">${r.rule}</td><td>${thr(r)}</td><td>${fmtInt(r.removed)}</td><td>${r.removed_tokens == null ? "-" : fmtTok(r.removed_tokens)}</td><td>${fmtInt(r.removed_val)}</td><td>${fmtInt(r.fails)}</td>
        <td class="l"><span class="legend">${r.text}${r.examples && r.examples.length ? html`<br />${r.examples.join(" · ")}` : ""}</span></td></tr>`)}
      <tr style="font-weight:600;border-top:2px solid #94a3b8"><td class="l">kept</td><td></td><td>${fmtInt(sel.train)}</td><td>${fmtTok(m.train_tokens)}</td><td>${fmtInt(sel.val)}</td><td></td>
        <td class="l"><span class="legend" style="font-weight:400">${(f.kept_examples || []).join(" · ")}</span></td></tr>
    </table>
    <div class="legend" style="margin-top:6px">A book is counted under the first rule it fails, so "removed" adds up to the funnel; "fail it at all" is how many
      books that rule alone would remove. The date rule ran before download (those books were never fetched). ${f.val_rule || ""}. Per-book statistics and
      verdicts: <code>books.jsonl</code> next to the manifest.</div>
  </div>`;
}

function Hist({ edges, counts, label, info }) {
  const max = Math.max(1, ...counts);
  return html`<div><h3>${label}${info && html`<${Info} k=${info} />`}</h3><div class="hist" style="margin-bottom:18px">${counts.map((c, i) => html`<div style=${"height:" + (c / max * 100).toFixed(1) + "%"} title=${`${fmtTok(edges[i])}+: ${fmtInt(c)}`}><span>${fmtTok(edges[i])}</span></div>`)}</div></div>`;
}

function Documents({ initial = null }) {
  const [ov, setOv] = useState(null);
  const [tag, setTag] = useState(null);
  const [source, setSource] = useState(initial);
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
    if (!ov || !tag || !source) return;  // before the overview lands we cannot tell tokenized from raw-only
    if (isTok) { api(`${base}/shards`).then(setShards).catch(() => setShards([])); api(`${base}/stats`).then(setStats).catch(() => setStats(null)); }
    else { setFile(0); setRg(0); api(`/api/data/raw/${source}/files`).then(setFiles).catch(() => setFiles([])); }
  }, [ov, source, split, tag]);
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
      ${isTok && ["doc", "window", "stats"].map((v) => html`<button class=${view === v ? "active" : ""} onClick=${() => setView(v)}>${v}</button>`)}${isTok && html`<${Info} k="browse_views" />`}
      <span class="muted" style="margin-left:10px">show as<${Info} k="show_as" /></span>
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
            <div class="tile"><div class="k">mean tokens/doc<${Info} k="doc_lengths" /></div><div class="v">${fmtNum(stats.mean_len, 0)}</div><div class="s">p10 ${fmtNum(stats.p10, 0)} · p50 ${fmtNum(stats.p50, 0)} · p90 ${fmtNum(stats.p90, 0)} · p99 ${fmtNum(stats.p99, 0)}</div></div>
            <div class="tile"><div class="k">docs ≥ 4K tokens</div><div class="v">${fmtInt(stats.docs_over_4k)}</div><div class="s">${fmtTok(stats.tokens_over_4k)} tokens (${(stats.tokens_over_4k / Math.max(1, stats.tokens) * 100).toFixed(1)}%)</div></div>
            <div class="tile"><div class="k">docs ≥ 8K tokens</div><div class="v">${fmtInt(stats.docs_over_8k)}</div><div class="s">${fmtTok(stats.tokens_over_8k)} tokens (${(stats.tokens_over_8k / Math.max(1, stats.tokens) * 100).toFixed(1)}%)</div></div>
          </div>
          <${Hist} edges=${stats.hist_edges} counts=${stats.hist} label="documents by length (tokens, log bins)" info="doc_hist" />
          <${Hist} edges=${stats.hist_edges} counts=${stats.tokens_by_bin} label="tokens by document-length bin" info="doc_hist" />
        </div>` : html`<div class="empty-note">computing stats…</div>`)}
      </div>
    </div>
  </div>`;
}

import { h } from "preact";
import { useEffect, useState } from "preact/hooks";
import htm from "htm";
import { api, fmtTok, fmtDur, fmtNum, fmtInt } from "../components/util.js";

const html = htm.bind(h);

// Project pipeline, in order. A run is assigned to a stage from its metadata.
const STAGES = [
  { id: "pretrain", label: "Pretraining (base model)", hint: "random init → base checkpoint; val loss on the pretraining mixture" },
  { id: "sft", label: "Instruction SFT", hint: "assistant-token loss on chat data; pretraining-val tracks base drift" },
  { id: "reasoning", label: "Reasoning SFT", hint: "mandatory <|think|> span + '#### answer'" },
  { id: "grpo", label: "RL (GRPO, verifiable rewards)", hint: "group-relative advantages; held-out accuracy is the metric" },
  { id: "ctx", label: "Context extension", hint: "RoPE scaling + continued training on long documents" },
];

function stageOf(r) {
  if (r.stage === "grpo") return "grpo";
  if (r.stage === "sft") return (r.mixture || []).some((m) => /reasoning|gsm8k|math/.test(m)) ? "reasoning" : "sft";
  if (/ctx|16k|32k/.test(r.run_name)) return "ctx";
  return "pretrain";
}

const bestMetric = (r) => r.heldout_acc != null ? `held-out acc ${(r.heldout_acc * 100).toFixed(1)}%` : r.best_val != null ? `best val ${fmtNum(r.best_val, 3)}` : "-";
const open = (r) => (location.hash = "#/runs/" + encodeURIComponent(r.run_name));

export function Home() {
  const [runs, setRuns] = useState(null);
  const [data, setData] = useState(null);
  const [gpu, setGpu] = useState(null);
  useEffect(() => {
    let alive = true;
    const tick = () => { api("/api/runs").then((r) => alive && setRuns(r)).catch(() => {}); api("/api/system/gpu").then((g) => alive && setGpu(g)).catch(() => {}); };
    tick();
    api("/api/data/sources").then(setData).catch(() => {});
    const id = setInterval(tick, 10000);
    return () => { alive = false; clearInterval(id); };
  }, []);
  const byStage = {};
  for (const r of runs || []) (byStage[stageOf(r)] ||= []).push(r);
  const live = (runs || []).filter((r) => r.status === "running");
  const tags = data ? data.tags : [];
  return html`<div>
    <h1>Overview</h1>
    <div class="sub">${runs ? `${runs.length} runs · ${live.length} live` : "loading…"}${gpu && gpu.available ? ` · GPU ${gpu.used_gib.toFixed(1)}/${gpu.total_gib.toFixed(0)} GiB, ${gpu.util.toFixed(0)}% busy` : ""}</div>
    ${live.map((r) => html`<a class="card" style="display:block;margin-bottom:10px" href=${"#/runs/" + encodeURIComponent(r.run_name)}>
      <div><b>${r.run_name}</b> <span class="status running">running</span> <span class="muted">· ${STAGES.find((s) => s.id === stageOf(r)).label}</span></div>
      <div class="bar"><div style=${"width:" + (r.progress * 100).toFixed(1) + "%"}></div></div>
      <div class="muted">${fmtTok(r.tokens)} / ${fmtTok(r.total_tokens)} · loss ${fmtNum(r.loss, 3)} · ${bestMetric(r)} · ${fmtInt(r.tok_s)} tok/s · ETA ${fmtDur(r.eta_s)}</div></a>`)}
    <h2>Pipeline and runs</h2>
    <table><tr><th class="l">stage</th><th class="l">run</th><th>status</th><th>tokens</th><th>progress</th><th>loss</th><th>result</th><th>tok/s</th><th>elapsed</th><th>ETA</th><th>started</th><th>git</th><th class="l">starts from</th></tr>
      ${STAGES.map((s) => { const rs = byStage[s.id] || []; return rs.length === 0
        ? html`<tr><td class="l"><b>${s.label}</b><div class="legend">${s.hint}</div></td><td class="l muted" colspan="12">not started</td></tr>`
        : rs.map((r, i) => html`<tr class="click" onClick=${() => open(r)}>${i === 0 ? html`<td class="l" rowspan=${rs.length}><b>${s.label}</b><div class="legend">${s.hint}</div></td>` : ""}
            <td class="l"><a href=${"#/runs/" + encodeURIComponent(r.run_name)}>${r.run_name}</a></td><td><span class=${"status " + r.status}>${r.status}</span></td>
            <td>${fmtTok(r.tokens)}${r.total_tokens ? ` / ${fmtTok(r.total_tokens)}` : ""}</td><td>${(r.progress * 100).toFixed(1)}%</td><td>${fmtNum(r.loss, 4)}</td><td>${bestMetric(r)}</td>
            <td>${fmtInt(r.tok_s)}</td><td>${fmtDur(r.elapsed_s)}</td><td>${r.status === "running" ? fmtDur(r.eta_s) : "-"}</td><td>${r.started || "-"}</td><td>${(r.git_commit || "").slice(0, 8)}</td>
            <td class="l muted">${r.init_from ? r.init_from.replace(/^runs[\\/]/, "").replace(/[\\/]checkpoints[\\/]/, " › ") : "random init"}</td></tr>`); })}
    </table>
    <div class="legend" style="margin-top:6px">Click a run for live charts, milestones, samples, checkpoints, events and config.</div>
    <h2>Data readiness</h2>
    ${!data ? html`<div class="muted">loading…</div>` : tags.length === 0 ? html`<div class="empty-note">no tokenized data yet</div>` : tags.map((t) => {
      const srcs = data.sources.filter((s) => s.tokenized[t]);
      const total = srcs.reduce((a, s) => a + s.tokenized[t].train_tokens, 0);
      const sft = (data.sft && data.sft[t]) ? Object.keys(data.sft[t]) : [];
      return html`<div class="panel" style="margin-bottom:8px"><b>tokenizer ${t}</b> · ${fmtTok(total)} pretraining tokens across ${srcs.length} sources (${srcs.map((s) => `${s.name} ${fmtTok(s.tokenized[t].train_tokens)}`).join(", ")}) · ${sft.length} SFT/reasoning sets
        <div class="legend">details on the <a href="#/data">Data</a> page</div></div>`; })}
  </div>`;
}

import { h } from "preact";
import { useEffect, useState } from "preact/hooks";
import htm from "htm";
import { api, fmtTok, fmtDur, fmtNum, fmtInt } from "../components/util.js";
import { Info } from "../components/info.js";

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
    <div class="sub">${runs ? `${runs.length} runs · ${live.length} live` : "loading…"}${gpu && gpu.available ? ` · GPU ${gpu.used_gib.toFixed(1)}/${gpu.total_gib.toFixed(0)} GiB, ${gpu.util.toFixed(0)}% busy${gpu.temp_c != null ? `, ${gpu.temp_c.toFixed(0)} °C` : ""}${gpu.power_w != null ? `, ${gpu.power_w.toFixed(0)} W` : ""}${gpu.throttled ? " ⚠ throttling" : ""}` : ""}</div>
    ${live.map((r) => html`<a class="card" style="display:block;margin-bottom:10px" href=${"#/runs/" + encodeURIComponent(r.run_name)}>
      <div><b>${r.run_name}</b> <span class="status running">running</span> <span class="muted">· ${STAGES.find((s) => s.id === stageOf(r)).label}</span></div>
      <div class="bar"><div style=${"width:" + (r.progress * 100).toFixed(1) + "%"}></div></div>
      <div class="muted">${r.total_steps ? `step ${fmtInt(r.update)} / ${fmtInt(r.total_steps)}` : `${fmtTok(r.tokens)} / ${fmtTok(r.total_tokens)}`} · ${r.is_rl ? `reward ${fmtNum(r.rl.reward_mean, 3)}` : `loss ${fmtNum(r.loss, 3)}`} · ${bestMetric(r)}${r.total_steps ? "" : ` · ${fmtInt(r.tok_s)} tok/s`} · ETA ${fmtDur(r.eta_s)}</div></a>`)}
    <h2>Pipeline and runs<${Info} k="pipeline" /></h2>
    <table><tr><th class="l">stage</th><th class="l">run</th><th>status<${Info} k="run_status" /></th><th>tokens<${Info} k="tokens_seen" /><div class="legend">this run · seen in total</div></th><th>progress</th><th>loss<${Info} k="loss_col" /></th><th>result<${Info} k="result_col" /></th><th>quality<${Info} k="quality_col" /><div class="legend">judged, 1-5</div></th><th>tok/s<${Info} k="tok_s" /></th><th>elapsed<div class="legend">ETA if running</div></th><th>started<div class="legend">git</div></th></tr>
      ${STAGES.map((s) => { const rs = byStage[s.id] || []; return rs.length === 0
        ? html`<tr><td class="l"><b>${s.label}</b><div class="legend">${s.hint}</div></td><td class="l muted" colspan="10">not started</td></tr>`
        : rs.map((r, i) => html`<tr class="click" onClick=${() => open(r)}>${i === 0 ? html`<td class="l" rowspan=${rs.length}><b>${s.label}</b><div class="legend">${s.hint}</div></td>` : ""}
            <td class="l"><a href=${"#/runs/" + encodeURIComponent(r.run_name)}>${r.run_name}</a><div class="legend">${r.init_from ? "from " + r.init_from.replace(/^runs[\\/]/, "").replace(/[\\/]checkpoints[\\/]/, " › ") : "random init"}</div></td><td><span class=${"status " + r.status}>${r.status}</span></td>
            <td>${r.total_steps ? `step ${fmtInt(r.update)} / ${fmtInt(r.total_steps)}` : `${fmtTok(r.tokens)}${r.total_tokens ? ` / ${fmtTok(r.total_tokens)}` : ""}`}${r.cumulative_tokens > r.tokens ? html`<div class="legend">${fmtTok(r.cumulative_tokens)} seen</div>` : ""}</td><td>${(r.progress * 100).toFixed(1)}%</td><td>${r.is_rl ? `reward ${fmtNum(r.rl.reward_mean, 3)}` : fmtNum(r.loss, 4)}</td><td>${bestMetric(r)}</td>
            <td>${r.quality ? html`${r.quality.overall.toFixed(2)}<div class="legend">at ${fmtTok(r.quality.tokens)} · ${r.quality.n_judged} pts</div>` : html`<span class="muted">-</span>`}</td>
            <td>${fmtInt(r.tok_s)}</td><td>${fmtDur(r.elapsed_s)}${r.status === "running" ? html`<div class="legend">ETA ${fmtDur(r.eta_s)}</div>` : ""}</td><td class="w">${r.started || "-"}${r.git_commit ? html`<div class="legend">${r.git_commit.slice(0, 8)}</div>` : ""}</td>
            </tr>`); })}
    </table>
    <div class="legend" style="margin-top:6px">Click a run for live charts, milestones, samples, checkpoints, events and config.</div>
    <h2>Data readiness<${Info} k="data_readiness" /></h2>
    ${!data ? html`<div class="muted">loading…</div>` : tags.length === 0 ? html`<div class="empty-note">no tokenized data yet</div>` : tags.map((t) => {
      const srcs = data.sources.filter((s) => s.tokenized[t]);
      const total = srcs.reduce((a, s) => a + (s.tokenized[t].train_tokens ?? 0), 0);  // a source with no count must not NaN the sum
      const sft = (data.sft && data.sft[t]) ? Object.keys(data.sft[t]) : [];
      return html`<div class="panel" style="margin-bottom:8px"><b>tokenizer ${t}</b> · ${fmtTok(total)} pretraining tokens across ${srcs.length} sources (${srcs.map((s) => `${s.name} ${s.tokenized[t].train_tokens == null ? "?" : fmtTok(s.tokenized[t].train_tokens)}`).join(", ")}) · ${sft.length} SFT/reasoning sets
        <div class="legend">details on the <a href="#/data">Data</a> page</div></div>`; })}
  </div>`;
}

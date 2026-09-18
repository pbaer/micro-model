import { h } from "preact";
import { useEffect, useMemo, useState } from "preact/hooks";
import htm from "htm";
import { api, fmtTok, fmtDur, fmtNum, fmtInt, fmtSci, fmtBytes, fmtTime } from "../components/util.js";
import { Chart } from "../components/chart.js";

const html = htm.bind(h);

function Tile({ k, v, s }) {
  return html`<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s || ""}</div></div>`;
}

const hasData = (a) => Array.isArray(a) && a.some((v) => v !== null && v !== undefined);

export function RunDetail({ run }) {
  const [info, setInfo] = useState(null);
  const [series, setSeries] = useState(null);
  const [ckpts, setCkpts] = useState([]);
  const [samples, setSamples] = useState([]);
  const [quality, setQuality] = useState(null);
  const [qualityTok, setQualityTok] = useState(null);
  const [qualityDetail, setQualityDetail] = useState(null);
  const [sample, setSample] = useState(null);
  const [sampleTok, setSampleTok] = useState(null);
  const [events, setEvents] = useState([]);
  const [xmode, setXmode] = useState("tokens");
  const [logy, setLogy] = useState(false);
  const [tab, setTab] = useState("charts");
  const [live, setLive] = useState(false);

  const reload = () => {
    api(`/api/runs/${encodeURIComponent(run)}`).then(setInfo).catch(() => {});
    api(`/api/runs/${encodeURIComponent(run)}/series`).then(setSeries).catch(() => {});
    api(`/api/runs/${encodeURIComponent(run)}/checkpoints`).then(setCkpts).catch(() => {});
    api(`/api/runs/${encodeURIComponent(run)}/events`).then(setEvents).catch(() => {});
    api(`/api/runs/${encodeURIComponent(run)}/samples`).then((s) => { setSamples(s); if (s.length && sampleTok == null) setSampleTok(s[s.length - 1].tokens); }).catch(() => {});
    api(`/api/runs/${encodeURIComponent(run)}/quality`).then((q) => { setQuality(q); const c = q.checkpoints || []; if (c.length && qualityTok == null) setQualityTok(c[c.length - 1].tokens); }).catch(() => {});
  };
  useEffect(reload, [run]);
  useEffect(() => {
    if (qualityTok == null) return;
    api(`/api/runs/${encodeURIComponent(run)}/quality/${qualityTok}`).then(setQualityDetail).catch(() => setQualityDetail(null));
  }, [run, qualityTok, quality]);

  useEffect(() => {
    if (sampleTok == null) return;
    api(`/api/runs/${encodeURIComponent(run)}/samples/${sampleTok}`).then(setSample).catch(() => setSample(null));
  }, [run, sampleTok]);

  // live tail: append records in place
  useEffect(() => {
    const es = new EventSource(`/api/runs/${encodeURIComponent(run)}/live`);
    let pending = [];
    let timer = null;
    const flush = () => {
      const recs = pending; pending = []; timer = null;
      const trains = recs.filter((r) => r.kind === "train"), evals = recs.filter((r) => r.kind === "eval");
      if (trains.length || evals.length) {
        setSeries((s) => {
          if (!s) return s;
          const t = { ...s.train }, e = { ...s.eval };
          for (const r of trains) { for (const k of Object.keys(t)) t[k] = [...t[k], k === "tokens" || k === "update" || k === "time" ? r[k] : r[k]]; }
          for (const r of evals) { for (const k of Object.keys(e)) e[k] = [...e[k], r[k]]; }
          return { ...s, train: t, eval: e, milestones: [...s.milestones, ...recs.filter((r) => r.kind === "milestone")] };
        });
      }
      if (recs.some((r) => ["eval", "milestone", "checkpoint", "stop", "finish", "start", "resume"].includes(r.kind))) reload();
      else api(`/api/runs/${encodeURIComponent(run)}`).then(setInfo).catch(() => {});
    };
    es.addEventListener("hello", () => setLive(true));
    es.addEventListener("record", (ev) => { pending.push(JSON.parse(ev.data)); if (!timer) timer = setTimeout(flush, 500); });
    es.onerror = () => setLive(false);
    return () => es.close();
  }, [run]);

  const xs = useMemo(() => {
    if (!series) return null;
    const t = series.train, e = series.eval;
    const tx = xmode === "tokens" ? t.tokens : xmode === "update" ? t.update : t.time.map((v) => v - (t.time[0] || v));
    const ex = xmode === "tokens" ? e.tokens : xmode === "update" ? (t.update.length ? e.tokens.map((tok) => nearestUpdate(t, tok)) : e.tokens) : e.time.map((v) => v - (t.time[0] || v));
    return { tx, ex };
  }, [series, xmode]);

  if (!info || !series || !xs) return html`<div>loading ${run}…</div>`;
  const s = info.summary, t = series.train, e = series.eval;
  const tokensX = xs.tx, evalX = xs.ex;
  const qc = (quality && quality.checkpoints || []).filter((c) => c.overall != null);
  const qualityX = qc.map((c) => xmode === "tokens" ? c.tokens : xmode === "update" ? nearestUpdate(t, c.tokens) : nearestTime(t, c.tokens));
  const QPAL = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4d7c0f", "#be185d", "#78716c"];
  const eta = s.status === "running" ? s.eta_s : null;
  return html`<div>
    <h1>${run} <span class=${"status " + s.status}>${s.status}</span> ${live && s.status === "running" ? html`<span class="muted" style="font-size:12px">● live</span>` : ""}</h1>
    <div class="sub">${s.stage} · started ${s.started || "?"} · ${s.gpu || ""} · git ${(s.git_commit || "").slice(0, 8)} · ${fmtInt(s.n_params)} params · ${s.has_report ? html`<a href=${`/api/runs/${encodeURIComponent(run)}/report`} target="_blank">report.html</a>` : ""}</div>
    <div class="bar"><div style=${"width:" + (s.progress * 100).toFixed(2) + "%"}></div></div>
    <div class="tiles">
      <${Tile} k="progress" v=${(s.progress * 100).toFixed(1) + "%"} s=${s.total_steps ? `step ${fmtInt(s.update)} / ${fmtInt(s.total_steps)} · ${fmtTok(s.tokens)} completion tokens` : `${fmtTok(s.tokens)} / ${fmtTok(s.total_tokens)} tokens`} />
      <${Tile} k="ETA" v=${eta != null ? fmtDur(eta) : "-"} s=${eta != null ? "finish ~" + new Date(Date.now() + eta * 1000).toLocaleString([], { weekday: "short", hour: "2-digit", minute: "2-digit" }) : ""} />
      <${Tile} k="elapsed" v=${fmtDur(s.elapsed_s)} s=${s.initial_estimate_s ? "initial estimate " + fmtDur(s.initial_estimate_s) : ""} />
      ${!s.is_rl && html`<${Tile} k="tokens/sec" v=${fmtInt(s.tok_s)} s=${"run avg " + fmtInt(s.tok_s_avg)} />`}
      ${s.is_rl ? html`
        <${Tile} k="reward (rollouts)" v=${fmtNum(s.rl.reward_mean, 3)} s=${`success ${(s.rl.success_rate * 100).toFixed(0)}% · step ${fmtInt(s.update)}`} />
        <${Tile} k="held-out accuracy" v=${s.heldout_acc != null ? (s.heldout_acc * 100).toFixed(1) + "%" : "-"} s="greedy, unseen prompts" />
        <${Tile} k="KL to reference" v=${fmtNum(s.rl.kl, 4)} s=${`entropy ${fmtNum(s.rl.entropy, 2)} · clip ${(s.rl.clip_frac * 100).toFixed(0)}%`} />
        <${Tile} k="completion length" v=${fmtNum(s.rl.len_mean, 0)} s=${`malformed ${(s.rl.malformed_rate * 100).toFixed(0)}% · no-signal groups ${(s.rl.groups_no_signal * 100).toFixed(0)}%`} />
        <${Tile} k="policy objective" v=${fmtNum(s.loss, 4)} s="≈0 by construction (zero-mean advantages)" />`
      : html`
        <${Tile} k="train loss" v=${fmtNum(s.loss, 4)} s=${"update " + fmtInt(s.update)} />
        <${Tile} k="val loss" v=${fmtNum(s.val_loss, 4)} s=${s.val_loss != null ? `best ${fmtNum(s.best_val, 4)} · ppl ${fmtNum(s.val_ppl, 1)}` : ""} />`}
      ${s.needle && html`<${Tile} k="needle (effective ctx)" v=${fmtInt(s.needle.effective)} s=${Object.entries(s.needle).filter(([k]) => /^\d+$/.test(k)).map(([k, v]) => `${k}: ${(v * 100).toFixed(0)}%`).join(" · ")} />`}
      <${Tile} k="lr" v=${fmtSci(s.lr)} />
      <${Tile} k="grad norm" v=${fmtNum(s.grad_norm, 3)} />
      <${Tile} k="VRAM peak" v=${fmtNum(s.vram_gib, 1) + " GiB"} />
      ${s.gpu_temp_c != null && html`<${Tile} k="GPU" v=${fmtNum(s.gpu_temp_c, 0) + " \u00b0C"} s=${`${fmtNum(s.gpu_power_w, 0)} W \u00b7 max ${fmtNum(s.gpu_temp_max_c, 0)} \u00b0C${s.gpu_temp_max_c >= 80 ? " \u26a0" : ""}`} />`}
      <${Tile} k="step time" v=${fmtNum(s.step_ms, 0) + " ms"} s=${`fwd ${fmtNum(s.fwd_ms, 0)} · bwd ${fmtNum(s.bwd_ms, 0)} · opt ${fmtNum(s.opt_ms, 0)} · data ${fmtNum(s.data_ms, 0)}`} />
    </div>
    <div class="row" style="margin:8px 0">
      ${["charts", "milestones", "samples", "quality", "checkpoints", "events", "config"].map((x) => html`<button class=${tab === x ? "active" : ""} onClick=${() => setTab(x)}>${x}</button>`)}
      ${tab === "charts" && html`<span class="muted" style="margin-left:14px">x:</span>
        ${["tokens", "update", "time"].map((x) => html`<button class=${xmode === x ? "active" : ""} onClick=${() => setXmode(x)}>${x}</button>`)}
        <button class=${logy ? "active" : ""} onClick=${() => setLogy(!logy)}>log y</button>`}
    </div>
    ${tab === "charts" && html`<div class="charts">
      <${Chart} title=${series.is_rl ? "policy objective (≈0 by construction; advantages are zero-mean per group)" : "train / val loss"} xmode=${xmode} logy=${logy} series=${series.is_rl ? [{ label: "objective", x: tokensX, y: t.loss }] : [{ label: "train", x: tokensX, y: t.loss }, { label: "val", x: evalX, y: e.val_loss, points: true, width: 2 }]} />
      ${!series.is_rl && html`<${Chart} title=${e.val_pt_loss && e.val_pt_loss.some((v) => v != null) ? "validation loss (task) vs pretraining-mixture val (drift)" : "validation loss"} xmode=${xmode} logy=${logy} series=${e.val_pt_loss && e.val_pt_loss.some((v) => v != null) ? [{ label: "val", x: evalX, y: e.val_loss, points: true, width: 2, color: "#dc2626" }, { label: "pretrain val", x: evalX, y: e.val_pt_loss, points: true, width: 2, color: "#9333ea" }] : [{ label: "val", x: evalX, y: e.val_loss, points: true, width: 2, color: "#dc2626" }]} />`}
      ${series.needle_keys && series.needle_keys.length > 0 && html`<${Chart} title="needle retrieval accuracy by context length (solid = mean over depths, dashed = minimum over depths)" xmode=${xmode} ymin=${0} series=${(() => { const means = series.needle_keys.filter((k) => !k.startsWith("needle_min_")); const palette = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]; return means.map((k, i) => ({ label: k.replace("needle_", "") + " mean", x: evalX, y: e[k], points: true, width: 2, color: palette[i % palette.length] })).concat(means.map((k, i) => ({ label: k.replace("needle_", "") + " min", x: evalX, y: e["needle_min_" + k.replace("needle_", "")], points: true, width: 1.5, dash: true, color: palette[i % palette.length] }))); })()} />`}
      ${qc.length > 0 && html`<${Chart} title=${`judged quality, 1-5 (${quality.n_prompts}-prompt suite ${quality.suite}, judge ${(quality.judges || []).join("/")}; grey = the 3 prompts sampled since M1)`} xmode=${xmode} ymin=${1} series=${[
        { label: "overall", x: qualityX, y: qc.map((c) => c.overall), points: true, width: 2.5, color: "#111827" },
        ...quality.rubrics.map((r, i) => ({ label: r, x: qualityX, y: qc.map((c) => c[r]), points: true, width: 1.5, color: QPAL[i] })),
        { label: "legacy 3", x: qualityX, y: qc.map((c) => c.legacy3), points: true, width: 1, dash: true, color: "#9ca3af" }]} />
      <${Chart} title="judged quality by category (overall, 1-5)" xmode=${xmode} ymin=${1} series=${quality.categories.map((cat, i) => ({ label: cat, x: qualityX, y: qc.map((c) => (c.categories[cat] || {}).overall), points: true, width: 1.5, color: QPAL[i % QPAL.length] }))} />`}
      ${!series.is_rl && html`<${Chart} title="tokens / sec" xmode=${xmode} ymin=${0} series=${[{ label: "tok/s", x: tokensX, y: t.tok_s }, { label: "ema", x: tokensX, y: t.tok_s_ema }]} />`}
      <${Chart} title="learning rate" xmode=${xmode} ymin=${0} series=${[{ label: "lr", x: tokensX, y: t.lr }]} />
      <${Chart} title="gradient norm" xmode=${xmode} ymin=${0} series=${[{ label: "grad norm", x: tokensX, y: t.grad_norm }]} />
      <${Chart} title="step time (ms)" xmode=${xmode} ymin=${0} series=${[{ label: "step", x: tokensX, y: t.step_ms }, { label: "fwd", x: tokensX, y: t.fwd_ms }, { label: "bwd", x: tokensX, y: t.bwd_ms }, { label: "opt", x: tokensX, y: t.opt_ms }, { label: "data", x: tokensX, y: t.data_ms }]} />
      <${Chart} title="VRAM (GiB): peak allocated and reserved" xmode=${xmode} ymin=${0} series=${[{ label: "peak", x: tokensX, y: t.vram_gib }, ...(hasData(t.vram_reserved_gib) ? [{ label: "reserved", x: tokensX, y: t.vram_reserved_gib }] : [])]} />
      ${t.gpu_temp_c && t.gpu_temp_c.some((v) => v != null) && html`<${Chart} title="GPU temperature (\u00b0C, left) / power (W, right)" xmode=${xmode} ymin=${0} ymin2=${0} series=${[{ label: "\u00b0C", x: tokensX, y: t.gpu_temp_c, color: "#dc2626" }, { label: "W", x: tokensX, y: t.gpu_power_w, scale: "y2", color: "#2563eb" }]} />`}
      ${series.is_rl && html`
        <${Chart} title="reward / success rate (train rollouts)" xmode=${xmode} ymin=${0} series=${[{ label: "reward", x: tokensX, y: t.reward_mean }, { label: "success", x: tokensX, y: t.success_rate }]} />
        <${Chart} title="held-out vs train accuracy (greedy)" xmode=${xmode} ymin=${0} series=${[{ label: "heldout", x: evalX, y: e.heldout_acc, points: true, width: 2 }, { label: "train", x: evalX, y: e.train_acc, points: true }]} />
        <${Chart} title="KL to reference / entropy" xmode=${xmode} ymin=${0} series=${[{ label: "kl", x: tokensX, y: t.kl }, { label: "entropy", x: tokensX, y: t.entropy }]} />
        <${Chart} title="completion length (tokens)" xmode=${xmode} ymin=${0} series=${[{ label: "mean", x: tokensX, y: t.len_mean }, { label: "correct", x: tokensX, y: t.len_correct }, { label: "wrong", x: tokensX, y: t.len_wrong }]} />
        <${Chart} title="malformed / length-terminated / no-signal groups" xmode=${xmode} ymin=${0} series=${[{ label: "malformed", x: tokensX, y: t.malformed_rate }, { label: "length-term", x: tokensX, y: t.length_term_rate }, { label: "no-signal", x: tokensX, y: t.groups_no_signal }]} />
        <${Chart} title="clip fraction / |advantage| / group reward std" xmode=${xmode} ymin=${0} series=${[{ label: "clip", x: tokensX, y: t.clip_frac }, { label: "|adv|", x: tokensX, y: t.adv_abs_mean }, { label: "group std", x: tokensX, y: t.group_std_mean }]} />`}
    </div>`}
    ${tab === "milestones" && html`<table><tr><th>tokens</th><th>segment time</th><th>elapsed</th><th>tok/s (segment)</th><th>train loss</th><th>val loss</th><th>at</th></tr>
      ${series.milestones.map((m) => html`<tr><td>${fmtTok(m.tokens)}</td><td>${fmtDur(m.segment_s)}</td><td>${fmtDur(m.elapsed_s)}</td><td>${fmtInt(m.tok_s)}</td><td>${fmtNum(m.loss, 4)}</td><td>${fmtNum(m.val_loss, 4)}</td><td>${fmtTime(m.time)}</td></tr>`)}
      ${series.milestones.length === 0 && html`<tr><td colspan="7" class="l">no milestones yet</td></tr>`}</table>`}
    ${tab === "samples" && html`<div>
      <div class="row" style="margin-bottom:8px"><span class="muted">checkpoint:</span>
        <select value=${sampleTok} onChange=${(ev) => setSampleTok(Number(ev.target.value))}>${samples.map((x) => html`<option value=${x.tokens}>${fmtTok(x.tokens)} tokens</option>`)}</select>
        ${samples.length > 1 && html`<input type="range" min="0" max=${samples.length - 1} value=${Math.max(0, samples.findIndex((x) => x.tokens === sampleTok))} onInput=${(ev) => setSampleTok(samples[Number(ev.target.value)].tokens)} style="width:300px" />`}
      </div>
      ${!sample ? html`<div class="empty-note">no samples yet</div>` : html`<div class="muted" style="font-size:12px;margin-bottom:6px">${sample.header}</div>
        <div class="samples">${sample.items.map((it) => html`<div class="sample"><div class="p">${it.prompt}</div>
          ${it.greedy != null && html`<div class="muted">greedy</div><pre>${it.greedy}</pre>`}
          <div class="muted">sampled</div><pre>${it.sampled}</pre></div>`)}</div>`}
    </div>`}
    ${tab === "quality" && html`<div>
      ${!(quality && quality.checkpoints && quality.checkpoints.length) ? html`<div class="empty-note">no judged-quality outputs for this run yet (python -m slm.eval.quality generate --run ${run}; see docs/quality_eval.md)</div>` : html`
        <div class="row" style="margin-bottom:8px"><span class="muted">checkpoint:</span>
          <select value=${qualityTok} onChange=${(ev) => setQualityTok(Number(ev.target.value))}>${quality.checkpoints.map((c) => html`<option value=${c.tokens}>${fmtTok(c.tokens)} tokens · ${c.checkpoint}${c.overall == null ? " (unjudged)" : ` · ${c.overall.toFixed(2)}`}</option>`)}</select>
          ${quality.checkpoints.length > 1 && html`<input type="range" min="0" max=${quality.checkpoints.length - 1} value=${Math.max(0, quality.checkpoints.findIndex((c) => c.tokens === qualityTok))} onInput=${(ev) => setQualityTok(quality.checkpoints[Number(ev.target.value)].tokens)} style="width:300px" />`}
          <span class="muted">suite ${quality.suite} · rubric ${quality.rubric} · judge ${(quality.judges || []).join(", ") || "-"}</span></div>
        ${(() => { const c = quality.checkpoints.find((x) => x.tokens === qualityTok); return c && c.overall != null ? html`<table style="margin-bottom:10px"><tr><th>category</th><th>n</th><th>overall</th>${quality.rubrics.map((r) => html`<th>${r}</th>`)}</tr>
          <tr><td><b>all</b></td><td>${c.n_scored}/${c.n_items}</td><td><b>${fmtNum(c.overall, 2)}</b></td>${quality.rubrics.map((r) => html`<td>${fmtNum(c[r], 2)}</td>`)}</tr>
          ${quality.categories.map((cat) => { const k = c.categories[cat] || {}; return html`<tr><td>${cat}</td><td>${k.n || 0}</td><td>${fmtNum(k.overall, 2)}</td>${quality.rubrics.map((r) => html`<td>${fmtNum(k[r], 2)}</td>`)}</tr>`; })}</table>` : html`<div class="muted" style="margin-bottom:8px">outputs generated, not judged yet</div>`; })()}
        ${!qualityDetail ? html`<div class="empty-note">…</div>` : html`<div class="muted" style="font-size:12px;margin-bottom:6px">${qualityDetail.header.checkpoint} · ${qualityDetail.header.stage} · generated ${qualityDetail.header.generated_at} on ${qualityDetail.header.device} in ${qualityDetail.header.seconds}s</div>
          <table><tr><th>prompt</th><th>category</th><th class="l">output (greedy)</th><th>correct</th><th>coherent</th><th>task</th><th class="l">judge note</th></tr>
          ${qualityDetail.items.map((it) => html`<tr><td class="l" title=${it.prompt}><b>${it.id}</b><div class="muted" style="white-space:pre-wrap;max-width:260px">${it.prompt}</div></td><td>${it.category}</td>
            <td class="l"><pre style="margin:0;max-height:160px;max-width:520px;overflow:auto;white-space:pre-wrap">${it.think ? "[think] " + it.think + "\n" : ""}${it.output}</pre></td>
            ${it.scores ? html`<td class=${scoreCls(it.scores.correctness)}>${it.scores.correctness}</td><td class=${scoreCls(it.scores.coherence)}>${it.scores.coherence}</td><td class=${scoreCls(it.scores.task)}>${it.scores.task}</td><td class="l">${it.note || ""}</td>` : html`<td colspan="4" class="l muted">not judged</td>`}</tr>`)}</table>`}`}
    </div>`}
    ${tab === "checkpoints" && html`<table><tr><th>file</th><th>kind</th><th>tokens</th><th>val loss</th><th>size</th><th>modified</th></tr>
      ${ckpts.map((c) => html`<tr><td>${c.name}</td><td>${c.kind}</td><td>${fmtTok(c.tokens)}</td><td>${fmtNum(c.val_loss, 4)}</td><td>${fmtBytes(c.bytes)}</td><td>${fmtTime(c.mtime)}</td></tr>`)}
      ${ckpts.length === 0 && html`<tr><td colspan="6" class="l">no checkpoints yet</td></tr>`}</table>`}
    ${tab === "events" && html`<table><tr><th>time</th><th>kind</th><th class="l">message</th></tr>
      ${events.slice().reverse().map((ev) => html`<tr><td>${fmtTime(ev.time)}</td><td>${ev.kind}</td><td class="l">${ev.msg || ""}</td></tr>`)}</table>`}
    ${tab === "config" && html`<div><h2>train config</h2><pre>${JSON.stringify(info.config, null, 1)}</pre><h2>model config</h2><pre>${JSON.stringify(info.model_config, null, 1)}</pre><h2>meta</h2><pre>${JSON.stringify(info.meta, null, 1)}</pre></div>`}
  </div>`;
}

function nearestTime(t, tokens) {
  let best = 0, bd = Infinity;
  for (let i = 0; i < t.tokens.length; i++) { const d = Math.abs(t.tokens[i] - tokens); if (d < bd) { bd = d; best = t.time[i] - (t.time[0] || t.time[i]); } }
  return best;
}

function scoreCls(v) { return v >= 4 ? "score-good" : v <= 2 ? "score-bad" : "score-mid"; }

function nearestUpdate(t, tokens) {
  let best = t.update[0], bd = Infinity;
  for (let i = 0; i < t.tokens.length; i++) { const d = Math.abs(t.tokens[i] - tokens); if (d < bd) { bd = d; best = t.update[i]; } }
  return best;
}

import { h } from "preact";
import { useEffect, useMemo, useState } from "preact/hooks";
import htm from "htm";
import { api, fmtTok, fmtDur, fmtNum, fmtInt, fmtSci, fmtBytes, fmtTime } from "../components/util.js";
import { Chart } from "../components/chart.js";

const html = htm.bind(h);

function Tile({ k, v, s }) {
  return html`<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s || ""}</div></div>`;
}

export function RunDetail({ run }) {
  const [info, setInfo] = useState(null);
  const [series, setSeries] = useState(null);
  const [ckpts, setCkpts] = useState([]);
  const [samples, setSamples] = useState([]);
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
  };
  useEffect(reload, [run]);
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
  const eta = s.status === "running" ? s.eta_s : null;
  return html`<div>
    <h1>${run} <span class=${"status " + s.status}>${s.status}</span> ${live && s.status === "running" ? html`<span class="muted" style="font-size:12px">● live</span>` : ""}</h1>
    <div class="sub">${s.stage} · started ${s.started || "?"} · ${s.gpu || ""} · git ${(s.git_commit || "").slice(0, 8)} · ${fmtInt(s.n_params)} params · ${s.has_report ? html`<a href=${`/api/runs/${encodeURIComponent(run)}/report`} target="_blank">report.html</a>` : ""}</div>
    <div class="bar"><div style=${"width:" + (s.progress * 100).toFixed(2) + "%"}></div></div>
    <div class="tiles">
      <${Tile} k="progress" v=${(s.progress * 100).toFixed(1) + "%"} s=${s.total_steps ? `step ${fmtInt(s.update)} / ${fmtInt(s.total_steps)} · ${fmtTok(s.tokens)} completion tokens` : `${fmtTok(s.tokens)} / ${fmtTok(s.total_tokens)} tokens`} />
      <${Tile} k="ETA" v=${eta != null ? fmtDur(eta) : "-"} s=${eta != null ? "finish ~" + new Date(Date.now() + eta * 1000).toLocaleString([], { weekday: "short", hour: "2-digit", minute: "2-digit" }) : ""} />
      <${Tile} k="elapsed" v=${fmtDur(s.elapsed_s)} s=${s.initial_estimate_s ? "initial estimate " + fmtDur(s.initial_estimate_s) : ""} />
      <${Tile} k="tokens/sec" v=${fmtInt(s.tok_s)} s=${"run avg " + fmtInt(s.tok_s_avg)} />
      <${Tile} k="train loss" v=${fmtNum(s.loss, 4)} s=${"update " + fmtInt(s.update)} />
      <${Tile} k="val loss" v=${fmtNum(s.val_loss, 4)} s=${s.val_loss != null ? `best ${fmtNum(s.best_val, 4)} · ppl ${fmtNum(s.val_ppl, 1)}` : ""} />
      <${Tile} k="lr" v=${fmtSci(s.lr)} />
      <${Tile} k="grad norm" v=${fmtNum(s.grad_norm, 3)} />
      <${Tile} k="VRAM peak" v=${fmtNum(s.vram_gib, 1) + " GiB"} />
      <${Tile} k="step time" v=${fmtNum(s.step_ms, 0) + " ms"} s=${`fwd ${fmtNum(s.fwd_ms, 0)} · bwd ${fmtNum(s.bwd_ms, 0)} · opt ${fmtNum(s.opt_ms, 0)} · data ${fmtNum(s.data_ms, 0)}`} />
    </div>
    <div class="row" style="margin:8px 0">
      ${["charts", "milestones", "samples", "checkpoints", "events", "config"].map((x) => html`<button class=${tab === x ? "active" : ""} onClick=${() => setTab(x)}>${x}</button>`)}
      ${tab === "charts" && html`<span class="muted" style="margin-left:14px">x:</span>
        ${["tokens", "update", "time"].map((x) => html`<button class=${xmode === x ? "active" : ""} onClick=${() => setXmode(x)}>${x}</button>`)}
        <button class=${logy ? "active" : ""} onClick=${() => setLogy(!logy)}>log y</button>`}
    </div>
    ${tab === "charts" && html`<div class="charts">
      <${Chart} title="train / val loss" xmode=${xmode} logy=${logy} series=${[{ label: "train", x: tokensX, y: t.loss }, { label: "val", x: evalX, y: e.val_loss, points: true, width: 2 }]} />
      <${Chart} title=${e.val_pt_loss && e.val_pt_loss.some((v) => v != null) ? "validation loss (task) vs pretraining-mixture val (drift)" : "validation loss"} xmode=${xmode} logy=${logy} series=${e.val_pt_loss && e.val_pt_loss.some((v) => v != null) ? [{ label: "val", x: evalX, y: e.val_loss, points: true, width: 2, color: "#dc2626" }, { label: "pretrain val", x: evalX, y: e.val_pt_loss, points: true, width: 2, color: "#9333ea" }] : [{ label: "val", x: evalX, y: e.val_loss, points: true, width: 2, color: "#dc2626" }]} />
      <${Chart} title="tokens / sec" xmode=${xmode} ymin=${0} series=${[{ label: "tok/s", x: tokensX, y: t.tok_s }, { label: "ema", x: tokensX, y: t.tok_s_ema }]} />
      <${Chart} title="learning rate" xmode=${xmode} ymin=${0} series=${[{ label: "lr", x: tokensX, y: t.lr }]} />
      <${Chart} title="gradient norm" xmode=${xmode} ymin=${0} series=${[{ label: "grad norm", x: tokensX, y: t.grad_norm }]} />
      <${Chart} title="step time (ms)" xmode=${xmode} ymin=${0} series=${[{ label: "step", x: tokensX, y: t.step_ms }, { label: "fwd", x: tokensX, y: t.fwd_ms }, { label: "bwd", x: tokensX, y: t.bwd_ms }, { label: "opt", x: tokensX, y: t.opt_ms }, { label: "data", x: tokensX, y: t.data_ms }]} />
      <${Chart} title="VRAM peak (GiB)" xmode=${xmode} ymin=${0} series=${[{ label: "GiB", x: tokensX, y: t.vram_gib }]} />
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
    ${tab === "checkpoints" && html`<table><tr><th>file</th><th>kind</th><th>tokens</th><th>val loss</th><th>size</th><th>modified</th></tr>
      ${ckpts.map((c) => html`<tr><td>${c.name}</td><td>${c.kind}</td><td>${fmtTok(c.tokens)}</td><td>${fmtNum(c.val_loss, 4)}</td><td>${fmtBytes(c.bytes)}</td><td>${fmtTime(c.mtime)}</td></tr>`)}
      ${ckpts.length === 0 && html`<tr><td colspan="6" class="l">no checkpoints yet</td></tr>`}</table>`}
    ${tab === "events" && html`<table><tr><th>time</th><th>kind</th><th class="l">message</th></tr>
      ${events.slice().reverse().map((ev) => html`<tr><td>${fmtTime(ev.time)}</td><td>${ev.kind}</td><td class="l">${ev.msg || ""}</td></tr>`)}</table>`}
    ${tab === "config" && html`<div><h2>train config</h2><pre>${JSON.stringify(info.config, null, 1)}</pre><h2>model config</h2><pre>${JSON.stringify(info.model_config, null, 1)}</pre><h2>meta</h2><pre>${JSON.stringify(info.meta, null, 1)}</pre></div>`}
  </div>`;
}

function nearestUpdate(t, tokens) {
  let best = t.update[0], bd = Infinity;
  for (let i = 0; i < t.tokens.length; i++) { const d = Math.abs(t.tokens[i] - tokens); if (d < bd) { bd = d; best = t.update[i]; } }
  return best;
}

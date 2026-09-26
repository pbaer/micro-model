import { h } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import htm from "htm";
import { api, readSSE } from "./util.js";
import { Info } from "./info.js";

const html = htm.bind(h);

/** Swarm mode of the Inference page (slm/swarm.py via POST /api/model/swarm): sample k answers, collapse them
 *  by final answer with sandbox evidence, then one greedy selector pass of the same model picks the answer. */

const STAGES = [
  ["sampling", "sample k"],
  ["collapsed", "collapse by answer"],
  ["selecting", "select"],
  ["done", "done"],
];

/** Approximates slm.swarm.answer_key for the optional expected answer: numbers by value, text by lowercase. */
const answerKey = (s) => {
  if (s == null) return null;
  const t = String(s).trim().replace(/^####\s*/, "");
  const num = t.replace(/[$,\s]/g, "").replace(/\.$/, "");
  if (num !== "" && isFinite(Number(num))) return String(Number(num));
  const txt = t.split(/\s+/).join(" ").replace(/^[ .]+|[ .]+$/g, "").toLowerCase();
  return txt || null;
};

const Mark = ({ ok }) => (ok == null ? html`<span class="muted">-</span>` : ok ? html`<span class="score-good">yes</span>` : html`<span class="score-bad">no</span>`);

function Candidate({ c }) {
  return html`<div class="swarm-cand">
    <div class="row" style="gap:6px">
      <b>#${c.idx}</b>
      ${c.verified ? html`<span class="stage-badge rl">verified</span>` : c.from_tool ? html`<span class="stage-badge reasoning" title="the answer came out of a call, but a call in this attempt errored">from tool, with errors</span>` : html`<span class="stage-badge">not computed</span>`}
      <span class="muted">${c.n_tokens} tokens · ${c.n_calls} call${c.n_calls === 1 ? "" : "s"}${c.n_errors ? ` (${c.n_errors} errored)` : ""}${c.terminated ? "" : " · did not close the turn"}</span>
    </div>
    ${c.think != null && html`<pre class="think">${c.think}</pre>`}
    ${c.calls.map(([code, result], i) => html`<div class="toolcall" key=${i}><pre class="code">${code}</pre><span class=${String(result).startsWith("error") ? "result err" : "result ok"}>${result}</span></div>`)}
    <pre class="swarm-answer">${c.answer || "(empty answer)"}</pre>
  </div>`;
}

function GroupRow({ g, cands, flags, open, toggle, gold }) {
  const correct = gold != null ? answerKey(g.answer) === gold : null;
  return html`<tr class=${"click" + (flags.final ? " sel" : "")} onClick=${toggle}>
      <td><b>${g.answer}</b> ${flags.final ? html`<span class="stage-badge rl">final</span>` : ""}${flags.majority ? html`<span class="stage-badge">majority</span>` : ""}${flags.vmaj ? html`<span class="stage-badge sft">verified maj.</span>` : ""}${correct ? html`<span class="stage-badge rl">expected</span>` : ""}</td>
      <td>${g.support}</td><td>${g.verified}</td><td><${Mark} ok=${flags.inPrompt} /></td>
      <td class="l" style="min-width:260px">${g.rationale ? html`<span class="swarm-rationale">${g.rationale}</span>` : html`<span class="muted">(no think span)</span>`}</td>
      <td>${open ? "hide" : "show"} ${g.members.length}</td>
    </tr>
    ${open && html`<tr><td colspan="6" class="l">${g.members.map((i) => html`<${Candidate} c=${cands[i]} key=${i} />`)}</td></tr>`}`;
}

export function SwarmPanel({ slots, onError, busy, setBusy }) {
  const [task, setTask] = useState("A baker bakes 7 trays of 12 muffins and sells 60 of them. How many muffins are left?");
  const [p, setP] = useState({ slot: "A", k: 16, temperature: 0.8, top_p: 0.95, max_new_tokens: 512, max_calls: 6, seed: "", budget_tokens: 2400, max_groups: 12, answer_suffix: true });
  const [gold, setGold] = useState("");
  const [stage, setStage] = useState(null);
  const [collapsed, setCollapsed] = useState(null);
  const [res, setRes] = useState(null);
  const [streamId, setStreamId] = useState(null);
  const [t0, setT0] = useState(null);
  const [now, setNow] = useState(Date.now());
  const [open, setOpen] = useState({});
  const [showUnparsed, setShowUnparsed] = useState(false);
  const abortRef = useRef(null);

  useEffect(() => { if (!t0) return; const id = setInterval(() => setNow(Date.now()), 500); return () => clearInterval(id); }, [t0]);
  const set = (k, v) => setP({ ...p, [k]: v });
  const num = (k, attrs = {}) => html`<input type="number" value=${p[k]} onChange=${(e) => set(k, Number(e.target.value))} style="width:70px" ...${attrs} />`;
  const loaded = slots[p.slot] && slots[p.slot].checkpoint;

  const run = async () => {
    onError(null); setRes(null); setCollapsed(null); setOpen({}); setShowUnparsed(false);
    setStage({ stage: "sampling" }); setT0(Date.now()); setBusy(true);
    const ctrl = new AbortController(); abortRef.current = ctrl;
    const body = { ...p, text: task, seed: p.seed === "" ? null : Number(p.seed) };
    try {
      const r = await fetch("/api/model/swarm", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal: ctrl.signal });
      await readSSE(r, (ev, data) => {
        if (ev === "start") setStreamId(data.stream_id);
        else if (ev === "stage") { setStage(data); if (data.stage === "collapsed") setCollapsed(data); }
        else if (ev === "done") { setStage({ stage: "done" }); setRes(data.result); }
        else if (ev === "error") { onError(data.error); setStage(null); }
      });
    } catch (e) { if (e.name !== "AbortError") onError(String(e)); setStage(null); }
    setBusy(false); setStreamId(null); setT0(null);
  };
  const cancel = () => { if (streamId) api(`/api/model/streams/${streamId}/cancel`, { method: "POST" }).catch(() => {}); };

  const groups = res ? res.groups : collapsed ? collapsed.groups : [];
  const cands = res ? res.candidates : [];
  const prompt = res && res.selector_messages.length ? res.selector_messages[0].content : "";
  const goldKey = gold.trim() ? answerKey(gold) : null;
  const ok = (a) => (goldKey == null || res == null ? null : a != null && answerKey(a) === goldKey);
  const finalKey = res && res.final != null ? answerKey(res.final) : null;
  const unparsed = cands.filter((c) => c.key == null);
  const cur = stage ? STAGES.findIndex(([s]) => s === stage.stage) : -1;
  const finalSource = !res ? "" : res.meta.cancelled ? "cancelled before the selector ran: verified majority" : res.meta.selector_parsed ? "picked by the selector" : res.groups.length ? "the selector gave no '####' line: verified majority" : "no candidate produced a parsable answer";

  return html`<div>
    <div class="panel">
      <div class="row" style="margin-bottom:6px">
        <b>Swarm</b><${Info} k="swarm" />
        <span class="muted">slot</span><select value=${p.slot} onChange=${(e) => set("slot", e.target.value)}><option>A</option><option>B</option></select>
        <span class="muted">k</span>${num("k", { min: 1, max: 64 })}<${Info} k="swarm_k" />
        <span class="muted">temp</span>${num("temperature", { step: 0.1, min: 0, max: 2 })}
        <span class="muted">top-p</span>${num("top_p", { step: 0.05, min: 0.05, max: 1 })}
        <span class="muted">max new</span>${num("max_new_tokens", { min: 1, max: 4096 })}
        <span class="muted">max tool calls</span>${num("max_calls", { min: 0, max: 16 })}
        <span class="muted">seed</span><input type="number" placeholder="random" value=${p.seed} onChange=${(e) => set("seed", e.target.value)} style="width:90px" />
      </div>
      <div class="row" style="margin-bottom:6px">
        <label class="muted"><input type="checkbox" checked=${p.answer_suffix} onChange=${(e) => set("answer_suffix", e.target.checked)} /> append answer instruction</label><${Info} k="swarm_suffix" />
        <span class="muted">selector budget</span>${num("budget_tokens", { min: 200, max: 8192, step: 100 })}<span class="muted">tokens</span>
        <span class="muted">max groups</span>${num("max_groups", { min: 1, max: 64 })}<${Info} k="swarm_budget" />
        <span class="muted">expected answer (optional)</span><input type="text" value=${gold} onInput=${(e) => setGold(e.target.value)} style="width:90px" /><${Info} k="swarm_oracle" />
      </div>
      <textarea value=${task} onInput=${(e) => setTask(e.target.value)} placeholder="the task prompt (a word problem with a single final answer works best)"></textarea>
      <div class="row" style="margin-top:6px">
        <button class="active" onClick=${run} disabled=${busy || !loaded || !task.trim()}>run swarm</button>
        <button onClick=${cancel} disabled=${!streamId}>cancel</button>
        ${!loaded && html`<span class="muted">load a checkpoint into slot ${p.slot} first (a reasoning / RL model with the Python tool; best.pt)</span>`}
        ${loaded && html`<span class="legend">${p.k} samples × up to ${p.max_new_tokens} tokens on ${slots[p.slot].device}${slots[p.slot].device === "cpu" ? " — slow on CPU; try k=4 and a small max new first" : ""}. Cancel takes effect at the next stage.</span>`}
      </div>
      ${stage && html`<div class="stage-steps">
        ${STAGES.map(([s, label], i) => html`<span class=${"step" + (i < cur ? " past" : i === cur ? (s === "done" ? " past" : " now") : "")} key=${s}>${label}</span>`)}
        <span class="muted">${t0 ? `${((now - t0) / 1000).toFixed(0)} s` : res ? `${res.seconds.toFixed(1)} s` : ""}${collapsed ? ` · ${collapsed.n_candidates} samples, ${collapsed.n_parsed} with a parsable answer, ${collapsed.n_verified} verified, ${collapsed.groups.length} distinct answers` : ""}${stage.stage === "selecting" ? ` · selector prompt ${stage.prompt_tokens} tokens, ${stage.groups_in_prompt} answers` : ""}</span>
      </div>`}
    </div>

    ${res && html`<div class="swarm-final">
      <div class="k">final answer<${Info} k="swarm_selector" /></div>
      <div class="v">${res.final ?? "none"} ${goldKey != null && html`<${Mark} ok=${ok(res.final)} />`}</div>
      <div class="muted">${finalSource} · seed ${res.meta.seed} <a href="#" onClick=${(e) => { e.preventDefault(); set("seed", String(res.meta.seed)); }}>reuse seed</a></div>
      <div class="row" style="margin-top:6px">
        <span>majority: <b>${res.majority ?? "-"}</b> ${goldKey != null && html`<${Mark} ok=${ok(res.majority)} />`}</span>
        <span>verified majority: <b>${res.verified_majority ?? "-"}</b> ${goldKey != null && html`<${Mark} ok=${ok(res.verified_majority)} />`}</span><${Info} k="swarm_majority" />
      </div>
    </div>`}

    ${res && goldKey != null && html`<h2>Ceilings for this task<${Info} k="swarm_oracle" /></h2>
      <table style="max-width:640px"><tr><th>stage</th><th class="l">the expected answer ...</th><th>holds</th></tr>
        <tr><td>oracle (pass@k)</td><td class="l">is among the ${res.k} samples</td><td><${Mark} ok=${cands.some((c) => answerKey(c.parsed) === goldKey)} /></td></tr>
        <tr><td>oracle, verified</td><td class="l">is among the verified samples</td><td><${Mark} ok=${cands.some((c) => c.verified && answerKey(c.parsed) === goldKey)} /></td></tr>
        <tr><td>in prompt</td><td class="l">survived into the selector prompt</td><td><${Mark} ok=${groups.some((g) => answerKey(g.answer) === goldKey && prompt.includes(`- Answer: ${g.answer} (`))} /></td></tr>
        <tr><td>selector</td><td class="l">was picked</td><td><${Mark} ok=${ok(res.final)} /></td></tr>
      </table>
      <div class="legend">Client-side match (numbers by value, text by lowercase), close to the eval's but not the same verifier.</div>`}

    ${groups.length > 0 && html`<h2>Distinct answers (${groups.length})<${Info} k="swarm_support" /></h2>
      <table>
        <tr><th>answer</th><th>support</th><th>verified</th><th>in selector prompt<${Info} k="swarm_budget" /></th><th class="l">representative rationale</th><th>members</th></tr>
        ${groups.map((g) => html`<${GroupRow} key=${g.key} g=${g} cands=${cands} gold=${goldKey} open=${!!open[g.key] && cands.length > 0}
          toggle=${() => cands.length && setOpen({ ...open, [g.key]: !open[g.key] })}
          flags=${{ final: res && finalKey != null && answerKey(g.answer) === finalKey, majority: g.answer === (res || collapsed).majority, vmaj: g.answer === (res || collapsed).verified_majority, inPrompt: res ? prompt.includes(`- Answer: ${g.answer} (`) : null }} />`)}
      </table>
      <div class="legend">Sorted by verified support, then support. Click a row for its member samples (think span, tool calls with results, answer).</div>`}
    ${res && unparsed.length > 0 && html`<div style="margin-top:6px"><a href="#" onClick=${(e) => { e.preventDefault(); setShowUnparsed(!showUnparsed); }}>${showUnparsed ? "hide" : "show"} ${unparsed.length} sample${unparsed.length === 1 ? "" : "s"} with no parsable answer</a>
      ${showUnparsed && unparsed.map((c) => html`<${Candidate} c=${c} key=${c.idx} />`)}</div>`}
    ${res && res.groups.length === 0 && html`<div class="panel warn" style="margin-top:8px">No sample produced a parsable final answer, so there was nothing to select. Check that the slot holds a reasoning / RL checkpoint, that the answer instruction is on, and that max new tokens leaves room to finish.</div>`}

    ${res && res.selector_messages.length > 0 && html`<h2>Selector pass<${Info} k="swarm_selector" /></h2>
      <details><summary class="muted">selector prompt (${res.meta.prompt_tokens} tokens, ${res.meta.groups_in_prompt} of ${res.groups.length} answers, budget ${res.meta.budget_tokens})</summary><pre>${prompt}</pre></details>
      ${res.meta.cancelled ? html`<div class="muted">cancelled: the selector did not run.</div>` : html`
        <div class="muted" style="margin:6px 0 2px">think${res.selector_calls ? ` · ${res.selector_calls} python call${res.selector_calls > 1 ? "s" : ""}` : ""}</div>
        <pre class="think">${res.selector_think ?? "(no think span)"}</pre>
        <div class="muted" style="margin:6px 0 2px">answer</div>
        <pre class="swarm-answer">${res.selector_answer || "(empty)"}</pre>`}`}
  </div>`;
}

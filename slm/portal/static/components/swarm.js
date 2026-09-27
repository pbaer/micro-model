import { h } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import htm from "htm";
import { api, readSSE } from "./util.js";
import { Info } from "./info.js";

const html = htm.bind(h);

/** Swarm mode of the Inference page (slm/swarm.py via POST /api/model/swarm): sample k answers, collapse them
 *  by final answer with sandbox evidence, then pick the answer with the same model: one selector pass over all the
 *  distinct answers ("select"), a pairwise single-elimination bracket ("tournament"), or both side by side. */

const MODES = [
  ["select", "select", "one selector prompt over all distinct answers"],
  ["tournament", "tournament", "pairwise single-elimination bracket; the champion is the final answer"],
  ["both", "both", "run both and compare; the final answer follows the tournament"],
];

const stagesFor = (mode) => [
  ["sampling", "sample k"],
  ["collapsed", "collapse by answer"],
  ...(mode !== "tournament" ? [["selecting", "select"]] : []),
  ...(mode !== "select" ? [["tournament", "tournament"]] : []),
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

// ------------------------------------------------------------------------------------------------ bracket model
/** The bracket as slm.swarm.tournament plays it, rebuilt from the seeded entrants and the round logs seen so far:
 *  seed_pairs (first vs last, the middle one gets a bye when the count is odd), every odd-indexed pair presented
 *  swapped, winners then byes go on to the next round. An entrant is a slot {answer | null, from}: `from` points at
 *  the item it came out of, so an undecided slot still knows it is "the winner of R1 M2". Rounds that have not been
 *  decided are laid out with such placeholder slots, so the whole shape is visible from the start. */
export function buildBracket(entrants, rounds) {
  const cols = [];
  let cur = entrants.map((a) => ({ answer: a, from: null }));
  while (cur.length > 1) {
    const r = cols.length, n = cur.length, log = rounds[r];
    const items = [];
    for (let j = 0; j < Math.floor(n / 2); j++) {
      const swapped = j % 2 === 1;
      const [x, y] = swapped ? [cur[n - 1 - j], cur[j]] : [cur[j], cur[n - 1 - j]];
      const m = log && log[j];
      items.push({ kind: "match", id: `R${r + 1} M${j + 1}`, rows: [x, y], swapped, decided: !!m, pick: m ? m.pick : undefined,
        winner: m ? m.winner : null, winnerRow: m ? (m.winner === x.answer ? 0 : 1) : null,
        mismatch: !!m && (m.a !== x.answer || m.b !== y.answer) });
    }
    if (n % 2) items.push({ kind: "bye", id: `R${r + 1} bye`, rows: [cur[Math.floor(n / 2)]], decided: cur[Math.floor(n / 2)].answer != null });
    cols.push({ round: r + 1, items, decided: !!log });
    cur = items.map((it, i) => ({
      answer: it.kind === "bye" ? it.rows[0].answer : it.winner,
      from: { col: r, item: i },
    }));
  }
  return { cols, champion: cur.length === 1 ? cur[0] : null };
}

const COL_W = 214, GAP_X = 46, ROW_H = 22, HEAD_H = 18, PAD = 3, GAP_Y = 10, TOP = 26;
const itemH = (it) => HEAD_H + PAD * 2 + ROW_H * it.rows.length;
const rowY = (it, i) => it.y + HEAD_H + PAD + ROW_H * i + ROW_H / 2;  // centre of row i
const outY = (it) => (it.kind === "match" && it.winnerRow != null ? rowY(it, it.winnerRow) : it.kind === "bye" ? rowY(it, 0) : it.y + itemH(it) / 2);

/** Positions: round 1 stacked top to bottom; every later item centred on the rows it was fed from, then pushed down
 *  just enough not to overlap its neighbour. */
function layout(br, nEntrants) {
  br.cols.forEach((col, c) => {
    col.x = c * (COL_W + GAP_X);
    const want = col.items.map((it) => {
      if (c === 0) return 0;
      const src = it.rows.map((s) => outY(br.cols[c - 1].items[s.from.item]));
      return src.reduce((a, b) => a + b, 0) / src.length - (HEAD_H + PAD + (ROW_H * it.rows.length) / 2);
    });
    const order = col.items.map((_, i) => i);
    if (c > 0) order.sort((a, b) => want[a] - want[b]);
    let bottom = TOP - GAP_Y;
    for (const i of order) {
      const it = col.items[i];
      it.y = c === 0 ? bottom + GAP_Y : Math.max(want[i], bottom + GAP_Y);
      bottom = it.y + itemH(it);
    }
  });
  const last = br.cols[br.cols.length - 1];
  const champ = { x: br.cols.length * (COL_W + GAP_X), h: 78 };
  champ.y = last ? Math.max(TOP, outY(last.items[0]) - champ.h / 2) : TOP;
  const height = Math.max(champ.y + champ.h, ...br.cols.flatMap((c) => c.items.map((it) => it.y + itemH(it)))) + 8;
  return { champ, width: champ.x + COL_W, height: nEntrants ? height : 0 };
}

/** Where each answer ended up: champion, the round and opponent it lost to, still in, or not entered. */
function fates(br, entrants) {
  const out = {};
  entrants.forEach((a) => { out[a] = { entered: true, lost: null, bye: [] }; });
  br.cols.forEach((col) => col.items.forEach((it) => {
    if (it.kind === "bye" && it.rows[0].answer != null && out[it.rows[0].answer]) out[it.rows[0].answer].bye.push(col.round);
    if (it.kind !== "match" || !it.decided) return;
    const loser = it.rows[1 - it.winnerRow].answer;
    if (out[loser]) out[loser].lost = { round: col.round, to: it.winner, pick: it.pick, id: it.id };
  }));
  return out;
}

function EntrantRow({ slot, label, info, state, tags }) {
  if (slot.answer == null) {
    return html`<div class="br-row pending"><span class="br-ab">${label}</span><span class="muted">winner of ${slot.from ? `${slot.fromId}` : "?"}</span></div>`;
  }
  const g = info[slot.answer] || {};
  return html`<div class=${"br-row " + state} title=${`${slot.answer} · seed ${g.seed ?? "?"} · support ${g.support ?? "?"} · verified ${g.verified ?? "?"}`}>
    ${label && html`<span class="br-ab">${label}</span>`}
    <span class="br-seed">#${g.seed ?? "?"}</span>
    <span class="br-ans">${slot.answer}</span>
    <span class="br-ev">×${g.support ?? "?"}${g.verified ? html` <span class="br-ver" title="verified: computed with a Python call that ran without error">✓${g.verified}</span>` : ""}</span>
    ${tags.map((t) => html`<span class=${"br-tag " + t[0]} title=${t[2]} key=${t[0]}>${t[1]}</span>`)}
  </div>`;
}

export function Bracket({ groups, entrants, rounds, running, champion, marks, external }) {
  const info = {};
  groups.forEach((g, i) => { info[g.answer] = { seed: i + 1, support: g.support, verified: g.verified }; });
  const br = buildBracket(entrants, rounds);
  const L = layout(br, entrants.length);
  br.cols.forEach((col) => col.items.forEach((it) => it.rows.forEach((s) => { if (s.from) s.fromId = br.cols[s.from.col].items[s.from.item].id; })));
  const current = br.cols.findIndex((c) => !c.decided);
  const tagsFor = (a) => {
    const t = [];
    if (marks.gold != null && answerKey(a) === marks.gold) t.push(["gold", "exp", "the expected answer you entered"]);
    if (marks.selector != null && a === marks.selector) t.push(["sel", "sel", "the selector's pick (one prompt over all answers)"]);
    if (a === marks.vmaj) t.push(["vmaj", "vmaj", "verified majority"]);
    if (a === marks.majority) t.push(["maj", "maj", "majority"]);
    return t;
  };
  const lines = [];
  br.cols.forEach((col, c) => {
    if (c === 0) return;
    col.items.forEach((it) => it.rows.forEach((s, ri) => {
      const src = br.cols[c - 1].items[s.from.item];
      const x1 = br.cols[c - 1].x + COL_W, y1 = outY(src), x2 = col.x, y2 = rowY(it, ri), mx = x1 + GAP_X / 2;
      const hot = champion != null && s.answer === champion;
      lines.push(html`<path d=${`M${x1} ${y1} H${mx} V${y2} H${x2}`} class=${"br-line" + (s.answer != null ? " on" : "") + (hot ? " hot" : "")} key=${`${c}-${it.id}-${ri}-${s.answer}`} />`);
    }));
  });
  const last = br.cols[br.cols.length - 1];
  if (last) {
    const src = last.items[0], x1 = last.x + COL_W, y1 = outY(src), x2 = L.champ.x, y2 = L.champ.y + L.champ.h / 2, mx = x1 + GAP_X / 2;
    lines.push(html`<path d=${`M${x1} ${y1} H${mx} V${y2} H${x2}`} class=${"br-line" + (champion != null ? " on hot" : "")} key=${"champ-" + champion} />`);
  }
  const champInfo = champion != null ? info[champion] || {} : null;
  return html`<div class="bracket-scroll"><div class="bracket" style=${`width:${L.width}px;height:${L.height}px`}>
    <svg class="br-lines" width=${L.width} height=${L.height}>${lines}</svg>
    ${br.cols.map((col, c) => html`<div class="br-colhead" style=${`left:${col.x}px;width:${COL_W}px`} key=${"h" + c}>
      Round ${col.round} · ${col.items.filter((i) => i.kind === "match").length} match${col.items.filter((i) => i.kind === "match").length === 1 ? "" : "es"}${col.items.some((i) => i.kind === "bye") ? ", 1 bye" : ""}
      ${col.decided ? "" : c === current && running ? html` <span class="br-live">deciding…</span>` : ""}</div>`)}
    <div class="br-colhead" style=${`left:${L.champ.x}px;width:${COL_W}px`}>champion</div>
    ${br.cols.map((col) => col.items.map((it) => {
      const cls = "br-item " + it.kind + (it.decided ? " decided" : "") + (it.kind === "match" && !it.decided && running && br.cols.indexOf(col) === current ? " live" : "");
      const head = it.kind === "bye"
        ? html`<span>${it.id}</span><span class="muted" title="odd number of entrants: the middle seed advances without a comparison">advances unopposed</span>`
        : html`<span>${it.id}</span>
          ${it.swapped ? html`<span class="br-swap" title="presented swapped: the better seed is shown as B. Every other pair is swapped so the model's position bias cancels across the bracket instead of deciding it.">⇄ swapped</span>` : ""}
          <span class="br-pick">${!it.decided ? "" : it.pick == null ? html`<span class="br-fallback" title="no parsable '#### A' / '#### B': the evidence order decides (verified, then support; ties to A)">no pick → evidence</span>` : `picked ${it.pick === 0 ? "A" : "B"}`}</span>`;
      return html`<div class=${cls} style=${`left:${col.x}px;top:${it.y}px;width:${COL_W}px`} key=${`${it.id}-${it.decided}-${it.rows.map((r) => r.answer).join("|")}`}>
        <div class="br-head">${head}</div>
        ${it.rows.map((s, ri) => html`<${EntrantRow} slot=${s} info=${info} key=${ri} label=${it.kind === "match" ? (ri === 0 ? "A" : "B") : ""}
          state=${it.kind === "bye" ? "bye" : !it.decided ? "" : ri === it.winnerRow ? "win" : "lose"} tags=${tagsFor(s.answer)} />`)}
        ${it.mismatch ? html`<div class="br-warn">server order differs from the replayed seeding</div>` : ""}
      </div>`;
    }))}
    <div class=${"br-item champ" + (champion != null ? " decided" : "")} style=${`left:${L.champ.x}px;top:${L.champ.y}px;width:${COL_W}px;height:${L.champ.h}px`} key=${"champ-" + champion}>
      ${champion != null ? html`<div class="br-champ-v">${champion}</div>
        <div class="muted">seed #${champInfo.seed ?? "?"} · support ${champInfo.support ?? "?"}${champInfo.verified ? ` · verified ${champInfo.verified}` : external ? " · verification n/a" : " · not computed"} ${tagsFor(champion).map((t) => html`<span class=${"br-tag " + t[0]} key=${t[0]}>${t[1]}</span>`)}</div>`
        : html`<div class="muted" style="padding:8px">${running ? "to be decided" : "no champion"}</div>`}
    </div>
  </div></div>`;
}

// ------------------------------------------------------------------------------------------------ candidates / groups
function Candidate({ c }) {
  return html`<div class="swarm-cand">
    <div class="row" style="gap:6px">
      <b>#${c.idx}</b>
      ${c.verified == null ? html`<span class="stage-badge external" title="external model: no sandbox, so no verification">verification n/a</span>` : c.verified ? html`<span class="stage-badge rl">verified</span>` : c.from_tool ? html`<span class="stage-badge reasoning" title="the answer came out of a call, but a call in this attempt errored">from tool, with errors</span>` : html`<span class="stage-badge">not computed</span>`}
      <span class="muted">${c.n_tokens} tokens · ${c.n_calls} call${c.n_calls === 1 ? "" : "s"}${c.n_errors ? ` (${c.n_errors} errored)` : ""}${c.terminated ? "" : " · did not close the turn"}</span>
    </div>
    ${c.think != null && html`<pre class="think">${c.think}</pre>`}
    ${c.calls.map(([code, result], i) => html`<div class="toolcall" key=${i}><pre class="code">${code}</pre><span class=${String(result).startsWith("error") ? "result err" : "result ok"}>${result}</span></div>`)}
    <pre class="swarm-answer">${c.answer || "(empty answer)"}</pre>
  </div>`;
}

const fateText = (f, champion, finished) => {
  if (!f) return html`<span class="muted" title="beyond max entrants: the bracket takes the first answers in evidence order">not entered</span>`;
  if (champion != null && f.champion) return html`<span class="stage-badge rl">champion</span>`;
  if (f.lost) return html`<span title=${f.lost.pick == null ? "no parsable pick: decided by the evidence order" : `the model picked ${f.lost.pick === 0 ? "A" : "B"}`}>lost R${f.lost.round} to <b>${f.lost.to}</b>${f.lost.pick == null ? " (evidence)" : ""}</span>`;
  return html`<span class="muted">${finished ? "-" : "still in"}</span>`;
};

function GroupRow({ g, cands, flags, open, toggle, gold, showPrompt, showBracket, external }) {
  const correct = gold != null ? answerKey(g.answer) === gold : null;
  const ncol = 5 + (showPrompt ? 1 : 0) + (showBracket ? 1 : 0);
  return html`<tr class=${"click" + (flags.final ? " sel" : "")} onClick=${toggle}>
      <td><b>${g.answer}</b> ${flags.final ? html`<span class="stage-badge rl">final</span>` : ""}${flags.majority ? html`<span class="stage-badge">majority</span>` : ""}${flags.vmaj ? html`<span class="stage-badge sft">verified maj.</span>` : ""}${flags.selector ? html`<span class="stage-badge reasoning">selector</span>` : ""}${correct ? html`<span class="stage-badge rl">expected</span>` : ""}</td>
      <td>${g.support}</td><td>${external ? html`<span class="muted">n/a</span>` : g.verified}</td>
      ${showPrompt && html`<td><${Mark} ok=${flags.inPrompt} /></td>`}
      ${showBracket && html`<td class="l">${flags.fate}</td>`}
      <td class="l" style="min-width:260px">${g.rationale ? html`<span class="swarm-rationale">${g.rationale}</span>` : html`<span class="muted">(no think span)</span>`}</td>
      <td>${open ? "hide" : "show"} ${g.members.length}</td>
    </tr>
    ${open && html`<tr><td colspan=${ncol} class="l">${g.members.map((i) => html`<${Candidate} c=${cands[i]} key=${i} />`)}</td></tr>`}`;
}

// ------------------------------------------------------------------------------------------------ panel
export function SwarmPanel({ slots, onError, busy, setBusy }) {
  const [task, setTask] = useState("A baker bakes 7 trays of 12 muffins and sells 60 of them. How many muffins are left?");
  const [p, setP] = useState({ slot: "A", k: 16, temperature: 0.8, top_p: 0.95, max_new_tokens: 512, max_calls: 6, seed: "", budget_tokens: 2400, max_groups: 12,
    answer_suffix: true, mode: "both", pair_budget_tokens: 1200, max_entrants: 16 });
  const [runP, setRunP] = useState(null);  // the settings of the run on screen (the form may change meanwhile)
  const [gold, setGold] = useState("");
  const [stage, setStage] = useState(null);
  const [collapsed, setCollapsed] = useState(null);
  const [seeded, setSeeded] = useState(null);  // tournament round 0: the entrants in seed order
  const [rounds, setRounds] = useState([]);    // decided rounds, each the library's match log
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
  const slotInfo = slots[p.slot] || {};
  const extSlot = !!slotInfo.external, extBase = extSlot && !slotInfo.is_chat;  // an external base model has no chat template: n/a

  const run = async () => {
    onError(null); setRes(null); setCollapsed(null); setSeeded(null); setRounds([]); setOpen({}); setShowUnparsed(false);
    setStage({ stage: "sampling" }); setT0(Date.now()); setBusy(true); setRunP({ ...p });
    const ctrl = new AbortController(); abortRef.current = ctrl;
    const body = { ...p, text: task, seed: p.seed === "" ? null : Number(p.seed) };
    try {
      const r = await fetch("/api/model/swarm", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal: ctrl.signal });
      await readSSE(r, (ev, data) => {
        if (ev === "start") setStreamId(data.stream_id);
        else if (ev === "stage") {
          setStage(data);
          if (data.stage === "collapsed") setCollapsed(data);
          if (data.stage === "tournament") {
            if (data.round === 0) setSeeded(data);
            else setRounds((rs) => [...rs.slice(0, data.round - 1), data.matches]);
          }
        }
        else if (ev === "done") { setStage({ stage: "done" }); setRes(data.result); }
        else if (ev === "error") { onError(data.error); setStage(null); }
      });
    } catch (e) { if (e.name !== "AbortError") onError(String(e)); setStage(null); }
    setBusy(false); setStreamId(null); setT0(null);
  };
  const cancel = () => { if (streamId) api(`/api/model/streams/${streamId}/cancel`, { method: "POST" }).catch(() => {}); };

  const mode = res ? res.meta.mode : runP ? runP.mode : p.mode;
  const STAGES = stagesFor(mode);
  const groups = res ? res.groups : collapsed ? collapsed.groups : [];
  const cands = res ? res.candidates : [];
  const prompt = res && res.selector_messages.length ? res.selector_messages[0].content : "";
  const goldKey = gold.trim() ? answerKey(gold) : null;
  const ok = (a) => (goldKey == null || res == null ? null : a != null && answerKey(a) === goldKey);
  const finalKey = res && res.final != null ? answerKey(res.final) : null;
  const unparsed = cands.filter((c) => c.key == null);
  const cur = stage ? STAGES.findIndex(([s]) => s === stage.stage) : -1;
  const hasSelector = mode !== "tournament";
  const hasBracket = mode !== "select";
  const selectorPick = res ? res.meta.selector_final : null;

  // the bracket: entrants from round 0 (or, before it arrives, the seeding the worker will use), rounds as they are decided
  const maxEntrants = res ? res.meta.max_entrants : runP ? runP.max_entrants : p.max_entrants;
  const entrants = seeded ? seeded.entrants : groups.slice(0, maxEntrants).map((g) => g.answer);
  const bracketRounds = res ? res.rounds : rounds;
  const champion = res ? res.tournament : null;
  const running = !!t0 && !res;
  const brModel = hasBracket && entrants.length ? buildBracket(entrants, bracketRounds) : null;
  const fate = brModel ? fates(brModel, entrants) : {};
  if (champion != null && fate[champion]) fate[champion].champion = true;
  const bracketDone = !!res;
  const disagree = res && mode === "both" && res.tournament != null && selectorPick != null && answerKey(res.tournament) !== answerKey(selectorPick);
  const selGroup = selectorPick != null ? groups.find((g) => answerKey(g.answer) === answerKey(selectorPick)) : null;
  const selFate = selGroup ? fate[selGroup.answer] : null;

  const fbName = res && res.meta.external ? "majority (verification n/a)" : "verified majority";
  const finalSource = !res ? "" : !res.groups.length ? "no candidate produced a parsable answer"
    : mode === "select" ? (res.meta.cancelled ? `cancelled before the selector ran: ${fbName}` : res.meta.selector_parsed ? "picked by the selector" : `the selector gave no '####' line: ${fbName}`)
    : res.tournament != null ? `the tournament champion (${res.rounds.length} round${res.rounds.length === 1 ? "" : "s"}, ${res.meta.n_entrants} entrants)`
    : res.meta.cancelled ? `cancelled before the bracket finished: ${fbName}` : fbName;
  const tstage = stage && stage.stage === "tournament" ? stage : null;
  const naRun = res ? !!res.meta.external : collapsed ? !!collapsed.external : extSlot;  // the run on screen had no sandbox: verification n/a

  return html`<div>
    <div class="panel">
      <div class="row" style="margin-bottom:6px">
        <b>Swarm</b><${Info} k="swarm" />
        <span class="muted">slot</span><select value=${p.slot} onChange=${(e) => set("slot", e.target.value)}><option>A</option><option>B</option></select>
        <span class="muted">k</span>${num("k", { min: 1, max: 64 })}<${Info} k="swarm_k" />
        <span class="muted">temp</span>${num("temperature", { step: 0.1, min: 0, max: 2 })}
        <span class="muted">top-p</span>${num("top_p", { step: 0.05, min: 0.05, max: 1 })}
        <span class="muted">max new</span>${num("max_new_tokens", { min: 1, max: 4096 })}
        <span class="muted">max tool calls</span>${num("max_calls", { min: 0, max: 16, disabled: extSlot, title: extSlot ? "n/a: an external model has no Python tool" : "" })}
        <span class="muted">seed</span><input type="number" placeholder="random" value=${p.seed} onChange=${(e) => set("seed", e.target.value)} style="width:90px" />
      </div>
      <div class="row" style="margin-bottom:6px">
        <span class="grp"><span class="muted">pick by</span>
          <span class="seg">${MODES.map(([m, label, tip]) => html`<button class=${p.mode === m ? "active" : ""} title=${tip} onClick=${() => set("mode", m)} key=${m}>${label}</button>`)}</span><${Info} k="swarm_tournament" /></span>
        ${p.mode !== "tournament" && html`<span class="grp"><span class="muted">selector budget</span>${num("budget_tokens", { min: 200, max: 8192, step: 100 })}
          <span class="muted">max groups</span>${num("max_groups", { min: 1, max: 64 })}<${Info} k="swarm_budget" /></span>`}
        ${p.mode !== "select" && html`<span class="grp"><span class="muted">pair budget</span>${num("pair_budget_tokens", { min: 200, max: 8192, step: 100 })}
          <span class="muted">max entrants</span>${num("max_entrants", { min: 2, max: 64 })}<${Info} k="swarm_seeding" /></span>`}
      </div>
      <div class="row" style="margin-bottom:6px">
        <label class="muted"><input type="checkbox" checked=${p.answer_suffix} onChange=${(e) => set("answer_suffix", e.target.checked)} /> append answer instruction</label><${Info} k="swarm_suffix" />
        <span class="muted">expected answer (optional)</span><input type="text" value=${gold} onInput=${(e) => setGold(e.target.value)} style="width:90px" /><${Info} k="swarm_oracle" />
      </div>
      <textarea value=${task} onInput=${(e) => setTask(e.target.value)} placeholder="the task prompt (a word problem with a single final answer works best)"></textarea>
      <div class="row" style="margin-top:6px">
        <button class="active" onClick=${run} disabled=${busy || !loaded || !task.trim() || extBase}>run swarm</button>
        <button onClick=${cancel} disabled=${!streamId}>cancel</button>
        ${!loaded && html`<span class="muted">load a checkpoint into slot ${p.slot} first (a reasoning / RL model with the Python tool; best.pt)</span>`}
        ${extBase && html`<span class="muted">swarm is n/a for ${slotInfo.name}: an external base model (it samples chat replies, and a base model gets no chat template)</span>`}
        ${loaded && !extBase && html`<span class="legend">${p.k} samples × up to ${p.max_new_tokens} tokens on ${slots[p.slot].device}${slots[p.slot].device === "cpu" ? " — slow on CPU; try k=4 and a small max new first" : ""}. Cancel takes effect at the next stage${p.mode !== "select" ? " or bracket round" : ""}.</span>`}
      </div>
      ${extSlot && !extBase && html`<div class="swarm-na"><span class="stage-badge external">external</span> <b>${slotInfo.name}</b>: samples, selector and pairwise prompts go through its own chat template (greedy for the judgments). <b>Verification n/a</b>: no Python tool, no sandbox, so no answer is ever verified; the verified majority is n/a and every fallback uses the plain majority.<${Info} k="external_slot" /></div>`}
      ${stage && html`<div class="stage-steps">
        ${STAGES.map(([s, label], i) => html`<span class=${"step" + (i < cur ? " past" : i === cur ? (s === "done" ? " past" : " now") : "")} key=${s}>${label}${s === "tournament" && tstage ? ` ${Math.min(tstage.round + 1, tstage.n_rounds_expected)}/${tstage.n_rounds_expected}` : ""}</span>`)}
        <span class="muted">${t0 ? `${((now - t0) / 1000).toFixed(0)} s` : res ? `${res.seconds.toFixed(1)} s` : ""}${collapsed ? ` · ${collapsed.n_candidates} samples, ${collapsed.n_parsed} with a parsable answer, ${collapsed.n_verified == null ? "verification n/a" : `${collapsed.n_verified} verified`}, ${collapsed.groups.length} distinct answers` : ""}${stage.stage === "selecting" ? ` · selector prompt ${stage.prompt_tokens} tokens, ${stage.groups_in_prompt} answers` : ""}${tstage ? ` · bracket: ${tstage.round} of ${tstage.n_rounds_expected} round${tstage.n_rounds_expected === 1 ? "" : "s"} decided` : ""}</span>
      </div>`}
    </div>

    ${res && html`<div class="swarm-final">
      <div class="k">final answer<${Info} k=${mode === "select" ? "swarm_selector" : "swarm_tournament"} /></div>
      <div class="v">${res.final ?? "none"} ${goldKey != null && html`<${Mark} ok=${ok(res.final)} />`}</div>
      <div class="muted">${finalSource} · seed ${res.meta.seed} <a href="#" onClick=${(e) => { e.preventDefault(); set("seed", String(res.meta.seed)); }}>reuse seed</a></div>
      <div class="row" style="margin-top:6px">
        ${mode === "both" && html`<span>tournament: <b>${res.tournament ?? "-"}</b> ${goldKey != null && html`<${Mark} ok=${ok(res.tournament)} />`}</span>
          <span>selector: <b>${selectorPick ?? (res.meta.cancelled ? "cancelled" : "no pick")}</b> ${goldKey != null && html`<${Mark} ok=${ok(selectorPick)} />`}</span>`}
        <span>majority: <b>${res.majority ?? "-"}</b> ${goldKey != null && html`<${Mark} ok=${ok(res.majority)} />`}</span>
        <span>verified majority: ${naRun ? html`<span class="stage-badge external" title="external model: no sandbox">n/a</span>` : html`<b>${res.verified_majority ?? "-"}</b> ${goldKey != null && html`<${Mark} ok=${ok(res.verified_majority)} />`}`}</span><${Info} k="swarm_majority" />
      </div>
      ${mode === "both" && res.tournament != null && selectorPick != null && html`<div class=${"swarm-cmp " + (disagree ? "disagree" : "agree")}>
        ${disagree ? html`<b>The selector and the tournament disagree.</b> The selector picked <b>${selectorPick}</b>; ${!selGroup ? "that is not one of the sampled answers, so it could not enter the bracket" : !selFate ? `it was not entered in the bracket (seed #${groups.indexOf(selGroup) + 1}, beyond max entrants ${maxEntrants})` : selFate.lost ? html`in the bracket it lost in round ${selFate.lost.round} (${selFate.lost.id}) to <b>${selFate.lost.to}</b>${selFate.lost.pick == null ? ", decided by the evidence fallback" : ""}` : "in the bracket it did not lose a match"}. The final answer follows the tournament.`
          : html`The selector and the tournament agree on <b>${res.tournament}</b>.`}
      </div>`}
    </div>`}

    ${res && goldKey != null && html`<h2>Ceilings for this task<${Info} k="swarm_oracle" /></h2>
      <table style="max-width:640px"><tr><th>stage</th><th class="l">the expected answer ...</th><th>holds</th></tr>
        <tr><td>oracle (pass@k)</td><td class="l">is among the ${res.k} samples</td><td><${Mark} ok=${cands.some((c) => answerKey(c.parsed) === goldKey)} /></td></tr>
        <tr><td>oracle, verified</td><td class="l">is among the verified samples</td><td>${naRun ? html`<span class="muted">n/a</span>` : html`<${Mark} ok=${cands.some((c) => c.verified && answerKey(c.parsed) === goldKey)} />`}</td></tr>
        ${hasSelector && html`<tr><td>in prompt</td><td class="l">survived into the selector prompt</td><td><${Mark} ok=${groups.some((g) => answerKey(g.answer) === goldKey && prompt.includes(`- Answer: ${g.answer} (`))} /></td></tr>
        <tr><td>selector</td><td class="l">was picked by the selector</td><td><${Mark} ok=${ok(selectorPick)} /></td></tr>`}
        ${hasBracket && html`<tr><td>in bracket</td><td class="l">was among the ${res.meta.n_entrants ?? entrants.length} entrants</td><td><${Mark} ok=${entrants.some((a) => answerKey(a) === goldKey)} /></td></tr>
        <tr><td>tournament</td><td class="l">won the bracket</td><td><${Mark} ok=${ok(res.tournament)} /></td></tr>`}
      </table>
      <div class="legend">Client-side match (numbers by value, text by lowercase), close to the eval's but not the same verifier.</div>`}

    ${hasBracket && groups.length > 0 && html`<h2>Tournament bracket${entrants.length ? ` (${entrants.length} entrant${entrants.length === 1 ? "" : "s"}${groups.length > entrants.length ? ` of ${groups.length} answers` : ""})` : ""}<${Info} k="swarm_bracket" /></h2>
      ${entrants.length === 1 ? html`<div class="muted">Only one distinct answer: it is the champion without a comparison.</div>`
        : html`<${Bracket} groups=${groups} entrants=${entrants} rounds=${bracketRounds} running=${running} champion=${champion}
          external=${naRun} marks=${{ gold: goldKey, selector: mode === "both" && selectorPick != null ? (groups.find((g) => answerKey(g.answer) === answerKey(selectorPick)) || {}).answer : null, majority: (res || collapsed || {}).majority, vmaj: (res || collapsed || {}).verified_majority }} />`}
      <div class="br-legend">Seeds are the evidence order (verified support, then support): #1 meets the last seed, #2 the second last,
        the middle one of an odd count gets a bye<${Info} k="swarm_seeding" />. Rows are shown in the order the model saw them (A on top);
        ⇄ marks a pair presented swapped<${Info} k="swarm_swap" />; "no pick → evidence" is a comparison decided by the fallback<${Info} k="swarm_fallback" />.
        ×n = support, ✓n = verified.${!seeded && !res ? " The seeding is shown ahead of the first round." : ""}${res && res.meta.cancelled && !res.meta.tournament_complete ? " Cancelled: the rounds after the last decided one were not played." : ""}</div>`}

    ${groups.length > 0 && html`<h2>Distinct answers (${groups.length})<${Info} k="swarm_support" /></h2>
      <table>
        <tr><th>answer</th><th>support</th><th>verified${naRun ? " (n/a)" : ""}</th>${hasSelector && html`<th>in selector prompt<${Info} k="swarm_budget" /></th>`}${hasBracket && html`<th class="l">bracket<${Info} k="swarm_bracket" /></th>`}<th class="l">representative rationale</th><th>members</th></tr>
        ${groups.map((g) => html`<${GroupRow} key=${g.key} g=${g} external=${naRun} cands=${cands} gold=${goldKey} open=${!!open[g.key] && cands.length > 0} showPrompt=${hasSelector} showBracket=${hasBracket}
          toggle=${() => cands.length && setOpen({ ...open, [g.key]: !open[g.key] })}
          flags=${{ final: res && finalKey != null && answerKey(g.answer) === finalKey, majority: g.answer === (res || collapsed).majority, vmaj: g.answer === (res || collapsed).verified_majority,
            selector: mode === "both" && selectorPick != null && answerKey(g.answer) === answerKey(selectorPick),
            inPrompt: res && prompt ? prompt.includes(`- Answer: ${g.answer} (`) : null, fate: fateText(fate[g.answer], champion, bracketDone) }} />`)}
      </table>
      <div class="legend">Sorted by verified support, then support (this is also the bracket's seed order). Click a row for its member samples (think span, tool calls with results, answer).</div>`}
    ${res && unparsed.length > 0 && html`<div style="margin-top:6px"><a href="#" onClick=${(e) => { e.preventDefault(); setShowUnparsed(!showUnparsed); }}>${showUnparsed ? "hide" : "show"} ${unparsed.length} sample${unparsed.length === 1 ? "" : "s"} with no parsable answer</a>
      ${showUnparsed && unparsed.map((c) => html`<${Candidate} c=${c} key=${c.idx} />`)}</div>`}
    ${res && res.groups.length === 0 && html`<div class="panel warn" style="margin-top:8px">No sample produced a parsable final answer, so there was nothing to select. ${naRun ? "An external model has to end its reply with a '#### <answer>' line: keep the answer instruction on and leave max new tokens room to finish." : "Check that the slot holds a reasoning / RL checkpoint, that the answer instruction is on, and that max new tokens leaves room to finish."}</div>`}

    ${res && res.selector_messages.length > 0 && html`<h2>Selector pass<${Info} k="swarm_selector" /></h2>
      <details><summary class="muted">selector prompt (${res.meta.prompt_tokens} tokens, ${res.meta.groups_in_prompt} of ${res.groups.length} answers, budget ${res.meta.budget_tokens})</summary><pre>${prompt}</pre></details>
      ${res.meta.cancelled && res.selector_think == null && !res.selector_answer ? html`<div class="muted">cancelled: the selector did not run.</div>` : html`
        <div class="muted" style="margin:6px 0 2px">think${res.selector_calls ? ` · ${res.selector_calls} python call${res.selector_calls > 1 ? "s" : ""}` : ""}</div>
        <pre class="think">${res.selector_think ?? "(no think span)"}</pre>
        <div class="muted" style="margin:6px 0 2px">answer</div>
        <pre class="swarm-answer">${res.selector_answer || "(empty)"}</pre>`}`}
  </div>`;
}

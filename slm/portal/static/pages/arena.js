import { h } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import htm from "htm";
import { api, readSSE } from "../components/util.js";
import { TextWithSpecials } from "../components/tokens.js";
import { Info } from "../components/info.js";
import { SlotCard } from "./model.js";

const html = htm.bind(h);

/** The Arena (slm/arena/world.py via POST /api/model/arena): an NxN grid of robots, each one a conversation with the
 *  model, every turn ONE batched generation over all of them. The page streams the episode turn by turn, draws the
 *  grid (robots, objects, comm range, messages as arcs, movement between turns), and keeps every turn so the scrubber
 *  can step back. With 32 robots only the selected robot's transcript is rendered; the rest is one compact table. */

const DEFAULTS = { slot: "A", task: "key_door", n: 8, n_agents: 4, seed: 0, turns: 12, max_new_tokens: 128, max_calls: 4, max_history: 1, stop_when_done: true };
const LIMITS = { n: [4, 16], n_agents: [1, 32], turns: [1, 50], max_new_tokens: [16, 1024], max_calls: [0, 16], max_history: [0, 10] };

// which unique tool makes a robot what it is (the common move / say / look say nothing about its role)
const ROLES = [
  ["read_key", "key holder", "#f59e0b"], ["open_door", "door opener", "#3b82f6"], ["read_code", "source", "#f59e0b"],
  ["submit", "sink", "#16a34a"], ["dig", "digger", "#dc2626"], ["sense", "sensor", "#0d9488"],
];
const roleOf = (tools) => {
  for (const [t, label, color] of ROLES) if ((tools || []).includes(t)) return { tool: t, label, color };
  return { tool: null, label: "relay", color: "#9ca3af" };
};
const COMMON = new Set(["move", "say", "look"]);
const xy = (p) => (p ? `(${p[0]}, ${p[1]})` : "-");
const clip = (s, n) => (s == null ? "" : s.length > n ? s.slice(0, n - 1) + "…" : s);
const fmtVal = (v) => (Array.isArray(v) ? xy(v) : v === true ? "yes" : v === false ? "no" : v == null ? "-" : String(v));
const doorsOpened = (events) => new Set((events || []).map((e) => /opened door (\d+)/.exec(e)).filter(Boolean).map((m) => Number(m[1])));

// ---------------------------------------------------------------------------------------------------- the grid
const C = 40, PAD = 18;  // cell size and the coordinate margin, in SVG user units

/** Robot centres: one robot in a cell sits in the middle; several share it on a small ring. */
function robotSpots(agents) {
  const byCell = {};
  agents.forEach((a) => { (byCell[`${a.x},${a.y}`] = byCell[`${a.x},${a.y}`] || []).push(a.id); });
  const out = {};
  for (const ids of Object.values(byCell)) {
    const k = ids.length;
    const r = k === 1 ? C * 0.32 : Math.max(C * 0.13, (C * 0.3) / Math.sqrt(k));
    ids.forEach((id, j) => {
      const a = agents.find((b) => b.id === id);
      const ang = (2 * Math.PI * j) / k - Math.PI / 2, ring = k === 1 ? 0 : C * 0.22;
      out[id] = { cx: PAD + a.x * C + C / 2 + ring * Math.cos(ang), cy: PAD + a.y * C + C / 2 + ring * Math.sin(ang), r };
    });
  }
  return out;
}

function arcPath(x1, y1, x2, y2) {
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
  const bend = Math.min(0.3 * len, C * 1.4);
  return `M${x1.toFixed(1)},${y1.toFixed(1)} Q${(mx - (dy / len) * bend).toFixed(1)},${(my + (dx / len) * bend).toFixed(1)} ${x2.toFixed(1)},${y2.toFixed(1)}`;
}

function ArenaGrid({ n, state, prev, turn, view, meta, focus, sel, setSel, setHover, showTarget }) {
  if (!state) return html`<div class="arena-empty muted">Run an episode to see the grid.</div>`;
  const W = PAD + n * C + 4;
  const agents = state.agents;
  const spots = robotSpots(agents);
  const prevSpots = prev ? robotSpots(prev.agents) : spots;
  const fa = focus != null ? agents.find((a) => a.id === focus) : null;
  const cells = [];
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) cells.push([x, y]);
  const d = (a, x, y) => Math.abs(a.x - x) + Math.abs(a.y - y);
  const opened = doorsOpened(state.events);
  const sc = state.score || {};
  const target = state.task === "triangulate" && Array.isArray(sc.target) ? sc.target : null;
  const revealed = target && (sc.success || showTarget);
  const msgs = turn ? turn.messages || [] : [];
  return html`<svg class="arena-grid" viewBox=${`0 0 ${W} ${W}`} role="img" aria-label=${`${n} by ${n} arena grid, turn ${state.turn}`}>
    <defs><marker id="arena-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="arena-arrowhead" /></marker></defs>
    ${Array.from({ length: n }, (_, i) => html`<text key=${"cx" + i} class="arena-axis" x=${PAD + i * C + C / 2} y=${PAD - 5}>${i}</text>
      <text key=${"cy" + i} class="arena-axis" x=${PAD - 5} y=${PAD + i * C + C / 2 + 3} text-anchor="end">${i}</text>`)}
    ${cells.map(([x, y]) => {
      const inComm = fa && d(fa, x, y) <= meta.comm_range, inSight = fa && d(fa, x, y) <= meta.sight;
      return html`<rect key=${`c${x},${y}`} class=${"arena-cell" + (inSight ? " sight" : inComm ? " comm" : "")} x=${PAD + x * C} y=${PAD + y * C} width=${C} height=${C} />`;
    })}
    ${state.objects.map((o) => {
      const num = (/(\d+)/.exec(o.what) || [])[1];
      const open = /door/.test(o.what) && opened.has(Number(num));
      return html`<g key=${"o" + o.what} class=${"arena-obj" + (open ? " open" : "")}><title>${o.what} at ${xy([o.x, o.y])}${open ? " (open)" : ""}</title>
        <rect x=${PAD + o.x * C + 3} y=${PAD + o.y * C + 3} width=${C - 6} height=${C - 6} rx="3" />
        <text x=${PAD + o.x * C + C / 2} y=${PAD + o.y * C + C - 6}>${o.what[0].toUpperCase()}${num || ""}${open ? " ✓" : ""}</text></g>`;
    })}
    ${sc.dug_at && !(target && sc.dug_at[0] === target[0] && sc.dug_at[1] === target[1]) && html`<g class="arena-dug"><title>last dig: ${xy(sc.dug_at)} (nothing there)</title>
      <path d=${`M${PAD + sc.dug_at[0] * C + 8},${PAD + sc.dug_at[1] * C + 8} l${C - 16},${C - 16} M${PAD + sc.dug_at[0] * C + C - 8},${PAD + sc.dug_at[1] * C + 8} l${-(C - 16)},${C - 16}`} /></g>`}
    ${revealed && html`<g class=${"arena-target" + (sc.success ? " found" : "")}><title>buried target at ${xy(target)}${sc.success ? " (dug up)" : " (hidden from the robots)"}</title>
      <rect x=${PAD + target[0] * C + 4} y=${PAD + target[1] * C + 4} width=${C - 8} height=${C - 8} rx="4" /><text x=${PAD + target[0] * C + C / 2} y=${PAD + target[1] * C + C - 7}>${sc.success ? "found" : "target"}</text></g>`}
    ${prev && agents.map((a) => {
      const p = prevSpots[a.id], c = spots[a.id];
      return p && (Math.abs(p.cx - c.cx) > 1 || Math.abs(p.cy - c.cy) > 1) ? html`<line key=${`tr${view}-${a.id}`} class="arena-trail" x1=${p.cx} y1=${p.cy} x2=${c.cx} y2=${c.cy} />` : null;
    })}
    ${agents.map((a) => {
      const c = spots[a.id], role = roleOf(a.tools);
      return html`<g key=${"r" + a.id} class=${"arena-robot" + (a.id === sel ? " sel" : "") + (a.id === focus ? " focus" : "")} style=${`transform:translate(${c.cx}px,${c.cy}px)`}
          onClick=${() => setSel(a.id)} onMouseEnter=${() => setHover(a.id)} onMouseLeave=${() => setHover(null)}>
        <title>${a.name} at ${xy([a.x, a.y])} · ${role.label} · tools: ${a.tools.join(", ")}</title>
        <circle r=${c.r} fill=${role.color} /><text y=${c.r * 0.36} style=${`font-size:${Math.max(7, c.r * 0.9)}px`}>${a.name}</text></g>`;
    })}
    ${msgs.map((m, i) => {
      const s = agents.find((a) => a.id === m.speaker_id), sp = s && spots[s.id];
      if (!sp) return null;
      const hot = focus == null || focus === m.speaker_id || (m.heard_by_ids || []).includes(focus);
      if (!(m.heard_by_ids || []).length) {
        return html`<circle key=${`m${view}-${i}`} class=${"arena-nobody" + (hot ? "" : " dim")} cx=${sp.cx} cy=${sp.cy} r=${sp.r + 4}><title>${m.speaker}: "${m.text}" (nobody within range)</title></circle>`;
      }
      return m.heard_by_ids.map((hid) => {
        const t = spots[hid];
        if (!t) return null;
        const ang = Math.atan2(t.cy - sp.cy, t.cx - sp.cx);
        return html`<path key=${`m${view}-${i}-${hid}`} class=${"arena-arc" + (hot ? "" : " dim")} pathLength="1" marker-end="url(#arena-arrow)"
          d=${arcPath(sp.cx, sp.cy, t.cx - Math.cos(ang) * (t.r + 2), t.cy - Math.sin(ang) * (t.r + 2))}><title>${m.speaker} → ${agents.find((a) => a.id === hid).name}: "${m.text}"</title></path>`;
      });
    })}
  </svg>`;
}

// ---------------------------------------------------------------------------------------------------- panels
function TaskPanel({ task, meta, state, result, showTarget }) {
  const sc = state ? state.score || {} : {};
  const events = state ? state.events : [];
  const hide = (k) => k === "target" && state && state.task === "triangulate" && !sc.success && !showTarget;
  return html`<div class="panel arena-task">
    <div class="row"><b>${task ? task.title : meta ? meta.task : "task"}</b><${Info} k=${"arena_" + (meta ? meta.task : task ? task.key : "key_door")} />
      ${state && html`<span class=${"stage-badge " + (sc.success ? "rl" : "base")}>${sc.success ? "success" : result ? "not solved" : "in progress"}</span>`}</div>
    ${task && html`<p class="arena-desc">${task.description}</p><div class="legend">roles: ${task.roles}</div>`}
    ${meta && html`<div class="legend">grid ${meta.n}×${meta.n} · ${meta.n_agents} robot${meta.n_agents === 1 ? "" : "s"} · comm range ${meta.comm_range} · sight ${meta.sight}<${Info} k="arena_range" /> · seed ${meta.seed} · ${meta.checkpoint || ""} on ${meta.device}</div>`}
    ${state && html`<h3 style="margin-top:8px">score at turn ${state.turn}</h3>
      <div class="evd-kv">${Object.entries(sc).map(([k, v]) => html`<span key=${k}><i>${k}</i>${hide(k) ? "hidden" : fmtVal(v)}</span>`)}</div>
      <h3 style="margin-top:8px">events</h3>
      ${events.length ? html`<ul class="arena-events">${events.map((e, i) => html`<li key=${i}>${e}</li>`)}</ul>` : html`<div class="muted">none yet</div>`}`}
    ${result && html`<div class=${"arena-result " + (result.score.success ? "ok" : "no")}>
      <b>${result.score.success ? `Solved in ${result.score.done_turn ?? result.turns} turn${(result.score.done_turn ?? result.turns) === 1 ? "" : "s"}` : result.cancelled ? "Cancelled" : "Not solved"}</b>
      · ${result.turns} turn${result.turns === 1 ? "" : "s"} played${result.cancelled ? " (stopped by you)" : ""} · ${result.seconds.toFixed(1)} s
      · ${result.transcript.length} robot turns · ${result.transcript.reduce((a, r) => a + r.n_calls, 0)} tool calls · ${result.messages.reduce((a, m) => a + m.length, 0)} messages
    </div>`}
  </div>`;
}

function RobotTable({ state, turn, robots, sel, setSel }) {
  if (!state) return null;
  const recs = turn ? Object.fromEntries(turn.records.map((r) => [r.agent_id, r])) : {};
  const said = {};
  (turn ? turn.messages : []).forEach((m) => { (said[m.speaker_id] = said[m.speaker_id] || []).push(m); });
  return html`<table class="arena-table">
    <tr><th>robot</th><th>pos</th><th class="l">unique tool</th><th class="l">declared this turn</th><th class="l">calls</th><th class="l">said → heard by</th><th class="l">answer (status line)</th></tr>
    ${state.agents.map((a) => {
      const r = recs[a.id], role = roleOf(a.tools);
      return html`<tr key=${a.id} class=${a.id === sel ? "sel" : ""} onClick=${() => setSel(a.id)}>
        <td><span class="arena-dot" style=${`background:${role.color}`}></span>${a.name}</td><td>${xy([a.x, a.y])}</td>
        <td class="l">${a.tools.filter((t) => !COMMON.has(t)).join(", ") || html`<span class="muted">-</span>`}</td>
        <td class="l">${r && (r.declared || r.tools) ? (r.declared || r.tools).join(", ") : html`<span class="muted">-</span>`}</td>
        <td class="l mono" title=${r ? r.calls.map((c) => `${c[0]}  ->  ${c[1]}`).join("\n") : ""}>${r ? (r.calls.length ? clip(r.calls.map((c) => c[0]).join("; "), 48) : html`<span class="muted">no call</span>`) : ""}</td>
        <td class="l" title=${(said[a.id] || []).map((m) => m.text).join("\n")}>${(said[a.id] || []).map((m, i) => html`<div key=${i}>"${clip(m.text, 40)}" → ${m.heard_by.length ? m.heard_by.join(", ") : html`<span class="muted">nobody</span>`}</div>`)}</td>
        <td class="l" title=${r ? r.answer || "" : ""}>${r ? clip((r.answer || "").trim(), 60) || html`<span class="muted">(empty)</span>` : ""}</td>
      </tr>`;
    })}
  </table>`;
}

function RobotTurn({ r, open }) {
  return html`<details class="arena-turn" open=${open}>
    <summary><b>turn ${r.turn}</b> · at ${xy(r.pos)} · ${r.n_calls} call${r.n_calls === 1 ? "" : "s"} · ${clip((r.answer || "").trim(), 70) || "(no status line)"}</summary>
    ${r.status && html`<div class="evd-lab">situation (the world's view at the end of the turn; not shown to the model)</div><div class="arena-status">${r.status}</div>`}
    <div class="evd-lab" style="margin-top:6px">user turn, exactly as the model saw it</div>
    <${TextWithSpecials} text=${r.observation} cls="arena-obs" />
    ${(r.declared || r.tools) && html`<div class="legend">declared this turn: ${(r.declared || r.tools).map((t) => html`<code key=${t}>${t}()</code> `)}</div>`}
    <div class="evd-lab" style="margin-top:6px">generated turn (think span with tool calls; inserted results are not sampled)<${Info} k="arena_turn" /></div>
    <${TextWithSpecials} text=${r.raw || (r.think != null ? r.think : "(no think span)")} cls="think" />
    ${r.calls.length > 0 && html`<div class="evd-lab" style="margin-top:6px">tool calls</div>
      ${r.calls.map((c, i) => html`<div class="toolcall" key=${i}><pre class="code">${c[0]}</pre><span class=${/^error|Traceback|Error/.test(c[1]) ? "result err" : "result ok"}>${c[1]}</span></div>`)}`}
    <div class="evd-lab" style="margin-top:6px">answer (the status line after the think span)</div>
    <${TextWithSpecials} text=${(r.answer || "").trim() || "(empty)"} cls="swarm-answer" />
  </details>`;
}

function RobotPanel({ robot, ep, view, state }) {
  if (!robot) return null;
  const a = state && state.agents.find((x) => x.id === robot.id);
  const role = roleOf(robot.tools.map((t) => t.name));
  const recs = ep.turns.slice(0, view).map((t) => t.records.find((r) => r.agent_id === robot.id)).filter(Boolean).reverse();
  return html`<div class="panel arena-robot-panel">
    <div class="row"><span class="arena-dot big" style=${`background:${role.color}`}></span><b>${robot.name}</b><span class="muted">${role.label}${a ? ` · at ${xy([a.x, a.y])} after turn ${state.turn}` : ""}</span></div>
    <div class="evd-lab" style="margin-top:6px">declared tools (the turn's question declares only the one it needs)<${Info} k="arena_scaffold" /></div>
    ${robot.tools.map((t) => html`<div class="arena-decl" key=${t.name}><code>${t.signature}</code> <span class="muted">${t.comment}</span></div>`)}
    <details style="margin-top:6px"><summary class="muted">briefing (world.system_prompt: the robot's task as the world writes it; the model sees only what the user turns carry)</summary><${TextWithSpecials} text=${robot.system_prompt} /></details>
    <h3 style="margin-top:10px">turns ${view ? `1-${view}` : ""} (newest first)</h3>
    ${recs.length ? recs.map((r) => html`<${RobotTurn} key=${r.turn} r=${r} open=${r.turn === view} />`) : html`<div class="muted">no turn yet at this point of the episode</div>`}
  </div>`;
}

// ---------------------------------------------------------------------------------------------------- page
export function ArenaPage() {
  const [status, setStatus] = useState(null);
  const [ckpts, setCkpts] = useState([]);
  const [tasks, setTasks] = useState([]);
  const [p, setP] = useState(DEFAULTS);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const [streamId, setStreamId] = useState(null);
  const [ep, setEp] = useState(null);  // {request, meta, robots, start, turns: [turn events], result, t0}
  const [view, setView] = useState(0);  // 0 = the start state, k = after turn k
  const [follow, setFollow] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [sel, setSel] = useState(0);
  const [hover, setHover] = useState(null);
  const [showTarget, setShowTarget] = useState(false);
  const [now, setNow] = useState(Date.now());
  const abortRef = useRef(null);

  const refresh = () => { api("/api/model/status").then(setStatus).catch(() => {}); api("/api/model/checkpoints").then(setCkpts).catch(() => {}); };
  useEffect(() => { refresh(); api("/api/model/arena/tasks").then(setTasks).catch((e) => setErr(String(e))); }, []);
  useEffect(() => { if (!busy) return; const id = setInterval(() => setNow(Date.now()), 500); return () => clearInterval(id); }, [busy]);
  const nTurns = ep ? ep.turns.length : 0;
  useEffect(() => { if (follow) setView(nTurns); }, [nTurns, follow]);
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setView((v) => { if (v >= nTurns) { setPlaying(false); return v; } return v + 1; }), 1300);
    return () => clearInterval(id);
  }, [playing, nTurns]);

  const slots = status ? status.slots : { A: {}, B: {} };
  const slotInfo = slots[p.slot] || {};
  const loaded = !!slotInfo.checkpoint, ext = !!slotInfo.external;
  const task = tasks.find((t) => t.key === p.task);
  const bad = p.task === "key_door" && p.n_agents % 2 ? "key_door pairs robots: use an even number" : p.task === "relay" && p.n_agents < 2 ? "relay needs at least 2 robots" : null;
  const set = (k, v) => setP((q) => ({ ...q, [k]: v }));
  const num = (k, attrs = {}) => html`<input type="number" value=${p[k]} min=${LIMITS[k] && LIMITS[k][0]} max=${LIMITS[k] && LIMITS[k][1]} onChange=${(e) => {
    const lim = LIMITS[k], v = Number(e.target.value);
    set(k, lim ? Math.max(lim[0], Math.min(lim[1], Math.round(v) || lim[0])) : v);
  }} style="width:64px" ...${attrs} />`;

  const load = async (slot, path, device, force) => {
    setBusy(true); setErr(null);
    try { await api(`/api/model/slots/${slot}/load`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ checkpoint: path, device, force_cuda: force }) }); } catch (e) { setErr(String(e)); }
    setBusy(false); refresh();
  };
  const unload = async (slot) => { setBusy(true); await api(`/api/model/slots/${slot}/unload`, { method: "POST" }); setBusy(false); refresh(); };

  const run = async () => {
    setErr(null); setPlaying(false); setFollow(true); setView(0); setSel(0);
    const req = { ...p, seed: Number(p.seed) || 0 };
    setNow(Date.now()); setEp({ request: req, meta: null, robots: [], start: null, turns: [], result: null, t0: Date.now() });
    setBusy(true);
    const ctrl = new AbortController(); abortRef.current = ctrl;
    try {
      const r = await fetch("/api/model/arena", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(req), signal: ctrl.signal });
      await readSSE(r, (ev, d) => {
        if (ev === "start" && d.stream_id) setStreamId(d.stream_id);
        else if (ev === "start") setEp((e) => ({ ...e, meta: d.meta, robots: d.robots, start: d.state }));
        else if (ev === "turn") setEp((e) => ({ ...e, turns: [...e.turns, d] }));
        else if (ev === "done") setEp((e) => ({ ...e, result: d.result }));
        else if (ev === "error") setErr(d.error);
      });
    } catch (e) { if (e.name !== "AbortError") setErr(String(e)); }
    setBusy(false); setStreamId(null); refresh();
  };
  const stop = () => { if (streamId) api(`/api/model/streams/${streamId}/cancel`, { method: "POST" }).catch(() => {}); };
  const download = () => {
    const m = ep.meta || ep.request;
    const blob = new Blob([JSON.stringify({ request: ep.request, meta: ep.meta, robots: ep.robots, start: ep.start, turns: ep.turns, result: ep.result }, null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `arena_${m.task}_n${m.n}_r${m.n_agents}_s${m.seed}.json`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  };
  const togglePlay = () => {
    if (playing) { setPlaying(false); return; }
    if (view >= nTurns) setView(0);
    setFollow(false); setPlaying(true);
  };
  const scrub = (v) => { setPlaying(false); const c = Math.max(0, Math.min(nTurns, v)); setView(c); setFollow(c === nTurns); };

  const meta = ep && ep.meta;
  const state = ep && ep.start ? (view === 0 ? ep.start : ep.turns[view - 1].state) : null;
  const prev = ep && ep.start && view > 0 ? (view === 1 ? ep.start : ep.turns[view - 2].state) : null;
  const turn = ep && view > 0 ? ep.turns[view - 1] : null;
  const robot = ep && ep.robots.find((r) => r.id === sel);
  const focus = hover != null ? hover : sel;
  const running = busy && ep && !ep.result;
  const roles = state ? [...new Map(state.agents.map((a) => roleOf(a.tools)).map((r) => [r.label, r])).values()] : [];
  const genSec = ep ? ep.turns.reduce((s, t) => s + t.seconds, 0) : 0;

  return html`<div>
    <h1>Arena<${Info} k="arena" /></h1>
    <div class="sub">${status ? (status.worker ? "worker alive" : "worker idle (no VRAM held)") : "…"}<${Info} k="worker" />
      ${status && status.live_runs && status.live_runs.length ? ` · ⚠ training live: ${status.live_runs.join(", ")} — loads default to CPU` : ""}
      · robots act through declared functions inside the think span: load one of our tool-trained checkpoints (an RL model).</div>
    ${err && html`<div class="panel warn">${err}</div>`}
    <div class="two">
      <${SlotCard} slot="A" info=${slots.A} ckpts=${ckpts} onLoad=${load} onUnload=${unload} busy=${busy} />
      <${SlotCard} slot="B" info=${slots.B} ckpts=${ckpts} onLoad=${load} onUnload=${unload} busy=${busy} />
    </div>

    <div class="panel" style="margin-top:12px">
      <div class="row" style="margin-bottom:6px">
        <span class="grp"><span class="muted">slot</span><select value=${p.slot} onChange=${(e) => set("slot", e.target.value)}><option>A</option><option>B</option></select></span>
        <span class="grp"><span class="muted">task</span><select value=${p.task} onChange=${(e) => set("task", e.target.value)}>
          ${(tasks.length ? tasks : [{ key: "key_door", title: "Key and door" }, { key: "relay", title: "Relay" }, { key: "triangulate", title: "Triangulate" }]).map((t) => html`<option value=${t.key} key=${t.key}>${t.title} (${t.key})</option>`)}
        </select><${Info} k=${"arena_" + p.task} /></span>
        <span class="grp"><span class="muted">grid</span>${num("n")}</span>
        <span class="grp"><span class="muted">robots</span>${num("n_agents", { step: p.task === "key_door" ? 2 : 1 })}</span>
        <span class="grp"><span class="muted">seed</span><input type="number" value=${p.seed} min="0" onChange=${(e) => set("seed", Math.max(0, Math.round(Number(e.target.value)) || 0))} style="width:70px" /></span>
        <span class="grp"><span class="muted">turns</span>${num("turns")}</span>
      </div>
      <div class="row" style="margin-bottom:6px">
        <span class="grp"><span class="muted">max new tokens</span>${num("max_new_tokens", { step: 16 })}</span>
        <span class="grp"><span class="muted">max tool calls</span>${num("max_calls")}</span>
        <span class="grp"><span class="muted">history</span>${num("max_history")}<${Info} k="arena_turn" /></span>
        <label class="muted"><input type="checkbox" checked=${p.stop_when_done} onChange=${(e) => set("stop_when_done", e.target.checked)} /> stop when done</label>
        <span class="legend">cost: ${p.n_agents} rows × up to ${p.max_new_tokens} tokens per turn, one batch<${Info} k="arena_cost" /></span>
      </div>
      ${task && html`<div class="legend" style="margin-bottom:6px">${task.description}</div>`}
      <div class="row">
        <button class="active" onClick=${run} disabled=${busy || !loaded || ext || !!bad}>run</button>
        <button onClick=${stop} disabled=${!streamId}>stop</button>
        <button onClick=${download} disabled=${!ep || !ep.start || running} title="the whole episode: settings, briefings, every turn's records, messages and states, and the result">download episode log (JSON)</button>
        ${!loaded && html`<span class="muted">load a checkpoint into slot ${p.slot} first (CPU works for a small grid; the 336M RL model on cuda is the real test)</span>`}
        ${ext && html`<span class="muted">the arena is n/a for an external slot: the robots act through our tool protocol (declared functions, <code>${"<|python_call|>"}</code> inside the think span), which ${slotInfo.name} does not have</span>`}
        ${bad && html`<span class="muted">${bad}</span>`}
        ${loaded && !ext && !bad && slotInfo.device === "cpu" && html`<span class="legend">slot ${p.slot} is on CPU: keep it small (2-4 robots, 64 new tokens)</span>`}
      </div>
      ${ep && html`<div class="stage-steps">
        <span class=${"step " + (ep.result ? "past" : "now")}>${ep.result ? (ep.result.cancelled ? "stopped" : "done") : ep.start ? `turn ${nTurns + 1} of ${ep.request.turns}` : "building the world"}</span>
        <span class="muted">${busy ? `${(Math.max(0, now - ep.t0) / 1000).toFixed(0)} s` : ep.result ? `${ep.result.seconds.toFixed(1)} s` : ""}${nTurns ? ` · ${nTurns} turn${nTurns === 1 ? "" : "s"}, ${genSec.toFixed(1)} s generating (${(genSec / nTurns).toFixed(1)} s per batched turn of ${ep.request.n_agents})` : ""}</span>
      </div>`}
    </div>

    ${ep && ep.start && html`<div class="arena-main">
      <div>
        <h2>Grid<${Info} k="arena_grid" /></h2>
        <${ArenaGrid} n=${meta.n} state=${state} prev=${prev} turn=${turn} view=${view} meta=${meta} focus=${focus} sel=${sel} setSel=${setSel} setHover=${setHover} showTarget=${showTarget} />
        <div class="arena-scrub">
          <button onClick=${() => scrub(0)} disabled=${view === 0} title="start">⏮</button>
          <button onClick=${() => scrub(view - 1)} disabled=${view === 0} title="previous turn">◀</button>
          <button onClick=${togglePlay} disabled=${!nTurns} title="play the episode">${playing ? "❚❚" : "▶ play"}</button>
          <button onClick=${() => scrub(view + 1)} disabled=${view >= nTurns} title="next turn">▶</button>
          <button onClick=${() => scrub(nTurns)} disabled=${view >= nTurns} title="latest">⏭</button>
          <input type="range" min="0" max=${nTurns} value=${view} onInput=${(e) => scrub(Number(e.target.value))} disabled=${!nTurns} />
          <span class="muted">${view === 0 ? "start" : `after turn ${view}`} of ${nTurns}${running && follow ? " · following live" : ""}${turn ? ` · ${turn.seconds.toFixed(1)} s` : ""}</span>
        </div>
        <div class="legend arena-legend">
          ${roles.map((r) => html`<span key=${r.label}><span class="arena-dot" style=${`background:${r.color}`}></span>${r.label}</span>`)}
          <span><span class="arena-sw comm"></span>comm range of the selected / hovered robot</span><span><span class="arena-sw sight"></span>its sight</span>
          <span><span class="arena-sw arc"></span>message this turn (speaker → hearer)</span><span><span class="arena-sw nobody"></span>said to nobody</span>
          ${state && state.task === "triangulate" && html`<label><input type="checkbox" checked=${showTarget} onChange=${(e) => setShowTarget(e.target.checked)} /> show the hidden target</label>`}
        </div>
        ${turn && turn.messages.length > 0 && html`<div class="arena-msgs">${turn.messages.map((m, i) => html`<div key=${i}><b>${m.speaker}</b> at ${xy(m.pos)}: "${m.text}" → ${m.heard_by.length ? m.heard_by.join(", ") : html`<span class="muted">nobody in range</span>`}</div>`)}</div>`}
      </div>
      <div><h2>Task</h2><${TaskPanel} task=${tasks.find((t) => t.key === meta.task)} meta=${meta} state=${state} result=${ep.result} showTarget=${showTarget} /></div>
    </div>
    <h2>Robots ${view ? `(turn ${view})` : "(start)"}<${Info} k="arena_turn" /></h2>
    <div class="arena-table-wrap"><${RobotTable} state=${state} turn=${turn} robots=${ep.robots} sel=${sel} setSel=${setSel} /></div>
    <div class="legend">Click a row or a robot on the grid to open its transcript below; only one robot's turns are rendered at a time.</div>
    <h2>${robot ? robot.name : "Robot"}</h2>
    <${RobotPanel} robot=${robot} ep=${ep} view=${view} state=${state} />`}
  </div>`;
}

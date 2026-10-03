import { h } from "preact";
import { useMemo, useState } from "preact/hooks";
import htm from "htm";
import { Info } from "./info.js";

const html = htm.bind(h);

/** The Arena tab's episode model and its at-a-glance views (pages/arena.js draws the grid and the transcript):
 *  - `normalizeEpisode` turns either episode file shape (the CLI's `--out`, the tab's download) into the live shape;
 *  - `analyze` indexes an episode once: positions, per-robot actions with glyphs and error flags, steps, inboxes,
 *    messages and the per-turn series;
 *  - `Timeline` (swimlanes: one row per robot, one column per turn, messages as arrows into the next column),
 *    `MessageLog` (every say, filterable) and `SummaryStrip` (small charts over turns, the verdict and the events). */

// ---------------------------------------------------------------------------------------------- shared helpers
// which unique tool makes a robot what it is (the common move / say / look say nothing about its role)
export const ROLES = [
  ["read_key", "key holder", "#f59e0b"], ["open_door", "door opener", "#3b82f6"], ["read_code", "source", "#f59e0b"],
  ["submit", "sink", "#16a34a"], ["dig", "digger", "#dc2626"], ["distance_to_target", "sensor", "#0d9488"],
];
export const roleOf = (tools) => {
  for (const [t, label, color] of ROLES) if ((tools || []).includes(t)) return { tool: t, label, color };
  return { tool: null, label: "relay", color: "#9ca3af" };
};
export const COMMON = new Set(["move", "say", "look"]);
export const xy = (p) => (p ? `(${p[0]}, ${p[1]})` : "-");
export const clip = (s, n) => (s == null ? "" : s.length > n ? s.slice(0, n - 1) + "…" : s);
export const fmtVal = (v) => (Array.isArray(v) ? xy(v) : v === true ? "yes" : v === false ? "no" : v == null ? "-" : String(v));
export const doorsOpened = (events) => new Set((events || []).map((e) => /opened door (\d+)/.exec(e)).filter(Boolean).map((m) => Number(m[1])));
/** One colour per robot (its identity: trails, message arcs, timeline arrows), well apart for up to 32 robots. */
export const robotColor = (id) => `hsl(${Math.round((id * 137.508) % 360)}, 70%, 42%)`;
const SUFFIX_RE = /\n?Think step by step, then give the final answer on its own line as '#### <answer>'\.\s*$/;
export const question = (obs) => (obs || "").replace(SUFFIX_RE, "").trim();

const TOOLS = ["move", "say", "look", "read_key", "read_code", "open_door", "submit", "dig", "distance_to_target"];
const TOOL_RE = new RegExp(`\\b(${TOOLS.join("|")})\\s*\\(`, "g");
const ERR_RE = /^\s*error\b|Traceback|Wrong code|is not the code/;      // the call failed (sandbox error, wrong code)
const FAIL_RE = /^\s*Blocked:|Unknown direction|Move there first|^\s*Nothing here/;  // the call ran but did nothing
const DIRS = { north: [0, -1], south: [0, 1], east: [1, 0], west: [-1, 0] };
const ARROW = { north: "↑", south: "↓", east: "→", west: "←" };

/** The glyph, colour and label of each action kind (timeline cells and the legend). */
export const KINDS = {
  move: { glyph: "→", color: "#e0e7ff", ink: "#3730a3", label: "move (arrow = direction)" },
  say: { glyph: "»", color: "#dbeafe", ink: "#1d4ed8", label: "say" },
  look: { glyph: "◉", color: "#f3f4f6", ink: "#4b5563", label: "look" },
  read: { glyph: "R", color: "#fef3c7", ink: "#92400e", label: "read_key / read_code" },
  goal: { glyph: "★", color: "#dcfce7", ink: "#166534", label: "open_door / submit / dig" },
  sense: { glyph: "◎", color: "#ccfbf1", ink: "#115e59", label: "distance_to_target" },
  code: { glyph: "{}", color: "#f5f5f4", ink: "#57534e", label: "code without a tool call" },
  none: { glyph: "·", color: "transparent", ink: "#d1d5db", label: "no call" },
};
const kindOf = (tool) => (tool === "read_key" || tool === "read_code" ? "read" : tool === "open_door" || tool === "submit" || tool === "dig" ? "goal"
  : tool === "distance_to_target" ? "sense" : tool);

/** Python repr of a str back to the str ('...' or "..." with backslash escapes). */
function unrepr(s) {
  s = (s || "").trim();
  if (s.length >= 2 && (s[0] === "'" || s[0] === '"') && s[s.length - 1] === s[0]) s = s.slice(1, -1);
  return s.replace(/\\(['"\\])/g, "$1").replace(/\\n/g, "\n");
}
const squash = (t) => String(t).split(/\s+/).filter(Boolean).join(" ").slice(0, 200);  // what World.say keeps

/** The inbox of a record: the harness's `heard` field, else the "Heard: ..." part of its status line (CLI files and
 *  downloads made before the field existed). Each entry {speaker, text, line}. */
export function heardOf(rec) {
  if (!rec) return [];
  let lines = rec.heard;
  if (!Array.isArray(lines)) {
    const s = rec.status || "", i = s.indexOf(". Heard: ");
    if (i < 0) return [];
    const body = s.slice(i + 9).replace(/\.$/, "");
    lines = body === "nothing" ? [] : body.split(/; (?=R\d+ said: )/);
  }
  return lines.map((line) => {
    const m = /^(R\d+) said: ([\s\S]*)$/.exec(line);
    return m ? { speaker: m[1], text: unrepr(m[2]), line } : { speaker: null, text: line, line };
  });
}

// ---------------------------------------------------------------------------------------------- normalization
/** "cli" (python -m slm.arena --out), "portal" (the tab's download / a live run) or null. */
export function episodeKind(raw) {
  if (!raw || typeof raw !== "object") return null;
  if (Array.isArray(raw.turns) && raw.start) return "portal";
  if (Array.isArray(raw.transcript)) return "cli";
  return null;
}

/** An episode in the live shape {request, meta, robots, start, turns: [{turn, seconds, records, state, messages}],
 *  result, notes} from a file. A CLI file needs `world` (GET /api/model/arena/world for its task, grid, robots and
 *  seed: the world is deterministic, so it gives the start state, objects, tools and briefings); its per-turn states
 *  take positions from the records, events from the event list, scores derived the way the task scores them (the
 *  final turn takes the file's own score), and messages from the say calls (text from the call's result or its
 *  literal argument, hearers by the delivery rule: within comm range of where the speaker ended the turn), plus any
 *  line in a robot's next inbox that no parsed call explains. */
export function normalizeEpisode(raw, world, name) {
  const kind = episodeKind(raw);
  if (kind === "portal") {
    const meta = raw.meta || raw.request || {};
    const turns = raw.turns.map((t) => ({ ...t, messages: t.messages || [], records: (t.records || []).map((r, i) => ({ agent_id: i, ...r })) }));
    return { request: raw.request || meta, meta, robots: raw.robots || [], start: raw.start, turns, result: raw.result || null, notes: [], name, kind };
  }
  if (kind !== "cli") throw new Error("not an arena episode: expected a CLI transcript (python -m slm.arena --out) or the Arena tab's download");
  if (!world) throw new Error("a CLI episode needs the rebuilt world");
  const notes = [`Rebuilt from a CLI file: the start state is World("${raw.task}", ${raw.n}, ${raw.agents}, seed ${raw.seed}); per-turn scores are derived from the transcript, the final one is the file's.`];
  const start = world.state, robots = world.robots, cr = world.meta.comm_range;
  const idOf = Object.fromEntries(robots.map((r) => [r.name, r.id]));
  const T = raw.turns || Math.max(0, ...raw.transcript.map((r) => r.turn));
  const byTurn = Array.from({ length: T + 1 }, () => []);
  for (const r of raw.transcript) if (r.turn >= 1 && r.turn <= T) byTurn[r.turn].push({ ...r, agent_id: idOf[r.agent], declared: r.declared || r.tools });
  byTurn.forEach((rs) => rs.sort((a, b) => a.agent_id - b.agent_id));
  // positions after each turn
  const pos = [start.agents.map((a) => [a.x, a.y])];
  for (let t = 1; t <= T; t++) pos.push(pos[t - 1].map((p, id) => { const r = byTurn[t].find((x) => x.agent_id === id); return r && r.pos ? [r.pos[0], r.pos[1]] : p; }));
  // a robot that made no successful move in turn 1 must still stand where the rebuilt world put it
  const off = byTurn[1] ? byTurn[1].filter((r) => !r.calls.some((c) => /^You moved/.test(c[1] || "")) && (r.pos[0] !== pos[0][r.agent_id][0] || r.pos[1] !== pos[0][r.agent_id][1])) : [];
  if (off.length) notes.push(`The rebuilt start does not match the transcript for ${off.map((r) => r.agent).join(", ")} (the file may predate a change to the world); positions follow the transcript from turn 1.`);
  const d1 = (a, b) => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
  const inbox = (t) => Object.fromEntries((byTurn[t] || []).map((r) => [r.agent_id, heardOf(r)]));
  const turns = [];
  for (let t = 1; t <= T; t++) {
    const msgs = [], next = t < T ? inbox(t + 1) : null, used = new Set();
    for (const r of byTurn[t]) {
      for (const [code, result] of r.calls) {
        const said = [...String(result || "").matchAll(/You said: ('(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*")\. Heard by: /g)].map((m) => unrepr(m[1]));
        const texts = said.length ? said : [...String(code || "").matchAll(/\bsay\s*\(\s*(?:message\s*=\s*)?('(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*")\s*\)/g)].map((m) => unrepr(m[1]));
        for (const text of texts) msgs.push({ speaker_id: r.agent_id, text: squash(text) });
      }
    }
    if (next) {  // inbox lines no parsed call explains (a say with a computed argument whose result was not printed)
      const known = new Set(msgs.map((m) => `${m.speaker_id}\u0000${m.text}`));
      for (const lines of Object.values(next)) for (const l of lines) {
        const sid = idOf[l.speaker], k = `${sid}\u0000${l.text}`;
        if (sid != null && !known.has(k) && !used.has(k)) { used.add(k); msgs.push({ speaker_id: sid, text: l.text }); }
      }
    }
    const messages = msgs.map((m) => {
      const p = pos[t][m.speaker_id];
      const near = robots.filter((b) => b.id !== m.speaker_id && d1(p, pos[t][b.id]) <= cr);
      return { speaker: robots[m.speaker_id].name, speaker_id: m.speaker_id, pos: p, text: m.text, heard_by: near.map((b) => b.name), heard_by_ids: near.map((b) => b.id) };
    });
    // an event's "turn k" is World.turn while step k+1 runs (0-based), so it is in the state after record turn k+1
    const events = (raw.events || []).filter((e) => { const m = /^turn (\d+):/.exec(e); return !m || Number(m[1]) + 1 <= t; });
    const state = { turn: t, n: start.n, task: start.task, agents: start.agents.map((a, id) => ({ ...a, x: pos[t][id][0], y: pos[t][id][1] })), objects: start.objects, events, derived: true };
    turns.push({ turn: t, seconds: null, records: byTurn[t], state, messages });
  }
  // scores, derived per task as Task.score computes them (the final turn keeps the file's own score fields)
  const dug = { at: null };
  let reached = 0, submitted = null;
  const target = raw.score && Array.isArray(raw.score.target) ? raw.score.target : null;
  const startDist = target ? d1(pos[0][0], target) : null;
  turns.forEach((tu, i) => {
    const t = tu.turn, ev = tu.state.events, n = robots.length;
    let sc;
    if (raw.task === "key_door") {
      const opened = doorsOpened(ev).size, of = Math.max(1, Math.floor(n / 2));
      sc = { opened, of, success: opened === of, done_turn: opened === of ? (raw.score || {}).done_turn ?? t : null, progress: +(opened / of).toFixed(2) };
    } else if (raw.task === "relay") {
      for (const r of tu.records) if (heardOf(r).some((l) => /\d{4}/.test(l.text))) reached = Math.max(reached, r.agent_id);
      for (const r of tu.records) for (const [code, res] of r.calls) { const m = /\bsubmit\s*\(\s*['"]?([^'")]*)/.exec(code || ""); if (m && /submit|not the code|Correct/.test(res || "")) submitted = (/\d{4}/.exec(m[1]) || [m[1]])[0]; }
      const ok = ev.some((e) => /submitted the right code/.test(e));
      sc = { submitted, success: ok, done_turn: ok ? (raw.score || {}).done_turn ?? t : null, hops: n - 1, hops_reached: reached, progress: +(reached / Math.max(1, n - 1)).toFixed(2) };
    } else {
      for (const r of tu.records) if (r.agent_id === 0 && r.calls.some((c) => /\bdig\s*\(/.test(c[0] || ""))) dug.at = r.pos;
      const cur = target ? d1(pos[t][0], target) : null, ok = ev.some((e) => /dug up the target/.test(e));
      sc = { target, dug_at: dug.at, success: ok, done_turn: ok ? (raw.score || {}).done_turn ?? t : null, digger_distance: cur, progress: startDist ? +(1 - cur / startDist).toFixed(2) : 0 };
    }
    tu.state.score = i === turns.length - 1 ? { ...sc, ...(raw.score || {}) } : sc;
  });
  const meta = { ...world.meta, task: raw.task, n: raw.n, n_agents: raw.agents, seed: raw.seed, turns: T, checkpoint: name || "CLI episode", device: "file" };
  const result = { task: raw.task, n: raw.n, agents: raw.agents, seed: raw.seed, turns: T, score: raw.score || {}, events: raw.events || [], seconds: raw.seconds || 0,
    cancelled: false, meta, messages: turns.map((t) => t.messages), transcript: raw.transcript };
  return { request: meta, meta, robots, start, turns, result, notes, name, kind };
}

// ---------------------------------------------------------------------------------------------- analysis
/** What a robot did in one turn: its primary action (first tool its calls invoked), every tool, direction, errors. */
export function classify(rec, before, after) {
  if (!rec) return null;
  const tools = [];
  let err = false, fail = false;
  for (const [code, res] of rec.calls || []) {
    for (const m of String(code || "").matchAll(TOOL_RE)) tools.push(m[1]);
    if (ERR_RE.test(String(res || ""))) err = true;
    else if (FAIL_RE.test(String(res || ""))) fail = true;
  }
  const first = tools[0];
  const kind = first ? kindOf(first) : (rec.calls || []).length ? "code" : "none";
  const moved = before && after ? Math.abs(after[0] - before[0]) + Math.abs(after[1] - before[1]) : 0;
  let dir = null;
  if (moved && before && after) {
    const dx = after[0] - before[0], dy = after[1] - before[1];
    dir = Math.abs(dx) >= Math.abs(dy) ? (dx > 0 ? "east" : "west") : dy > 0 ? "south" : "north";
  } else if (first === "move") {
    const m = /\bmove\s*\(\s*(?:direction\s*=\s*)?['"](\w+)['"]/.exec((rec.calls[0] || [""])[0]);
    dir = m && DIRS[m[1].toLowerCase()] ? m[1].toLowerCase() : null;
  }
  const glyph = kind === "move" ? (dir ? ARROW[dir] : "?") : KINDS[kind].glyph;
  const goalOk = kind === "goal" && (rec.calls || []).some((c) => /Task complete|is open/.test(c[1] || ""));
  return { kind, tools, glyph, dir, moved, err, fail, goalOk, multi: new Set(tools).size > 1 || tools.length > 1 };
}

/** Everything the views read, computed once per episode (and again only when a turn arrives). */
export function analyze(ep) {
  const N = ep.start ? ep.start.agents.length : 0, T = ep.turns.length;
  const states = [ep.start, ...ep.turns.map((t) => t.state)];
  const pos = states.map((s) => (s ? s.agents.map((a) => [a.x, a.y]) : []));
  const rec = [null, ...ep.turns.map((t) => { const o = {}; (t.records || []).forEach((r, i) => { o[r.agent_id ?? i] = r; }); return o; })];
  const act = [null], heard = [null], steps = [Array(N).fill(0)];
  for (let t = 1; t <= T; t++) {
    const a = {}, hd = {}, st = [];
    for (let id = 0; id < N; id++) {
      a[id] = classify(rec[t][id], pos[t - 1][id], pos[t][id]);
      hd[id] = heardOf(rec[t][id]);
      const mv = pos[t - 1][id] && pos[t][id] ? Math.abs(pos[t][id][0] - pos[t - 1][id][0]) + Math.abs(pos[t][id][1] - pos[t - 1][id][1]) : 0;
      st.push(steps[t - 1][id] + mv);
    }
    act.push(a); heard.push(hd); steps.push(st);
  }
  const messages = [];
  ep.turns.forEach((tu, i) => (tu.messages || []).forEach((m, j) => messages.push({ ...m, turn: i + 1, idx: j, key: `${i + 1}:${j}` })));
  const progressOf = (sc) => {
    if (!sc) return null;
    if (typeof sc.progress === "number") return sc.progress;
    if (typeof sc.opened === "number" && sc.of) return sc.opened / sc.of;
    if (typeof sc.hops_reached === "number" && sc.hops) return sc.hops_reached / sc.hops;
    return sc.success ? 1 : null;
  };
  const series = { progress: [], msgs: [], calls: [], errors: [], fails: [] };
  for (let t = 1; t <= T; t++) {
    series.progress.push(progressOf(states[t].score));
    series.msgs.push((ep.turns[t - 1].messages || []).length);
    series.calls.push((ep.turns[t - 1].records || []).reduce((s, r) => s + (r.n_calls ?? (r.calls || []).length), 0));
    series.errors.push(Object.values(act[t]).filter((a) => a && a.err).length);
    series.fails.push(Object.values(act[t]).filter((a) => a && a.fail).length);
  }
  // the turn after which each event first shows up in the state (an event's own "turn k" label is World.turn, 0-based)
  const eventTurns = {};
  for (let t = 1; t <= T; t++) {
    const before = new Set((states[t - 1] && states[t - 1].events) || []);
    const fresh = ((states[t] && states[t].events) || []).filter((e) => !before.has(e));
    if (fresh.length) eventTurns[t] = fresh;
  }
  return { N, T, states, pos, rec, act, heard, steps, messages, series, eventTurns };
}

/** One arrowhead marker per robot colour (`${prefix}-${id}`): markers cannot inherit the stroke of the path everywhere. */
export function ArrowDefs({ prefix, n }) {
  return html`<defs>${Array.from({ length: n }, (_, id) => html`<marker key=${id} id=${`${prefix}-${id}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill=${robotColor(id)} /></marker>`)}</defs>`;
}

// ---------------------------------------------------------------------------------------------- legend
export function GlyphLegend({ compact }) {
  return html`<div class="legend arena-glyphs">
    ${Object.entries(KINDS).map(([k, v]) => html`<span key=${k}><span class="arena-glyph" style=${`background:${v.color};color:${v.ink}`}>${v.glyph}</span>${v.label}</span>`)}
    <span><span class="arena-glyph err">!</span>call error (error:, wrong code, not the code)</span>
    <span><span class="arena-glyph fail">·</span>call did nothing (blocked, not on the door, nothing here)</span>
    ${!compact && html`<span><span class="arena-sw arc"></span>message: speaker's turn → hearer's next turn (delivery), coloured per speaker</span>
      <span><span class="arena-sw nobody"></span>said to nobody</span>`}
  </div>`;
}

// ---------------------------------------------------------------------------------------------- tooltip
function CellTip({ tip, ep, ix }) {
  if (!tip) return null;
  const { t, id, x, y } = tip;
  const r = ix.rec[t] && ix.rec[t][id], a = ix.act[t] && ix.act[t][id], robot = ep.robots.find((b) => b.id === id);
  const said = (ep.turns[t - 1].messages || []).filter((m) => m.speaker_id === id);
  const left = Math.min(x + 14, window.innerWidth - 440), top = Math.min(y + 12, window.innerHeight - 320);
  return html`<div class="arena-tip" style=${`left:${Math.max(8, left)}px;top:${Math.max(8, top)}px`}>
    <div><b>${robot ? robot.name : `R${id + 1}`}</b> · turn ${t}${r ? ` · at ${xy(r.pos)}` : ""}${a && a.moved ? ` · moved ${a.moved}` : ""}</div>
    ${!r ? html`<div class="muted">no record this turn</div>` : html`
      <div class="evd-lab">heard</div>${ix.heard[t][id].length ? ix.heard[t][id].map((l, i) => html`<div key=${i} class="arena-tip-heard">${l.line}</div>`) : html`<div class="muted">nothing</div>`}
      <div class="evd-lab">question</div><div>${clip(question(r.observation), 260)}</div>
      <div class="evd-lab">call${r.calls.length === 1 ? "" : "s"}</div>
      ${r.calls.length ? r.calls.map((c, i) => html`<div key=${i} class="arena-tip-call"><code>${clip(c[0], 160)}</code><div class=${ERR_RE.test(c[1] || "") ? "err" : FAIL_RE.test(c[1] || "") ? "fail" : "ok"}>→ ${clip(c[1], 200)}</div></div>`) : html`<div class="muted">no call</div>`}
      ${said.length > 0 && html`<div class="evd-lab">said</div>${said.map((m, i) => html`<div key=${i}>"${clip(m.text, 120)}" → ${m.heard_by.length ? m.heard_by.join(", ") : "nobody within range"}</div>`)}`}
      <div class="evd-lab">answer</div><div>${clip((r.answer || "").trim(), 160) || html`<span class="muted">(empty)</span>`}</div>`}
  </div>`;
}

// ---------------------------------------------------------------------------------------------- timeline
/** Swimlanes: rows = robots, columns = turns. A cell is the robot's action that turn; a message is an arrow from the
 *  speaker's cell to each hearer's cell in the next column (delivery is next turn). Hover = tooltip, click = select. */
export function Timeline({ ep, ix, view, sel, focus, onPick, setHover }) {
  const [tip, setTip] = useState(null);
  const { N, T } = ix;
  if (!N || !T) return html`<div class="muted arena-empty">The timeline fills in as turns arrive.</div>`;
  const RH = N > 16 ? 14 : N > 8 ? 18 : 24, CW = T <= 10 ? 52 : T <= 20 ? 38 : 30, HEAD = 18;
  const lastMsgs = (ep.turns[T - 1].messages || []).length > 0;
  const cols = T + (lastMsgs ? 1 : 0), W = cols * CW + 4, H = HEAD + N * RH + 2;
  const cx = (t) => (t - 1) * CW, cy = (id) => HEAD + id * RH;
  const fs = Math.max(9, Math.min(13, RH - 5));
  const hot = (m) => focus == null || m.speaker_id === focus || m.heard_by_ids.includes(focus);
  const pick = (t, id) => onPick(id, t);
  const cells = [], arrows = [], marks = [];
  for (let t = 1; t <= T; t++) {
    for (let id = 0; id < N; id++) {
      const a = ix.act[t][id];
      const k = a ? KINDS[a.kind] : null;
      cells.push(html`<g key=${`c${t}-${id}`} class="tl-cell" onMouseEnter=${(e) => { setTip({ t, id, x: e.clientX, y: e.clientY }); setHover && setHover(id); }}
          onMouseMove=${(e) => setTip({ t, id, x: e.clientX, y: e.clientY })} onMouseLeave=${() => { setTip(null); setHover && setHover(null); }} onClick=${() => pick(t, id)}>
        <rect x=${cx(t) + 1} y=${cy(id) + 1} width=${CW - 2} height=${RH - 2} rx="2" fill=${k ? k.color : "transparent"} class=${a && a.err ? "err" : a && a.goalOk ? "ok" : ""} />
        ${a && html`<text x=${cx(t) + CW / 2} y=${cy(id) + RH / 2 + fs * 0.36} style=${`font-size:${fs}px;fill:${k.ink}`}>${a.glyph}${a.multi ? "+" : ""}</text>`}
        ${a && a.err && html`<path d=${`M${cx(t) + CW - 1},${cy(id) + 1} l0,${Math.min(8, RH - 4)} l${-Math.min(8, RH - 4)},${-Math.min(8, RH - 4)} z`} class="tl-err" />`}
        ${a && !a.err && a.fail && html`<circle cx=${cx(t) + CW - 5} cy=${cy(id) + 5} r="2.5" class="tl-fail" />`}
      </g>`);
    }
    (ep.turns[t - 1].messages || []).forEach((m, j) => {
      const x1 = cx(t) + CW * 0.72, y1 = cy(m.speaker_id) + RH / 2, col = robotColor(m.speaker_id), on = hot(m);
      if (!m.heard_by_ids.length) {
        arrows.push(html`<circle key=${`n${t}-${j}`} cx=${cx(t) + CW - 6} cy=${y1} r=${Math.min(4, RH / 3)} class=${"tl-nobody" + (on ? "" : " dim")}><title>${m.speaker} (turn ${t}): "${m.text}" (nobody within range)</title></circle>`);
        return;
      }
      for (const hid of m.heard_by_ids) {
        const x2 = cx(t + 1) + CW * 0.18, y2 = cy(hid) + RH / 2, mx = (x1 + x2) / 2;
        arrows.push(html`<path key=${`a${t}-${j}-${hid}`} d=${`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`} stroke=${col} class=${"tl-arrow" + (on ? "" : " dim")}
          marker-end=${`url(#tl-head-${m.speaker_id})`}><title>${m.speaker} (turn ${t}) → ${(ep.robots[hid] || {}).name || `R${hid + 1}`} (heard at turn ${t + 1}): "${m.text}"</title></path>`);
      }
    });
    if (ix.eventTurns[t]) marks.push(html`<text key=${`e${t}`} class="tl-event" x=${cx(t) + CW - 7} y=${HEAD - 5}><title>${ix.eventTurns[t].join("\n")}</title>★</text>`);
  }
  return html`<div class="tl-wrap">
    <div class="tl-names" style=${`padding-top:${HEAD}px`}>
      ${ep.robots.map((r) => { const role = roleOf(r.tools.map((x) => x.name)); return html`<div key=${r.id} class=${"tl-name" + (r.id === sel ? " sel" : "")} style=${`height:${RH}px;line-height:${RH}px;font-size:${Math.min(12, RH - 3)}px`}
          onClick=${() => onPick(r.id, null)} title=${`${r.name} · ${role.label} · moved ${ix.steps[Math.min(view, T)][r.id]} cells by turn ${Math.min(view, T)}`}>
        <span class="arena-dot" style=${`background:${role.color};width:${Math.min(9, RH - 6)}px;height:${Math.min(9, RH - 6)}px`}></span><span style=${`color:${robotColor(r.id)}`}>${r.name}</span></div>`; })}
    </div>
    <div class="tl-scroll">
      <svg class="tl-svg" width=${W} height=${H} viewBox=${`0 0 ${W} ${H}`} role="img" aria-label=${`interaction timeline: ${N} robots over ${T} turns`}>
        <${ArrowDefs} prefix="tl-head" n=${N} />
        ${sel != null && sel < N && html`<rect x="0" y=${cy(sel)} width=${W} height=${RH} class="tl-selrow" />`}
        ${view >= 1 && view <= T && html`<rect x=${cx(view)} y="0" width=${CW} height=${H} class="tl-now" />`}
        ${Array.from({ length: cols }, (_, i) => html`<text key=${"h" + i} class=${"tl-head" + (i + 1 === view ? " now" : "")} x=${cx(i + 1) + CW / 2} y=${HEAD - 5}
            onClick=${() => i + 1 <= T && onPick(null, i + 1)}>${i + 1 <= T ? i + 1 : "next"}</text>`)}
        ${Array.from({ length: N + 1 }, (_, i) => html`<line key=${"g" + i} x1="0" x2=${W} y1=${cy(i)} y2=${cy(i)} class="tl-grid" />`)}
        ${cells}${marks}
        <g class="tl-arrows">${arrows}</g>
      </svg>
    </div>
    <${CellTip} tip=${tip} ep=${ep} ix=${ix} />
  </div>`;
}

// ---------------------------------------------------------------------------------------------- message log
const LOG_CAP = 400;
export function MessageLog({ ep, ix, view, focusMsg, onPick }) {
  const [who, setWho] = useState("");
  const [q, setQ] = useState("");
  const T = ix.T;
  const rows = useMemo(() => {
    const id = who === "" ? null : Number(who), needle = q.trim().toLowerCase();
    return ix.messages.filter((m) => (id == null || m.speaker_id === id || m.heard_by_ids.includes(id))
      && (!needle || m.text.toLowerCase().includes(needle) || m.speaker.toLowerCase() === needle || m.heard_by.some((n) => n.toLowerCase() === needle)));
  }, [ix, who, q]);
  return html`<div class="panel arena-log">
    <div class="row" style="margin-bottom:6px">
      <span class="grp"><span class="muted">robot</span><select value=${who} onChange=${(e) => setWho(e.target.value)}>
        <option value="">all</option>${ep.robots.map((r) => html`<option key=${r.id} value=${r.id}>${r.name}</option>`)}</select></span>
      <input type="text" placeholder="filter text or a robot name" value=${q} onInput=${(e) => setQ(e.target.value)} style="width:220px" />
      <span class="muted">${rows.length} of ${ix.messages.length} message${ix.messages.length === 1 ? "" : "s"}${who !== "" ? ` said or heard by ${ep.robots[Number(who)].name}` : ""}</span>
    </div>
    ${rows.length ? html`<div class="arena-log-list">
      ${rows.slice(0, LOG_CAP).map((m) => html`<div key=${m.key} class=${"arena-log-row" + (m.turn === view ? " now" : "") + (focusMsg === m.key ? " sel" : "")} onClick=${() => onPick(m)}>
        <span class="t">t${m.turn}</span>
        <span class="who"><b style=${`color:${robotColor(m.speaker_id)}`}>${m.speaker}</b> → ${m.heard_by.length ? m.heard_by.join(", ") : html`<span class="muted">nobody within range</span>`}</span>
        <span class="txt">"${m.text}"</span>
        <span class="dl">${!m.heard_by.length ? "" : m.turn < T ? html`<span class="arena-delivered" title=${`delivered at the start of turn ${m.turn + 1}`}>✓ heard t${m.turn + 1}</span>` : html`<span class="muted" title="the episode ended before the next turn">not delivered</span>`}</span>
      </div>`)}
      ${rows.length > LOG_CAP && html`<div class="muted">showing the first ${LOG_CAP} of ${rows.length}; filter to narrow</div>`}
    </div>` : html`<div class="muted">${ix.messages.length ? "no message matches the filter" : "nobody has said anything yet"}</div>`}
  </div>`;
}

// ---------------------------------------------------------------------------------------------- summary strip
function Spark({ title, info, values, view, onPick, kind = "bar", color = "#2563eb", max, fmt = (v) => String(v), marks = {} }) {
  const W = 240, H = 56, P = 3, n = values.length;
  const top = max ?? Math.max(1, ...values.filter((v) => v != null));
  const bw = n ? (W - 2 * P) / n : 0;
  const X = (i) => P + i * bw + bw / 2, Y = (v) => H - 2 - (v / top) * (H - 8);
  const total = values.reduce((s, v) => s + (v || 0), 0);
  const last = [...values].reverse().find((v) => v != null);
  return html`<div class="arena-spark">
    <div class="arena-spark-t">${title}${info && html`<${Info} k=${info} />`}<span class="muted">${kind === "line" ? (last == null ? "n/a" : fmt(last)) : `${total} total`}</span></div>
    <svg viewBox=${`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label=${title}>
      <line x1=${P} x2=${W - P} y1=${H - 1} y2=${H - 1} class="sp-axis" />
      ${view >= 1 && view <= n && html`<rect x=${P + (view - 1) * bw} y="0" width=${bw} height=${H} class="sp-now" />`}
      ${kind === "bar" ? values.map((v, i) => html`<rect key=${i} x=${P + i * bw + bw * 0.15} y=${Y(v || 0)} width=${bw * 0.7} height=${Math.max(0, H - 2 - Y(v || 0))} fill=${color} />`)
        : html`<polyline fill="none" stroke=${color} stroke-width="1.8" points=${values.map((v, i) => (v == null ? null : `${X(i)},${Y(v)}`)).filter(Boolean).join(" ")} />
          ${values.map((v, i) => (v == null ? null : html`<circle key=${i} cx=${X(i)} cy=${Y(v)} r="2" fill=${color} />`))}`}
      ${Object.keys(marks).map((t) => html`<line key=${"m" + t} x1=${X(Number(t) - 1)} x2=${X(Number(t) - 1)} y1="0" y2=${H} class="sp-mark" />`)}
      ${values.map((v, i) => html`<rect key=${"hit" + i} x=${P + i * bw} y="0" width=${bw} height=${H} class="sp-hit" onClick=${() => onPick(i + 1)}><title>turn ${i + 1}: ${v == null ? "n/a" : fmt(v)}${marks[i + 1] ? "\n" + marks[i + 1].join("\n") : ""}</title></rect>`)}
    </svg>
    <div class="sp-labs"><span>turn 1</span><span>${n}</span></div>
  </div>`;
}

export function SummaryStrip({ ep, ix, view, onPick }) {
  if (!ix.T) return null;
  const s = ix.series, res = ep.result, sc = res ? res.score : (ix.states[ix.T] || {}).score || {};
  const progLabel = ep.meta && ep.meta.task === "relay" ? "hops reached" : ep.meta && ep.meta.task === "triangulate" ? "digger closer (1 - dist / start)" : "doors opened";
  return html`<div class="panel arena-summary">
    <div class="arena-sparks">
      <${Spark} title=${`task progress (${progLabel})`} info="arena_summary" values=${s.progress} view=${view} onPick=${onPick} kind="line" color="#16a34a" max=${1} fmt=${(v) => `${Math.round(v * 100)}%`} marks=${ix.eventTurns} />
      <${Spark} title="messages per turn" values=${s.msgs} view=${view} onPick=${onPick} color="#2563eb" />
      <${Spark} title="tool calls per turn" values=${s.calls} view=${view} onPick=${onPick} color="#6366f1" />
      <${Spark} title="call errors per turn" values=${s.errors} view=${view} onPick=${onPick} color="#dc2626" max=${Math.max(1, ...s.errors, ...s.fails)} />
    </div>
    <div class="arena-verdict">
      <span class=${"stage-badge " + (sc.success ? "rl" : "base")}>${sc.success ? `solved in ${sc.done_turn ?? ix.T} turn${(sc.done_turn ?? ix.T) === 1 ? "" : "s"}` : res ? (res.cancelled ? "stopped" : "not solved") : "in progress"}</span>
      <span class="muted">${ix.T} turn${ix.T === 1 ? "" : "s"} · ${ix.messages.length} messages · ${s.calls.reduce((a, b) => a + b, 0)} tool calls · ${s.errors.reduce((a, b) => a + b, 0)} with errors · ${s.fails.reduce((a, b) => a + b, 0)} did nothing</span>
      ${(res ? res.events : (ix.states[ix.T] || {}).events || []).map((e, i) => {
        const at = Object.keys(ix.eventTurns).find((t) => ix.eventTurns[t].includes(e));
        return html`<span key=${i} class="arena-ev" title=${at ? `shows up after turn ${at}` : ""} onClick=${() => at && onPick(Number(at))}>★ ${e}</span>`;
      })}
    </div>
  </div>`;
}

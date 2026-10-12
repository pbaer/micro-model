/** The Arena in 3D: a small hand-written WebGL renderer (no library: the portal vendors its dependencies and has no build
 *  step) drawing toon-shaded, outlined cartoon robots on a tile floor, with an orbit camera, WASD to move the centre,
 *  click-to-follow, and an HTML overlay for the things text does better (name tags, speech bubbles, emotes, flying mail,
 *  the verdict banner).
 *
 *  It is a pure view of the same data the 2D map draws: `ep` (the normalized episode), `ix` (analyze(ep): positions per
 *  turn, actions), `view` (0 = start, k = after turn k). Turn k's animations (hops, mail, coins, tags, confetti) play when
 *  the view arrives at k; scrubbing far snaps with a short glide.
 *
 *  Controls: drag = orbit · right-drag or Shift+drag = pan · wheel = zoom · W A S D = move the centre · Q / E = turn ·
 *  R = reset · T = top-down · click a robot = select and follow · Esc = stop following. Keys act only while the pointer
 *  is over the view or it has focus, so typing in the form is never hijacked. */
import { h } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import htm from "htm";
import { roleOf, robotColor, doorsOpened, clip } from "./arena_views.js";

const html = htm.bind(h);

// ------------------------------------------------------------------------------------------------ math (column-major)
const M4 = {
  ident: () => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
  mul(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
    return o;
  },
  persp(fov, asp, n, f) {
    const t = 1 / Math.tan(fov / 2), o = new Float32Array(16);
    o[0] = t / asp; o[5] = t; o[10] = (f + n) / (n - f); o[11] = -1; o[14] = (2 * f * n) / (n - f);
    return o;
  },
  look(e, c, u) {
    let zx = e[0] - c[0], zy = e[1] - c[1], zz = e[2] - c[2];
    let l = Math.hypot(zx, zy, zz); zx /= l; zy /= l; zz /= l;
    let xx = u[1] * zz - u[2] * zy, xy = u[2] * zx - u[0] * zz, xz = u[0] * zy - u[1] * zx;
    l = Math.hypot(xx, xy, xz); xx /= l; xy /= l; xz /= l;
    const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
    return new Float32Array([xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0, -(xx * e[0] + xy * e[1] + xz * e[2]), -(yx * e[0] + yy * e[1] + yz * e[2]), -(zx * e[0] + zy * e[1] + zz * e[2]), 1]);
  },
  /** translate * rotY * rotX * rotZ * scale */
  trs(tx, ty, tz, ry = 0, rx = 0, rz = 0, sx = 1, sy = sx, sz = sx) {
    const cy = Math.cos(ry), sy_ = Math.sin(ry), cx = Math.cos(rx), sx_ = Math.sin(rx), cz = Math.cos(rz), sz_ = Math.sin(rz);
    // R = Ry * Rx * Rz
    const r00 = cy * cz + sy_ * sx_ * sz_, r01 = -cy * sz_ + sy_ * sx_ * cz, r02 = sy_ * cx;
    const r10 = cx * sz_, r11 = cx * cz, r12 = -sx_;
    const r20 = -sy_ * cz + cy * sx_ * sz_, r21 = sy_ * sz_ + cy * sx_ * cz, r22 = cy * cx;
    return new Float32Array([r00 * sx, r10 * sx, r20 * sx, 0, r01 * sy, r11 * sy, r21 * sy, 0, r02 * sz, r12 * sz, r22 * sz, 0, tx, ty, tz, 1]);
  },
};
const lerp = (a, b, t) => a + (b - a) * t;
const ROBOT = 1.3;  // robot scale: chunky enough to read their googly eyes from the default camera
const ease = (t) => (t < 0 ? 0 : t > 1 ? 1 : t * t * (3 - 2 * t));
const backOut = (t, c = 1.9) => { const u = Math.max(0, Math.min(1, t)) - 1; return 1 + (c + 1) * u * u * u + c * u * u; };
/** A cartoon scoot over one cell, `tt` seconds into the turn: a wind-up (lean back, squash, a little step back), a dash
 *  that overshoots and settles (leaning into it), then a rocking stop. p = progress 0..1(+), lean in radians (+ = forward). */
const SCOOT = 0.62;
const scootPose = (tt) => {
  if (tt < 0.12) { const u = Math.sin((tt / 0.12) * Math.PI / 2); return { p: -0.07 * u, lean: -0.22 * u, squash: 1 - 0.12 * u }; }
  if (tt < SCOOT) { const u = (tt - 0.12) / (SCOOT - 0.12); return { p: -0.07 + 1.07 * backOut(u), lean: 0.36 * Math.sin(Math.min(1, u * 1.3) * Math.PI) - 0.22 * Math.max(0, 1 - u * 4), squash: 1 + 0.07 * Math.sin(u * Math.PI) }; }
  const s = tt - SCOOT;
  return { p: 1, lean: 0.3 * Math.exp(-4 * s) * Math.sin(16 * s), squash: 1 - 0.07 * Math.exp(-6 * s) * Math.sin(18 * s) };
};
/** A bump: lunge 0.4 of a cell towards the occupied cell, hit at BUMP_HIT, bounce back past the start and settle. */
const BUMP_HIT = 0.16;
const bumpOffset = (tt) => (tt < BUMP_HIT ? 0.4 * ease(tt / BUMP_HIT) : 0.4 * Math.exp(-7 * (tt - BUMP_HIT)) * Math.cos(13 * (tt - BUMP_HIT)));
const hex = (c) => { const m = /^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i.exec(c); return m ? [1, 2, 3].map((i) => parseInt(m[i], 16) / 255) : [0.6, 0.6, 0.6]; };
const hsl = (s) => {  // "hsl(h, s%, l%)" -> rgb 0..1
  const m = /hsl\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*\)/.exec(s);
  if (!m) return hex(s);
  const H = +m[1] / 360, S = +m[2] / 100, L = +m[3] / 100;
  const q = L < 0.5 ? L * (1 + S) : L + S - L * S, p = 2 * L - q;
  const f = (t) => { t = (t + 1) % 1; return t < 1 / 6 ? p + (q - p) * 6 * t : t < 1 / 2 ? q : t < 2 / 3 ? p + (q - p) * (2 / 3 - t) * 6 : p; };
  return [f(H + 1 / 3), f(H), f(H - 1 / 3)];
};

// ------------------------------------------------------------------------------------------------ geometry
function box() {
  const P = [], N = [], I = [];
  const F = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], [[0, 1, 0], [0, 0, 1], [1, 0, 0]], [[0, -1, 0], [0, 0, -1], [1, 0, 0]], [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], [[0, 0, -1], [0, 1, 0], [1, 0, 0]]];
  for (const [n, u, v] of F) {
    const b = P.length / 3;
    for (const [a, c] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      P.push(0.5 * (n[0] + a * v[0] + c * u[0]), 0.5 * (n[1] + a * v[1] + c * u[1]), 0.5 * (n[2] + a * v[2] + c * u[2]));
      N.push(...n);
    }
    I.push(b, b + 1, b + 2, b, b + 2, b + 3);
  }
  return { P, N, I };
}
function sphere(seg = 16, rings = 12) {
  const P = [], N = [], I = [];
  for (let r = 0; r <= rings; r++) {
    const th = (r / rings) * Math.PI;
    for (let s = 0; s <= seg; s++) {
      const ph = (s / seg) * Math.PI * 2, x = Math.sin(th) * Math.cos(ph), y = Math.cos(th), z = Math.sin(th) * Math.sin(ph);
      P.push(x * 0.5, y * 0.5, z * 0.5); N.push(x, y, z);
    }
  }
  for (let r = 0; r < rings; r++) for (let s = 0; s < seg; s++) {
    const a = r * (seg + 1) + s, b = a + seg + 1;
    I.push(a, b, a + 1, b, b + 1, a + 1);
  }
  return { P, N, I };
}
function cylinder(seg = 18, top = 0.5, bottom = 0.5) {
  const P = [], N = [], I = [];
  for (let s = 0; s <= seg; s++) {
    const a = (s / seg) * Math.PI * 2, c = Math.cos(a), sn = Math.sin(a);
    P.push(c * bottom, -0.5, sn * bottom, c * top, 0.5, sn * top); N.push(c, 0, sn, c, 0, sn);
  }
  for (let s = 0; s < seg; s++) { const a = s * 2; I.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
  for (const [y, r, ny] of [[0.5, top, 1], [-0.5, bottom, -1]]) {
    const c0 = P.length / 3; P.push(0, y, 0); N.push(0, ny, 0);
    for (let s = 0; s <= seg; s++) { const a = (s / seg) * Math.PI * 2; P.push(Math.cos(a) * r, y, Math.sin(a) * r); N.push(0, ny, 0); }
    for (let s = 0; s < seg; s++) ny > 0 ? I.push(c0, c0 + s + 2, c0 + s + 1) : I.push(c0, c0 + s + 1, c0 + s + 2);
  }
  return { P, N, I };
}
function disc(seg = 24) {
  const P = [0, 0, 0], N = [0, 1, 0], I = [];
  for (let s = 0; s <= seg; s++) { const a = (s / seg) * Math.PI * 2; P.push(Math.cos(a) * 0.5, 0, Math.sin(a) * 0.5); N.push(0, 1, 0); }
  for (let s = 0; s < seg; s++) I.push(0, s + 2, s + 1);
  return { P, N, I };
}
function ground(n) {  // one mesh: n x n bevelled tiles with vertex colours (a candy checkerboard)
  const P = [], N = [], C = [], I = [];
  const b = box();
  const A = [0.80, 0.94, 0.86], B = [0.98, 0.95, 0.85];
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const col = (x + y) % 2 ? A : B, jit = ((x * 7 + y * 13) % 5) * 0.004, base = P.length / 3;
    for (let i = 0; i < b.P.length; i += 3) {
      P.push(x - n / 2 + 0.5 + b.P[i] * 0.94, -0.06 + b.P[i + 1] * (0.12 + jit), y - n / 2 + 0.5 + b.P[i + 2] * 0.94);
      N.push(b.N[i], b.N[i + 1], b.N[i + 2]); C.push(...col);
    }
    for (const k of b.I) I.push(base + k);
  }
  return { P, N, I, C };
}

/** Make every triangle wind counter-clockwise seen from outside (its face normal agrees with its vertex normals), so
 *  back-face culling and the inverted-hull outline work whatever order a builder emitted. */
function fixWinding(g) {
  const P = g.P, N = g.N, I = g.I.slice();
  for (let t = 0; t < I.length; t += 3) {
    const [a, b, c] = [I[t], I[t + 1], I[t + 2]];
    const ux = P[b * 3] - P[a * 3], uy = P[b * 3 + 1] - P[a * 3 + 1], uz = P[b * 3 + 2] - P[a * 3 + 2];
    const vx = P[c * 3] - P[a * 3], vy = P[c * 3 + 1] - P[a * 3 + 1], vz = P[c * 3 + 2] - P[a * 3 + 2];
    const fx = uy * vz - uz * vy, fy = uz * vx - ux * vz, fz = ux * vy - uy * vx;
    const nx = N[a * 3] + N[b * 3] + N[c * 3], ny = N[a * 3 + 1] + N[b * 3 + 1] + N[c * 3 + 1], nz = N[a * 3 + 2] + N[b * 3 + 2] + N[c * 3 + 2];
    if (fx * nx + fy * ny + fz * nz < 0) { I[t + 1] = c; I[t + 2] = b; }
  }
  return { ...g, I };
}

// ------------------------------------------------------------------------------------------------ renderer
const VS = `attribute vec3 aP; attribute vec3 aN; attribute vec3 aC;
uniform mat4 uM; uniform mat4 uVP; uniform float uInflate;
varying vec3 vN; varying vec3 vW; varying vec3 vC;
void main() {
  vec4 w = uM * vec4(aP, 1.0);
  // the ink hull: pushed away from the part's centre in world space, after scaling. Pushed along the vertex normals in
  // the mesh's own space (until 2026-10-11) it took the scale with it (a flattened coin or a board-sized box got a thick
  // slab on one side and none on another) and split at hard edges, where the rim and the caps have different normals.
  if (uInflate > 0.0) { vec3 c = (uM * vec4(0.0, 0.0, 0.0, 1.0)).xyz; w.xyz += normalize(w.xyz - c + vec3(1e-5)) * uInflate * 0.7; }
  vW = w.xyz; vN = mat3(uM) * aN; vC = aC; gl_Position = uVP * w;
}`;
const FS = `precision mediump float;
uniform vec3 uColor; uniform vec3 uLight; uniform vec3 uCam; uniform vec3 uFog; uniform float uAlpha; uniform float uVC; uniform float uFlat; uniform float uEmit; uniform float uFogFar;
varying vec3 vN; varying vec3 vW; varying vec3 vC;
void main() {
  vec3 base = mix(uColor, vC, uVC);
  if (uFlat > 0.5) { gl_FragColor = vec4(base, uAlpha); return; }
  vec3 n = normalize(vN); float d = dot(n, normalize(uLight));
  float band = d > 0.55 ? 1.0 : d > 0.1 ? 0.8 : 0.6;
  vec3 v = normalize(uCam - vW); float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.28;
  vec3 col = base * (band + uEmit) + vec3(rim);
  float f = smoothstep(uFogFar * 0.45, uFogFar, length(uCam - vW));
  gl_FragColor = vec4(mix(col, uFog, f * 0.75), uAlpha);
}`;

class Renderer {
  constructor(canvas) {
    const gl = canvas.getContext("webgl", { antialias: true, alpha: true, premultipliedAlpha: false });
    if (!gl) throw new Error("WebGL is not available in this browser");
    this.gl = gl;
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    gl.useProgram(p);
    this.p = p;
    this.a = { P: gl.getAttribLocation(p, "aP"), N: gl.getAttribLocation(p, "aN"), C: gl.getAttribLocation(p, "aC") };
    this.u = {};
    for (const k of ["uM", "uVP", "uInflate", "uColor", "uLight", "uCam", "uFog", "uAlpha", "uVC", "uFlat", "uEmit", "uFogFar"]) this.u[k] = gl.getUniformLocation(p, k);
    gl.enable(gl.DEPTH_TEST); gl.enable(gl.CULL_FACE);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    this.mesh = { box: this.upload(box()), sphere: this.upload(sphere()), cyl: this.upload(cylinder()), cone: this.upload(cylinder(18, 0.02, 0.5)), disc: this.upload(disc()) };
  }
  upload(g) {
    g = fixWinding(g);
    const gl = this.gl, buf = (data, T = Float32Array, target = gl.ARRAY_BUFFER) => { const b = gl.createBuffer(); gl.bindBuffer(target, b); gl.bufferData(target, new T(data), gl.STATIC_DRAW); return b; };
    return { p: buf(g.P), n: buf(g.N), c: g.C ? buf(g.C) : null, i: buf(g.I, g.P.length / 3 > 65535 ? Uint32Array : Uint16Array, gl.ELEMENT_ARRAY_BUFFER), count: g.I.length,
      big: g.P.length / 3 > 65535 };
  }
  frame(vp, cam, fogFar, bg) {
    const gl = this.gl, u = this.u;
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);  // the page's sky shows through
    gl.uniformMatrix4fv(u.uVP, false, vp); gl.uniform3fv(u.uCam, cam); gl.uniform3fv(u.uLight, [0.45, 1.0, 0.35]);
    gl.uniform3fv(u.uFog, bg); gl.uniform1f(u.uFogFar, fogFar);
  }
  /** One mesh: color rgb, model matrix; opts: outline (cartoon ink line), alpha, emit, flat, vc (vertex colours). */
  draw(mesh, M, color, o = {}) {
    const gl = this.gl, u = this.u, a = this.a;
    gl.bindBuffer(gl.ARRAY_BUFFER, mesh.p); gl.enableVertexAttribArray(a.P); gl.vertexAttribPointer(a.P, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, mesh.n); gl.enableVertexAttribArray(a.N); gl.vertexAttribPointer(a.N, 3, gl.FLOAT, false, 0, 0);
    if (mesh.c && a.C >= 0) { gl.bindBuffer(gl.ARRAY_BUFFER, mesh.c); gl.enableVertexAttribArray(a.C); gl.vertexAttribPointer(a.C, 3, gl.FLOAT, false, 0, 0); }
    else if (a.C >= 0) { gl.disableVertexAttribArray(a.C); gl.vertexAttrib3f(a.C, 1, 1, 1); }
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.i);
    gl.uniformMatrix4fv(u.uM, false, M);
    const type = mesh.big ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
    const alpha = o.alpha ?? 1;
    if (alpha < 1) { gl.enable(gl.BLEND); gl.depthMask(false); }
    if (o.twoSided) gl.disable(gl.CULL_FACE);
    if (o.outline) {  // inverted hull: back faces pushed out along the normals, drawn flat dark
      gl.cullFace(gl.FRONT);
      gl.uniform1f(u.uInflate, o.outline); gl.uniform3fv(u.uColor, [0.1, 0.09, 0.14]); gl.uniform1f(u.uFlat, 1); gl.uniform1f(u.uVC, 0); gl.uniform1f(u.uAlpha, alpha);
      gl.drawElements(gl.TRIANGLES, mesh.count, type, 0);
      gl.cullFace(gl.BACK);
    }
    gl.uniform1f(u.uInflate, 0); gl.uniform3fv(u.uColor, color); gl.uniform1f(u.uFlat, o.flat ? 1 : 0); gl.uniform1f(u.uVC, o.vc ? 1 : 0);
    gl.uniform1f(u.uAlpha, alpha); gl.uniform1f(u.uEmit, o.emit || 0);
    gl.drawElements(gl.TRIANGLES, mesh.count, type, 0);
    if (alpha < 1) { gl.disable(gl.BLEND); gl.depthMask(true); }
    if (o.twoSided) gl.enable(gl.CULL_FACE);
  }
}

// ------------------------------------------------------------------------------------------------ what a turn did, per robot
/** The robot this turn's move bumped into, {into, at: [x, y]}: the record's `bumped` (2026-10-11), else the move's result. */
const bumpOf = (rec) => {
  if (!rec) return null;
  if (rec.bumped) return rec.bumped;
  for (const c of rec.calls || []) {
    const m = /Bump! (R\d+) is standing at \((\d+), (\d+)\)/.exec(String(c[1] || ""));
    if (m) return { into: m[1], at: [Number(m[2]), Number(m[3])] };
  }
  return null;
};
const EMOTE = (rec, said) => {
  if (!rec) return null;
  const res = (rec.calls || []).map((c) => String(c[1] || ""));
  const code = (rec.calls || []).map((c) => String(c[0] || ""));
  if (!rec.calls || !rec.calls.length) return { e: "❓", why: "no tool call this turn" };
  if (res.some((r) => /^\s*error\b|Traceback/.test(r))) return { e: "💥", why: "a call failed" };
  const bump = bumpOf(rec);
  if (bump) return { e: "💫", why: `bumped into ${bump.into}`, bump: true };
  if (res.some((r) => /You tagged/.test(r))) return { e: "🫵", why: "tagged a runner!", big: true };
  if (res.some((r) => /picked up the coin/.test(r))) return { e: "🪙", why: "+1 coin", big: true };
  if (res.some((r) => /is open|Task complete|Correct!|finished|found it/i.test(r))) return { e: "🎉", why: "goal!", big: true };
  if (res.some((r) => /Blocked|too far|already tagged|no coin|Wrong code|not the code|not your team|not a runner|Move there first/.test(r))) return { e: "🧱", why: "that did not work" };
  if (said) return null;  // the speech bubble says it
  if (code.some((c) => /^\s*look\s*\(/.test(c))) return { e: "👀", why: "looked around" };
  if (code.some((c) => /read_(key|code)/.test(c))) return { e: "🤫", why: "read the secret" };
  return null;
};

// ------------------------------------------------------------------------------------------------ the component
export function Arena3D({ ep, ix, view, sel, setSel, showMsgs = "all", showRange = false, trails = "all", showTarget = false, titleOf }) {
  const wrap = useRef(null), canvas = useRef(null), layer = useRef(null);
  const [err, setErr] = useState(null);
  const [follow, setFollow] = useState(false);  // follow starts with a click on a robot, not on load
  const live = useRef({});  // everything the animation loop reads: updated each render, read every frame
  live.current = { ep, ix, view, sel, showMsgs, showRange, trails, showTarget, follow, setSel, titleOf, setFollow };

  useEffect(() => {
    let R;
    // one renderer per canvas, reused when another episode opens: tearing the context down on cleanup (loseContext) left
    // the same canvas with a dead context, and the next episode's shaders failed to compile with no message
    try { R = canvas.current.__r || (canvas.current.__r = new Renderer(canvas.current)); } catch (e) { setErr(String(e.message || e)); return; }
    const L = live.current;
    const n = L.ep.start.n;
    const groundMesh = R.upload(ground(n));
    const cam = { yaw: -0.6, pitch: 0.85, dist: n * 1.25 + 4, tx: 0, tz: 0 };
    const home = { ...cam };
    const keys = new Set();
    let hover = false, raf = 0, last = performance.now(), turnStart = performance.now(), shownView = -1, prevView = -1;
    const anim = {};  // id -> {x, z, yaw, fromX, fromZ, toX, toZ}
    const particles = [];  // confetti / sparkles / smoke: {x,y,z,vx,vy,vz,life,max,color,s}
    const els = {};  // overlay elements by key
    const W = (x) => x - n / 2 + 0.5, Wz = (y) => y - n / 2 + 0.5;
    const ui = layer.current;

    const el = (key, cls, text) => {
      let d = els[key];
      if (!d) { d = document.createElement("div"); d.className = cls; ui.appendChild(d); els[key] = d; }
      if (text != null && d.__t !== text) { d.textContent = text; d.__t = text; }
      d.__seen = true;
      return d;
    };
    const burst = (x, y, z, kind, k = 26, back = null) => {
      const COLS = { confetti: [[0.98, 0.4, 0.5], [0.4, 0.75, 0.98], [0.99, 0.84, 0.3], [0.55, 0.9, 0.55], [0.8, 0.55, 0.98]], sparkle: [[1, 0.88, 0.3], [1, 0.97, 0.7]], smoke: [[0.55, 0.55, 0.6], [0.72, 0.72, 0.76]],
        dust: [[0.93, 0.88, 0.78], [0.86, 0.8, 0.7]] }[kind];
      if (kind === "dust") {  // a scoot's launch puffs: low, quick, kicked out behind the robot
        for (let i = 0; i < k; i++) {
          const j = (Math.random() - 0.5) * 1.2;
          particles.push({ x: x + (Math.random() - 0.5) * 0.2, y, z: z + (Math.random() - 0.5) * 0.2, vx: (back[0] + back[1] * j) * (0.9 + Math.random()), vy: 0.35 + Math.random() * 0.4,
            vz: (back[1] - back[0] * j) * (0.9 + Math.random()), life: 0, max: 0.45 + Math.random() * 0.25, color: COLS[i % 2], s: 0.09 + Math.random() * 0.07, kind: "smoke", spin: 0 });
        }
        return;
      }
      for (let i = 0; i < k; i++) {
        const a = Math.random() * Math.PI * 2, sp = kind === "smoke" ? 0.6 : 1.4 + Math.random() * 1.6;
        particles.push({ x, y, z, vx: Math.cos(a) * sp * (kind === "confetti" ? 0.8 : 0.5), vy: kind === "smoke" ? 0.8 + Math.random() : 2.5 + Math.random() * 2.5, vz: Math.sin(a) * sp * (kind === "confetti" ? 0.8 : 0.5),
          life: 0, max: kind === "smoke" ? 1.1 : 1.6 + Math.random(), color: COLS[i % COLS.length], s: kind === "smoke" ? 0.18 : 0.08, kind, spin: Math.random() * 6 });
      }
    };

    // --- input
    const cv = canvas.current;
    let drag = null;
    const onDown = (e) => { cv.focus(); drag = { x: e.clientX, y: e.clientY, sx: e.clientX, sy: e.clientY, pan: e.button === 2 || e.shiftKey, moved: false }; cv.setPointerCapture(e.pointerId); };
    const onMove = (e) => {
      const rect = cv.getBoundingClientRect();
      live.current.mouse = { x: e.clientX - rect.left, y: e.clientY - rect.top };
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      if (Math.abs(e.clientX - drag.sx) + Math.abs(e.clientY - drag.sy) > 4) drag.moved = true;
      if (drag.pan) {
        const s = cam.dist * 0.0022, c = Math.cos(cam.yaw), sn = Math.sin(cam.yaw);
        cam.tx -= (dx * c + dy * sn) * s; cam.tz -= (-dx * sn + dy * c) * s;
        live.current.setFollow(false);
      } else {
        cam.yaw -= dx * 0.008; cam.pitch = Math.max(0.12, Math.min(1.5, cam.pitch + dy * 0.006));
      }
    };
    const pick = (mx, my) => {  // the robot under the cursor: ray against each robot's bounding sphere
      const rect = cv.getBoundingClientRect(), aspect = rect.width / rect.height;
      const ndcX = (mx / rect.width) * 2 - 1, ndcY = 1 - (my / rect.height) * 2;
      const eye = eyePos(), fwd = norm([cam.tx - eye[0], -eye[1], cam.tz - eye[2]]);
      const right = norm(cross(fwd, [0, 1, 0])), up = cross(right, fwd), t = Math.tan(FOV / 2);
      const dir = norm([fwd[0] + right[0] * ndcX * t * aspect + up[0] * ndcY * t, fwd[1] + right[1] * ndcX * t * aspect + up[1] * ndcY * t, fwd[2] + right[2] * ndcX * t * aspect + up[2] * ndcY * t]);
      let best = null, bd = 1e9;
      for (const [id, a] of Object.entries(anim)) {
        const c = [a.x, 0.55 * ROBOT, a.z], oc = [eye[0] - c[0], eye[1] - c[1], eye[2] - c[2]];
        const b = oc[0] * dir[0] + oc[1] * dir[1] + oc[2] * dir[2], q = oc[0] ** 2 + oc[1] ** 2 + oc[2] ** 2 - (0.6 * ROBOT) ** 2;
        const disc_ = b * b - q;
        if (disc_ >= 0) { const tt = -b - Math.sqrt(disc_); if (tt > 0 && tt < bd) { bd = tt; best = +id; } }
      }
      return best;
    };
    const onUp = (e) => {
      if (drag && !drag.moved && e.button === 0) {
        const rect = cv.getBoundingClientRect(), id = pick(e.clientX - rect.left, e.clientY - rect.top);
        if (id != null) { live.current.setSel(id); live.current.setFollow(true); }
      }
      drag = null;
    };
    const onWheel = (e) => { e.preventDefault(); cam.dist = Math.max(3, Math.min(n * 3 + 10, cam.dist * (1 + Math.sign(e.deltaY) * 0.1))); };
    const typing = (e) => /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);
    const onKey = (e) => {
      if (typing(e) || !(hover || document.activeElement === cv)) return;
      const k = e.key.toLowerCase();
      if ("wasdqe".includes(k) && k.length === 1) { keys.add(k); e.preventDefault(); if ("wasd".includes(k)) live.current.setFollow(false); }
      if (k === "r") { Object.assign(cam, home); live.current.setFollow(false); }
      if (k === "t") { cam.pitch = 1.5; }
      if (k === "escape") live.current.setFollow(false);
      if (k === "f") live.current.setFollow(true);
    };
    const onKeyUp = (e) => keys.delete(e.key.toLowerCase());
    const onEnter = () => { hover = true; }, onLeave = () => { hover = false; keys.clear(); live.current.mouse = null; };
    cv.addEventListener("pointerdown", onDown); cv.addEventListener("pointermove", onMove); cv.addEventListener("pointerup", onUp);
    const noMenu = (e) => e.preventDefault();
    cv.addEventListener("wheel", onWheel, { passive: false }); cv.addEventListener("contextmenu", noMenu);
    cv.addEventListener("pointerenter", onEnter); cv.addEventListener("pointerleave", onLeave);
    window.addEventListener("keydown", onKey); window.addEventListener("keyup", onKeyUp);

    const FOV = 0.85;
    const norm = (v) => { const l = Math.hypot(...v) || 1; return v.map((x) => x / l); };
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const eyePos = () => [cam.tx + Math.cos(cam.pitch) * Math.sin(cam.yaw) * cam.dist, Math.sin(cam.pitch) * cam.dist, cam.tz + Math.cos(cam.pitch) * Math.cos(cam.yaw) * cam.dist];

    const BG = [0.93, 0.94, 0.99];
    const tick = (now) => {
      raf = requestAnimationFrame(tick);
      const dt = Math.min(0.05, (now - last) / 1000); last = now;
      const S = live.current, E = S.ep, X = S.ix;
      if (!E || !E.start) return;
      const v = Math.min(S.view, E.turns.length);
      const state = v === 0 ? { ...E.start, agents: E.start.agents } : E.turns[v - 1].state;
      const turn = v > 0 ? E.turns[v - 1] : null;
      // a new view: start its animations (a step of one plays the turn; a jump glides)
      if (v !== shownView) {
        const step = v === shownView + 1;
        prevView = shownView < 0 ? v : shownView; shownView = v; turnStart = now;
        const idOf = Object.fromEntries(state.agents.map((a) => [a.name, a.id]));
        for (const a of Object.values(anim)) { a.bump = null; a.jolt = null; }
        for (const a of state.agents) {
          const cur = anim[a.id], tx = W(a.x), tz = Wz(a.y);
          if (!cur) { anim[a.id] = { x: tx, z: tz, yaw: 0, fromX: tx, fromZ: tz, toX: tx, toZ: tz, scoot: false, bump: null, jolt: null }; continue; }
          cur.fromX = cur.x; cur.fromZ = cur.z; cur.toX = tx; cur.toZ = tz; cur.scoot = step && (cur.fromX !== tx || cur.fromZ !== tz);
          if (tx !== cur.fromX || tz !== cur.fromZ) cur.yaw = Math.atan2(tx - cur.fromX, tz - cur.fromZ);
          if (cur.scoot) {  // dust kicked out behind at launch
            const dx = tx - cur.fromX, dz = tz - cur.fromZ, l = Math.hypot(dx, dz) || 1;
            burst(cur.fromX - (dx / l) * 0.3, 0.08, cur.fromZ - (dz / l) * 0.3, "dust", 7, [-(dx / l) * 1.2, -(dz / l) * 1.2]);
          }
        }
        if (step && turn) {  // a move into a robot: the mover lunges, bounces back and sees stars; the robot it hit is jolted
          for (const r of turn.records) {
            const b = bumpOf(r), A = anim[r.agent_id];
            if (!b || !A || A.scoot) continue;
            const dx = W(b.at[0]) - A.toX, dz = Wz(b.at[1]) - A.toZ, l = Math.hypot(dx, dz) || 1;
            A.bump = { dx: dx / l, dz: dz / l, fired: false }; A.yaw = Math.atan2(dx, dz);
            const H = anim[idOf[b.into]];
            if (H && (b.kind || "occupied") === "occupied") H.jolt = { dx: dx / l, dz: dz / l };  // contested / head-on: both lunge
          }
        }
        if (step && turn) {  // the turn's one-shot effects
          const prevObjs = new Set((v >= 2 ? E.turns[v - 2].state.objects : E.start.objects).filter((o) => o.what === "coin").map((o) => `${o.x},${o.y}`));
          const nowObjs = new Set(state.objects.filter((o) => o.what === "coin").map((o) => `${o.x},${o.y}`));
          for (const k of prevObjs) if (!nowObjs.has(k)) { const [cx, cy] = k.split(",").map(Number); burst(W(cx), 0.6, Wz(cy), "sparkle", 18); }
          for (const r of turn.records) {
            const res = (r.calls || []).map((c) => String(c[1] || "")).join(" ");
            const a = state.agents.find((q) => q.id === r.agent_id);
            if (!a) continue;
            if (/^\s*error\b/m.test(res)) burst(W(a.x), 0.9, Wz(a.y), "smoke", 10);
            if (/You tagged|is open|Correct!|finished|found it/i.test(res)) burst(W(a.x), 1.2, Wz(a.y), "confetti", 40);
          }
          if (state.score && state.score.verdict && !(v >= 2 && E.turns[v - 2].state.score && E.turns[v - 2].state.score.verdict)) {
            for (let i = 0; i < 6; i++) burst((Math.random() - 0.5) * n, 4 + Math.random() * 2, (Math.random() - 0.5) * n, "confetti", 50);
          }
        }
      }
      const tt = (now - turnStart) / 1000, k = ease(tt / 0.6);
      for (const a of Object.values(anim)) { const p = a.scoot ? scootPose(tt).p : k; a.x = lerp(a.fromX, a.toX, p); a.z = lerp(a.fromZ, a.toZ, p); }
      // camera: keys, follow
      const c = Math.cos(cam.yaw), sn = Math.sin(cam.yaw), sp = (cam.dist * 0.9 + 4) * dt;
      if (keys.has("w")) { cam.tx -= sn * sp; cam.tz -= c * sp; }
      if (keys.has("s")) { cam.tx += sn * sp; cam.tz += c * sp; }
      if (keys.has("a")) { cam.tx -= c * sp; cam.tz += sn * sp; }
      if (keys.has("d")) { cam.tx += c * sp; cam.tz -= sn * sp; }
      if (keys.has("q")) cam.yaw -= 1.6 * dt;
      if (keys.has("e")) cam.yaw += 1.6 * dt;
      if (S.follow && S.sel != null && anim[S.sel]) { const a = anim[S.sel]; cam.tx = lerp(cam.tx, a.x, 1 - Math.exp(-6 * dt)); cam.tz = lerp(cam.tz, a.z, 1 - Math.exp(-6 * dt)); }
      cam.tx = Math.max(-n, Math.min(n, cam.tx)); cam.tz = Math.max(-n, Math.min(n, cam.tz));
      // matrices
      const rect = cv.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1);
      const w = Math.max(1, Math.round(rect.width * dpr)), hh = Math.max(1, Math.round(rect.height * dpr));
      if (cv.width !== w || cv.height !== hh) { cv.width = w; cv.height = hh; }
      R.gl.viewport(0, 0, w, hh);
      const eye = eyePos(), V = M4.look(eye, [cam.tx, 0, cam.tz], [0, 1, 0]), P = M4.persp(FOV, rect.width / Math.max(1, rect.height), 0.1, 200), VP = M4.mul(P, V);
      R.frame(VP, eye, cam.dist * 2.2 + n, BG);
      const T = now / 1000;
      // floor, and a far ground disc so the board floats on a pastel plane
      R.draw(R.mesh.box, M4.trs(0, -0.42, 0, 0, 0, 0, n + 0.7, 0.62, n + 0.7), [0.72, 0.66, 0.95], {});  // the board's plinth (no ink outline: the hull is inflated before scaling, a slab at this size)
      R.draw(R.mesh.box, M4.trs(0, -0.9, 0, 0, 0, 0, n + 0.2, 0.3, n + 0.2), [0.58, 0.52, 0.86], {});
      R.draw(groundMesh, M4.ident(), [1, 1, 1], { vc: true });
      // objects
      const opened = doorsOpened(state.events || []);
      for (const o of state.objects) {
        const ox = W(o.x), oz = Wz(o.y);
        if (o.what === "coin") {
          const bob = 0.45 + Math.sin(T * 3 + o.x + o.y) * 0.08;
          R.draw(R.mesh.disc, M4.trs(ox, 0.012, oz, 0, 0, 0, 0.45), [0.1, 0.1, 0.15], { alpha: 0.18 });
          R.draw(R.mesh.cyl, M4.trs(ox, bob, oz, T * 2.2 + o.x, Math.PI / 2, 0, 0.42, 0.09, 0.42), [0.99, 0.79, 0.18], { outline: 0.06, emit: 0.15 });
          continue;
        }
        const num = +((/(\d+)/.exec(o.what) || [])[1]), open = /door/.test(o.what) && opened.has(num);
        const col = open ? [0.45, 0.85, 0.5] : [0.98, 0.72, 0.3];
        R.draw(R.mesh.box, M4.trs(ox - 0.38, 0.45, oz, 0, 0, 0, 0.12, 0.9, 0.12), col, { outline: 0.05 });
        R.draw(R.mesh.box, M4.trs(ox + 0.38, 0.45, oz, 0, 0, 0, 0.12, 0.9, 0.12), col, { outline: 0.05 });
        R.draw(R.mesh.box, M4.trs(ox, 0.94, oz, 0, 0, 0, 0.9, 0.14, 0.16), col, { outline: 0.05 });
        const swing = open ? Math.min(1, (now - turnStart) / 600) * 1.6 : 0;
        R.draw(R.mesh.box, M4.mul(M4.trs(ox - 0.32, 0.43, oz, -swing), M4.trs(0.32, 0, 0, 0, 0, 0, 0.62, 0.82, 0.06)), open ? [0.62, 0.92, 0.66] : [0.86, 0.52, 0.22], { outline: 0.04 });
        const tag = el(`obj:${o.what}`, "a3-objtag", `${open ? "🔓" : "🚪"} ${o.what}`);
        place(tag, [ox, 1.25, oz], VP, rect);
      }
      // triangulate target / dig marks
      const sc = state.score || {};
      if (state.task === "triangulate" && Array.isArray(sc.target) && (sc.success || S.showTarget)) {
        const tx = W(sc.target[0]), tz = Wz(sc.target[1]);
        R.draw(R.mesh.box, M4.trs(tx, 0.02, tz, Math.PI / 4, 0, 0, 0.7, 0.03, 0.12), [0.9, 0.2, 0.25], { flat: true });
        R.draw(R.mesh.box, M4.trs(tx, 0.02, tz, -Math.PI / 4, 0, 0, 0.7, 0.03, 0.12), [0.9, 0.2, 0.25], { flat: true });
        if (sc.success) R.draw(R.mesh.sphere, M4.trs(tx, 0.5 + Math.sin(T * 4) * 0.1, tz, T, 0, 0, 0.35), [0.5, 0.85, 1.0], { outline: 0.05, emit: 0.3 });
      }
      if (sc.dug_at && !(Array.isArray(sc.target) && sc.dug_at[0] === sc.target[0] && sc.dug_at[1] === sc.target[1])) {
        R.draw(R.mesh.disc, M4.trs(W(sc.dug_at[0]), 0.015, Wz(sc.dug_at[1]), 0, 0, 0, 0.6), [0.35, 0.25, 0.2], { alpha: 0.8 });
      }
      // trails: footprints along each robot's path so far
      if (S.trails !== "none" && X && X.pos) {
        const ids = S.trails === "sel" ? (S.sel != null ? [S.sel] : []) : state.agents.map((a) => a.id);
        for (const id of ids) {
          const col = hsl(robotColor(id));
          for (let t = Math.max(0, v - 10); t < v; t++) {
            const p = X.pos[t] && X.pos[t][id];
            if (!p) continue;
            R.draw(R.mesh.disc, M4.trs(W(p[0]), 0.013, Wz(p[1]), 0, 0, 0, 0.18), col, { alpha: 0.2 + 0.5 * ((t - v + 10) / 10), flat: true });
          }
        }
      }
      // robots
      const hearers = {};
      const msgs = turn && S.showMsgs !== "none" ? turn.messages || [] : [];
      const said = new Set(msgs.map((m) => m.speaker_id));
      for (const a of state.agents) {
        const A = anim[a.id] || { x: W(a.x), z: Wz(a.y), yaw: 0 };
        const role = roleOf(a.tools, a.team), body = hex(role.color), self = hsl(robotColor(a.id));
        const rec = turn ? turn.records.find((r) => r.agent_id === a.id) : null;
        const em = EMOTE(rec, said.has(a.id));
        const pose = A.scoot ? scootPose(tt) : { lean: 0, squash: 1 };
        // a bump: the mover's lunge and bounce, a squash on impact, a dizzy sway while the stars circle; the robot it hit rocks
        let ox = 0, oz = 0, squash = pose.squash, sway = 0;
        if (A.bump) {
          const d = bumpOffset(tt);
          ox += A.bump.dx * d; oz += A.bump.dz * d;
          squash *= 1 - 0.2 * Math.exp(-(((tt - BUMP_HIT) / 0.05) ** 2));
          sway = tt > BUMP_HIT ? 0.13 * Math.sin(T * 8) * Math.max(0, 1 - (tt - BUMP_HIT) / 2.4) : 0;
          if (!A.bump.fired && tt >= BUMP_HIT) { A.bump.fired = true; burst(A.toX + A.bump.dx * 0.5, 0.75, A.toZ + A.bump.dz * 0.5, "sparkle", 12); }
        }
        if (A.jolt && tt > BUMP_HIT) { const s = tt - BUMP_HIT, e = 0.12 * Math.exp(-6 * s) * Math.sin(20 * s); ox += A.jolt.dx * e; oz += A.jolt.dz * e; }
        const PX = A.x + ox, PZ = A.z + oz;
        const bob = Math.sin(T * 3.1 + a.id * 1.7) * 0.04;
        const shake = em && em.e === "💥" && tt < 0.8 ? Math.sin(T * 60) * 0.06 * (1 - tt / 0.8) : 0;
        const down = a.tagged ? Math.min(1, tt / 0.5) * (Math.PI / 2) : 0;  // a tagged runner keels over
        const isSel = a.id === S.sel;
        const base = M4.mul(M4.mul(M4.trs(PX + shake, bob, PZ, A.yaw), M4.trs(0, 0, 0, 0, down + pose.lean, sway)),
          M4.trs(0, 0, 0, 0, 0, 0, ROBOT * (2 - squash) ** 0.5, ROBOT * squash, ROBOT * (2 - squash) ** 0.5));
        // shadow
        R.draw(R.mesh.disc, M4.trs(PX, 0.014, PZ, 0, 0, 0, 0.7 * ROBOT), [0.1, 0.1, 0.18], { alpha: 0.22 });
        if (isSel) R.draw(R.mesh.disc, M4.trs(PX, 0.016, PZ, 0, 0, 0, 1.0 + Math.sin(T * 5) * 0.08), [1, 0.85, 0.2], { alpha: 0.45, flat: true });
        // wheels, body, head
        for (const sx of [-0.24, 0.24]) R.draw(R.mesh.cyl, M4.mul(base, M4.trs(sx, 0.11, 0, 0, 0, Math.PI / 2, 0.2, 0.1, 0.2)), [0.25, 0.25, 0.32], { outline: 0.05 });
        R.draw(R.mesh.cyl, M4.mul(base, M4.trs(0, 0.42, 0, 0, 0, 0, 0.5, 0.42, 0.5)), a.tagged ? [0.7, 0.7, 0.72] : body, { outline: 0.05 });
        R.draw(R.mesh.cyl, M4.mul(base, M4.trs(0, 0.42, 0.001, 0, 0, 0, 0.51, 0.08, 0.51)), self, { outline: 0 });  // the robot's own colour band
        R.draw(R.mesh.sphere, M4.mul(base, M4.trs(0, 0.66, 0, 0, 0, 0, 0.5)), a.tagged ? [0.78, 0.78, 0.8] : [0.92, 0.93, 0.97], { outline: 0.05 });
        // googly eyes: whites on the front of the dome, pupils jiggle (and look around on a look() turn)
        const look = em && em.e === "👀" ? Math.sin(T * 6) * 0.04 : 0;
        for (const ex of [-0.11, 0.11]) {
          R.draw(R.mesh.sphere, M4.mul(base, M4.trs(ex, 0.72, 0.2, 0, 0, 0, 0.17)), [1, 1, 1], { outline: 0.06 });
          const jig = Math.sin(T * 9 + a.id + ex * 20) * 0.012;
          R.draw(R.mesh.sphere, M4.mul(base, M4.trs(ex + look + jig, a.tagged ? 0.72 : 0.71 + jig, 0.28, 0, 0, 0, a.tagged ? 0.03 : 0.08)), [0.05, 0.05, 0.08], {});
        }
        // antenna with a bulb: glows when speaking, red after an error, green on a success
        const wob = Math.sin(T * 7 + a.id) * 0.18 + (A.scoot && tt < 1.2 ? Math.sin(tt * 20) * 0.4 * (1 - tt / 1.2) : 0) + (A.bump && tt > BUMP_HIT && tt < 1.2 ? Math.sin(tt * 26) * 0.5 * (1 - tt / 1.2) : 0);
        R.draw(R.mesh.cyl, M4.mul(base, M4.trs(0, 0.98, 0, 0, 0, wob, 0.03, 0.28, 0.03)), [0.3, 0.3, 0.36], {});
        const bulb = em && em.e === "💥" ? [1, 0.3, 0.3] : em && em.big ? [0.4, 1, 0.5] : said.has(a.id) ? [1, 0.9, 0.3] : [0.75, 0.75, 0.82];
        R.draw(R.mesh.sphere, M4.mul(base, M4.trs(-Math.sin(wob) * 0.14, 1.13, 0, 0, 0, 0, 0.11)), bulb, { outline: 0.04, emit: said.has(a.id) || (em && em.big) ? 0.6 + Math.sin(T * 12) * 0.3 : 0 });
        // little arms: wave on a speaking / goal turn
        const wave = (said.has(a.id) || (em && em.big)) && tt < 1.6 ? Math.sin(T * 14) * 0.6 : 0.2;
        for (const s of [-1, 1]) R.draw(R.mesh.box, M4.mul(base, M4.trs(s * 0.3, 0.48, 0, 0, 0, s * (0.6 + wave), 0.06, 0.3, 0.06)), [0.3, 0.3, 0.36], { outline: 0.03 });
        // overlay: name tag, emote
        const tagEl = el(`tag:${a.id}`, "a3-tag" + (isSel ? " sel" : "") + (a.tagged ? " out" : ""), `${a.name}${a.player === "B" ? " ·B" : ""}${a.coins ? ` 🪙${a.coins}` : ""}`);
        tagEl.className = "a3-tag" + (isSel ? " sel" : "") + (a.tagged ? " out" : "");
        tagEl.style.setProperty("--c", role.color);
        place(tagEl, [PX, 1.45 * ROBOT, PZ], VP, rect);
        if (em && !em.bump && tt < 2.2) {
          const e = el(`emote:${a.id}`, "a3-emote", em.e);
          e.title = em.why;
          place(e, [PX + 0.35, 1.65 * ROBOT + tt * 0.25, PZ], VP, rect, 1 - Math.max(0, tt - 1.6) / 0.6);
        }
        if (A.bump && tt > BUMP_HIT && tt < 2.8) {  // stars circling the head
          const fade = Math.min(1, (tt - BUMP_HIT) / 0.15) * Math.min(1, (2.8 - tt) / 0.5);
          for (let i = 0; i < 4; i++) {
            const ang = T * 5.5 + (i * Math.PI) / 2;
            const st = el(`star:${a.id}:${i}`, "a3-star", i % 2 ? "✨" : "⭐");
            st.title = em ? em.why : "bump!";
            place(st, [PX + Math.cos(ang) * 0.45, 1.32 * ROBOT + Math.sin(ang * 2) * 0.06, PZ + Math.sin(ang) * 0.45], VP, rect, fade);
          }
        }
        if (a.tagged) { const z = el(`zz:${a.id}`, "a3-emote small", "x_x"); place(z, [PX, 0.9, PZ], VP, rect); }
        hearers[a.id] = A;
      }
      // comm range: the square of cells around the selected robot (or every robot) that hears it, clipped to the board
      // (hearing counts diagonals since 2026-10-11: range 2 = the 5x5 cells with the robot in the middle)
      const ranges = S.showRange ? state.agents.map((a) => a.id) : S.sel != null ? [S.sel] : [];
      const cr = (E.meta && E.meta.comm_range) || 2, lo = W(0) - 0.5, hi = W(n - 1) + 0.5, loz = Wz(0) - 0.5, hiz = Wz(n - 1) + 0.5;
      for (const id of ranges) {
        const A = anim[id];
        if (!A) continue;
        const col = hsl(robotColor(id));
        const x0 = Math.max(lo, A.x - cr - 0.5), x1 = Math.min(hi, A.x + cr + 0.5), z0 = Math.max(Math.min(loz, hiz), A.z - cr - 0.5), z1 = Math.min(Math.max(loz, hiz), A.z + cr + 0.5);
        R.draw(R.mesh.box, M4.trs((x0 + x1) / 2, 0.016, (z0 + z1) / 2, 0, 0, 0, x1 - x0, 0.01, z1 - z0), col, { flat: true, alpha: 0.1 });
        for (const [cx, cz, sx, sz] of [[(x0 + x1) / 2, z0, x1 - x0, 0.05], [(x0 + x1) / 2, z1, x1 - x0, 0.05], [x0, (z0 + z1) / 2, 0.05, z1 - z0], [x1, (z0 + z1) / 2, 0.05, z1 - z0]]) {
          R.draw(R.mesh.box, M4.trs(cx, 0.02, cz, 0, 0, 0, sx, 0.02, sz), col, { flat: true, alpha: 0.45 });
        }
      }
      // messages: a speech bubble over the speaker, an envelope flying to every hearer
      msgs.forEach((m, i) => {
        const sA = hearers[m.speaker_id];
        if (!sA) return;
        const bubble = el(`say:${i}`, "a3-bubble", S.showMsgs === "all" ? `“${clip(m.text, 40)}”` : "💬");
        place(bubble, [sA.x, 2.05 * ROBOT + (i % 3) * 0.32, sA.z], VP, rect, tt < 4 ? 1 : 0.85);
        (m.heard_by_ids || []).forEach((hid, j) => {
          const tA = hearers[hid];
          if (!tA) return;
          const f = Math.max(0, Math.min(1, (tt - 0.25 - j * 0.07) / 0.9));
          if (f <= 0 || f >= 1) return;
          const x = lerp(sA.x, tA.x, f), z = lerp(sA.z, tA.z, f), y = 1.2 + Math.sin(f * Math.PI) * 1.4;
          place(el(`mail:${i}:${hid}`, "a3-mail", "✉️"), [x, y, z], VP, rect);
        });
      });
      // particles
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.life += dt;
        if (p.life > p.max) { particles.splice(i, 1); continue; }
        p.vy -= (p.kind === "smoke" ? -0.3 : 6) * dt; p.x += p.vx * dt; p.y += p.vy * dt; p.z += p.vz * dt;
        if (p.y < 0.03 && p.kind !== "smoke") { p.y = 0.03; p.vy *= -0.3; p.vx *= 0.7; p.vz *= 0.7; }
        const fade = 1 - p.life / p.max;
        R.draw(p.kind === "smoke" ? R.mesh.sphere : R.mesh.box, M4.trs(p.x, p.y, p.z, p.spin * p.life, p.spin * p.life * 0.7, 0, p.s * (p.kind === "smoke" ? 1 + p.life * 2 : 1)),
          p.color, { flat: p.kind !== "smoke", alpha: p.kind === "smoke" ? 0.5 * fade : Math.min(1, fade * 2) });
      }
      // the verdict banner
      if (sc.verdict) el("banner", "a3-banner", `🏆 ${sc.verdict}`).style.opacity = "1";
      // hover label
      if (S.mouse && !drag) {
        const id = pick(S.mouse.x, S.mouse.y);
        cv.style.cursor = id != null ? "pointer" : "grab";
        if (id != null && S.titleOf) { const tip = el("hover", "a3-hover", S.titleOf(id)); tip.style.left = `${S.mouse.x + 14}px`; tip.style.top = `${S.mouse.y + 10}px`; tip.style.opacity = "1"; }
      }
      // drop overlay elements nobody drew this frame
      for (const [key, d] of Object.entries(els)) { if (!d.__seen) { d.remove(); delete els[key]; } else d.__seen = false; }
    };
    const place = (d, p, VP, rect, op = 1) => {
      const x = VP[0] * p[0] + VP[4] * p[1] + VP[8] * p[2] + VP[12], y = VP[1] * p[0] + VP[5] * p[1] + VP[9] * p[2] + VP[13];
      const w = VP[3] * p[0] + VP[7] * p[1] + VP[11] * p[2] + VP[15];
      if (w <= 0.01) { d.style.opacity = "0"; return; }
      d.style.transform = `translate(-50%, -100%) translate(${((x / w + 1) / 2) * rect.width}px, ${((1 - y / w) / 2) * rect.height}px)`;
      d.style.opacity = String(op);
      d.style.zIndex = String(1000 - Math.round(w * 10));
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("keydown", onKey); window.removeEventListener("keyup", onKeyUp);
      // the canvas outlives this effect (another episode reuses it): its handlers must go with this camera
      cv.removeEventListener("pointerdown", onDown); cv.removeEventListener("pointermove", onMove); cv.removeEventListener("pointerup", onUp);
      cv.removeEventListener("wheel", onWheel); cv.removeEventListener("contextmenu", noMenu);
      cv.removeEventListener("pointerenter", onEnter); cv.removeEventListener("pointerleave", onLeave);
      for (const d of Object.values(els)) d.remove();
      for (const b of [groundMesh.p, groundMesh.n, groundMesh.c, groundMesh.i]) if (b) R.gl.deleteBuffer(b);  // the episode's floor
    };
  }, [ep && ep.start && ep.start.n, ep && ep.meta && ep.meta.seed, ep && ep.name, ep && ep.t0]);

  if (err) return html`<div class="a3-err">The 3D view needs WebGL (${err}); use the map view.</div>`;
  const selName = sel != null && ep && ep.robots[sel] ? ep.robots[sel].name : null;
  return html`<div class="a3" ref=${wrap}>
    <canvas ref=${canvas} class="a3-canvas" tabIndex="0" aria-label="3D arena: drag to orbit, WASD to move, click a robot to follow it"></canvas>
    <div class="a3-layer" ref=${layer}></div>
    <div class="a3-hud">
      <span class="a3-help">🖱️ drag orbit · right-drag / ⇧ pan · wheel zoom · <b>WASD</b> move · <b>Q/E</b> turn · <b>R</b> reset · <b>T</b> top-down · click a robot to follow</span>
      ${selName && html`<button class=${"a3-btn" + (follow ? " on" : "")} onClick=${() => setFollow(!follow)} title="the camera follows the selected robot (F / Esc)">${follow ? "📍 following" : "📍 follow"} ${selName}</button>`}
    </div>
  </div>`;
}

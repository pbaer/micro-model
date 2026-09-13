export const fmtTok = (n) => n == null ? "-" : n >= 1e9 ? (n / 1e9).toFixed(2) + "B" : n >= 1e6 ? (n / 1e6).toFixed(0) + "M" : n >= 1e3 ? (n / 1e3).toFixed(0) + "K" : String(n);
export const fmtDur = (s) => {
  if (s == null || !isFinite(s)) return "-";
  s = Math.max(0, Math.floor(s));
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (d) return `${d}d ${h}h ${String(m).padStart(2, "0")}m`;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  return `${m}m ${String(sec).padStart(2, "0")}s`;
};
export const fmtNum = (x, d = 4) => x == null || !isFinite(x) ? "-" : Number(x).toFixed(d);
export const fmtInt = (x) => x == null ? "-" : Math.round(x).toLocaleString();
export const fmtSci = (x) => x == null ? "-" : Number(x).toExponential(2);
export const fmtBytes = (b) => b == null ? "-" : b >= 2 ** 30 ? (b / 2 ** 30).toFixed(2) + " GiB" : (b / 2 ** 20).toFixed(0) + " MiB";
export const fmtTime = (t) => t == null ? "-" : new Date(t * 1000).toLocaleString([], { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
export const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
};

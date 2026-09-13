"""Self-contained HTML progress report for a run, regenerated from metrics.jsonl.

No external assets: charts are drawn as inline SVG by a small embedded script. The page
auto-refreshes every 60 s so it can be left open in a browser tab while a run is in progress.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any

from slm.utils.logging import MetricsLogger, fmt_duration, fmt_tokens

_JS = r"""
function chart(id, series, opts){
  opts = opts||{};
  const W=560,H=260,L=60,R=16,T=16,B=36;
  const svg=document.getElementById(id); if(!svg) return;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  let xs=[],ys=[];
  series.forEach(s=>s.pts.forEach(p=>{xs.push(p[0]);ys.push(p[1]);}));
  if(!xs.length){svg.innerHTML='<text x="20" y="40" fill="#888">no data yet</text>';return;}
  let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  if(opts.ymin!==undefined) y0=opts.ymin;
  if(x1===x0) x1=x0+1; if(y1===y0) y1=y0+1;
  const pad=(y1-y0)*0.05; y0-=pad; y1+=pad;
  const sx=x=>L+(x-x0)/(x1-x0)*(W-L-R), sy=y=>T+(1-(y-y0)/(y1-y0))*(H-T-B);
  let g='';
  for(let i=0;i<=4;i++){const y=y0+(y1-y0)*i/4, yy=sy(y);
    g+=`<line x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}" stroke="#e5e5e5"/><text x="${L-6}" y="${yy+4}" text-anchor="end" font-size="10" fill="#666">${opts.ylog?y.toFixed(3):y.toPrecision(4)}</text>`;}
  for(let i=0;i<=4;i++){const x=x0+(x1-x0)*i/4, xx=sx(x);
    g+=`<text x="${xx}" y="${H-B+14}" text-anchor="middle" font-size="10" fill="#666">${opts.xfmt?opts.xfmt(x):x.toPrecision(3)}</text>`;}
  const colors=['#2563eb','#dc2626','#16a34a','#9333ea','#ea580c'];
  series.forEach((s,i)=>{
    const d=s.pts.map((p,j)=>(j?'L':'M')+sx(p[0]).toFixed(1)+' '+sy(p[1]).toFixed(1)).join(' ');
    g+=`<path d="${d}" fill="none" stroke="${colors[i%colors.length]}" stroke-width="${s.w||1.5}"/>`;
    if(s.dots) s.pts.forEach(p=>{g+=`<circle cx="${sx(p[0])}" cy="${sy(p[1])}" r="2.5" fill="${colors[i%colors.length]}"/>`;});
    g+=`<text x="${W-R}" y="${T+12+i*13}" text-anchor="end" font-size="11" fill="${colors[i%colors.length]}">${s.name}</text>`;
  });
  g+=`<text x="${(L+W-R)/2}" y="${H-4}" text-anchor="middle" font-size="11" fill="#444">${opts.xlabel||''}</text>`;
  svg.innerHTML=g;
}
const fmtTok=x=>x>=1e9?(x/1e9).toFixed(2)+'B':x>=1e6?(x/1e6).toFixed(0)+'M':(x/1e3).toFixed(0)+'K';
"""

_CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;padding:20px 28px;color:#222;background:#fafafa;max-width:1280px}
h1{margin:0 0 4px 0;font-size:22px} h2{font-size:16px;margin:26px 0 8px;border-bottom:1px solid #ddd;padding-bottom:4px}
.sub{color:#666;font-size:13px;margin-bottom:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin:12px 0}
.tile{background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:10px 12px}
.tile .k{font-size:11px;color:#777;text-transform:uppercase;letter-spacing:.04em}.tile .v{font-size:20px;font-weight:600;margin-top:2px}
.tile .s{font-size:11px;color:#888}
.bar{height:10px;background:#e5e7eb;border-radius:5px;overflow:hidden;margin:6px 0 2px}.bar>div{height:100%;background:#2563eb}
.charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:14px}
.chart{background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:8px}.chart h3{margin:0 0 4px 6px;font-size:13px;font-weight:600;color:#444}
svg{width:100%;height:auto;display:block}
table{border-collapse:collapse;font-size:12.5px;background:#fff;width:100%}th,td{border:1px solid #e3e3e3;padding:4px 8px;text-align:right}th{background:#f3f4f6}td:first-child,th:first-child{text-align:left}
pre{background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:10px;font-size:12px;white-space:pre-wrap;word-break:break-word;max-height:520px;overflow:auto}
.status{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:600}
.running{background:#dbeafe;color:#1d4ed8}.finished{background:#dcfce7;color:#15803d}.stopped{background:#fee2e2;color:#b91c1c}.stale{background:#fef3c7;color:#b45309}
"""


def _tile(k: str, v: str, s: str = "") -> str:
    return f'<div class="tile"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div><div class="s">{html.escape(s)}</div></div>'


def build_report(run_dir: Path, status: str | None = None) -> str:
    run_dir = Path(run_dir)
    recs = MetricsLogger.read(run_dir / "metrics.jsonl")
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8")) if (run_dir / "run.json").exists() else {}
    cfg = meta.get("config", {})
    train = [r for r in recs if r["kind"] == "train"]
    evals = [r for r in recs if r["kind"] == "eval"]
    miles = [r for r in recs if r["kind"] == "milestone"]
    events = [r for r in recs if r["kind"] in ("start", "resume", "stop", "finish", "checkpoint")]
    last = train[-1] if train else {}
    total_tokens = cfg.get("schedule", {}).get("total_tokens", 0)
    tokens = max((r.get("tokens") or 0 for r in recs), default=0)  # finish/milestone records carry the exact total
    now = time.time()
    last_time = recs[-1]["time"] if recs else now
    if status is None:
        if any(r["kind"] == "finish" for r in recs):
            status = "finished"
        elif recs and recs[-1]["kind"] == "stop":
            status = "stopped"
        elif now - last_time > 900:
            status = "stale"
        else:
            status = "running"
    # wall-clock: sum of active segments (start/resume -> last record before next start/resume)
    elapsed = _active_seconds(recs)
    tps_recent = last.get("tok_s_ema", last.get("tok_s", 0.0)) or 0.0
    tps_avg = tokens / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total_tokens - tokens)
    eta = remaining / tps_recent if tps_recent > 0 else float("nan")
    initial_est = meta.get("initial_estimate_s")
    best_val = min((e["val_loss"] for e in evals), default=float("nan"))

    tiles = [
        _tile("progress", f"{tokens / total_tokens * 100:.1f}%" if total_tokens else "-", f"{fmt_tokens(tokens)} / {fmt_tokens(total_tokens)} tokens"),
        _tile("ETA", fmt_duration(eta), f"finish ~{time.strftime('%a %H:%M', time.localtime(now + eta))}" if eta == eta else ""),
        _tile("elapsed", fmt_duration(elapsed), f"initial estimate {fmt_duration(initial_est)}" if initial_est else ""),
        _tile("tokens/sec", f"{tps_recent:,.0f}", f"run avg {tps_avg:,.0f}"),
        _tile("train loss", f"{last.get('loss', float('nan')):.4f}", f"update {last.get('update', 0)}"),
        _tile("val loss", f"{evals[-1]['val_loss']:.4f}" if evals else "-", f"best {best_val:.4f}  ppl {evals[-1]['val_ppl']:.1f}" if evals else ""),
        _tile("lr", f"{last.get('lr', 0):.2e}", ""),
        _tile("grad norm", f"{last.get('grad_norm', 0):.3f}", ""),
        _tile("VRAM peak", f"{last.get('vram_gib', 0):.1f} GiB", ""),
        _tile("step time", f"{last.get('step_ms', 0):.0f} ms", f"fwd {last.get('fwd_ms', 0):.0f} bwd {last.get('bwd_ms', 0):.0f} opt {last.get('opt_ms', 0):.0f} data {last.get('data_ms', 0):.0f}"),
    ]
    prog = tokens / total_tokens * 100 if total_tokens else 0

    data = {
        "loss": [[r["tokens"], r["loss"]] for r in train],
        "val": [[r["tokens"], r["val_loss"]] for r in evals],
        "lr": [[r["tokens"], r["lr"]] for r in train],
        "tps": [[r["tokens"], r["tok_s"]] for r in train],
        "gn": [[r["tokens"], r["grad_norm"]] for r in train],
        "vram": [[r["tokens"], r["vram_gib"]] for r in train],
    }
    # thin the train series for the page
    for k in ("loss", "lr", "tps", "gn", "vram"):
        pts = data[k]
        if len(pts) > 1500:
            step = len(pts) // 1500 + 1
            data[k] = pts[::step]

    mile_rows = "".join(
        f"<tr><td>{fmt_tokens(m['tokens'])}</td><td>{fmt_duration(m['segment_s'])}</td><td>{fmt_duration(m['elapsed_s'])}</td>"
        f"<td>{m['tok_s']:,.0f}</td><td>{m['loss']:.4f}</td><td>{m.get('val_loss', float('nan')):.4f}</td><td>{time.strftime('%m-%d %H:%M', time.localtime(m['time']))}</td></tr>"
        for m in miles
    )
    ev_rows = "".join(
        f"<tr><td>{time.strftime('%m-%d %H:%M:%S', time.localtime(e['time']))}</td><td>{e['kind']}</td><td style='text-align:left'>{html.escape(str(e.get('msg', '')))}</td></tr>"
        for e in events[-30:]
    )
    samples_path = run_dir / "samples" / "latest.txt"
    samples = samples_path.read_text(encoding="utf-8") if samples_path.exists() else "(no samples yet)"

    return f"""<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="60">
<title>{html.escape(meta.get('run_name', run_dir.name))}</title><style>{_CSS}</style></head><body>
<h1>{html.escape(meta.get('run_name', run_dir.name))} <span class="status {status}">{status}</span></h1>
<div class="sub">generated {time.strftime('%Y-%m-%d %H:%M:%S')} · started {meta.get('started', '?')} · {html.escape(str(meta.get('env', {}).get('gpu', '')))} · git {str(meta.get('env', {}).get('git_commit', ''))[:8]} · model {meta.get('n_params', 0):,} params</div>
<div class="bar"><div style="width:{prog:.2f}%"></div></div>
<div class="tiles">{''.join(tiles)}</div>
<div class="charts">
<div class="chart"><h3>train loss</h3><svg id="c_loss"></svg></div>
<div class="chart"><h3>validation loss</h3><svg id="c_val"></svg></div>
<div class="chart"><h3>tokens / sec</h3><svg id="c_tps"></svg></div>
<div class="chart"><h3>learning rate</h3><svg id="c_lr"></svg></div>
<div class="chart"><h3>gradient norm</h3><svg id="c_gn"></svg></div>
<div class="chart"><h3>VRAM peak (GiB)</h3><svg id="c_vram"></svg></div>
</div>
<h2>Milestones (every {fmt_tokens(cfg.get('milestone_tokens', 0))} tokens)</h2>
<table><tr><th>tokens</th><th>segment time</th><th>elapsed</th><th>tok/s (segment)</th><th>train loss</th><th>val loss</th><th>at</th></tr>{mile_rows or '<tr><td colspan=7>none yet</td></tr>'}</table>
<h2>Latest samples</h2><pre>{html.escape(samples)}</pre>
<h2>Events</h2><table><tr><th>time</th><th>kind</th><th>message</th></tr>{ev_rows}</table>
<h2>Config</h2><pre>{html.escape(json.dumps(cfg, indent=1))}</pre>
<script>{_JS}
const D={json.dumps(data)};
chart('c_loss',[{{name:'train',pts:D.loss}},{{name:'val',pts:D.val,dots:true,w:2}}],{{xfmt:fmtTok,xlabel:'tokens'}});
chart('c_val',[{{name:'val',pts:D.val,dots:true,w:2}}],{{xfmt:fmtTok,xlabel:'tokens'}});
chart('c_tps',[{{name:'tok/s',pts:D.tps}}],{{xfmt:fmtTok,xlabel:'tokens',ymin:0}});
chart('c_lr',[{{name:'lr',pts:D.lr}}],{{xfmt:fmtTok,xlabel:'tokens',ymin:0}});
chart('c_gn',[{{name:'grad norm',pts:D.gn}}],{{xfmt:fmtTok,xlabel:'tokens',ymin:0}});
chart('c_vram',[{{name:'GiB',pts:D.vram}}],{{xfmt:fmtTok,xlabel:'tokens',ymin:0}});
</script></body></html>"""


def _active_seconds(recs: list[dict]) -> float:
    """Sum wall time between each start/resume and the last record before the next one."""
    total = 0.0
    seg_start = None
    prev_t = None
    for r in recs:
        if r["kind"] in ("start", "resume"):
            if seg_start is not None and prev_t is not None:
                total += prev_t - seg_start
            seg_start = r["time"]
        prev_t = r["time"]
    if seg_start is not None and prev_t is not None:
        total += prev_t - seg_start
    return total


def write_report(run_dir: Path, status: str | None = None) -> Path:
    out = Path(run_dir) / "report.html"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(build_report(run_dir, status), encoding="utf-8")
    tmp.replace(out)
    return out


if __name__ == "__main__":
    import sys

    print(write_report(Path(sys.argv[1])))

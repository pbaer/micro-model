"""The Eval tab: every benchmark and homebrew eval we run, one row per evaluated checkpoint (torch-free).

Result files live beside the run they measure (`runs/<run>/lm_eval_limit2000.json`, `facts_s200.json`, ...). A file is
attributed to the run whose directory holds it and to the checkpoint named by its `checkpoint` field (file name only:
runs were renamed after some evals, so the directory part of that path is not trusted, e.g. `m4_sft_rehearsal_149m`'s
files still say `m4_sft_149m`). Checkpoints of one run with the same token count are the same weights (`best.pt` =
`step_00150.pt`), so rows are keyed by (run, tokens) and the judged-quality summary, which names snapshots, lands on
the row the other files made. When two files fill the same cell the higher-priority one wins (see `_PRIO`) and the
other is listed in the cell's detail.

Returned shape: `{columns: [{key, label, group, super, higher_is_better, fmt, min, max}], rows: [{run, checkpoint,
aliases, stage, params, own_tokens, tokens, cells: {key: {value, source, detail, t}}}]}`. `t` in [0, 1] is the colour
position inside its column (1 = best, 0 = worst, 0.5 when the column has a single distinct value).

Every file that measured a cell (not only the winner) is kept per row for the detail pages (`cell_refs`, `detail`,
services/eval_detail.py).

External comparison models (slm.eval.external) live in `runs/ext_<name>/` beside a `model.json` (the registry entry):
the same result file names, `"checkpoint": "external:<name>"`. Their rows carry `group: "external models"`, stage
`external`, the registry's params, the published training tokens and a `model` block (hf id, params, license, chat or
base) for the hover card, and sort after ours; the colour scale spans all rows. Our rows carry `group: "ours"`.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from slm.utils.stage import run_stage, run_tools

# ------------------------------------------------------------------------------------------------ columns
# fmt: pct1 = x100 with one decimal (the lm-eval tables in docs/results.md), pct = x100 with one decimal and a % sign
# (facts probe), frac = 3 significant digits, score = judged 1-5, int = a whole number (tokens).
_LM = "lm-eval harness"
COLUMNS: list[dict] = [
    # metric per task follows docs/results.md: acc_norm for HellaSwag and OpenBookQA (raw acc is at chance on OBQA),
    # acc elsewhere. Both are in the cell's detail.
    {"key": "hellaswag", "label": "HellaSwag (norm)", "group": _LM, "super": "public", "higher_is_better": True, "fmt": "pct1", "task": "hellaswag", "metric": "acc_norm"},
    {"key": "arc_easy", "label": "ARC-Easy", "group": _LM, "super": "public", "higher_is_better": True, "fmt": "pct1", "task": "arc_easy", "metric": "acc"},
    {"key": "piqa", "label": "PIQA", "group": _LM, "super": "public", "higher_is_better": True, "fmt": "pct1", "task": "piqa", "metric": "acc"},
    {"key": "lambada", "label": "LAMBADA", "group": _LM, "super": "public", "higher_is_better": True, "fmt": "pct1", "task": "lambada_openai", "metric": "acc"},
    {"key": "openbookqa", "label": "OBQA (norm)", "group": _LM, "super": "public", "higher_is_better": True, "fmt": "pct1", "task": "openbookqa", "metric": "acc_norm"},
    {"key": "sciq", "label": "SciQ", "group": _LM, "super": "public", "higher_is_better": True, "fmt": "pct1", "task": "sciq", "metric": "acc"},
    {"key": "facts", "label": "facts probe", "group": "facts", "super": "homebrew", "higher_is_better": True, "fmt": "pct"},
    {"key": "r_arith2", "label": "arith2", "group": "reasoning", "super": "homebrew", "higher_is_better": True, "fmt": "frac", "task": "arith2"},
    {"key": "r_arith2mul", "label": "arith2mul", "group": "reasoning", "super": "homebrew", "higher_is_better": True, "fmt": "frac", "task": "arith2mul"},
    {"key": "r_algebra", "label": "algebra", "group": "reasoning", "super": "homebrew", "higher_is_better": True, "fmt": "frac", "task": "algebra"},
    {"key": "r_word", "label": "word", "group": "reasoning", "super": "homebrew", "higher_is_better": True, "fmt": "frac", "task": "word"},
    {"key": "r_gsm8k", "label": "GSM8K", "group": "reasoning", "super": "homebrew", "higher_is_better": True, "fmt": "frac", "task": "gsm8k_test"},
    {"key": "r_svamp", "label": "SVAMP", "group": "reasoning", "super": "homebrew", "higher_is_better": True, "fmt": "frac", "task": "svamp_test"},
    {"key": "r_mean", "label": "mean", "group": "reasoning", "super": "homebrew", "higher_is_better": True, "fmt": "frac"},
    {"key": "r_tool_use", "label": "tool use", "group": "reasoning", "super": "homebrew", "higher_is_better": True, "fmt": "frac"},
    {"key": "mt_recall", "label": "recall", "group": "multi-turn", "super": "homebrew", "higher_is_better": True, "fmt": "frac"},
    {"key": "mt_recall_absent", "label": "not stated", "group": "multi-turn", "super": "homebrew", "higher_is_better": True, "fmt": "frac"},
    {"key": "mt_revise", "label": "revise", "group": "multi-turn", "super": "homebrew", "higher_is_better": True, "fmt": "frac"},
    {"key": "mt_sysrule", "label": "sysrule", "group": "multi-turn", "super": "homebrew", "higher_is_better": True, "fmt": "frac"},
    {"key": "mt_format", "label": "format", "group": "multi-turn", "super": "homebrew", "higher_is_better": True, "fmt": "frac"},
    {"key": "mt_misfire", "label": "misfire", "group": "multi-turn", "super": "homebrew", "higher_is_better": False, "fmt": "frac"},
    {"key": "needle", "label": "effective ctx", "group": "needle", "super": "homebrew", "higher_is_better": True, "fmt": "int"},
    {"key": "judged", "label": "overall", "group": "judged", "super": "homebrew", "higher_is_better": True, "fmt": "score"},
    {"key": "judged_misfire", "label": "misfire", "group": "judged", "super": "homebrew", "higher_is_better": False, "fmt": "frac"},
    {"key": "pk_gsm8k", "label": "GSM8K", "group": "pass@k", "super": "homebrew", "higher_is_better": True, "fmt": "frac", "set": "gsm8k"},
    {"key": "pk_svamp", "label": "SVAMP", "group": "pass@k", "super": "homebrew", "higher_is_better": True, "fmt": "frac", "set": "svamp"},
]
SWARM_METHODS = [("greedy", "greedy"), ("majority", "majority"), ("verified_majority", "verified maj."), ("selector", "selector"), ("oracle", "oracle")]
for _set, _name in (("gsm8k", "GSM8K"), ("svamp", "SVAMP")):
    for _m, _label in SWARM_METHODS:
        COLUMNS.append({"key": f"sw_{_set}_{_m}", "label": _label, "group": f"swarm · {_name}", "super": "homebrew",
                        "higher_is_better": True, "fmt": "frac", "set": _set, "method": _m})
_BY_KEY = {c["key"]: c for c in COLUMNS}
_PUBLIC_FIELDS = ("key", "label", "group", "super", "higher_is_better", "fmt")

# ------------------------------------------------------------------------------------------------ files
_LM_RE = re.compile(r"^lm_eval(_[A-Za-z0-9]+)?\.json$")
_FACTS_RE = re.compile(r"^facts(_[A-Za-z0-9]+)?\.json$")
_REASON_RE = re.compile(r"^reasoning(_eval(_tools|_notools)?|_svamp|_s\d+|_step\d+|_snap[A-Za-z0-9]+)\.json$")
_MT_RE = re.compile(r"^multiturn(_[A-Za-z0-9]+)?\.json$")
# needle: v2 (real-text haystack) and its per-checkpoint copies. Not needle.json (v1, filler-era, superseded), not the
# filler control, not the depth probes (needle_depth0 / needle_shallow measure a few cells, not the gate).
_NEEDLE_RE = re.compile(r"^needle(_v2|_s\d+|_step\d+|_snap[A-Za-z0-9]+)\.json$")
_SWARM_RE = re.compile(r"^swarm_eval(_[A-Za-z0-9]+)?\.json$")
_SUFFIX_CKPT = [  # file-name suffix -> checkpoint file, for a result file without a `checkpoint` field
    (re.compile(r"_s(\d+)\.json$"), lambda m: f"step_{int(m.group(1)):05d}.pt"),
    (re.compile(r"_step(\d+)\.json$"), lambda m: f"step_{int(m.group(1)):05d}.pt"),
    (re.compile(r"_snap([A-Za-z0-9]+)\.json$"), lambda m: f"snap_{m.group(1)}.pt"),
]
_STAGE_RANK = {"base": 0, "sft": 1, "reasoning": 2, "tool": 2, "rl": 3}
EXTERNAL_GROUP = "external models"
_MODEL_FIELDS = ("name", "hf_id", "params", "license", "is_chat", "max_positions", "train_tokens", "notes")
_LABEL_PREF = ("final.pt", "best.pt")


def _read(p: Path) -> dict | None:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _num(x) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if x == x else None  # NaN -> None


def _ckpt_name(d: dict, fname: str) -> str:
    ck = d.get("checkpoint")
    if isinstance(ck, str) and ck.strip():
        return Path(ck.replace("\\", "/")).name
    for rx, f in _SUFFIX_CKPT:
        m = rx.search(fname)
        if m:
            return f(m)
    return "final.pt"


def _ckpt_run(d: dict) -> str | None:
    """The run directory named inside the file's `checkpoint` path, if any (used only for the detail note)."""
    ck = d.get("checkpoint")
    if not isinstance(ck, str):
        return None
    parts = Path(ck.replace("\\", "/")).parts
    return parts[-3] if len(parts) >= 3 and parts[-2] == "checkpoints" else None


def effective_context(summary: dict, threshold: float) -> tuple[int, int | None, float | None]:
    """(largest length L such that every measured length <= L has min-over-depths >= threshold, first failing
    length, its min). 0 when even the shortest length fails."""
    rows = sorted((int(k), v) for k, v in summary.items() if isinstance(v, dict) and _num(v.get("min")) is not None)
    eff = 0
    for L, v in rows:
        if v["min"] >= threshold:
            eff = L
        else:
            return eff, L, float(v["min"])
    return eff, None, None


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


class EvalIndex:
    """Scans `runs/*/` for result files; rebuilds only when a file's mtime/size (or a run.json / index.json) changes."""

    def __init__(self, runs_root: Path, run_index=None) -> None:
        self.root = Path(runs_root)
        self.runs = run_index
        self._sig: tuple | None = None
        self._table: dict | None = None
        # (run, checkpoint name) -> {"row": row, "refs": {key: [ref, ...]}}: every file that measured a cell, winner
        # first, with what the detail page needs to find the item inside it (rebuilt with the table)
        self._cells: dict[tuple[str, str], dict] = {}
        self.lock = threading.Lock()
        self._detail = None

    # -------------------------------------------------------------------------------- file discovery
    def _files(self) -> dict[str, list[Path]]:
        out: dict[str, list[Path]] = {}
        if not self.root.is_dir():
            return out
        for d in sorted(self.root.iterdir()):
            if not d.is_dir():
                continue
            fs = []
            for p in d.iterdir():
                n = p.name
                if p.is_file() and (n in ("run.json", "model.json", "bench_ll.json", "pass_at_k.json") or _LM_RE.match(n) or _FACTS_RE.match(n)
                                    or _REASON_RE.match(n) or _MT_RE.match(n) or _NEEDLE_RE.match(n) or _SWARM_RE.match(n)):
                    fs.append(p)
            for extra in (d / "quality" / "summary.json", d / "checkpoints" / "index.json"):
                if extra.is_file():
                    fs.append(extra)
            if fs:
                out[d.name] = sorted(fs)
        return out

    @staticmethod
    def _signature(files: dict[str, list[Path]]) -> tuple:
        sig = []
        for run, fs in files.items():
            for p in fs:
                try:
                    st = p.stat()
                except OSError:
                    continue
                sig.append((run, p.name, p.parent.name, st.st_mtime_ns, st.st_size))
        return tuple(sig)

    def table(self) -> dict:
        with self.lock:
            files = self._files()
            sig = self._signature(files)
            if self._table is None or sig != self._sig:
                self._table = self._build(files)
                self._sig = sig
            return self._table

    def cell_refs(self, run: str, checkpoint: str) -> dict | None:
        """{"row", "refs"} for a row by any of its checkpoint names (label or alias), or None."""
        self.table()
        with self.lock:
            return self._cells.get((run, checkpoint))

    def detail(self, run: str, checkpoint: str, key: str, **kw) -> dict:
        """Everything stored for one (checkpoint, column) cell: see services/eval_detail.py."""
        from slm.portal.services.eval_detail import EvalDetail

        if self._detail is None:
            self._detail = EvalDetail(self)
        return self._detail.detail(run, checkpoint, key, **kw)

    # -------------------------------------------------------------------------------- build
    def _build(self, files: dict[str, list[Path]]) -> dict:
        rows: dict[tuple, dict] = {}
        index_cache: dict[str, dict] = {}

        def index(run: str) -> dict:
            if run not in index_cache:
                index_cache[run] = _read(self.root / run / "checkpoints" / "index.json") or {}
            return index_cache[run]

        def row_for(run: str, ckpt: str, create: bool = True) -> dict | None:
            idx = index(run)
            tok = (idx.get(ckpt) or {}).get("tokens")
            key = (run, tok) if tok is not None else (run, ckpt)
            r = rows.get(key)
            if r is None and create:
                same = sorted(n for n, v in idx.items() if tok is not None and (v or {}).get("tokens") == tok)
                r = rows[key] = {"run": run, "own_tokens": tok, "names": set(same) | {ckpt}, "referenced": [], "cells": {}, "_prio": {}, "_alt": {}, "_refs": {}}
            if r is not None and ckpt not in r["referenced"]:
                r["referenced"].append(ckpt)
            return r

        def put(r: dict, key: str, value, source: str, detail: str, prio: float = 0.0, ref: dict | None = None) -> None:
            v = _num(value)
            if v is None:
                return
            r["_refs"].setdefault(key, []).append({"source": source, "value": v, "prio": prio, "detail": detail, **(ref or {})})
            cur = r["_prio"].get(key)
            if cur is not None and cur >= prio:
                r["_alt"].setdefault(key, []).append(f"{source} = {v:.4g}")
                return
            if cur is not None:
                old = r["cells"][key]
                r["_alt"].setdefault(key, []).append(f"{old['source']} = {old['value']:.4g}")
            r["cells"][key] = {"value": v, "source": source, "detail": detail}
            r["_prio"][key] = prio

        for run, fs in files.items():
            for p in fs:
                n = p.name
                if n in ("run.json", "index.json", "model.json") or (n == "summary.json"):
                    continue
                d = _read(p)
                if d is None:
                    continue
                ckpt = _ckpt_name(d, n)
                src = f"{run}/{n}"
                note = ""
                crun = _ckpt_run(d)
                if crun and crun != run:
                    note = f" · the file's checkpoint field says runs/{crun}/ (run renamed after the eval); attributed to {run}"
                if _LM_RE.match(n) or n == "bench_ll.json":
                    self._lm(row_for(run, ckpt), d, src, note, bench=(n == "bench_ll.json"), put=put)
                elif _FACTS_RE.match(n):
                    self._facts(row_for(run, ckpt), d, src, note, put)
                elif _REASON_RE.match(n):
                    self._reasoning(row_for(run, ckpt), d, n, src, note, put)
                elif _MT_RE.match(n):
                    self._multiturn(row_for(run, ckpt), d, src, note, put)
                elif _NEEDLE_RE.match(n):
                    self._needle(row_for(run, ckpt), d, src, note, put)
                elif n == "pass_at_k.json":
                    self._pass_at_k(row_for(run, ckpt), d, src, note, put)
                elif _SWARM_RE.match(n):
                    self._swarm(row_for(run, ckpt), d, src, note, put)

        # judged quality: attach to the rows other files made; a run with no other row gets its last judged checkpoint
        for run, fs in files.items():
            qp = self.root / run / "quality" / "summary.json"
            if qp not in fs:
                continue
            q = _read(qp)
            if not q:
                continue
            judged = [c for c in q.get("checkpoints") or [] if _num(c.get("overall")) is not None and c.get("checkpoint")]
            has_row = any(r["run"] == run for r in rows.values())
            for i, c in enumerate(judged):
                last = i == len(judged) - 1
                r = row_for(run, c["checkpoint"], create=not has_row and last)
                if r is None:
                    continue
                src = f"{run}/quality/summary.json"
                det = (f"judged {c.get('n_scored')}/{c.get('n_items')} prompts ({c.get('checkpoint')}, {c.get('stage', '')}) by "
                       f"{', '.join(q.get('judges') or []) or '?'} · correctness {c.get('correctness')}, coherence {c.get('coherence')}, task {c.get('task')}"
                       + (f" · excludes {', '.join(q.get('excluded_from_overall') or [])}" if q.get("excluded_from_overall") else ""))
                qref = {"tokens": c.get("tokens"), "qckpt": c.get("checkpoint")}
                put(r, "judged", c.get("overall"), src, det, ref=qref)
                if _num(c.get("tool_misfire")) is not None:
                    cats = ", ".join(c.get("tool_misfire_cats") or [])
                    put(r, "judged_misfire", c.get("tool_misfire"), src,
                        f"share of {c.get('tool_misfire_n')} no-tool prompts answered with a tool call" + (f" (misfired on: {cats})" if cats else ""), ref=qref)

        return self._finish(rows)

    # -------------------------------------------------------------------------------- per eval kind
    @staticmethod
    def _lm(r: dict, d: dict, src: str, note: str, bench: bool, put) -> None:
        res = d.get("results") or {}
        limit = d.get("limit")
        # the same limit everywhere is what makes rows comparable: 2000 (every 336M row) > full set > 1000 (the
        # benchmark-revision candidates in bench_ll.json)
        prio = {2000: 3.0, None: 2.0, 1000: 1.0}.get(limit, 0.5) if not bench else 0.9
        for col in COLUMNS:
            if "metric" not in col or col["task"] not in res:
                continue
            t = res[col["task"]]
            v = t.get(f"{col['metric']},none")
            extra = f"acc {t.get('acc,none', float('nan')):.3f}" + (f", acc_norm {t['acc_norm,none']:.3f}" if "acc_norm,none" in t else "")
            if "perplexity,none" in t:
                extra += f", perplexity {t['perplexity,none']:.1f}"
            put(r, col["key"], v, src, f"{col['metric']} · n={t.get('sample_len')} · limit={limit or 'full set'} · {extra}{note}", prio)

    @staticmethod
    def _facts(r: dict, d: dict, src: str, note: str, put) -> None:
        cats = ", ".join(f"{k} {_pct(v)}" for k, v in (d.get("per_category") or {}).items() if _num(v) is not None)
        put(r, "facts", d.get("accuracy"), src, f"n={d.get('n')} · mode {d.get('mode')} · {cats}{note}")

    @staticmethod
    def _reasoning(r: dict, d: dict, fname: str, src: str, note: str, put) -> None:
        tools = bool(d.get("tools"))
        if fname == "reasoning_svamp.json":
            prio = 0.0  # a separate SVAMP pass: only fills tasks the main file lacks
        elif "_notools" in fname:
            prio = 1.0
        else:
            prio = 3.0 if tools else 2.0
        per = d.get("per_task") or {}
        mode = "tool on" if tools else "tool off"
        for col in COLUMNS:
            if col["group"] != "reasoning" or "task" not in col or col["task"] not in per:
                continue
            t = per[col["task"]]
            tu = t.get("tool_use_rate")
            put(r, col["key"], t.get("accuracy"), src,
                f"n={t.get('n')} · {mode}" + (f" · tool use {tu:.2f}" if _num(tu) is not None else "")
                + (f" · malformed {t['malformed_rate']:.2f}" if _num(t.get("malformed_rate")) is not None else "") + note, prio)
        tasks = [k for k in per if _num((per[k] or {}).get("accuracy")) is not None]
        put(r, "r_mean", d.get("mean_accuracy"), src, f"mean over {len(tasks)} tasks: {', '.join(tasks)} · {mode}{note}", prio)
        rates = [per[k]["tool_use_rate"] for k in tasks if _num(per[k].get("tool_use_rate")) is not None]
        if tools and rates:
            put(r, "r_tool_use", sum(rates) / len(rates), src, f"share of answers that called the sandbox, mean over {len(rates)} tasks{note}", prio)

    @staticmethod
    def _multiturn(r: dict, d: dict, src: str, note: str, put) -> None:
        s = d.get("summary") or {}
        counts = s.get("counts") if isinstance(s.get("counts"), dict) else {"recall": s.get("n")}  # files before the kinds: recall only
        kinds = " · ".join(f"{k} {v}" for k, v in counts.items())
        tail = f" · seed {s.get('seed')} · {s.get('mean_answer_tokens')} tokens/turn" + (f" · system prompt: {s['system_prompt']}" if s.get("system_prompt") else "") + note
        put(r, "mt_recall", s.get("recall"), src, f"n={counts.get('recall')} recall conversations · templated {s.get('templated')}{tail}")
        put(r, "mt_recall_absent", s.get("recall_absent"), src,
            f"n={counts.get('recall_absent')} conversations asking about something never stated: names nothing and says so{tail}")
        put(r, "mt_revise", s.get("revise"), src,
            f"n={counts.get('revise')} rewrites · mean share of constraints kept · all kept {s.get('revise_all')}{tail}")
        put(r, "mt_sysrule", s.get("sysrule"), src,
            f"n={counts.get('sysrule')} conversations · mean share of the system rule's parts kept · all kept {s.get('sysrule_all')}{tail}")
        put(r, "mt_format", s.get("format"), src, f"over every kind ({kinds}){tail}")
        put(r, "mt_misfire", s.get("misfire"), src, f"share of generated turns, every kind ({kinds}){tail}")

    @staticmethod
    def _needle(r: dict, d: dict, src: str, note: str, put) -> None:
        if d.get("haystack", "real") != "real" or not isinstance(d.get("summary"), dict):
            return  # the filler control never counts (CLAUDE.md: real-text haystack only)
        thr = _num(d.get("threshold")) or 0.8
        eff, fail_len, fail_min = effective_context(d["summary"], thr)
        mins = ", ".join(f"{k}: {_pct(v['min'])}" for k, v in sorted(d["summary"].items(), key=lambda kv: int(kv[0])) if isinstance(v, dict))
        det = f"n={d.get('n')} per cell · {len(d.get('depths') or [])} depths · gate min over depths >= {thr:g} · min per length {mins}"
        if fail_len is not None:
            det += f" · first failing length {fail_len} ({_pct(fail_min)})"
        if d.get("effective_context") is not None and d.get("effective_context") != eff:
            det += f" · the file says {d.get('effective_context')}"
        if (d.get("n") or 0) < 16:
            det += " · n < 16: not enough for a claim"
        put(r, "needle", eff, src, det + note)

    @staticmethod
    def _pass_at_k(r: dict, d: dict, src: str, note: str, put) -> None:
        res = d.get("results") or {}
        for col in COLUMNS:
            if col["group"] != "pass@k" or col["set"] not in res:
                continue
            s = (res[col["set"]] or {}).get("summary") or {}
            put(r, col["key"], s.get("pass_at_k"), src,
                f"pass@{s.get('k')} over {s.get('n_problems')} problems · T={d.get('temperature')} top_p={d.get('top_p')} · pass@1 {s.get('pass_at_1')} · "
                f"majority {s.get('majority')} · {s.get('mean_distinct_answers')} distinct answers per {s.get('k')}{note}")

    @staticmethod
    def _swarm(r: dict, d: dict, src: str, note: str, put) -> None:
        res = d.get("results") or {}
        for col in COLUMNS:
            if not col["key"].startswith("sw_") or col["set"] not in res:
                continue
            s = (res[col["set"]] or {}).get("summary") or {}
            put(r, col["key"], s.get(col["method"]), src,
                f"k={s.get('k', d.get('k'))} samples, n={s.get('n')} problems, T={d.get('temperature')} · oracle {s.get('oracle')}, "
                f"oracle verified {s.get('oracle_verified')}, in prompt {s.get('in_prompt')} · {s.get('mean_groups')} groups / "
                f"{s.get('mean_verified')} verified per k{note}")

    # -------------------------------------------------------------------------------- rows, order, colour
    def _run_info(self) -> tuple[dict, dict]:
        """({run: summary}, {run: meta}) from the shared RunIndex (cumulative tokens need the whole init_from chain)."""
        if self.runs is None:
            from slm.portal.services.runs import RunIndex

            self.runs = RunIndex(self.root)
        summaries = {s["run_name"]: s for s in self.runs.summaries()}
        metas = {}
        for n in summaries:
            try:
                metas[n] = self.runs.get(n).meta()
            except KeyError:
                metas[n] = {}
        return summaries, metas

    def _finish(self, rows: dict) -> dict:
        summaries, metas = self._run_info()
        out = []
        self._cells = {}
        ext_cache: dict[str, dict | None] = {}
        for r in rows.values():
            if not r["cells"]:
                continue
            run = r["run"]
            if run not in ext_cache:
                ext_cache[run] = _read(self.root / run / "model.json")
            ext = ext_cache[run]
            meta = metas.get(run) or {}
            stage = run_stage(meta) if meta else "base"
            if stage in ("sft", "reasoning") and run_tools(meta):
                stage = "tool"
            names = r["names"]
            label = next((n for n in _LABEL_PREF if n in names), r["referenced"][0])
            s = summaries.get(run) or {}
            own = r["own_tokens"] if r["own_tokens"] is not None else (s.get("tokens") if label in _LABEL_PREF else None)
            chain = self.runs.chain(run, summaries) if run in summaries else []
            cum = (sum(e["tokens_used"] for e in chain[:-1]) + own) if own is not None and chain else own
            for k, alts in r["_alt"].items():
                if k in r["cells"] and alts:
                    r["cells"][k]["detail"] += " · also measured: " + "; ".join(alts)
            row = {"run": run, "checkpoint": label, "aliases": sorted(n for n in names if n != label and n != "latest.pt"),
                   "stage": stage, "params": s.get("n_params") or (meta.get("n_params") if meta else None),
                   "own_tokens": own, "tokens": cum, "cells": r["cells"], "group": "ours"}
            if ext is not None:  # an external comparison model: its registry entry, not a run of ours
                tt = _num(ext.get("train_tokens"))
                row.update(stage="external", group=EXTERNAL_GROUP, params=ext.get("params"), own_tokens=int(tt) if tt else None,
                           tokens=int(tt) if tt else None, model={k: ext.get(k) for k in _MODEL_FIELDS})
            out.append(row)
            refs = {k: [x for _, x in sorted(enumerate(v), key=lambda iv: (-iv[1]["prio"], iv[0]))] for k, v in r["_refs"].items()}
            for n in names | set(r["referenced"]):
                self._cells[(run, n)] = {"row": row, "refs": refs}
        out.sort(key=lambda r: (r["group"] == EXTERNAL_GROUP, -(r["params"] or 0), _STAGE_RANK.get(r["stage"], 9), r["run"], r["own_tokens"] or 0,
                                r["checkpoint"]))
        cols = []
        for c in COLUMNS:
            vals = [r["cells"][c["key"]]["value"] for r in out if c["key"] in r["cells"]]
            lo, hi = (min(vals), max(vals)) if vals else (None, None)
            for r in out:
                cell = r["cells"].get(c["key"])
                if cell is None:
                    continue
                if hi == lo:
                    cell["t"] = 0.5
                else:
                    t = (cell["value"] - lo) / (hi - lo)
                    cell["t"] = round(t if c["higher_is_better"] else 1.0 - t, 4)
            cols.append({**{k: c[k] for k in _PUBLIC_FIELDS}, "min": lo, "max": hi, "n": len(vals)})
        return {"columns": cols, "rows": out}

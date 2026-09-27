"""The Eval tab's detail pages: everything stored for one (checkpoint, column) cell (torch-free).

The table (services/evals.py) records, per cell, every result file that measured it, winner first. `EvalDetail.detail`
opens one of those files and returns what it holds:

- `summary`: labelled numbers (the cell's own statistics first),
- `tables`: aggregates such as lm-eval's per-task accuracies or the needle grid, with the cell's row highlighted,
- `items`: the per-item rows where the eval stored them (prompt, model output, gold, verdict, judge scores, tool
  calls), paged and filterable by verdict and text; `{available: false, note}` where it stored only aggregates,
- `sources`, `run_info` and the tail of the eval's `.log` beside the file.

Where the rows live, per eval: the judged suite in `quality/outputs/<tokens>.jsonl` joined with `quality/scores.jsonl`;
reasoning in the `reasoning_dump_<tag>.jsonl` written beside `reasoning_eval_<tag>.json` when the eval ran with
`--dump` (older runs only); facts, multi-turn, needle (failures per cell only), pass@k and swarm (flags per problem,
no sample text) inside the result file; lm-eval files never hold them (run without `--log_samples`). GSM8K / SVAMP
problem text for pass@k and swarm rows is looked up by id in the local test parquet the evals read.

Parsed files are cached by (mtime, size), a few dozen at most; a file above `MAX_FILE_BYTES` is not parsed.
"""

from __future__ import annotations

import json
import re
import threading
from collections import OrderedDict
from pathlib import Path

from slm.portal.services.evals import _BY_KEY, _FACTS_RE, _LM_RE, _MT_RE, _NEEDLE_RE, _PUBLIC_FIELDS, _REASON_RE, _SWARM_RE, _num, _read, effective_context

MAX_FILE_BYTES = 64 << 20  # nothing the evals write comes near this; a runaway dump is refused, not loaded
PAGE_DEFAULT = 100
PAGE_MAX = 500
SHORT = 160  # characters of a text field in the collapsed row; the full text is in the expansion
_CACHE_N = 32
_LOG_BYTES = 8192
_TQDM = re.compile(r"\d+%\|")

_Q: dict[str, dict[str, str]] = {}
_Q_LOCK = threading.Lock()


def dataset_questions(name: str) -> dict[str, str]:
    """id -> problem text for the GSM8K / SVAMP test sets, numbered the way slm.eval.reasoning numbers them
    (`gsm8k-test-<row>`); {} when the parquet is not on this machine."""
    with _Q_LOCK:
        if name in _Q:
            return _Q[name]
        out: dict[str, str] = {}
        try:
            import pyarrow.parquet as pq

            from slm.data.sources import SOURCES

            col = {"gsm8k": "question", "svamp": "question_concat"}[name]
            files = [p for p in SOURCES[name].local_dir.rglob("*.parquet") if "test" in p.name]
            if files:
                rows = pq.read_table(files[0], columns=[col]).column(col).to_pylist()
                out = {f"{name}-test-{i}": str(q or "").strip() for i, q in enumerate(rows)}
        except Exception:
            out = {}
        _Q[name] = out
        return out


def _short(s, n: int = SHORT):
    if not isinstance(s, str):
        return s
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _blk(label: str, value, kind: str = "text") -> dict | None:
    """One block of an expanded row: text (pre-wrap), think (the think span), kv (an object), tools ([[code, result]])."""
    if value is None or value == "" or value == [] or value == {}:
        return None
    return {"label": label, "kind": kind, "value": value}


def _row(ok, c: dict, x: list, search: str = "") -> dict:
    return {"ok": ok, "c": c, "x": [b for b in x if b], "_s": search.lower()}


def _items(rows: list[dict], columns: list[dict], labels: dict, a: dict, note: str = "", source: str = "") -> dict:
    counts = {"pass": sum(1 for r in rows if r["ok"] is True), "fail": sum(1 for r in rows if r["ok"] is False),
              "other": sum(1 for r in rows if r["ok"] is None)}
    for i, r in enumerate(rows):
        r["i"] = i + 1
    flt, q = a["filter"], a["q"].strip().lower()
    want = {"pass": True, "fail": False, "other": None}.get(flt, "all")
    sel = [r for r in rows if (want == "all" or r["ok"] is want) and (not q or q in r["_s"])]
    page = [{k: v for k, v in r.items() if k != "_s"} for r in sel[a["offset"]: a["offset"] + a["limit"]]]
    return {"available": True, "columns": columns, "labels": labels, "counts": counts, "total": len(rows),
            "n_filtered": len(sel), "offset": a["offset"], "limit": a["limit"], "rows": page, "note": note, "source": source}


def _none(note: str) -> dict:
    return {"available": False, "note": note}


def _cols(*spec) -> list[dict]:
    """("key", "label", "type") triples -> column specs. type: text | id | num | int | frac | pct | pct1 | score (1-5) | bool (a
    verdict: ✓ green / ✗ red) | flag (a neutral yes / no)."""
    return [{"key": k, "label": lab, "type": t} for k, lab, t in spec]


def _sv(label: str, value, fmt: str | None = None) -> dict | None:
    return None if value is None else {"label": label, "value": value, "fmt": fmt}


def _log_tail(path: Path) -> dict | None:
    cands = [path.with_suffix(".log")]
    if _LM_RE.match(path.name) and not (path.parent / "lm_eval.json").exists():
        cands.append(path.parent / "lm_eval.log")  # the limit-2000 evals of the 336M runs log to lm_eval.log
    for p in cands:
        try:
            size = p.stat().st_size
            with p.open("rb") as f:
                f.seek(max(0, size - _LOG_BYTES))
                raw = f.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        lines = [ln.rstrip("\r").split("\r")[-1].rstrip() for ln in raw.split("\n")]  # CRLF logs; a bare \r is a progress-bar redraw
        if size > _LOG_BYTES:
            lines = lines[1:]  # the first line is cut
        lines = [ln for ln in lines if ln.strip() and not _TQDM.search(ln)]
        return {"file": p.name, "size": size, "lines": lines[-40:]}
    return None


class EvalDetail:
    def __init__(self, index) -> None:
        self.idx = index
        self._cache: OrderedDict[tuple, tuple] = OrderedDict()
        self.lock = threading.Lock()
        self.questions = dataset_questions  # replaceable in tests

    # ------------------------------------------------------------------------------------------ cached parsing
    def _load(self, path: Path, kind: str = "json"):
        st = path.stat()
        if st.st_size > MAX_FILE_BYTES:
            raise ValueError(f"{path.name} is {st.st_size >> 20} MiB, above the {MAX_FILE_BYTES >> 20} MiB the portal parses")
        key, sig = (str(path), kind), (st.st_mtime_ns, st.st_size)
        with self.lock:
            hit = self._cache.get(key)
            if hit and hit[0] == sig:
                self._cache.move_to_end(key)
                return hit[1]
        text = path.read_text(encoding="utf-8")
        if kind == "json":
            obj = json.loads(text)
        else:  # jsonl; a torn last line (a file being written) is skipped
            obj = []
            for ln in text.splitlines():
                if ln.strip():
                    try:
                        obj.append(json.loads(ln))
                    except ValueError:
                        continue
        with self.lock:
            self._cache[key] = (sig, obj)
            while len(self._cache) > _CACHE_N:
                self._cache.popitem(last=False)
        return obj

    # ------------------------------------------------------------------------------------------ entry point
    def detail(self, run: str, checkpoint: str, key: str, file: str | None = None, filter: str = "all", q: str = "",
               offset: int = 0, limit: int = PAGE_DEFAULT) -> dict:
        col = _BY_KEY.get(key)
        if col is None:
            raise KeyError(f"unknown eval column {key!r}")
        ent = self.idx.cell_refs(run, checkpoint)
        if ent is None:
            raise KeyError(f"no evaluated checkpoint {run}/{checkpoint}")
        row, refs = ent["row"], ent["refs"].get(key) or []
        if not refs:
            raise KeyError(f"{col['label']} ({col['group']}) was never measured on {run}/{checkpoint}")
        ref = next((r for r in refs if r["source"] == file), None) if file else refs[0]
        if ref is None:
            raise KeyError(f"{file} did not measure this cell")
        path = self.idx.root / ref["source"]
        tcol = next((c for c in self.idx.table()["columns"] if c["key"] == key), {})
        sources = []
        for i, r in enumerate(refs):
            p = self.idx.root / r["source"]
            try:
                st = p.stat()
                size, mtime = st.st_size, st.st_mtime
            except OSError:
                size = mtime = None
            sources.append({"source": r["source"], "value": r["value"], "detail": r["detail"], "winner": i == 0,
                            "size": size, "mtime": mtime, "shown": r is ref})
        out = {"run": row["run"], "checkpoint": row["checkpoint"], "requested": checkpoint, "aliases": row["aliases"],
               "stage": row["stage"], "params": row["params"], "tokens": row["tokens"], "own_tokens": row["own_tokens"], "model": row.get("model"),
               "key": key, "column": {**{k: col[k] for k in _PUBLIC_FIELDS}, "min": tcol.get("min"), "max": tcol.get("max")},
               "cell": row["cells"].get(key), "value": ref["value"], "source": ref["source"], "source_detail": ref["detail"],
               "sources": sources, "run_info": self._run_info(row["run"]), "log": _log_tail(path),
               "file_fields": {}, "summary": [], "tables": [], "items": None, "notes": []}
        a = {"filter": filter if filter in ("all", "pass", "fail", "other") else "all", "q": q or "",
             "offset": max(0, int(offset)), "limit": max(1, min(PAGE_MAX, int(limit)))}
        n = path.name
        try:
            if n == "summary.json" and path.parent.name == "quality":
                self._judged(path.parent.parent, ref, col, out, a)
            else:
                d = self._load(path)
                if not isinstance(d, dict):
                    raise ValueError(f"{n} is not a JSON object")
                out["file_fields"] = {k: v for k, v in d.items() if isinstance(v, (str, int, float, bool)) or v is None}
                if _LM_RE.match(n) or n == "bench_ll.json":
                    self._lm(d, col, out)
                elif _FACTS_RE.match(n):
                    self._facts(d, out, a, ref["source"])
                elif _REASON_RE.match(n):
                    self._reasoning(path, d, col, out, a)
                elif _MT_RE.match(n):
                    self._multiturn(d, col, out, a, ref["source"])
                elif _NEEDLE_RE.match(n):
                    self._needle(d, out, a, ref["source"])
                elif n == "pass_at_k.json":
                    self._pass_at_k(d, col, out, a, ref["source"])
                elif _SWARM_RE.match(n):
                    self._swarm(d, col, out, a, ref["source"])
        except (OSError, ValueError) as e:
            out["notes"].append(f"could not read {ref['source']}: {e}")
        out["summary"] = [s for s in out["summary"] if s]
        if out["items"] is None:
            out["items"] = _none("This eval stores no per-item data.")
        return out

    def _run_info(self, run: str) -> dict:
        meta = _read(self.idx.root / run / "run.json") or {}
        cfg = meta.get("config") if isinstance(meta.get("config"), dict) else {}
        env = meta.get("env") if isinstance(meta.get("env"), dict) else {}
        return {"stage": meta.get("stage"), "n_params": meta.get("n_params"), "started": meta.get("started"),
                "git_commit": env.get("git_commit"), "init_from": cfg.get("init_from"),
                "notes": cfg.get("notes") or cfg.get("description") or meta.get("notes"), "config": cfg or None}

    # ------------------------------------------------------------------------------------------ lm-eval
    @staticmethod
    def _lm(d: dict, col: dict, out: dict) -> None:
        res = d.get("results") or {}
        limit = d.get("limit")
        t = res.get(col.get("task")) or {}
        m = col.get("metric", "acc")
        out["summary"] += [_sv("metric", m), _sv(m, t.get(f"{m},none"), "pct1"), _sv("± stderr", t.get(f"{m}_stderr,none"), "pct1"),
                           _sv("samples", t.get("sample_len"), "int"), _sv("limit", limit if limit is not None else "full set"),
                           _sv("perplexity", t.get("perplexity,none"), "num")]
        rows, hi = [], None
        for task, v in res.items():
            if not isinstance(v, dict):
                continue
            if task == col.get("task"):
                hi = len(rows)
            rows.append([task, v.get("sample_len"), v.get("acc,none"), v.get("acc_stderr,none"), v.get("acc_norm,none"),
                         v.get("acc_norm_stderr,none"), v.get("perplexity,none")])
        out["tables"].append({"title": "every task in this file", "highlight": hi,
                              "columns": _cols(("task", "task", "text"), ("n", "n", "int"), ("acc", "acc", "pct1"), ("acc_se", "± se", "pct1"),
                                               ("acc_norm", "acc_norm", "pct1"), ("norm_se", "± se", "pct1"), ("ppl", "perplexity", "num")),
                              "rows": rows, "note": "accuracies × 100; ± is lm-eval's bootstrap standard error"})
        out["items"] = _none("lm-evaluation-harness was run without --log_samples, so this file stores only the per-task "
                             "aggregates above (accuracy, standard error, sample count): no per-question predictions were saved.")

    # ------------------------------------------------------------------------------------------ facts
    @staticmethod
    def _facts(d: dict, out: dict, a: dict, src: str) -> None:
        mode = d.get("mode")
        out["summary"] += [_sv("accuracy", d.get("accuracy"), "pct"), _sv("n", d.get("n"), "int"), _sv("mode", mode)]
        cats = d.get("per_category") or {}
        if cats:
            out["tables"].append({"title": "per category", "columns": _cols(("cat", "category", "text"), ("acc", "accuracy", "pct")),
                                  "rows": [[k, v] for k, v in cats.items()]})
        rows = d.get("rows")
        if not isinstance(rows, list) or not rows:
            out["items"] = _none("This facts file predates per-question rows: only the accuracy is stored.")
            return
        items = []
        for r in rows:
            prompt = r.get("completion") if mode == "completion" else r.get("question")
            prompt = prompt or r.get("question") or r.get("completion")
            items.append(_row(r.get("correct"), {"cat": r.get("cat"), "prompt": _short(prompt), "answers": _short(r.get("answers"), 60),
                                                 "output": _short(r.get("output")), "correct": r.get("correct")},
                              [_blk("prompt", prompt), _blk("model output", r.get("output")), _blk("accepted answers", r.get("answers")),
                               _blk("other form of the question", r.get("question") if prompt != r.get("question") else r.get("completion"))],
                              f"{prompt} {r.get('output')} {r.get('answers')} {r.get('cat')}"))
        out["items"] = _items(items, _cols(("cat", "category", "text"), ("prompt", "prompt", "text"), ("answers", "gold", "text"),
                                           ("output", "model output", "text"), ("correct", "correct", "bool")),
                              {"pass": "correct", "fail": "wrong"}, a,
                              "Greedy output, scored by whole-word match against the accepted answers (| separates alternatives).", src)

    # ------------------------------------------------------------------------------------------ reasoning
    def _reasoning(self, path: Path, d: dict, col: dict, out: dict, a: dict) -> None:
        per = d.get("per_task") or {}
        task = col.get("task")
        tools = bool(d.get("tools"))
        t = per.get(task) or {}
        out["summary"] += [_sv("tool", "on" if tools else "off"), _sv("accuracy", t.get("accuracy"), "frac"), _sv("n", t.get("n"), "int"),
                           _sv("malformed", t.get("malformed_rate"), "frac"), _sv("tool use", t.get("tool_use_rate"), "frac"),
                           _sv("calls per answer", t.get("tool_calls_mean"), "num"), _sv("tool errors", t.get("tool_error_rate"), "frac"),
                           _sv("answer from tool", t.get("answer_from_tool_rate"), "frac"), _sv("mean length (tokens)", t.get("mean_len"), "num"),
                           _sv("mean accuracy (file)", d.get("mean_accuracy"), "frac"), _sv("seconds", d.get("seconds"), "num")]
        rows, hi = [], None
        for k, v in per.items():
            if not isinstance(v, dict):
                continue
            if k == task:
                hi = len(rows)
            rows.append([k, v.get("n"), v.get("accuracy"), v.get("malformed_rate"), v.get("tool_use_rate"), v.get("tool_calls_mean"),
                         v.get("tool_error_rate"), v.get("answer_from_tool_rate"), v.get("mean_len")])
        out["tables"].append({"title": "every task in this file", "highlight": hi, "rows": rows,
                              "columns": _cols(("task", "task", "text"), ("n", "n", "int"), ("acc", "accuracy", "frac"), ("mal", "malformed", "frac"),
                                               ("tu", "tool use", "frac"), ("calls", "calls / answer", "num"), ("terr", "tool errors", "frac"),
                                               ("aft", "answer from tool", "frac"), ("len", "mean length", "num"))})
        dump = path.parent / (re.sub(r"^reasoning(_eval)?", "reasoning_dump", path.stem) + ".jsonl")
        if not dump.is_file():
            out["items"] = _none(f"slm.eval.reasoning keeps per-problem traces only when run with --dump, and this evaluation was not "
                                 f"(no {dump.name} beside the file); only the per-task aggregates above were saved.")
            return
        lines = self._load(dump, "jsonl")
        blocks, why = _segment(lines, per)
        if task:
            sel = blocks.get(task)
            if sel is None:
                out["items"] = _none(f"{dump.name} exists but its problems could not be matched to {task} ({why}).")
                return
        else:
            sel = lines
        use_tool = col["key"] == "r_tool_use"
        items = []
        for r in sel:
            ok = bool(r.get("tool_calls")) if use_tool else r.get("correct")
            items.append(_row(ok, {"task": r.get("task"), "id": r.get("id"), "prompt": _short(r.get("prompt")), "gold": r.get("gold"),
                                   "answer": _short(r.get("answer"), 40), "correct": r.get("correct"), "calls": r.get("tool_calls"),
                                   "malformed": r.get("malformed")},
                              [_blk("prompt", r.get("prompt")), _blk("model output (<<code=result>> marks a tool call)", r.get("text")),
                               _blk("tool calls", r.get("tool_results"), "tools"),
                               _blk("grading", {"gold": r.get("gold"), "parsed answer": r.get("answer"), "correct": r.get("correct"),
                                                "malformed": r.get("malformed"), "tool calls": r.get("tool_calls"), "tool errors": r.get("tool_errors"),
                                                "answer from tool": r.get("answer_from_tool"), "tokens": r.get("n_tokens")}, "kv")],
                              f"{r.get('prompt')} {r.get('text')} {r.get('id')}"))
        labels = {"pass": "called the tool", "fail": "no tool call"} if use_tool else {"pass": "correct", "fail": "wrong"}
        out["items"] = _items(items, _cols(("task", "task", "text"), ("id", "id", "id"), ("prompt", "prompt", "text"), ("gold", "gold", "text"),
                                           ("answer", "parsed", "text"), ("correct", "correct", "bool"), ("calls", "tool calls", "int"),
                                           ("malformed", "malformed", "flag")),
                              labels, a, f"Greedy traces from {dump.name}. The answer is what follows the last ####; malformed = no parsable answer.",
                              f"{path.parent.name}/{dump.name}")

    # ------------------------------------------------------------------------------------------ multi-turn
    @staticmethod
    def _multiturn(d: dict, col: dict, out: dict, a: dict, src: str) -> None:
        s = d.get("summary") or {}
        out["summary"] += [_sv("recall", s.get("recall"), "frac"), _sv("format", s.get("format"), "frac"), _sv("misfire", s.get("misfire"), "frac"),
                           _sv("templated", s.get("templated"), "frac"), _sv("conversations", s.get("n"), "int"), _sv("seed", s.get("seed"), "int"),
                           _sv("tokens per answer", s.get("mean_answer_tokens"), "num"), _sv("seconds", s.get("seconds"), "num")]
        convs = d.get("conversations")
        if not isinstance(convs, list) or not convs:
            out["items"] = _none("This multi-turn file stores only the summary.")
            return
        k = col["key"]
        items = []
        for cv in convs:
            turns, asst = cv.get("turns") or [], cv.get("assistant") or []
            ok = cv.get("recall") if k == "mt_recall" else cv.get("format_ok") if k == "mt_format" else (cv.get("misfires") or 0) == 0
            x = []
            for j, u in enumerate(turns):
                x.append(_blk(f"user {j + 1}", u))
                if j < len(asst) and isinstance(asst[j], dict):
                    t = asst[j]
                    x.append(_blk(f"assistant {j + 1}: think", t.get("think"), "think"))
                    x.append(_blk(f"assistant {j + 1}", t.get("answer") if t.get("answer") != "" else "(empty answer)"))
                    x.append(_blk(f"assistant {j + 1}: flags", {"ended with <|end|>": t.get("terminated"), "tool calls": t.get("tool_calls"),
                                                                 "tokens": t.get("n_tokens")}, "kv"))
            last = asst[-1] if asst and isinstance(asst[-1], dict) else {}
            items.append(_row(ok, {"fact": cv.get("fact"), "question": _short(turns[-1] if turns else None), "answer": _short(last.get("answer")),
                                   "recall": cv.get("recall"), "format": cv.get("format_ok"), "misfires": cv.get("misfires"),
                                   "templated": cv.get("templated")},
                              x + [_blk("grading", {"fact": cv.get("fact"), "recall": cv.get("recall"), "format ok": cv.get("format_ok"),
                                                     "misfires": cv.get("misfires"), "templated": cv.get("templated")}, "kv")],
                              " ".join(str(t) for t in turns) + " " + " ".join(str((t or {}).get("answer")) for t in asst)))
        labels = {"mt_recall": {"pass": "recalled", "fail": "missed"}, "mt_format": {"pass": "every turn ended", "fail": "ran on"},
                  "mt_misfire": {"pass": "no tool call", "fail": "misfired"}}[k]
        out["items"] = _items(items, _cols(("fact", "fact", "text"), ("question", "last user turn", "text"), ("answer", "last answer", "text"),
                                           ("recall", "recall", "bool"), ("format", "format", "bool"), ("misfires", "misfires", "int"),
                                           ("templated", "templated", "flag")),
                              labels, a, "Three turns per conversation; expand a row for every turn with its think span.", src)

    # ------------------------------------------------------------------------------------------ needle
    @staticmethod
    def _needle(d: dict, out: dict, a: dict, src: str) -> None:
        thr = _num(d.get("threshold")) or 0.8
        summ = d.get("summary") if isinstance(d.get("summary"), dict) else {}
        eff, fail_len, fail_min = effective_context(summ, thr)
        out["summary"] += [_sv("effective context", eff, "int"), _sv("gate: min over depths ≥", thr, "num"), _sv("first failing length", fail_len, "int"),
                           _sv("its worst depth", fail_min, "pct"), _sv("trials per cell", d.get("n"), "int"), _sv("haystack", d.get("haystack", "real")),
                           _sv("the file says", d.get("effective_context"), "int"), _sv("max_seq_len", d.get("max_seq_len"), "int")]
        res = [r for r in d.get("results") or [] if isinstance(r, dict)]
        depths = sorted({r.get("depth") for r in res if r.get("depth") is not None}) or list(d.get("depths") or [])
        lengths = sorted({r.get("length") for r in res if r.get("length") is not None}) or sorted(int(k) for k in summ)
        grid = {(r.get("length"), r.get("depth")): r.get("accuracy") for r in res}
        rows = [[L, *[grid.get((L, dp)) for dp in depths], (summ.get(str(L)) or {}).get("min"), (summ.get(str(L)) or {}).get("mean")] for L in lengths]
        out["tables"].append({"title": "retrieval accuracy per length × depth", "rows": rows,
                              "highlight": lengths.index(fail_len) if fail_len in lengths else None,
                              "columns": _cols(("len", "length", "int"), *[(f"d{dp}", f"depth {dp:g}", "pct") for dp in depths],
                                               ("min", "min", "pct"), ("mean", "mean", "pct")),
                              "note": "highlighted: the first length whose worst depth falls below the gate"})
        if not res:
            out["items"] = _none("This needle file stores only the per-length summary.")
            return
        items = []
        for r in res:
            fails = [f for f in r.get("failures") or [] if isinstance(f, dict)]
            text = "\n".join(f"gold {f.get('gold')}  →  {json.dumps(f.get('out'), ensure_ascii=False)}" for f in fails)
            items.append(_row((_num(r.get("accuracy")) or 0) >= thr, {"length": r.get("length"), "depth": r.get("depth"), "accuracy": r.get("accuracy"),
                                                                      "n": r.get("n"), "failures": len(fails)},
                              [_blk(f"failures ({len(fails)}): the hidden number and the model's output", text)], text))
        out["items"] = _items(items, _cols(("length", "length", "int"), ("depth", "depth", "num"), ("accuracy", "accuracy", "pct"),
                                           ("n", "trials", "int"), ("failures", "failures", "int")),
                              {"pass": f"cell ≥ {thr:g}", "fail": f"cell < {thr:g}"}, a,
                              "One row per (length, depth) cell. The haystack prompts are not stored; each cell keeps its failed trials "
                              "(the hidden number and what the model answered).", src)

    # ------------------------------------------------------------------------------------------ pass@k
    def _pass_at_k(self, d: dict, col: dict, out: dict, a: dict, src: str) -> None:
        res = d.get("results") or {}
        st = col.get("set")
        s = (res.get(st) or {}).get("summary") or {}
        out["summary"] += [_sv(f"pass@{s.get('k')}", s.get("pass_at_k"), "frac"), _sv("pass@1", s.get("pass_at_1"), "frac"),
                           _sv("majority", s.get("majority"), "frac"), _sv("problems", s.get("n_problems"), "int"), _sv("k", s.get("k"), "int"),
                           _sv("tool use", s.get("tool_use"), "frac"), _sv("distinct answers per k", s.get("mean_distinct_answers"), "num"),
                           _sv("temperature", d.get("temperature"), "num"), _sv("top_p", d.get("top_p"), "num"), _sv("seconds", s.get("seconds"), "num")]
        ks = sorted({int(k) for v in res.values() for k in ((v or {}).get("summary") or {}).get("pass_at_curve", {})})
        if ks:
            rows, hi = [], None
            for name, v in res.items():
                if name == st:
                    hi = len(rows)
                cur = ((v or {}).get("summary") or {}).get("pass_at_curve") or {}
                rows.append([name, *[cur.get(str(k)) for k in ks]])
            out["tables"].append({"title": "pass@k curve (unbiased estimator)", "highlight": hi, "rows": rows,
                                  "columns": _cols(("set", "set", "text"), *[(f"k{k}", f"pass@{k}", "frac") for k in ks])})
        probs = (res.get(st) or {}).get("problems")
        if not isinstance(probs, list) or not probs:
            out["items"] = _none("This pass@k file stores only the summary.")
            return
        qs = self.questions(st)
        items = []
        for p in probs:
            q = qs.get(p.get("id"))
            items.append(_row(p.get("any_correct"), {"id": p.get("id"), "question": _short(q), "gold": p.get("gold"),
                                                     "correct": f"{p.get('n_correct')}/{p.get('k')}", "majority": p.get("majority_correct"),
                                                     "distinct": p.get("n_distinct_answers"), "tool": p.get("tool_use")},
                              [_blk("problem", q), _blk("pass@ per budget", p.get("pass_at"), "kv"),
                               _blk("grading", {"gold": p.get("gold"), "correct samples": p.get("n_correct"), "k": p.get("k"),
                                                "any correct": p.get("any_correct"), "majority correct": p.get("majority_correct"),
                                                "distinct answers": p.get("n_distinct_answers"), "tool use": p.get("tool_use")}, "kv")],
                              f"{p.get('id')} {q or ''}"))
        out["items"] = _items(items, _cols(("id", "id", "id"), ("question", "problem", "text"), ("gold", "gold", "text"), ("correct", "correct / k", "text"),
                                           ("majority", "majority right", "bool"), ("distinct", "distinct answers", "int"), ("tool", "tool use", "frac")),
                              {"pass": "solved by some sample", "fail": "no sample right"}, a,
                              "The k samples themselves were not saved: per problem the file keeps counts and flags."
                              + ("" if qs else " Problem text unavailable (the test parquet is not on this machine)."), src)

    # ------------------------------------------------------------------------------------------ swarm
    def _swarm(self, d: dict, col: dict, out: dict, a: dict, src: str) -> None:
        res = d.get("results") or {}
        st, m = col.get("set"), col.get("method")
        s = (res.get(st) or {}).get("summary") or {}
        out["summary"] += [_sv(col["label"], s.get(m), "frac"), _sv("problems", s.get("n"), "int"), _sv("k", s.get("k", d.get("k")), "int"),
                           _sv("temperature", d.get("temperature"), "num"), _sv("oracle", s.get("oracle"), "frac"),
                           _sv("oracle verified", s.get("oracle_verified"), "frac"), _sv("in selector prompt", s.get("in_prompt"), "frac"),
                           _sv("selector called tool", s.get("selector_called_tool"), "frac"), _sv("groups per k", s.get("mean_groups"), "num"),
                           _sv("verified per k", s.get("mean_verified"), "num"), _sv("seconds", s.get("seconds"), "num")]
        meth = ["greedy", "majority", "verified_majority", "selector", "oracle", "oracle_verified", "in_prompt"]
        rows, hi = [], None
        for name, v in res.items():
            if name == st:
                hi = len(rows)
            ss = (v or {}).get("summary") or {}
            rows.append([name, ss.get("n"), *[ss.get(x) for x in meth]])
        out["tables"].append({"title": "every method, every set in this file", "highlight": hi, "rows": rows,
                              "columns": _cols(("set", "set", "text"), ("n", "n", "int"), *[(x, x.replace("_", " "), "frac") for x in meth])})
        probs = (res.get(st) or {}).get("problems")
        if not isinstance(probs, list) or not probs:
            out["items"] = _none("This swarm file stores only the summary.")
            return
        qs = self.questions(st)
        items = []
        for p in probs:
            q = qs.get(p.get("id"))
            items.append(_row(p.get(m), {"id": p.get("id"), "question": _short(q), "gold": p.get("gold"), "greedy": p.get("greedy"),
                                         "majority": p.get("majority_answer"), "maj_ok": p.get("majority"), "vmaj": p.get("verified_majority"),
                                         "selector": p.get("selector_final"), "sel_ok": p.get("selector"), "oracle": p.get("oracle"),
                                         "groups": p.get("n_groups")},
                              [_blk("problem", q),
                               _blk("grading", {"gold": p.get("gold"), "greedy correct": p.get("greedy"), "majority answer": p.get("majority_answer"),
                                                "majority correct": p.get("majority"), "verified majority correct": p.get("verified_majority"),
                                                "selector answer": p.get("selector_final"), "selector correct": p.get("selector"),
                                                "selector called tool": p.get("selector_called_tool"), "oracle (a sample was right)": p.get("oracle"),
                                                "oracle verified": p.get("oracle_verified"), "right answer in selector prompt": p.get("in_prompt"),
                                                "answer groups": p.get("n_groups"), "verified samples": p.get("n_verified"), "seconds": p.get("seconds")}, "kv")],
                              f"{p.get('id')} {q or ''}"))
        out["items"] = _items(items, _cols(("id", "id", "id"), ("question", "problem", "text"), ("gold", "gold", "text"), ("greedy", "greedy", "bool"),
                                           ("majority", "majority answer", "text"), ("maj_ok", "maj.", "bool"), ("vmaj", "verified maj.", "bool"),
                                           ("selector", "selector answer", "text"), ("sel_ok", "sel.", "bool"), ("oracle", "oracle", "bool"),
                                           ("groups", "groups", "int")),
                              {"pass": f"{col['label']} right", "fail": f"{col['label']} wrong"}, a,
                              "The k candidates, their groups and the selector prompt were not saved: per problem the file keeps each "
                              "method's verdict and the majority / selector answers." + ("" if qs else " Problem text unavailable (the test parquet is not on this machine)."),
                              src)

    # ------------------------------------------------------------------------------------------ judged quality
    def _judged(self, run_dir: Path, ref: dict, col: dict, out: dict, a: dict) -> None:
        from slm.eval.quality import _ran_code, item_id, outputs_path
        from slm.eval.quality_suite import EXCLUDED_FROM_OVERALL, JUDGE_INSTRUCTIONS, TOOL_DEFENSIBLE

        q = self._load(run_dir / "quality" / "summary.json")
        q = q if isinstance(q, dict) else {}
        tok = ref.get("tokens")
        c = next((x for x in q.get("checkpoints") or [] if isinstance(x, dict) and x.get("tokens") == tok), {})
        rub = list(q.get("rubrics") or ["correctness", "coherence", "task"])
        out["file_fields"] = {"suite": q.get("suite"), "rubric": q.get("rubric"), "judges": ", ".join(q.get("judges") or []),
                              "judged checkpoint": c.get("checkpoint"), "tokens": tok, "stage": c.get("stage"),
                              "excluded from overall": ", ".join(q.get("excluded_from_overall") or [])}
        out["summary"] += [_sv("overall", c.get("overall"), "score"), *[_sv(r, c.get(r), "score") for r in rub],
                           _sv("scored", f"{c.get('n_scored')}/{c.get('n_items')}" if c.get("n_items") is not None else None),
                           _sv("overall incl. excluded", c.get("overall_all"), "score"), _sv("legacy 3 prompts", c.get("legacy3"), "score"),
                           _sv("stopped on their own", c.get("stopped_frac"), "frac"), _sv("tool misfire", c.get("tool_misfire"), "frac"),
                           _sv("misfire pool", c.get("tool_misfire_n"), "int"), _sv("misfired on", ", ".join(c.get("tool_misfire_cats") or []) or None),
                           _sv("sandbox errors", c.get("tool_errors"), "int")]
        cats = c.get("categories") or {}
        if cats:
            excl = set(q.get("excluded_from_overall") or [])
            out["tables"].append({"title": "per category (mean of the judge's 1-5 scores)", "rows": [
                [k + (" (excluded from overall)" if k in excl else ""), v.get("n"), v.get("overall"), *[v.get(r) for r in rub]] for k, v in cats.items()],
                "columns": _cols(("cat", "category", "text"), ("n", "n", "int"), ("overall", "overall", "score"), *[(r, r, "score") for r in rub])})
        out["rubric_text"] = JUDGE_INSTRUCTIONS if isinstance(JUDGE_INSTRUCTIONS, str) else None
        if tok is None:
            out["items"] = _none("The summary entry has no token count, so its outputs file cannot be found.")
            return
        op = outputs_path(run_dir, int(tok))
        if not op.is_file():
            out["items"] = _none(f"No outputs file for this checkpoint ({op.name}); only the summary numbers survive.")
            return
        lines = self._load(op, "jsonl")
        h = lines[0] if lines and lines[0].get("header") else {}
        prompts = [x for x in lines if not x.get("header")]
        sp = run_dir / "quality" / "scores.jsonl"
        by_id: dict[str, list] = {}
        if sp.is_file():
            for s in self._load(sp, "jsonl"):
                by_id.setdefault(s.get("item_id"), []).append(s)
        out["file_fields"].update({"generated": h.get("generated_at"), "device": h.get("device"), "outputs": f"quality/outputs/{op.name}"})
        misfire = col["key"] == "judged_misfire"
        items = []
        for it in prompts:
            iid = item_id(h.get("run", run_dir.name), h.get("tokens", tok), it.get("id"), h.get("suite", q.get("suite") or "v1"))
            recs = by_id.get(iid) or []
            sc = (recs[-1].get("scores") if recs else None) or {}
            vals = [sc[r] for r in rub if _num(sc.get(r)) is not None]
            mean = sum(vals) / len(vals) if vals else None
            cat = it.get("category")
            ran = _ran_code(it)
            if misfire:
                ok = None if cat in EXCLUDED_FROM_OVERALL or cat in TOOL_DEFENSIBLE else not ran
            else:
                cr = _num(sc.get("correctness"))
                ok = None if cr is None else True if cr >= 4 else False if cr <= 2 else None
            judges = [_blk(f"judge {s.get('judge') or '?'}" + (" (used)" if i == len(recs) - 1 and len(recs) > 1 else "")
                           + (f", {s.get('judged_at')}" if s.get("judged_at") else ""),
                           {**(s.get("scores") or {}), "note": s.get("note")}, "kv") for i, s in enumerate(recs)]
            items.append(_row(ok, {"id": it.get("id"), "category": cat + (" (excl.)" if cat in EXCLUDED_FROM_OVERALL else ""), "mode": it.get("mode"),
                                   "prompt": _short(it.get("prompt"), 90), "output": _short(it.get("output")), **{r: sc.get(r) for r in rub},
                                   "mean": mean, "ran": ran},
                              [_blk("prompt", it.get("prompt")), _blk("what a good answer contains (shown to the judge)", it.get("expect")),
                               _blk("think span", it.get("think"), "think"), _blk("model output (the judged answer)", it.get("output") or "(empty)"),
                               *(judges or [_blk("judge", "not judged yet")]),
                               _blk("generation", {"termination": it.get("termination"), "stopped": it.get("stopped"), "malformed": it.get("malformed"),
                                                   "tool calls": it.get("tool_calls"), "tool errors": it.get("tool_errors"), "ran code (misfire test)": ran,
                                                   "new tokens": f"{it.get('n_new')}/{it.get('max_new')}", "seconds": it.get("seconds")}, "kv")],
                              f"{it.get('id')} {cat} {it.get('prompt')} {it.get('output')} {(recs[-1].get('note') if recs else '') or ''}"))
        labels = ({"pass": "no tool call", "fail": "misfire (ran code)", "other": "exempt (arithmetic, excluded)"} if misfire else
                  {"pass": "correct (correctness 4-5)", "fail": "wrong (1-2)", "other": "partly right (3) / unjudged"})
        out["items"] = _items(items, _cols(("id", "prompt id", "id"), ("category", "category", "text"), ("mode", "mode", "text"),
                                           ("prompt", "prompt", "text"), ("output", "model output", "text"), *[(r, r[:5] + ".", "score") for r in rub],
                                           ("mean", "mean", "score"), ("ran", "ran code", "flag")),
                              labels, a, f"Greedy outputs from quality/outputs/{op.name}, joined with the judge's scores in quality/scores.jsonl by item id.",
                              f"{run_dir.name}/quality/outputs/{op.name}")


def _segment(lines: list[dict], per: dict) -> tuple[dict[str, list], str]:
    """Split a reasoning dump into the result file's tasks. The dump is written task by task in `per_task` order with n
    rows each, and arith2mul rows carry task "arith2", so consecutive blocks sized by n are the reliable split; the
    name match is the fallback when the sizes disagree (a dump from a different invocation)."""
    tasks = [(k, (v or {}).get("n")) for k, v in per.items() if isinstance(v, dict)]
    base = lambda k: k.removesuffix("_test")
    if tasks and all(isinstance(n, int) for _, n in tasks) and sum(n for _, n in tasks) == len(lines):
        out, i = {}, 0
        for k, n in tasks:
            blk = lines[i: i + n]
            i += n
            if not all(k == r.get("task") or k.startswith(str(r.get("task"))) or base(k) == r.get("task") for r in blk):
                break
            out[k] = blk
        else:
            return out, ""
    names = {r.get("task") for r in lines}
    out = {k: [r for r in lines if r.get("task") in (k, base(k))] for k, _ in tasks if k in names or base(k) in names}
    if "arith2mul" in per and "arith2mul" not in names:
        out.pop("arith2", None)  # arith2 and arith2mul rows both say "arith2": without the block sizes they cannot be told apart
    return out, "the dump's row count does not match the file's per-task n, and its task names do not separate this task"

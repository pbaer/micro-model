"""The Eval tab: services/evals.py on a synthetic runs tree, /api/evals, and the page's JS + cards."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from slm.eval.quality import item_id
from slm.portal.app import create_app
from slm.portal.services import eval_detail
from slm.portal.services.eval_detail import EvalDetail
from slm.portal.services.evals import COLUMNS, EvalIndex, effective_context
from slm.portal.settings import PortalSettings


def _w(p: Path, obj) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _run(root: Path, name: str, stage: str = "pretrain", config: dict | None = None, index: dict | None = None, n_params: int = 100) -> Path:
    d = root / name
    _w(d / "run.json", {"run_name": name, "stage": stage, "n_params": n_params, "config": config or {}})
    if index is not None:
        _w(d / "checkpoints" / "index.json", {k: {"tokens": v} for k, v in index.items()})
    return d


def _lm(ckpt: str, limit, hs: float, arc: float = 0.5) -> dict:
    return {"checkpoint": ckpt, "limit": limit, "results": {
        "hellaswag": {"sample_len": 2000, "acc,none": hs - 0.05, "acc_norm,none": hs},
        "arc_easy": {"sample_len": 2000, "acc,none": arc, "acc_norm,none": arc - 0.04}}}


def _needle(ckpt: str, mins: dict, haystack: str = "real") -> dict:
    return {"checkpoint": ckpt, "n": 16, "haystack": haystack, "depths": [0.0, 0.5, 1.0], "threshold": 0.8,
            "summary": {str(k): {"mean": v, "min": v} for k, v in mins.items()}, "effective_context": 0}


def make_tree(root: Path) -> Path:
    runs = root / "runs"
    base = _run(runs, "base_a", index={"final.pt": 1000, "best.pt": 1000, "latest.pt": 1000})
    _w(base / "lm_eval_limit2000.json", _lm("runs/base_a/checkpoints/final.pt", 2000, 0.30))
    _w(base / "lm_eval.json", _lm("runs/base_a/checkpoints/final.pt", None, 0.28))  # full set: lower priority, kept as "also measured"
    _w(base / "facts.json", {"checkpoint": "runs/base_a/checkpoints/final.pt", "accuracy": 0.4, "n": 194, "mode": "completion", "per_category": {"capitals": 0.6}})
    _w(base / "needle_v2.json", _needle("runs/base_a/checkpoints/final.pt", {1024: 0.9, 2048: 0.85, 4096: 0.5, 8000: 0.9}))
    _w(base / "needle_v2_filler.json", _needle("runs/base_a/checkpoints/final.pt", {1024: 1.0, 8000: 1.0}, haystack="filler"))
    _w(base / "needle.json", _needle("runs/base_a/checkpoints/final.pt", {8000: 1.0}))  # v1: superseded, never read

    sft = _run(runs, "chat_c", stage="sft", config={"init_from": "runs/base_a/checkpoints/final.pt", "data": {"kind": "sft", "mixture": {"chat": 1.0}}},
               index={"final.pt": 200})
    # renamed after the eval: the path inside names the old run; the file belongs to the directory that holds it
    _w(sft / "facts.json", {"checkpoint": "runs/old_name/checkpoints/final.pt", "accuracy": 0.5, "n": 194, "mode": "chat", "per_category": {}})

    rl = _run(runs, "rl_b", stage="grpo", config={"init_from": "runs/chat_c/checkpoints/final.pt", "group_size": 8, "tools": True},
              index={"best.pt": 50, "step_00050.pt": 50, "step_00100.pt": 80, "step_00025.pt": 20})
    _w(rl / "lm_eval_limit2000.json", _lm("runs/rl_b/checkpoints/best.pt", 2000, 0.35))
    _w(rl / "lm_eval_s100.json", _lm("runs/rl_b/checkpoints/step_00100.pt", 2000, 0.33))
    _w(rl / "multiturn.json", {"checkpoint": "runs/rl_b/checkpoints/best.pt", "summary": {"n": 64, "recall": 0.5, "format": 0.9, "misfire": 0.1}})
    _w(rl / "multiturn_s100.json", {"checkpoint": "runs/rl_b/checkpoints/step_00100.pt", "summary": {"n": 64, "recall": 0.6, "format": 0.8, "misfire": 0.0}})
    _w(rl / "reasoning_eval.json", {"checkpoint": "runs/rl_b/checkpoints/best.pt", "tools": True, "mean_accuracy": 0.5, "per_task": {
        "gsm8k_test": {"accuracy": 0.04, "n": 200, "tool_use_rate": 0.7}, "algebra": {"accuracy": 0.96, "n": 100, "tool_use_rate": 1.0}}})
    _w(rl / "reasoning_svamp.json", {"checkpoint": "runs/rl_b/checkpoints/best.pt", "tools": True, "mean_accuracy": 0.3, "per_task": {
        "gsm8k_test": {"accuracy": 0.9, "n": 200}, "svamp_test": {"accuracy": 0.09, "n": 300, "tool_use_rate": 0.8}}})
    _w(rl / "pass_at_k.json", {"checkpoint": "runs/rl_b/checkpoints/best.pt", "temperature": 0.8, "top_p": 0.95,
                               "results": {"gsm8k": {"summary": {"k": 32, "n_problems": 30, "pass_at_k": 0.4}}}})
    _w(rl / "swarm_eval_s100.json", {"checkpoint": "runs/rl_b/checkpoints/step_00100.pt", "k": 16, "temperature": 0.8, "results": {
        "svamp": {"summary": {"greedy": 0.1, "majority": 0.14, "verified_majority": 0.12, "selector": 0.1, "oracle": 0.58, "n": 50, "k": 16}}}})
    _w(rl / "quality" / "summary.json", {"run": "rl_b", "judges": ["judge-x"], "checkpoints": [
        {"checkpoint": "step_00025.pt", "tokens": 20, "overall": 2.0, "n_scored": 35, "n_items": 35},   # no row of its own
        {"checkpoint": "step_00050.pt", "tokens": 50, "overall": 3.5, "n_scored": 35, "n_items": 35, "tool_misfire": 0.2, "tool_misfire_n": 28},
        {"checkpoint": "step_00100.pt", "tokens": 80, "overall": 4.0, "n_scored": 35, "n_items": 35}]})

    judged = _run(runs, "judged_only", index={"snap_1K.pt": 1000, "final.pt": 2000}, n_params=50)
    _w(judged / "quality" / "summary.json", {"run": "judged_only", "judges": ["j"], "checkpoints": [
        {"checkpoint": "snap_1K.pt", "tokens": 1000, "overall": 1.5}, {"checkpoint": "final.pt", "tokens": 2000, "overall": 2.5}]})
    _run(runs, "no_evals", index={"final.pt": 5})
    return runs


def _row(tab, run, ckpt):
    return next(r for r in tab["rows"] if r["run"] == run and r["checkpoint"] == ckpt)


def test_effective_context_is_the_longest_passing_prefix():
    s = {"1024": {"min": 0.9}, "2048": {"min": 0.8}, "4096": {"min": 0.79}, "8000": {"min": 1.0}}
    assert effective_context(s, 0.8) == (2048, 4096, 0.79)
    assert effective_context({"512": {"min": 0.5}}, 0.8) == (0, 512, 0.5)
    assert effective_context({"512": {"min": 0.95}}, 0.8) == (512, None, None)


def test_eval_table_attribution_priority_and_colour(tmp_path):
    runs = make_tree(tmp_path)
    tab = EvalIndex(runs).table()
    assert [c["key"] for c in tab["columns"]] == [c["key"] for c in COLUMNS]
    assert all({"key", "label", "group", "higher_is_better"} <= set(c) for c in tab["columns"])
    got = [(r["run"], r["checkpoint"]) for r in tab["rows"]]
    # size (params) first, then stage base -> sft -> rl, then run name, then tokens
    assert got == [("base_a", "final.pt"), ("chat_c", "final.pt"), ("rl_b", "best.pt"), ("rl_b", "step_00100.pt"), ("judged_only", "final.pt")]

    base = _row(tab, "base_a", "final.pt")
    assert base["stage"] == "base" and base["aliases"] == ["best.pt"] and base["tokens"] == 1000
    hs = base["cells"]["hellaswag"]
    assert hs["value"] == pytest.approx(0.30) and hs["source"] == "base_a/lm_eval_limit2000.json"
    assert "limit=2000" in hs["detail"] and "base_a/lm_eval.json = 0.28" in hs["detail"]  # the full-set run is listed, not shown
    assert base["cells"]["arc_easy"]["value"] == pytest.approx(0.5)  # acc for ARC-Easy, acc_norm for HellaSwag
    assert base["cells"]["needle"]["value"] == 2048 and "first failing length 4096" in base["cells"]["needle"]["detail"]
    assert "the file says 0" in base["cells"]["needle"]["detail"]

    chat = _row(tab, "chat_c", "final.pt")
    assert chat["stage"] == "sft" and chat["tokens"] == 1200 and chat["own_tokens"] == 200
    assert chat["cells"]["facts"]["source"] == "chat_c/facts.json" and "runs/old_name/" in chat["cells"]["facts"]["detail"]

    best = _row(tab, "rl_b", "best.pt")
    assert best["stage"] == "rl" and best["aliases"] == ["step_00050.pt"] and best["tokens"] == 1250  # 1000 + 200 + 50
    assert best["cells"]["judged"]["value"] == 3.5 and best["cells"]["judged_misfire"]["value"] == 0.2  # step_00050.pt = best.pt
    assert best["cells"]["r_gsm8k"]["value"] == 0.04 and best["cells"]["r_gsm8k"]["source"] == "rl_b/reasoning_eval.json"
    assert best["cells"]["r_svamp"]["value"] == 0.09 and best["cells"]["r_svamp"]["source"] == "rl_b/reasoning_svamp.json"
    assert best["cells"]["r_mean"]["value"] == 0.5 and best["cells"]["r_tool_use"]["value"] == pytest.approx(0.85)
    assert best["cells"]["pk_gsm8k"]["value"] == 0.4 and "pk_svamp" not in best["cells"]
    s100 = _row(tab, "rl_b", "step_00100.pt")
    assert s100["cells"]["judged"]["value"] == 4.0 and s100["cells"]["sw_svamp_selector"]["value"] == 0.1
    assert "sw_gsm8k_selector" not in s100["cells"]
    assert not any(r["own_tokens"] == 20 for r in tab["rows"])  # a judged snapshot with no other eval is not a row
    only = _row(tab, "judged_only", "final.pt")
    assert set(only["cells"]) == {"judged"} and only["cells"]["judged"]["value"] == 2.5
    assert not any(r["run"] == "no_evals" for r in tab["rows"])

    # colour: 1 = best, 0 = worst, linear in between; lower-is-better columns flip; a single value is the midpoint
    col = {c["key"]: c for c in tab["columns"]}
    assert col["hellaswag"]["min"] == pytest.approx(0.30) and col["hellaswag"]["max"] == pytest.approx(0.35)
    assert best["cells"]["hellaswag"]["t"] == 1.0 and base["cells"]["hellaswag"]["t"] == 0.0
    assert s100["cells"]["hellaswag"]["t"] == pytest.approx(0.6)
    assert col["mt_misfire"]["higher_is_better"] is False
    assert s100["cells"]["mt_misfire"]["t"] == 1.0 and best["cells"]["mt_misfire"]["t"] == 0.0
    assert best["cells"]["pk_gsm8k"]["t"] == 0.5


def test_eval_table_cache_follows_mtime(tmp_path):
    runs = make_tree(tmp_path)
    idx = EvalIndex(runs)
    t1 = idx.table()
    assert idx.table() is t1  # nothing changed: cached
    p = _w(runs / "base_a" / "facts.json", {"checkpoint": "runs/base_a/checkpoints/final.pt", "accuracy": 0.45, "n": 194, "mode": "completion"})
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
    t2 = idx.table()
    assert t2 is not t1 and _row(t2, "base_a", "final.pt")["cells"]["facts"]["value"] == 0.45
    _w(runs / "chat_c" / "multiturn.json", {"checkpoint": "runs/chat_c/checkpoints/final.pt", "summary": {"recall": 0.7}})
    assert _row(idx.table(), "chat_c", "final.pt")["cells"]["mt_recall"]["value"] == 0.7  # a new file is picked up


def test_api_evals_endpoint_and_page(tmp_path):
    runs = make_tree(tmp_path)
    c = TestClient(create_app(PortalSettings(runs_root=runs, open_browser=False)))
    d = c.get("/api/evals").json()
    assert {"columns", "rows"} <= set(d) and len(d["rows"]) == 5
    assert _row(d, "rl_b", "best.pt")["cells"]["hellaswag"]["t"] == 1.0
    pages = [p["id"] for p in c.get("/api/meta").json()["pages"]]
    assert pages[0] == "home" and "evals" in pages
    for asset in ("/static/pages/evals.js", "/static/components/cards.js", "/static/app.js"):
        assert c.get(asset).status_code == 200, asset
    empty = TestClient(create_app(PortalSettings(runs_root=tmp_path / "nothing", open_browser=False))).get("/api/evals").json()
    assert empty["rows"] == [] and len(empty["columns"]) == len(COLUMNS)


def test_every_eval_column_has_an_info_card():
    """The page asks for `ev_<key>` (swarm columns share `ev_sw_<method>`, pass@k shares `ev_pass_at_k`); each must exist."""
    cards = Path("slm/portal/static/components/cards.js").read_text(encoding="utf-8")
    keys = set(re.findall(r"^  (ev_[a-z0-9_]+): \{", cards, re.M))
    want = {"ev_table", "ev_colour", "ev_sort", "ev_detail"}
    for c in COLUMNS:
        want.add("ev_sw_" + c["method"] if c["key"].startswith("sw_") else "ev_pass_at_k" if c["key"].startswith("pk_") else "ev_" + c["key"])
    assert want <= keys, sorted(want - keys)
    page = Path("slm/portal/static/pages/evals.js").read_text(encoding="utf-8")
    assert '"ev_sw_" + c.key.replace(/^sw_[a-z0-9]+_/, "")' in page and '"ev_pass_at_k"' in page


def test_evals_js_modules_parse():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    for f in ("slm/portal/static/pages/evals.js", "slm/portal/static/components/cards.js", "slm/portal/static/app.js"):
        tmp = Path(tempfile.gettempdir()) / "slm_check_evals.mjs"
        shutil.copy(f, tmp)
        r = subprocess.run([node, "--check", str(tmp)], capture_output=True, text=True)
        assert r.returncode == 0, f"{f}: {r.stderr[:400]}"


# ------------------------------------------------------------------------------------------------ detail pages
def _jsonl(p: Path, rows: list) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")


def _conv(fact: str, rec: bool, fmt: bool, mis: int) -> dict:
    asst = [{"answer": "ok", "think": "t", "terminated": True, "tool_calls": 0, "n_tokens": 3}] * 2
    asst.append({"answer": fact if rec else "no idea", "think": "", "terminated": fmt, "tool_calls": mis, "n_tokens": 4})
    return {"fact": fact, "turns": [f"My number is {fact}.", "Hi?", "What is my number?"], "assistant": asst,
            "recall": rec, "format_ok": fmt, "misfires": mis, "templated": False}


def make_detail_tree(root: Path) -> Path:
    """make_tree plus per-item data of every kind, added to rows that already exist (the table's rows do not change)."""
    runs = make_tree(root)
    base, rl = runs / "base_a", runs / "rl_b"
    _w(base / "facts.json", {"checkpoint": "runs/base_a/checkpoints/final.pt", "accuracy": 0.5, "n": 2, "mode": "completion", "per_category": {"capitals": 0.5},
                             "rows": [{"cat": "capitals", "completion": "The capital of France is", "question": "What is the capital of France?",
                                       "answers": "Paris", "output": "Paris.", "correct": True},
                                      {"cat": "capitals", "completion": "The capital of Peru is", "question": "What is the capital of Peru?",
                                       "answers": "Lima", "output": "Quito, a city", "correct": False}]})
    nd = _needle("runs/base_a/checkpoints/final.pt", {1024: 1.0, 2048: 0.5})
    nd["results"] = [{"length": 1024, "depth": 0.0, "accuracy": 1.0, "n": 16, "failures": []},
                     {"length": 2048, "depth": 0.0, "accuracy": 0.5, "n": 16, "failures": [{"gold": 123456, "out": "654321"}]}]
    _w(base / "needle_v2.json", nd)
    (base / "lm_eval_limit2000.log").write_text("  5%|#   | 1/20 [00:01<00:10]\nhellaswag {'acc,none': 0.25}\n", encoding="utf-8")
    # reasoning with a --dump companion: arith2mul rows carry task "arith2", so only the block sizes separate them
    _w(rl / "reasoning_s100.json", {"checkpoint": "runs/rl_b/checkpoints/step_00100.pt", "tools": True, "mean_accuracy": 0.75, "per_task": {
        "arith2": {"accuracy": 1.0, "n": 2, "tool_use_rate": 1.0}, "arith2mul": {"accuracy": 0.5, "n": 2, "tool_use_rate": 0.5}}})
    _jsonl(rl / "reasoning_dump_s100.jsonl", [
        {"task": "arith2", "id": f"arith2-{i}", "prompt": f"What is {i} {op} 3?", "gold": g, "text": f"<<x>>#### {a}", "answer": a, "correct": g == a,
         "malformed": False, "tool_calls": t, "tool_errors": 0, "answer_from_tool": bool(t), "tool_results": [["x", a]] if t else [], "n_tokens": 9}
        for i, op, g, a, t in [(1, "+", "4", "4", 1), (2, "-", "-1", "-1", 1), (3, "*", "9", "9", 1), (4, "*", "12", "11", 0)]])
    _w(rl / "multiturn.json", {"checkpoint": "runs/rl_b/checkpoints/best.pt", "summary": {"n": 3, "recall": 0.67, "format": 0.67, "misfire": 0.33},
                               "conversations": [_conv("7", True, True, 0), _conv("8", True, False, 1), _conv("9", False, True, 0)]})
    _w(rl / "pass_at_k.json", {"checkpoint": "runs/rl_b/checkpoints/best.pt", "temperature": 0.8, "top_p": 0.95, "results": {"gsm8k": {
        "summary": {"k": 4, "n_problems": 2, "pass_at_k": 0.5, "pass_at_curve": {"1": 0.25, "4": 0.5}},
        "problems": [{"id": "gsm8k-test-0", "gold": "18", "n_correct": 1, "k": 4, "any_correct": True, "pass_at": {"1": False, "4": True}},
                     {"id": "gsm8k-test-1", "gold": "3", "n_correct": 0, "k": 4, "any_correct": False, "pass_at": {"1": False, "4": False}}]}}})
    sw = json.loads((rl / "swarm_eval_s100.json").read_text(encoding="utf-8"))
    sw["results"]["svamp"]["problems"] = [
        {"id": "svamp-test-0", "gold": "27", "greedy": False, "majority": True, "verified_majority": True, "selector": True, "oracle": True,
         "majority_answer": "27", "selector_final": "27", "n_groups": 5},
        {"id": "svamp-test-1", "gold": "5", "greedy": False, "majority": False, "verified_majority": False, "selector": False, "oracle": True,
         "majority_answer": "4", "selector_final": "6", "n_groups": 9}]
    _w(rl / "swarm_eval_s100.json", sw)
    # judged suite: outputs for tokens 50 (= step_00050.pt = best.pt) joined with scores by item id
    q = rl / "quality"
    head = {"header": True, "run": "rl_b", "tokens": 50, "checkpoint": "step_00050.pt", "stage": "rl +tools", "suite": "v1", "device": "cpu"}
    _jsonl(q / "outputs" / f"{50:012d}.jsonl", [head,
        {"id": "cap_france", "category": "facts", "mode": "chat", "prompt": "What is the capital of France?", "expect": "Paris.", "output": "Paris.",
         "think": "", "tool_calls": 0, "termination": "stop"},
        {"id": "gold_symbol", "category": "facts", "mode": "chat", "prompt": "Symbol for gold?", "expect": "Au.", "output": "Ag", "think": "hmm",
         "tool_calls": 1, "termination": "stop"},
        {"id": "add", "category": "arithmetic", "mode": "chat", "prompt": "2+2?", "expect": "4", "output": "4", "tool_calls": 1}])

    def sc(pid: str, c: int, note: str, judge: str = "judge-x") -> dict:
        return {"item_id": item_id("rl_b", 50, pid, "v1"), "tokens": 50, "prompt_id": pid, "note": note, "judge": judge,
                "scores": {"correctness": c, "coherence": 5, "task": 4}}

    _jsonl(q / "scores.jsonl", [sc("cap_france", 2, "first pass", "judge-old"), sc("cap_france", 5, "right"), sc("gold_symbol", 1, "Ag is silver"), sc("add", 3, "ok")])
    summ = json.loads((q / "summary.json").read_text(encoding="utf-8"))
    summ["checkpoints"][1]["categories"] = {"facts": {"n": 2, "overall": 3.5, "correctness": 3.0}}
    _w(q / "summary.json", summ)
    return runs


def _detail_index(root: Path) -> EvalIndex:
    idx = EvalIndex(make_detail_tree(root))
    idx._detail = EvalDetail(idx)
    idx._detail.questions = lambda name: {f"{name}-test-0": f"{name} problem zero"}  # not the real parquet
    return idx


def test_detail_judged_joins_outputs_scores_and_aliases(tmp_path):
    idx = _detail_index(tmp_path)
    d = idx.detail("rl_b", "step_00050.pt", "judged")  # an alias of best.pt
    assert d["checkpoint"] == "best.pt" and d["requested"] == "step_00050.pt" and d["value"] == 3.5
    assert d["source"] == "rl_b/quality/summary.json" and d["sources"][0]["winner"] and d["tables"][0]["rows"][0][0] == "facts"
    it = d["items"]
    assert it["available"] and it["total"] == 3 and it["counts"] == {"pass": 1, "fail": 1, "other": 1}  # correctness 5 / 1 / 3
    rows = {r["c"]["id"]: r for r in it["rows"]}
    assert rows["cap_france"]["c"]["correctness"] == 5 and rows["cap_france"]["c"]["mean"] == pytest.approx(14 / 3)
    judges = [b for b in rows["cap_france"]["x"] if b["label"].startswith("judge")]
    assert len(judges) == 2 and judges[1]["value"]["note"] == "right" and "(used)" in judges[1]["label"]  # every judge record, the last one counts
    assert any(b["label"].startswith("what a good answer") and b["value"] == "Paris." for b in rows["cap_france"]["x"])
    assert d["rubric_text"]
    mis = idx.detail("rl_b", "best.pt", "judged_misfire")["items"]
    assert {r["c"]["id"]: r["ok"] for r in mis["rows"]} == {"cap_france": True, "gold_symbol": False, "add": None}  # arithmetic is exempt


def test_detail_reasoning_dump_is_split_by_block_size(tmp_path):
    idx = _detail_index(tmp_path)
    d = idx.detail("rl_b", "step_00100.pt", "r_arith2mul")
    assert d["source"] == "rl_b/reasoning_s100.json" and d["tables"][0]["highlight"] == 1
    it = d["items"]
    assert it["total"] == 2 and [r["c"]["prompt"] for r in it["rows"]] == ["What is 3 * 3?", "What is 4 * 3?"]
    assert it["counts"] == {"pass": 1, "fail": 1, "other": 0} and it["source"] == "rl_b/reasoning_dump_s100.jsonl"
    assert any(b["kind"] == "tools" for b in it["rows"][0]["x"])
    assert idx.detail("rl_b", "step_00100.pt", "r_mean")["items"]["total"] == 4  # the mean page lists every task
    tu = idx.detail("rl_b", "step_00100.pt", "r_tool_use")["items"]
    assert tu["labels"]["pass"] == "called the tool" and tu["counts"]["fail"] == 1
    nd = idx.detail("rl_b", "best.pt", "r_gsm8k")  # no dump beside the main file: aggregates only, and the page says why
    assert nd["items"]["available"] is False and "--dump" in nd["items"]["note"] and nd["tables"][0]["rows"]


def test_detail_per_item_kinds_and_paging(tmp_path):
    idx = _detail_index(tmp_path)
    f = idx.detail("base_a", "final.pt", "facts")["items"]
    assert f["total"] == 2 and f["rows"][0]["c"]["prompt"] == "The capital of France is"  # completion mode shows the completion form
    wrong = idx.detail("base_a", "final.pt", "facts", filter="fail")["items"]
    assert wrong["n_filtered"] == 1 and wrong["rows"][0]["c"]["answers"] == "Lima" and wrong["rows"][0]["i"] == 2
    assert idx.detail("base_a", "final.pt", "facts", q="QUITO")["items"]["n_filtered"] == 1
    page = idx.detail("base_a", "final.pt", "facts", offset=1, limit=1)["items"]
    assert page["offset"] == 1 and len(page["rows"]) == 1 and page["n_filtered"] == 2

    n = idx.detail("base_a", "final.pt", "needle")
    assert n["value"] == 1024 and n["items"]["counts"] == {"pass": 1, "fail": 1, "other": 0}
    assert "654321" in n["items"]["rows"][1]["x"][0]["value"]

    mt = {k: idx.detail("rl_b", "best.pt", k)["items"] for k in ("mt_recall", "mt_format", "mt_misfire")}
    assert [r["ok"] for r in mt["mt_recall"]["rows"]] == [True, True, False]
    assert [r["ok"] for r in mt["mt_format"]["rows"]] == [True, False, True]
    assert [r["ok"] for r in mt["mt_misfire"]["rows"]] == [True, False, True]
    assert sum(1 for b in mt["mt_recall"]["rows"][0]["x"] if b["label"].startswith("user")) == 3

    pk = idx.detail("rl_b", "best.pt", "pk_gsm8k")
    assert pk["items"]["rows"][0]["c"]["question"] == "gsm8k problem zero" and pk["items"]["rows"][0]["c"]["correct"] == "1/4"
    assert pk["items"]["counts"]["pass"] == 1 and "not saved" in pk["items"]["note"] and pk["tables"][0]["title"].startswith("pass@k curve")

    sw = idx.detail("rl_b", "step_00100.pt", "sw_svamp_selector")["items"]
    assert [r["ok"] for r in sw["rows"]] == [True, False] and sw["rows"][1]["c"]["selector"] == "6"
    assert [r["ok"] for r in idx.detail("rl_b", "step_00100.pt", "sw_svamp_oracle")["items"]["rows"]] == [True, True]


def test_detail_lm_eval_aggregate_only_sources_and_errors(tmp_path, monkeypatch):
    idx = _detail_index(tmp_path)
    d = idx.detail("base_a", "final.pt", "hellaswag")
    assert d["items"]["available"] is False and "--log_samples" in d["items"]["note"]
    assert [s["source"] for s in d["sources"]] == ["base_a/lm_eval_limit2000.json", "base_a/lm_eval.json"]  # winner first, then "also measured"
    t = d["tables"][0]
    assert [r[0] for r in t["rows"]] == ["hellaswag", "arc_easy"] and t["highlight"] == 0
    assert d["log"]["file"] == "lm_eval_limit2000.log" and d["log"]["lines"] == ["hellaswag {'acc,none': 0.25}"]  # progress bars dropped
    alt = idx.detail("base_a", "final.pt", "hellaswag", file="base_a/lm_eval.json")
    assert alt["value"] == pytest.approx(0.28) and alt["sources"][1]["shown"] and alt["file_fields"]["limit"] is None
    assert d["run_info"]["stage"] == "pretrain"
    for args in [("base_a", "final.pt", "nope"), ("base_a", "missing.pt", "facts"), ("base_a", "final.pt", "r_gsm8k"), ("no_evals", "final.pt", "facts")]:
        with pytest.raises(KeyError):
            idx.detail(*args)
    with pytest.raises(KeyError):
        idx.detail("base_a", "final.pt", "hellaswag", file="base_a/facts.json")  # only files that measured the cell
    monkeypatch.setattr(eval_detail, "MAX_FILE_BYTES", 10)
    big = idx.detail("base_a", "final.pt", "facts")
    assert big["items"]["available"] is False and "above the" in big["notes"][0]


def test_detail_cache_follows_mtime(tmp_path):
    idx = _detail_index(tmp_path)
    p = idx.root / "base_a" / "facts.json"
    det = idx._detail
    a = det._load(p)
    assert det._load(p) is a
    d = json.loads(p.read_text(encoding="utf-8"))
    d["rows"] = d["rows"][:1]
    p.write_text(json.dumps(d), encoding="utf-8")
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
    assert idx.detail("base_a", "final.pt", "facts")["items"]["total"] == 1


def test_api_eval_detail_endpoint(tmp_path):
    runs = make_detail_tree(tmp_path)
    c = TestClient(create_app(PortalSettings(runs_root=runs, open_browser=False)))
    r = c.get("/api/evals/detail", params={"run": "rl_b", "checkpoint": "best.pt", "key": "judged", "filter": "fail", "limit": 5})
    assert r.status_code == 200
    d = r.json()
    assert d["items"]["n_filtered"] == 1 and d["items"]["rows"][0]["c"]["id"] == "gold_symbol" and "_s" not in d["items"]["rows"][0]
    assert c.get("/api/evals/detail", params={"run": "rl_b", "checkpoint": "best.pt", "key": "facts"}).status_code == 404  # an n/a cell
    assert c.get("/api/evals/detail", params={"run": "rl_b", "checkpoint": "best.pt", "key": "zzz"}).status_code == 404
    assert c.get("/api/evals/detail", params={"run": "../x", "checkpoint": "best.pt", "key": "judged"}).status_code == 404
    big = c.get("/api/evals/detail", params={"run": "rl_b", "checkpoint": "best.pt", "key": "judged", "limit": 10_000}).json()
    assert big["items"]["limit"] == eval_detail.PAGE_MAX
    page = Path("slm/portal/static/pages/evals.js").read_text(encoding="utf-8")
    assert "/api/evals/detail?" in page and "export function EvalDetailPage" in page and "export function sortRows" in page
    assert "EvalDetailPage" in Path("slm/portal/static/app.js").read_text(encoding="utf-8")

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

from slm.portal.app import create_app
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
    want = {"ev_table", "ev_colour"}
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

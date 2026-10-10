"""The Overview's run lineage graph: layoutLineage / parentOf / relatives in pages/home.js, run under node on a
realistic forest (the project's own shape: two base families, a stub, a run whose parent is gone, an orphan)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

RUNS = [
    {"run_name": "m2_base_149m", "init_from": "", "started": "2026-09-10 10:00:00", "stage": "pretrain"},
    {"run_name": "m3_base_stable_149m", "init_from": "runs/m2_base_149m/checkpoints/snap_800M.pt", "started": "2026-09-11 10:00:00", "stage": "pretrain"},
    {"run_name": "m4_sft_149m", "init_from": "runs/m3_base_stable_149m/checkpoints/final.pt", "started": "2026-09-12 10:00:00", "stage": "sft"},
    {"run_name": "m6_rl_149m", "init_from": "runs\\m4_sft_149m\\checkpoints\\final.pt", "started": "2026-09-13 10:00:00", "stage": "grpo", "is_rl": True},
    {"run_name": "m8_base_stable_336m", "init_from": "", "started": "2026-09-16 10:00:00", "stage": "pretrain"},
    {"run_name": "m8_base_4k_336m", "init_from": "C:/x/runs/m8_base_stable_336m/checkpoints/final.pt", "started": "2026-09-18 10:00:00", "stage": "pretrain"},
    {"run_name": "m9_sft_336m", "init_from": "runs/m8_base_4k_336m/checkpoints/final.pt", "started": "2026-09-20 10:00:00", "stage": "sft"},
    {"run_name": "m10_sft_336m", "init_from": "runs/m8_base_4k_336m/checkpoints/latest.pt", "started": "2026-10-01 10:00:00", "stage": "sft"},
    {"run_name": "orphan_run", "init_from": "runs/deleted_parent/checkpoints/final.pt", "started": "2026-09-05 10:00:00", "stage": "sft"},
    {"run_name": "m1_tinystories_26m", "init_from": "", "started": "2026-09-01 10:00:00", "stage": "pretrain"},
]

SCRIPT = r"""
import { layoutLineage, parentOf, relatives, GEOM } from "./home_under_test.mjs";
const runs = JSON.parse(process.argv[2]);
const L = layoutLineage(runs);
const at = Object.fromEntries(L.nodes.map((n) => [n.id, n]));
const rel = [...relatives("m8_base_4k_336m", L.parent, L.kids)].sort();
console.log(JSON.stringify({
  nodes: L.nodes.map((n) => ({ id: n.id, ghost: n.ghost, depth: n.depth, x: n.x, y: n.y })),
  edges: L.edges.map((e) => [e.from, e.to, e.file]), W: L.W, H: L.H, geom: GEOM, rel,
  parse: [parentOf("runs/a/checkpoints/final.pt"), parentOf("D:\\r\\runs\\b\\checkpoints\\snap_1K.pt"), parentOf(""), parentOf("weird.pt")],
}));
"""


def _run_layout(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    src = Path("slm/portal/static/pages/home.js").read_text(encoding="utf-8")
    # the module imports preact / htm through the import map; stub them so the pure layout functions load under node
    for name in ("preact", "preact/hooks", "htm"):
        src = src.replace(f'from "{name}";', f'from "./stub_{name.replace("/", "_")}.mjs";')
    src = src.replace('from "../components/util.js";', 'from "./stub_util.mjs";').replace('from "../components/info.js";', 'from "./stub_info.mjs";')
    (tmp_path / "home_under_test.mjs").write_text(src, encoding="utf-8")
    (tmp_path / "stub_preact.mjs").write_text("export const h = () => null;\n", encoding="utf-8")
    (tmp_path / "stub_preact_hooks.mjs").write_text("export const useEffect = () => {}, useLayoutEffect = () => {}, useMemo = (f) => f(), useRef = () => ({}), useState = (v) => [v, () => {}];\n", encoding="utf-8")
    (tmp_path / "stub_htm.mjs").write_text("export default { bind: () => () => null };\n", encoding="utf-8")
    (tmp_path / "stub_util.mjs").write_text("export const api = () => Promise.resolve(), fmtTok = String, fmtDur = String, fmtNum = String, fmtInt = String;\n", encoding="utf-8")
    (tmp_path / "stub_info.mjs").write_text("export const Info = () => null;\n", encoding="utf-8")
    (tmp_path / "t.mjs").write_text(SCRIPT, encoding="utf-8")
    r = subprocess.run([node, str(tmp_path / "t.mjs"), json.dumps(RUNS)], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-800:]
    return json.loads(r.stdout)


def test_lineage_layout_is_a_left_to_right_forest(tmp_path):
    out = _run_layout(tmp_path)
    at = {n["id"]: n for n in out["nodes"]}
    g = out["geom"]
    # one node per run plus a ghost for the parent that is not in the list
    assert len(at) == len(RUNS) + 1 and at["ghost:deleted_parent"]["ghost"] is True
    # every init_from becomes an edge, whatever the path form (relative, absolute, backslashes)
    edges = {(a, b): f for a, b, f in out["edges"]}
    assert edges[("m2_base_149m", "m3_base_stable_149m")] == "snap_800M.pt"
    assert edges[("m4_sft_149m", "m6_rl_149m")] == "final.pt"
    assert edges[("m8_base_stable_336m", "m8_base_4k_336m")] == "final.pt"
    assert edges[("ghost:deleted_parent", "orphan_run")] == "final.pt"
    assert len(edges) == sum(1 for r in RUNS if r["init_from"]) == 7
    # a child sits one column right of its parent; a parent is centred on its children
    for (a, b) in edges:
        assert at[b]["depth"] == at[a]["depth"] + 1 and at[b]["x"] > at[a]["x"]
    kids = [at["m9_sft_336m"]["y"], at["m10_sft_336m"]["y"]]
    assert at["m8_base_4k_336m"]["y"] == pytest.approx(sum(kids) / 2)
    # newest family first: the 336M tree (latest start 2026-10-01) is above the 149M tree, the 26M run last
    assert at["m8_base_stable_336m"]["y"] < at["m2_base_149m"]["y"] < at["m1_tinystories_26m"]["y"]
    # siblings oldest first
    assert at["m9_sft_336m"]["y"] < at["m10_sft_336m"]["y"]
    # no two nodes overlap
    boxes = sorted((n["x"], n["y"]) for n in out["nodes"])
    for i, (x, y) in enumerate(boxes):
        for (x2, y2) in boxes[i + 1:]:
            assert x2 >= x + g["nodeW"] or abs(y2 - y) >= g["nodeH"], f"overlap at {(x, y)} / {(x2, y2)}"
    assert out["W"] >= max(n["x"] for n in out["nodes"]) + g["nodeW"] and out["H"] >= max(n["y"] for n in out["nodes"]) + g["nodeH"]


def test_lineage_relatives_and_parent_parsing(tmp_path):
    out = _run_layout(tmp_path)
    assert out["rel"] == sorted(["m8_base_4k_336m", "m8_base_stable_336m", "m9_sft_336m", "m10_sft_336m"])
    a, b, empty, weird = out["parse"]
    assert a == {"run": "a", "file": "final.pt", "raw": "runs/a/checkpoints/final.pt"}
    assert b["run"] == "b" and b["file"] == "snap_1K.pt"
    assert empty is None and weird["run"] is None and weird["file"] == "weird.pt"


def test_overview_page_wiring():
    src = Path("slm/portal/static/pages/home.js").read_text(encoding="utf-8")
    assert "LineageGraph" in src and 'href=${"#/runs/" + encodeURIComponent(r.run_name)}' in src, "a node is a link to its run page"
    assert "HoverCard" in src and 'k="run_graph"' in src
    cards = Path("slm/portal/static/components/cards.js").read_text(encoding="utf-8")
    assert "  run_graph: {" in cards

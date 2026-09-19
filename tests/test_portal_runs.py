import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from slm.portal.app import create_app
from slm.portal.settings import PortalSettings
from slm.utils import metrics as M
from slm.utils.logging import MetricsLogger


def make_run(root: Path, name: str, n: int = 20, finished: bool = False) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"run_name": name, "stage": "pretrain", "n_params": 123, "config": {"schedule": {"total_tokens": 100 * n * 2}, "milestone_tokens": 500}, "env": {"gpu": "g", "git_commit": "abcdef12"}}))
    lg = MetricsLogger(d)
    lg.log("start", msg="go")
    for i in range(1, n + 1):
        lg.log("train", tokens=i * 100, update=i, loss=5 - i * 0.05, lr=1e-4, grad_norm=1.0, tok_s=1000.0, tok_s_ema=1000.0, step_ms=100.0, fwd_ms=30.0, bwd_ms=60.0, opt_ms=10.0, data_ms=1.0, vram_gib=3.0, elapsed_s=i, eta_s=5)
        if i % 5 == 0:
            lg.log("eval", tokens=i * 100, update=i, val_loss=4.9 - i * 0.05, val_ppl=100.0, best=True, eval_s=1)
            lg.log("milestone", tokens=i * 100, update=i, segment_s=5.0, elapsed_s=i, tok_s=1000.0, loss=4.6, val_loss=4.5)
    lg.log("checkpoint", tokens=n * 100, msg="latest.pt saved")
    if finished:
        lg.log("finish", tokens=n * 100, msg="done")
    lg.close()
    ck = d / "checkpoints"
    ck.mkdir()
    for f in ("latest.pt", "best.pt", "snap_1K.pt", "snap_1p50K.pt"):
        (ck / f).write_bytes(b"x" * 10)
    (d / "samples").mkdir()
    (d / "samples" / f"{1000:012d}.txt").write_text("tokens=1K update=10 t\n" + "=" * 80 + "\nPROMPT: 'Once'\n--- greedy:\nupon a time\n--- sampled:\nthere was\n", encoding="utf-8")
    return d


def test_jsonl_tail_partial_line(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"kind":"train","time":1,"tokens":1}\n{"kind":"tr', encoding="utf-8")
    t = M.JsonlTail(p)
    assert len(t.refresh()) == 1
    with open(p, "a", encoding="utf-8") as f:
        f.write('ain","time":2,"tokens":2}\n')
    assert [r["tokens"] for r in t.refresh()] == [2]
    assert t.refresh() == []


def test_snapshot_name_parse():
    assert M.parse_snapshot_tokens("snap_400M.pt") == 400_000_000
    assert M.parse_snapshot_tokens("snap_1p20B.pt") == 1_200_000_000
    assert M.parse_snapshot_tokens("best.pt") is None


def test_api_runs(tmp_path):
    make_run(tmp_path / "runs", "alpha", finished=True)
    make_run(tmp_path / "runs", "beta")
    app = create_app(PortalSettings(runs_root=tmp_path / "runs", open_browser=False))
    c = TestClient(app)
    runs = c.get("/api/runs").json()
    assert {r["run_name"] for r in runs} == {"alpha", "beta"}
    a = next(r for r in runs if r["run_name"] == "alpha")
    assert a["status"] == "finished" and a["tokens"] == 2000 and a["best_val"] < 4.0
    d = c.get("/api/runs/alpha").json()
    assert d["summary"]["progress"] == 0.5 and d["config"]["schedule"]["total_tokens"] == 4000
    s = c.get("/api/runs/alpha/series?max_points=5").json()
    assert len(s["train"]["tokens"]) <= 6 and len(s["eval"]["val_loss"]) == 4 and len(s["milestones"]) == 4
    ck = c.get("/api/runs/alpha/checkpoints").json()
    kinds = {x["name"]: x for x in ck}
    assert kinds["snap_1K.pt"]["tokens"] == 1000 and kinds["snap_1K.pt"]["val_loss"] is not None
    assert kinds["snap_1p50K.pt"]["tokens"] == 1500 and kinds["best.pt"]["kind"] == "best" and kinds["latest.pt"]["tokens"] == 2000
    smp = c.get("/api/runs/alpha/samples").json()
    assert smp == [{"tokens": 1000, "path": smp[0]["path"]}]
    one = c.get("/api/runs/alpha/samples/1000").json()
    assert one["items"][0]["greedy"] == "upon a time" and one["items"][0]["sampled"] == "there was"
    assert c.get("/api/runs/nope").status_code == 404
    assert c.get("/api/meta").json()["pages"][0]["id"] == "home"
    assert c.get("/").status_code == 200 and "app.js" in c.get("/").text
    for asset in ("/static/app.js", "/static/pages/runs.js", "/static/components/chart.js", "/static/vendor/preact.mjs", "/static/vendor/uplot.iife.min.js"):
        assert c.get(asset).status_code == 200, asset


def test_portal_js_modules_parse():
    """Every ES module in the portal must parse (a broken template literal blanks the whole app)."""
    import shutil
    import subprocess
    from pathlib import Path

    node = shutil.which("node")
    if not node:
        import pytest

        pytest.skip("node not installed")
    static = Path("slm/portal/static")
    for f in list(static.glob("*.js")) + list((static / "pages").glob("*.js")) + list((static / "components").glob("*.js")):
        tmp = Path(__import__("tempfile").gettempdir()) / "slm_check.mjs"
        shutil.copy(f, tmp)
        r = subprocess.run([node, "--check", str(tmp)], capture_output=True, text=True)
        assert r.returncode == 0, f"{f}: {r.stderr[:400]}"


def test_tokenized_split_notices_replaced_shards(tmp_path):
    """A data swap replaces shard files under the same names; the cached document index must follow."""
    import time

    import numpy as np

    from slm.portal.services.datasets import TokenizedSplit

    d = tmp_path / "src" / "train"
    d.mkdir(parents=True)
    np.arange(1000, dtype=np.uint16).tofile(d / "shard_00000.bin")
    np.save(d / "shard_00000.idx.npy", np.array([0, 100, 500], dtype=np.int64))
    ts = TokenizedSplit(d)
    assert ts.shards()[0]["docs"] == 3
    time.sleep(0.05)
    np.arange(4000, dtype=np.uint16).tofile(d / "shard_00000.bin")
    np.save(d / "shard_00000.idx.npy", np.array([0, 10, 20, 30, 40], dtype=np.int64))
    assert ts.shards()[0]["docs"] == 5 and ts.shards()[0]["tokens"] == 4000 and len(ts.idx(0)) == 5


def _world(root):
    """Minimal data root with one manifest of each of the four dialects plus one SFT set."""
    import numpy as np

    def tokset(name, m, tokens=200):
        d = root / "tokenized" / "v1" / name
        (d / "train").mkdir(parents=True)
        np.arange(tokens, dtype=np.uint16).tofile(d / "train" / "shard_00000.bin")
        np.save(d / "train" / "shard_00000.idx.npy", np.array([0], dtype=np.int64))
        (d / "manifest.json").write_text(json.dumps(m))

    tokset("prose", {"source": "fineweb-edu", "name": "prose", "kind": "prose", "min_doc_tokens": 16, "train_tokens": 1000, "val_tokens": 10, "files": ["a.parquet"]})
    tokset("chatstream", {"source": "chatstream", "from_sft": ["setA", "setB"], "train": {"tokens": 500, "docs": 7, "shards": 1}, "val": {"tokens": 5, "docs": 1, "shards": 1}})
    tokset("synth", {"name": "synth-v2", "train_tokens": 300, "train_docs": 3, "val_tokens": 3, "backdrop": str(root / "tokenized" / "v1" / "prose" / "train"), "seed": 7, "min_len": 512, "max_len": 4096})
    d = root / "sft" / "v1" / "setA"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"source": "gsm8k", "name": "setA", "max_len": 2048, "think_required": True, "tools": True,
                                                 "train_examples": 10, "train_tokens": 400, "train_targets": 200, "val_examples": 2, "val_tokens": 40}))
    return root


def test_manifest_normalizer_and_recipe(tmp_path):
    from slm.portal.services.datasets import DataCatalog, normalize_manifest

    root = _world(tmp_path / "data")
    cat = DataCatalog(root, tmp_path / "cache")
    p = cat.prepared("v1")
    assert p["prose"]["made_by"] == "prepare" and p["prose"]["train_tokens"] == 1000 and p["prose"]["from"] == ["fineweb-edu"]
    assert p["chatstream"]["made_by"] == "sft_to_pretrain" and p["chatstream"]["train_tokens"] == 500 and p["chatstream"]["from"] == ["setA", "setB"]
    assert p["synth"]["made_by"] == "synth_retrieval" and p["synth"]["from"] == ["prose"]  # never "?" (the writer stores no `source`)
    assert p["sft:setA"]["made_by"] == "sft" and p["sft:setA"]["train_tokens"] == 400 and p["sft:setA"]["train_docs"] == 10
    assert normalize_manifest("x", {"name": "x", "tasks": ["arith1"], "train_examples": 3, "train_tokens": 30})["made_by"] == "rl.synth"

    tok_root = str(root / "tokenized" / "v1")
    cfg = {"data": {"kind": "pretrain", "tokenized_root": tok_root, "mixture": {"prose": 0.5, "chatstream": 0.5}, "seq_len": 64,
                    "extra_val_mixture": {"prose": 1.0}}, "schedule": {"total_tokens": 2000}}
    r = cat.recipe(cfg, "pretrain")
    rows = {x["source"]: x for x in r["rows"]}
    assert rows["chatstream"]["available_tokens"] == 500 and abs(rows["chatstream"]["epochs"] - 2.0) < 1e-9  # the `train: {tokens}` dialect used to read 0
    assert "SFT sets in chat format" in rows["chatstream"]["provenance"]
    assert [x["source"] for x in r["extra_val"]] == ["prose"] and r["extra_val"][0]["val_tokens"] == 10

    sft_cfg = {"data": {"kind": "sft", "tokenized_root": tok_root, "sft_root": str(root / "sft" / "v1"), "mixture": {"setA": 1.0}, "seq_len": 64},
               "schedule": {"total_tokens": 0, "epochs": 2.0}}
    rs = cat.recipe(sft_cfg, "sft")
    assert rs["stage"] == "sft" and rs["total_tokens"] == 800 and rs["rows"][0]["available_tokens"] == 400  # epochs resolved like the trainer does


def test_recipe_api_handles_rl_yaml(tmp_path):
    """RL yaml has unknown keys for TrainConfig; the recipe endpoint must return the prompt panel, not 500."""
    make_run(tmp_path / "runs", "alpha", finished=True)
    root = _world(tmp_path / "data")
    cfgs = tmp_path / "configs" / "train"
    cfgs.mkdir(parents=True)
    (cfgs / "rl_x.yaml").write_text("\n".join(["run_name: rl_x", "tasks: [gsm8k, gsm8k, word]", "reward_scheme: tool", "tools: true", "total_steps: 7"]))
    tok_root = str(root / "tokenized" / "v1").replace("\\", "/")
    (cfgs / "alpha.yaml").write_text("\n".join(["run_name: alpha", "data: {tokenized_root: '%s', mixture: {prose: 1.0}}" % tok_root]))
    app = create_app(PortalSettings(runs_root=tmp_path / "runs", data_root=root, configs_root=tmp_path / "configs", cache_dir=tmp_path / "cache", open_browser=False))
    c = TestClient(app)
    rl = c.get("/api/data/recipe", params={"id": "config:%s" % (cfgs / "rl_x.yaml")}).json()
    assert rl["stage"] == "rl" and rl["rl"]["task_weights"]["gsm8k"] == 2 / 3 and "Python call" in rl["rl"]["reward_rule"]
    rs = c.get("/api/data/recipes").json()
    ids = {x["id"]: x for x in rs}
    assert any(k.startswith("config:") and k.endswith("rl_x.yaml") for k in ids)
    assert ids["run:alpha"]["kind"] == "run" and ids["run:alpha"]["plan_differs"] is True  # run.json config != alpha.yaml
    assert c.get("/api/data/recipe", params={"id": "run:nope"}).status_code == 404
    assert c.get("/api/data/recipe", params={"id": "bogus"}).status_code == 400


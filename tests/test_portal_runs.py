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


def _tokenizer(root):
    """A real 400-piece tokenizer under <data_root>/tokenizer/v1, which the registry discovers by tag."""
    from slm.data.tokenizer import SlmTokenizer, train_bpe

    d = root / "tokenizer" / "v1"
    if not (d / "tokenizer.json").exists():
        SlmTokenizer(train_bpe(["the cat sat on the mat " * 80, "1 + 2 = 3 " * 80], vocab_size=400)).save(d)
    return d


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



def test_sft_split_window_carries_mask_and_example_bounds(tmp_path):
    """The packed SFT training row (tokens + mask + example starts) has no reader outside the trainer."""
    import numpy as np

    from slm.portal.services.datasets import SftSplit

    d = tmp_path / "set" / "train"
    d.mkdir(parents=True)
    np.arange(300, dtype=np.uint16).tofile(d / "tokens_00000.bin")
    np.concatenate([np.zeros(50, np.uint8), np.ones(50, np.uint8), np.zeros(200, np.uint8)]).tofile(d / "mask_00000.bin")
    np.save(d / "idx_00000.npy", np.array([0, 100, 220], dtype=np.int64))
    s = SftSplit(d)
    assert s.shards() == [{"shard": 0, "name": "tokens_00000.bin", "tokens": 300, "examples": 3}]  # tokens, not bytes
    ex = s.example(0, 1)
    assert ex["start"] == 100 and ex["length"] == 120 and ex["n_target"] == 0
    assert s.example(0, 0)["n_target"] == 50 and s.example(0, 2)["length"] == 80  # last example runs to the end of the shard
    w = s.window(0, 80, 160)
    assert w["example_starts"] == [20, 140] and w["n_target"] == 20 and w["shard_tokens"] == 300
    assert s.stats()["targets"] == 50 and abs(s.stats()["target_share"] - 50 / 300) < 1e-9


def test_portal_main_process_is_torch_free():
    """Importing the app must not pull torch (and with it CUDA init) into the portal process: a live
    training run owns the GPU. Every torch-touching route imports inside the handler.

    Checked in a subprocess: this session has torch imported already by other tests.
    """
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-c", "import sys; import slm.portal.app; print('torch' in sys.modules)"],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stdout


def test_data_tab_shows_a_filtered_source(tmp_path):
    """gutenberg-pg19: registry fields (kind, license) plus the manifest's filter block reach the source page."""
    import numpy as np

    root = _world(tmp_path / "data")
    d = root / "tokenized" / "v1" / "gutenberg-pg19"
    (d / "train").mkdir(parents=True)
    np.arange(300, dtype=np.uint16).tofile(d / "train" / "shard_00000.bin")
    np.save(d / "train" / "shard_00000.idx.npy", np.array([0, 150], dtype=np.int64))
    flt = {"books_in": {"train": 10, "validation": 1, "test": 1}, "selected": {"train": 2, "val": 1}, "candidates": 5, "dialogue_cutoff": 0.28,
           "config": {"target_tokens": 300}, "rules": [{"rule": "date", "threshold": 1850, "text": "t", "removed": 3, "removed_val": 0, "fails": 3, "examples": []}],
           "kept_examples": ["A Novel (1901)"], "val_rule": "PG-19 val"}
    (d / "manifest.json").write_text(json.dumps({"source": "gutenberg-pg19", "name": "gutenberg-pg19", "kind": "prose", "train_tokens": 300, "train_docs": 2,
                                                 "val_tokens": 0, "files": ["a.parquet"], "books": {"train": 2, "val": 1, "mean_tokens_per_book": 150}, "filter": flt}))
    app = create_app(PortalSettings(runs_root=tmp_path / "runs", data_root=root, configs_root=tmp_path / "configs", cache_dir=tmp_path / "cache", open_browser=False))
    c = TestClient(app)
    s = c.get("/api/data/source/gutenberg-pg19").json()
    assert s["kind"] == "prose" and "public-domain" in s["license"] and s["repo"] == "deepmind/pg19"
    p = s["prepared"]["tokenized:v1"]
    assert p["train_tokens"] == 300 and p["manifest"]["filter"]["dialogue_cutoff"] == 0.28
    assert "filtered to 2 + 1 val of 12 books" in p["provenance"]
    ov = {x["name"]: x for x in c.get("/api/data/sources").json()["sources"]}
    assert ov["gutenberg-pg19"]["prepared"]["v1"]["manifest"]["books"]["train"] == 2


def test_trace_of_a_book_reads_the_recorded_verdict(tmp_path):
    from slm.data.tokenizer import SlmTokenizer
    from slm.portal.api.data import _trace_book

    tok = SlmTokenizer.load(_tokenizer(tmp_path))
    bj = tmp_path / "books.jsonl"
    base = {"title": "T", "year": 1900, "words": 5000, "dialogue": 0.3, "caps_share": 0.0, "verse_share": 0.0, "archaic_per_1k": 0.0, "tokens": 900}
    bj.write_text("\n".join(json.dumps({**base, "book_id": i, "selected": s, "verdict": v, "split": "train", "segments": 1})
                            for i, s, v in [("1", True, None), ("2", False, "verse")]), encoding="utf-8")

    class Reg:
        def pieces(self, tag, ids):
            return [{"id": i} for i in ids]

    kept = _trace_book(bj, {"book_id": "1", "text": "the cat sat on the mat\n\nthe cat"}, tok, Reg(), "v1")
    assert kept["kept"] and kept["split"] == "train" and kept["ids"][0] == tok.bos_id and "900 tokens in 1 documents" in kept["reason"]
    dropped = _trace_book(bj, {"book_id": "2", "text": "x"}, tok, Reg(), "v1")
    assert not dropped["kept"] and "'verse'" in dropped["reason"]
    assert "not prepared" in _trace_book(bj, {"book_id": "3"}, tok, Reg(), "v1")["reason"]


def test_chain_walks_init_from_back_to_the_root(tmp_path):
    """The chain is what the weights saw: this run's tokens plus the checkpoint each parent was cut at."""
    runs = tmp_path / "runs"
    make_run(runs, "root", n=20, finished=True)
    make_run(runs, "mid", n=10, finished=True)
    make_run(runs, "leaf", n=5, finished=True)
    for child, parent, ckpt in (("mid", "root", "snap_1K.pt"), ("leaf", "mid", "latest.pt")):
        meta = json.loads((runs / child / "run.json").read_text())
        meta["config"]["init_from"] = f"runs/{parent}/checkpoints/{ckpt}"
        (runs / child / "run.json").write_text(json.dumps(meta))
    (runs / "root" / "checkpoints" / "index.json").write_text(json.dumps({"snap_1K.pt": {"tokens": 1000}}))
    from slm.portal.services.runs import RunIndex

    idx = RunIndex(runs)
    ch = idx.chain("leaf")
    assert [c["run"] for c in ch] == ["root", "mid", "leaf"]
    assert ch[0]["tokens_used"] == 1000 and ch[0]["checkpoint"] == "snap_1K.pt"  # exact count from index.json
    assert ch[1]["tokens_used"] == 1000 and ch[2]["tokens_used"] == 500  # mid has no index entry -> its own total
    assert sum(c["tokens_used"] for c in ch) == next(s["cumulative_tokens"] for s in idx.summaries() if s["run_name"] == "leaf")
    assert idx.chain("root") == [c for c in idx.chain("root")] and len(idx.chain("root")) == 1


def test_rollouts_reader_drops_logprobs_and_lists_steps(tmp_path):
    """The RL "training row" is the model's own sample; the per-token logprob arrays never reach the page."""
    d = tmp_path / "runs" / "rl"
    (d / "rollouts").mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"run_name": "rl", "stage": "grpo", "config": {"tokenizer_dir": "nope"}}))
    for step, n in ((1, 2), (7, 3)):
        (d / "rollouts" / f"step_{step:05d}.jsonl").write_text("\n".join(
            json.dumps({"prompt_id": f"p{i}", "task": "gsm8k", "prompt": "q", "gold": "1", "prompt_ids": [1, 2], "completion_ids": [3],
                        "text": "a", "parsed": "1", "reward": 1.0, "malformed": False, "n_tokens": 1,
                        "old_logprobs": [0.1] * 3, "ref_logprobs": [0.2] * 3}) for i in range(n)) + "\n")
    from slm.portal.services.runs import RunReader

    r = RunReader(d)
    assert r.rollout_steps() == [1, 7]
    out = r.rollouts()  # no step -> the latest
    assert out["step"] == 7 and out["n"] == 3 and len(out["rollouts"]) == 3
    assert all("old_logprobs" not in x and "ref_logprobs" not in x for x in out["rollouts"])
    assert r.rollouts(step=1, offset=1, limit=5)["rollouts"][0]["prompt_id"] == "p1"
    assert RunReader(tmp_path / "runs" / "none").rollouts()["rollouts"] == []


def test_rl_prompts_endpoint_renders_the_generation_prompt(tmp_path):
    """make_tasks + format_chat(add_generation_prompt=True) is what the rollouts start from, torch-free."""
    make_run(tmp_path / "runs", "alpha", finished=True)
    root = _world(tmp_path / "data")
    tok_dir = _tokenizer(root)
    cfgs = tmp_path / "configs" / "train"
    cfgs.mkdir(parents=True)
    (cfgs / "rl_y.yaml").write_text("\n".join([
        "run_name: rl_y", "tasks: [arith1, arith2]", "n_train_prompts: 12", "n_heldout_prompts: 4",
        "think_required: true", f"tokenizer_dir: '{str(tok_dir).replace(chr(92), '/')}'"]))
    app = create_app(PortalSettings(runs_root=tmp_path / "runs", data_root=root, configs_root=tmp_path / "configs", cache_dir=tmp_path / "cache", open_browser=False))
    c = TestClient(app)
    rid = "config:%s" % (cfgs / "rl_y.yaml")
    d = c.get("/api/data/rl/prompts", params={"id": rid, "split": "train", "limit": 3}).json()
    assert d["n"] == 12 and len(d["prompts"]) == 3
    p = d["prompts"][0]
    text = "".join(x["piece"] for x in p["pieces"])
    assert text.startswith("<|bos|><|user|>") and text.endswith("<|assistant|><|think|>"), text[-60:]
    assert p["gold"] and p["task"] in ("arith1", "arith2")
    assert c.get("/api/data/rl/prompts", params={"id": rid, "split": "heldout"}).json()["n"] == 4
    # the prompt list is seeded, so two calls agree; train and held-out never share a prompt
    again = c.get("/api/data/rl/prompts", params={"id": rid, "split": "train", "limit": 3}).json()
    assert [x["prompt_id"] for x in again["prompts"]] == [x["prompt_id"] for x in d["prompts"]]
    held = {x["prompt_id"] for x in c.get("/api/data/rl/prompts", params={"id": rid, "split": "heldout", "limit": 4}).json()["prompts"]}
    assert not held & {x["prompt_id"] for x in d["prompts"]}
    assert c.get("/api/data/rl/prompts", params={"id": "config:%s" % (cfgs / "nope.yaml")}).status_code == 404


# ------------------------------------------------------------------- text view: reserved tokens as chips
def test_text_runs_keep_specials_from_the_ids(tmp_path):
    """The text view's runs come from the ids: real specials split, a literal "<|user|>" stays text, the mask rides along."""
    from slm.data.chat import format_chat
    from slm.data.tokenizer import SlmTokenizer
    from slm.portal.services.tokenizer import text_runs

    tok = SlmTokenizer.load(_tokenizer(tmp_path))
    ids = tok.encode_document("the cat <|user|> sat café")
    runs = text_runs(tok, ids)
    assert runs == [{"special": "<|bos|>", "i": 0}, {"text": "the cat <|user|> sat café"}, {"special": "<|eos|>", "i": len(ids) - 1}]
    enc = format_chat(tok, [{"role": "user", "content": "1 + 2"}, {"role": "assistant", "think": "3", "content": "#### 3"}], think_required=True)
    runs = text_runs(tok, enc.ids, enc.loss_mask)
    assert "".join(r.get("special") or r["text"] for r in runs) == tok.decode(enc.ids)  # nothing dropped, nothing added
    specials = [(r["special"], r["loss"]) for r in runs if "special" in r]
    assert specials[:4] == [("<|bos|>", False), ("<|user|>", False), ("<|end|>", False), ("<|assistant|>", False)]
    assert ("<|think|>", True) in specials and ("<|/think|>", True) in specials and specials[-1] == ("<|eos|>", False)
    assert [r for r in runs if r.get("text") == "#### 3"][0]["loss"] is True and [r for r in runs if r.get("text") == "1 + 2"][0]["loss"] is False
    assert all(r["i"] < len(enc.ids) and enc.ids[r["i"]] == tok.special(r["special"]) for r in runs if "special" in r)


def test_data_endpoints_return_runs_and_the_first_document(tmp_path):
    """doc / window / tokenizer encode carry `runs`; /api/tokenizers lists the registered specials; doc 0 is servable."""
    import numpy as np

    from slm.data.tokenizer import SPECIAL_TOKENS, SlmTokenizer, apply_named

    root = _world(tmp_path / "data")
    tok = SlmTokenizer.load(_tokenizer(root))
    docs = [tok.encode_document("the cat sat on the mat"), tok.encode_document("1 + 2 = 3")]
    d = root / "tokenized" / "v1" / "prose" / "train"
    np.array(docs[0] + docs[1], dtype=np.uint16).tofile(d / "shard_00000.bin")
    np.save(d / "shard_00000.idx.npy", np.array([0, len(docs[0])], dtype=np.int64))
    app = create_app(PortalSettings(runs_root=tmp_path / "runs", data_root=root, configs_root=tmp_path / "configs", cache_dir=tmp_path / "cache", open_browser=False))
    c = TestClient(app)
    tags = c.get("/api/tokenizers").json()
    assert tags[0]["specials"] == apply_named(SPECIAL_TOKENS) and "<|python_call|>" in tags[0]["specials"]
    base = "/api/data/tokenized/v1/prose/train"
    lst = c.get(f"{base}/docs", params={"shard": 0}).json()
    first = lst["docs"][0]["doc"]  # what the browser opens when the source is selected
    doc = c.get(f"{base}/doc", params={"shard": 0, "doc": first}).json()
    assert doc["runs"] == [{"special": "<|bos|>", "i": 0}, {"text": "the cat sat on the mat"}, {"special": "<|eos|>", "i": len(docs[0]) - 1}]
    assert doc["text"] == "the cat sat on the mat"  # the plain decode (specials skipped) is unchanged
    win = c.get(f"{base}/window", params={"shard": 0, "start": 0, "length": 64}).json()
    assert [r["special"] for r in win["runs"] if "special" in r] == ["<|bos|>", "<|eos|>", "<|bos|>", "<|eos|>"]
    assert {r["i"] for r in win["runs"] if r.get("special") == "<|bos|>"} == set(win["doc_starts"])  # boundary outline lines up
    raw = c.post("/api/tokenizers/v1/encode", json={"mode": "raw", "text": "a <|user|> b"}).json()
    assert raw["runs"] == [{"text": "a <|user|> b"}], "raw text never produces a reserved token"
    chat = c.post("/api/tokenizers/v1/encode", json={"mode": "chat", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}).json()
    assert [r["special"] for r in chat["runs"] if "special" in r][:2] == ["<|bos|>", "<|user|>"] and all("loss" in r for r in chat["runs"])


def test_split_specials_matches_the_registered_list_only(tmp_path):
    """components/specials.js splits plain text on the tokenizer's actual special strings and nothing else."""
    import shutil
    import subprocess

    import pytest

    from slm.data.chat import format_chat
    from slm.data.tokenizer import SPECIAL_TOKENS, SlmTokenizer, apply_named
    from slm.portal.services.tokenizer import text_runs

    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    specials = apply_named(SPECIAL_TOKENS)
    tok = SlmTokenizer.load(_tokenizer(tmp_path))
    enc = format_chat(tok, [{"role": "user", "content": "2 + 3"}, {"role": "assistant", "think": "5", "content": "#### 5"}], think_required=True)
    cases = {
        "chat": "<|bos|><|user|>hi<|end|><|assistant|><|think|>x<|python_call|>print(1)<|/python_call|><|python_result|>1<|/python_result|><|/think|>#### 1<|end|><|eos|>",
        "lookalikes": "a <|notatoken|> b <| user |> c <|USER|> d <|user| e |user|> f <|",
        "reserved": "<|reserved_1|><|reserved_10|><|reserved_47|>",
        "mixed": "x<|eos|><|eos|>y",
        "empty": "",
        "formatted": tok.decode(enc.ids),
    }
    mod = Path("slm/portal/static/components/specials.js").resolve().as_uri()
    script = tmp_path / "split.mjs"
    script.write_text(
        f'import {{ splitSpecials, joinRuns }} from "{mod}";\n'
        f"const specials = {json.dumps(specials)};\nconst cases = {json.dumps(cases)};\nconst out = {{}};\n"
        "for (const [k, v] of Object.entries(cases)) { const r = splitSpecials(v, specials); out[k] = { runs: r, back: joinRuns(r) === v }; }\n"
        'out.nolist = splitSpecials("<|bos|>a", []);\n'
        "console.log(JSON.stringify(out));\n", encoding="utf-8")
    r = subprocess.run([node, str(script)], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr[:600]
    out = json.loads(r.stdout)
    assert all(v["back"] for k, v in out.items() if k != "nolist"), "splitting must be lossless"

    def sp(k):
        return [x["special"] for x in out[k]["runs"] if "special" in x]

    assert sp("chat") == ["<|bos|>", "<|user|>", "<|end|>", "<|assistant|>", "<|think|>", "<|python_call|>", "<|/python_call|>",
                          "<|python_result|>", "<|/python_result|>", "<|/think|>", "<|end|>", "<|eos|>"]
    assert out["lookalikes"]["runs"] == [{"text": cases["lookalikes"]}], "unregistered <|...|> shapes stay text"
    assert sp("reserved") == ["<|reserved_1|>", "<|reserved_10|>", "<|reserved_47|>"] and all(s in specials for s in sp("reserved"))
    assert out["mixed"]["runs"] == [{"text": "x"}, {"special": "<|eos|>"}, {"special": "<|eos|>"}, {"text": "y"}]
    assert out["empty"]["runs"] == [] and out["nolist"] == [{"text": "<|bos|>a"}]
    # on formatted chat text the client split agrees with the server's id-based runs
    assert out["formatted"]["runs"] == [{k: v for k, v in x.items() if k in ("text", "special")} for x in text_runs(tok, enc.ids)]


def test_text_view_is_the_default_everywhere():
    """Every text / tokens toggle in the portal starts on text, and both data browsers open the first document."""
    import re

    static = Path("slm/portal/static")
    found = {}
    for f in list((static / "pages").glob("*.js")) + list((static / "components").glob("*.js")):
        for m in re.finditer(r"const \[(mode|view), set\w+\] = useState\(\"(\w+)\"\)", f.read_text(encoding="utf-8")):
            found.setdefault(f.name, []).append((m.group(1), m.group(2)))
    flat = [(f, k, v) for f, xs in found.items() for k, v in xs]
    assert not [x for x in flat if x[2] in ("tokens", "ids")], flat
    assert [v for k, v in found["data.js"] if k == "mode"] == ["text"] * 4  # inspector, RL prompts, rollouts, source browser
    assert ("view", "text") in found["tokenizer.js"] and ("view", "text") in found["model.js"]
    data = (static / "pages" / "data.js").read_text(encoding="utf-8")
    assert "openTok(docs.docs[0].doc)" in data and "openRaw(file, rg, page.docs[0].row)" in data
    assert "const first = (isSft ? list.examples : list.docs)[0];" in data
    # the text views go through the shared component, never a hand-rolled special marker
    assert "boundary-mark" not in data and "isSpecial" not in (static / "pages" / "model.js").read_text(encoding="utf-8")
    for name in ("data.js", "model.js", "tokenizer.js", "evals.js", "runs.js"):
        assert "TextWithSpecials" in (static / "pages" / name).read_text(encoding="utf-8"), name
    assert "TextWithSpecials" in (static / "components" / "swarm.js").read_text(encoding="utf-8")

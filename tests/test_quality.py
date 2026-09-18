"""Judged-quality eval: suite integrity, checkpoint selection, blind packets, score validation and ingest,
the per-checkpoint summary, and the portal endpoints that serve it. CPU only, no model."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from slm.eval import quality as Q
from slm.eval.quality_suite import CATEGORIES, JUDGE_INSTRUCTIONS, RUBRICS, SUITE
from slm.portal.app import PortalSettings, create_app


def test_suite_is_well_formed():
    ids = [p["id"] for p in SUITE]
    assert len(ids) == len(set(ids)) and len(SUITE) >= 30
    for p in SUITE:
        assert p["category"] in CATEGORIES and p["completion"] and p["chat"] and p["expect"] and 16 <= p["max_new_tokens"] <= 128
    assert all(k in JUDGE_INSTRUCTIONS for k in RUBRICS) and "1 to 5" in JUDGE_INSTRUCTIONS
    assert all(k in {p["id"] for p in SUITE} for k in Q.LEGACY3)


def _fake_run(root: Path, name: str = "r", stage: str = "pretrain") -> Path:
    d = root / name
    (d / "checkpoints").mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"run_name": name, "stage": stage, "config": {"data": {"kind": "pretrain"}, "schedule": {"total_tokens": 1000}}}), encoding="utf-8")
    idx = {"latest.pt": {"kind": "latest", "tokens": 400}, "best.pt": {"kind": "best", "tokens": 300},
           "snap_100.pt": {"kind": "snapshot", "tokens": 100}, "snap_200.pt": {"kind": "snapshot", "tokens": 200}, "snap_300.pt": {"kind": "snapshot", "tokens": 300},
           "final.pt": {"kind": "final", "tokens": 400}}
    for n in idx:
        (d / "checkpoints" / n).write_bytes(b"x")
    (d / "checkpoints" / "index.json").write_text(json.dumps(idx), encoding="utf-8")
    return d


def _write_outputs(run_dir: Path, tokens: int, outputs: dict[str, str]) -> None:
    p = Q.outputs_path(run_dir, tokens)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"header": True, "run": run_dir.name, "tokens": tokens, "checkpoint": f"snap_{tokens}.pt", "stage": "base", "suite": "v1", "device": "cpu",
                            "n_items": len(SUITE), "generated_at": "t", "seconds": 1.0, "n_new_total": 10}) + "\n")
        for sp in SUITE:
            f.write(json.dumps({"id": sp["id"], "category": sp["category"], "mode": "completion", "prompt": sp["completion"], "expect": sp["expect"],
                                "output": outputs.get(sp["id"], "…"), "n_new": 5, "max_new": 32, "seconds": 0.1, "stopped": False}) + "\n")


def test_checkpoint_selection(tmp_path):
    d = _fake_run(tmp_path)
    assert Q.selected_checkpoints(d, None, 1) == [("snap_100.pt", 100), ("snap_200.pt", 200), ("snap_300.pt", 300), ("final.pt", 400)]
    assert Q.selected_checkpoints(d, None, 2) == [("snap_100.pt", 100), ("snap_300.pt", 300), ("final.pt", 400)]  # every 2nd, last snapshot always kept
    assert Q.selected_checkpoints(d, None, 1, include_final=False)[-1] == ("snap_300.pt", 300)
    assert Q.selected_checkpoints(d, ["best.pt"], 1) == [("best.pt", 300)]
    (d / "checkpoints" / "snap_200.pt").unlink()
    assert ("snap_200.pt", 200) not in Q.selected_checkpoints(d, None, 1)  # deleted files are skipped


def test_pack_ingest_summary_roundtrip(tmp_path, capsys):
    d = _fake_run(tmp_path)
    _write_outputs(d, 100, {"cap_france": " Paris is a city of France France France"})
    _write_outputs(d, 200, {"cap_france": " Paris."})
    ids = {Q.item_id("r", t, sp["id"]) for t in (100, 200) for sp in SUITE}
    assert len(ids) == 2 * len(SUITE) and all(len(i) == 12 for i in ids)
    items = Q.unjudged_items(d)
    # 34 prompts have byte-identical outputs at both checkpoints and are packed once; cap_france differs -> 2
    assert len(items) == len(SUITE) + 1 and set(items[0]) == {"item_id", "category", "mode", "prompt", "output", "expect"}
    # pack: blind, shuffled, batched, carries the rubric
    Q.cmd_pack(_ns(runs_root=tmp_path, run="r", batch=20, seed=1))
    packets = sorted((d / "quality" / "packets").glob("r-*.json"))
    assert len(packets) == 2
    pk = json.loads(packets[0].read_text(encoding="utf-8"))
    assert pk["instructions"] == JUDGE_INSTRUCTIONS and len(pk["items"]) == 20 and "tokens" not in pk["items"][0] and "checkpoint" not in pk["items"][0]
    # a judge answers: score everything in the two packets, plus one bogus row and one out-of-range row
    rows = []
    for p in packets:
        for it in json.loads(p.read_text(encoding="utf-8"))["items"]:
            good = it["output"].strip().startswith("Paris.")
            rows.append({"item_id": it["item_id"], "correctness": 5 if good else 3, "coherence": 5 if good else 2, "task": 4, "note": "ok"})
    rows.append({"item_id": "deadbeef0000", "correctness": 5, "coherence": 5, "task": 5})
    rows.append({"item_id": rows[0]["item_id"], "correctness": 9, "coherence": 5, "task": 5})
    sf = tmp_path / "scores.json"
    sf.write_text(json.dumps(rows), encoding="utf-8")
    Q.cmd_ingest(_ns(runs_root=tmp_path, run="r", scores=[str(sf)], judge="test-judge", replace=False))
    out = capsys.readouterr().out
    assert "rejected 2" in out and f"ingested {len(SUITE) + 1} scores" in out and f"{len(SUITE) - 1} copied to identical outputs" in out
    # re-ingesting the same file adds nothing
    Q.cmd_ingest(_ns(runs_root=tmp_path, run="r", scores=[str(sf)], judge="test-judge", replace=False))
    assert "ingested 0 scores" in capsys.readouterr().out
    assert not Q.unjudged_items(d)
    copied = [r for r in Q.read_scores(d).values() if r.get("copied_from")]
    assert len(copied) == len(SUITE) - 1 and all(Q.read_scores(d)[c["copied_from"]]["scores"] == c["scores"] for c in copied)
    s = json.loads((d / "quality" / "summary.json").read_text(encoding="utf-8"))
    assert s["judges"] == ["test-judge"] and [c["tokens"] for c in s["checkpoints"]] == [100, 200]
    c100, c200 = s["checkpoints"]
    assert c100["n_scored"] == len(SUITE) and c200["overall"] > c100["overall"]  # the better cap_france answer lifts the mean
    assert c200["categories"]["facts"]["overall"] > c100["categories"]["facts"]["overall"] and c200["categories"]["python"]["overall"] == c100["categories"]["python"]["overall"]
    assert c100["legacy3_n"] == 3 and c100["legacy3"] is not None
    det = Q.checkpoint_detail(d, 200)
    it = next(i for i in det["items"] if i["id"] == "cap_france")
    assert it["scores"] == {"correctness": 5, "coherence": 5, "task": 4} and it["judge"] == "test-judge"


def test_portal_serves_quality(tmp_path):
    d = _fake_run(tmp_path / "runs", "q")
    from slm.utils.logging import MetricsLogger

    lg = MetricsLogger(d)
    lg.log("start", msg="go")
    lg.log("train", tokens=100, update=1, loss=3.0, lr=1e-4, grad_norm=1.0, tok_s=1.0, tok_s_ema=1.0, step_ms=1.0, fwd_ms=0, bwd_ms=0, opt_ms=0, data_ms=0, vram_gib=1.0, elapsed_s=1, eta_s=1)
    lg.close()
    app = create_app(PortalSettings(runs_root=tmp_path / "runs", open_browser=False))
    c = TestClient(app)
    assert c.get("/api/runs/q/quality").json() == {"run": "q", "checkpoints": []}
    assert c.get("/api/runs/q/quality/100").status_code == 404
    _write_outputs(d, 100, {})
    Q.write_summary(d)
    q = c.get("/api/runs/q/quality").json()
    assert q["checkpoints"][0]["tokens"] == 100 and q["checkpoints"][0]["overall"] is None and q["rubrics"] == list(RUBRICS)
    det = c.get("/api/runs/q/quality/100").json()
    assert len(det["items"]) == len(SUITE) and det["items"][0]["scores"] is None and det["header"]["checkpoint"] == "snap_100.pt"


def _ns(**kw):
    import argparse

    return argparse.Namespace(**kw)


def test_generate_suite_on_tiny_model_both_modes(tmp_path):
    """Completion mode for base checkpoints, chat mode (with and without a forced think span) for SFT/RL ones;
    CPU, untrained tiny model, so only shapes and bookkeeping are checked."""
    import torch

    from slm.config import ModelConfig, load_config
    from slm.data.tokenizer import SlmTokenizer, train_bpe
    from slm.model import Transformer

    tok = SlmTokenizer(train_bpe(["The capital of France is Paris. def fibonacci(n): return n " * 40, "once upon a time there lived a small village by the sea " * 40], vocab_size=400))
    cfg = load_config(ModelConfig, "configs/model/tiny.yaml")
    cfg.vocab_size, cfg.max_seq_len = tok.vocab_size, 512
    torch.manual_seed(0)
    model = Transformer(cfg).eval()
    base = Q.generate_suite(model, tok, "base", "cpu", max_new_cap=8)
    assert len(base) == len(SUITE) and all(i["mode"] == "completion" and 0 < i["n_new"] <= 8 and "output" in i for i in base)
    assert base[0]["prompt"] == SUITE[0]["completion"] and "think" not in base[0]
    sft = Q.generate_suite(model, tok, "sft", "cpu", max_new_cap=8)
    assert all(i["mode"] == "chat" and i["prompt"] == p["chat"] and "malformed" in i for i, p in zip(sft, SUITE))
    rl = Q.generate_suite(model, tok, "rl", "cpu", max_new_cap=8)
    assert all(i["n_new"] <= 8 + 64 for i in rl)  # room for the think span
    p = Q.write_outputs(tmp_path, "tiny", 123, "snap_123.pt", "base", "cpu", base, 0.5)
    h, items = Q.read_outputs(p)
    assert h["tokens"] == 123 and h["n_items"] == len(SUITE) and len(items) == len(SUITE)
    assert json.loads((tmp_path / "quality" / "summary.json").read_text(encoding="utf-8"))["checkpoints"][0]["n_scored"] == 0

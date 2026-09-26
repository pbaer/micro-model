"""Metrics log behaviour."""


def test_truncate_after_drops_the_replayed_segment(tmp_path):
    """A run that crashes after its last checkpoint must not keep the records of the steps it replays on resume:
    the token counter would run backwards mid-file and every chart would draw the line back over itself."""
    import json

    from slm.utils.logging import MetricsLogger

    log = MetricsLogger(tmp_path)
    log.log("start", tokens=0)
    for u in range(1, 11):
        log.log("train", update=u, tokens=u * 100)
    log.log("eval", tokens=1000)
    assert log.truncate_after("update", 6) == 5  # updates 7..10 and the eval after them
    log.log("resume", tokens=600)
    log.log("train", update=7, tokens=700)
    log.close()
    recs = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    updates = [r["update"] for r in recs if r["kind"] == "train"]
    assert updates == [1, 2, 3, 4, 5, 6, 7] and [r["kind"] for r in recs][-2:] == ["resume", "train"]
    assert all(b["tokens"] >= a["tokens"] for a, b in zip(recs, recs[1:]) if "tokens" in a and "tokens" in b)


def test_repair_keeps_the_last_occurrence(tmp_path):
    import json

    from slm.utils.logging import MetricsLogger

    p = tmp_path / "metrics.jsonl"
    rows = [{"kind": "train", "update": u, "tokens": u * 100, "seg": "crashed"} for u in range(1, 6)]
    rows += [{"kind": "resume"}] + [{"kind": "train", "update": u, "tokens": u * 100, "seg": "replayed"} for u in range(3, 7)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert MetricsLogger.repair(p, "update") == 3  # crashed 3, 4, 5 are superseded by the replay
    recs = [json.loads(l) for l in p.read_text().splitlines()]
    assert [r["update"] for r in recs if r["kind"] == "train"] == [1, 2, 3, 4, 5, 6]
    assert all(r["seg"] == "replayed" for r in recs if r.get("update", 0) >= 3)

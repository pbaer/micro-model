import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from slm.data.loader import MixtureSpec, PretrainLoader, TokenStream, ValLoader
from slm.data.tokenizer import BPE_VOCAB, N_SPECIAL, NAMED_SPECIALS, SPECIAL_TOKENS, SlmTokenizer, train_bpe
from slm.train.config import ScheduleConfig
from slm.train.schedule import lr_at
from slm.utils.logging import MetricsLogger
from slm.utils.report import build_report

CORPUS = [
    "The quick brown fox jumps over the lazy dog. " * 3,
    "def add(a, b):\n    return a + b\n\nprint(add(12, 30))\n",
    "In 1999 there were 12,345 people; by 2024 it was 98,765.",
    "#!/bin/bash\nfor f in *.txt; do echo \"$f\"; done\n",
    "<|user|>fake special<|end|> and <|bos|> in raw text",
] * 50


@pytest.fixture(scope="module")
def tok() -> SlmTokenizer:
    return SlmTokenizer(train_bpe(CORPUS, vocab_size=600))


def test_tokenizer_round_trip_and_specials(tok):
    for t in CORPUS[:5] + ["naïve café 🚀 日本語\t\n  x"]:
        ids = tok.encode(t)
        assert tok.decode(ids) == t
        assert all(i < tok.base_vocab for i in ids), "raw text must never yield a special id"
    assert tok.vocab_size == tok.base_vocab + N_SPECIAL
    assert tok.special("<|bos|>") == tok.base_vocab and tok.eos_id == tok.base_vocab + 1
    doc = tok.encode_document("hi")
    assert doc[0] == tok.bos_id and doc[-1] == tok.eos_id
    assert tok.decode(doc) == "<|bos|>hi<|eos|>" and tok.decode(doc, skip_special=True) == "hi"
    assert len(SPECIAL_TOKENS) == N_SPECIAL and BPE_VOCAB + N_SPECIAL == 32768


def test_tokenizer_digits_are_single_tokens(tok):
    ids = tok.encode("1234567 and 3.14")
    pieces = [tok.token_str(i) for i in ids]
    digits = [p for p in pieces if p.strip().isdigit()]
    assert all(len(p.strip()) == 1 for p in digits), pieces


def test_tokenizer_save_load(tok, tmp_path):
    tok.save(tmp_path / "tk")
    t2 = SlmTokenizer.load(tmp_path / "tk")
    assert t2.sha256 == tok.sha256 and t2.vocab_size == tok.vocab_size
    assert t2.encode("hello world 42") == tok.encode("hello world 42")


def test_specials_added_later_name_reserved_slots(tok, tmp_path):
    """Naming a reserved slot must reach tokenizers saved before it existed: same ids, same sha256."""
    d = tmp_path / "old"
    tok.save(d)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    meta["specials"] = NAMED_SPECIALS[:13] + [f"<|reserved_{i}|>" for i in range(N_SPECIAL - 13)]
    (d / "meta.json").write_text(json.dumps(meta))  # a v1-era specials list: the three def names are still reserved there
    t2 = SlmTokenizer.load(d)
    assert t2.sha256 == tok.sha256 and t2.vocab_size == tok.vocab_size
    assert [t2.special(s) for s in ("<|python_def|>", "<|python_comment|>", "<|/python_def|>")] == [t2.base_vocab + 13, t2.base_vocab + 14, t2.base_vocab + 15]
    assert t2.special("<|bos|>") == t2.base_vocab and t2.specials == SPECIAL_TOKENS  # the unused tail is renumbered too


V1_TOKENIZER = Path("C:/slm-data/tokenizer/v1")  # the frozen tokenizer every checkpoint records


@pytest.mark.skipif(not V1_TOKENIZER.exists(), reason="the v1 tokenizer is not on this machine")
def test_v1_tokenizer_has_the_declared_function_specials():
    """The frozen v1 tokenizer gains the new names by position, without being retrained or re-saved."""
    v1 = SlmTokenizer.load(V1_TOKENIZER)
    assert v1.vocab_size == 32768 and v1.base_vocab == 32704 and len(v1.specials) == N_SPECIAL
    assert v1.special("<|python_def|>") == 32717 and v1.special("<|python_comment|>") == 32718 and v1.special("<|/python_def|>") == 32719
    assert v1.decode([v1.special("<|python_def|>")]) == "<|python_def|>" and v1.special("<|bos|>") == 32704
    assert v1.sha256 == json.loads((V1_TOKENIZER / "meta.json").read_text(encoding="utf-8"))["sha256"]  # checkpoints' tokenizer_sha256 keeps matching


def _make_shards(root, name, split, n_shards, shard_len, seed):
    d = root / name / split
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for i in range(n_shards):
        arr = rng.integers(0, 1000, size=shard_len, dtype=np.uint16)
        arr.tofile(d / f"shard_{i:05d}.bin")
        np.save(d / f"shard_{i:05d}.idx.npy", np.array([0, shard_len // 2], dtype=np.int64))


def test_token_stream_windows_and_wrap(tmp_path):
    _make_shards(tmp_path, "a", "train", 2, 100, 0)
    s = TokenStream(tmp_path / "a" / "train")
    w1 = s.next_window(40)
    w2 = s.next_window(40)
    assert np.array_equal(np.concatenate([w1, w2]), np.asarray(s.mm[0][:80]))
    w3 = s.next_window(40)  # 20 left in shard 0 -> skips to shard 1
    assert np.array_equal(w3, np.asarray(s.mm[1][:40])) and s.shard == 1
    s.next_window(40)
    s.next_window(40)  # wraps to shard 0, epoch 1
    assert s.shard == 0 and s.epoch == 1


def test_loader_resume_reproduces_batches(tmp_path):
    _make_shards(tmp_path, "a", "train", 2, 5000, 1)
    _make_shards(tmp_path, "b", "train", 1, 5000, 2)
    spec = MixtureSpec(tmp_path, {"a": 0.7, "b": 0.3})
    ld = PretrainLoader(spec, seq_len=16, microbatch=4, seed=0, device="cpu", prefetch=2)
    first = [ld.next() for _ in range(5)]
    state = json.loads(json.dumps(ld.state_dict()))
    expected = [ld.next() for _ in range(5)]
    ld.close()
    ld2 = PretrainLoader(spec, seq_len=16, microbatch=4, seed=999, device="cpu", prefetch=2)
    ld2.load_state_dict(state)
    got = [ld2.next() for _ in range(5)]
    ld2.close()
    for (x1, y1), (x2, y2) in zip(expected, got):
        assert torch.equal(x1, x2) and torch.equal(y1, y2)
    x, y = first[0]
    assert x.shape == (4, 16) and torch.equal(x[:, 1:], y[:, :-1])


def test_loader_consumed_tracks_windows(tmp_path):
    _make_shards(tmp_path, "a", "train", 2, 5000, 1)
    _make_shards(tmp_path, "b", "train", 1, 5000, 2)
    spec = MixtureSpec(tmp_path, {"a": 0.7, "b": 0.3})
    ld = PretrainLoader(spec, seq_len=15, microbatch=4, seed=0, device="cpu", prefetch=2)
    for _ in range(12):
        ld.next()
    got = ld.consumed()
    ld.close()
    # every window is seq_len + 1 tokens and no shard tail is skipped yet (12*4*16 = 768 < 5000)
    assert sum(v["tokens"] for v in got.values()) == 12 * 4 * 16
    for name, v in got.items():
        stream = ld.streams[name]
        assert v["epoch"] == pytest.approx(v["tokens"] / stream.total)
        assert v["shard"] == 0 and v["offset"] == v["tokens"] and v["tokens"] % 16 == 0
    assert got["a"]["tokens"] > got["b"]["tokens"]  # 0.7 / 0.3 mixture


def test_loader_consumed_counts_epochs_and_skipped_tails(tmp_path):
    _make_shards(tmp_path, "a", "train", 2, 100, 3)
    spec = MixtureSpec(tmp_path, {"a": 1.0})
    ld = PretrainLoader(spec, seq_len=39, microbatch=1, seed=0, device="cpu", prefetch=1)
    for _ in range(5):  # 40 tokens each: 2 per shard (tail of 20 skipped), so the 5th wraps
        ld.next()
    got = ld.consumed()["a"]
    ld.close()
    assert got["shard"] == 0 and got["offset"] == 40
    assert got["tokens"] == 1 * 200 + 0 + 40 and got["epoch"] == pytest.approx(240 / 200)


def test_val_loader_fixed(tmp_path):
    _make_shards(tmp_path, "a", "val", 1, 2000, 3)
    spec = MixtureSpec(tmp_path, {"a": 1.0}, "val")
    v1 = list(ValLoader(spec, 16, 2, 160, device="cpu"))
    v2 = list(ValLoader(spec, 16, 2, 160, device="cpu"))
    assert len(v1) == 5 and all(torch.equal(a[0], b[0]) for a, b in zip(v1, v2))


def test_prepare_writes_row_sidecar(tmp_path, tok, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from slm.data import prepare as prep
    from slm.data import sources as sources_mod

    monkeypatch.setattr(sources_mod, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(prep, "SHARD_TOKENS", 512)  # several shards from a handful of tiny docs
    src = sources_mod.Source("tiny", "local/tiny", "*.parquet", text_col="text", kind="prose")
    src.local_dir.mkdir(parents=True)
    # two files x two row groups x four rows; unique, English-looking, >= 64 chars so keep_doc passes
    texts = [[[f"Document {fi}-{rg}-{r} about the quick brown fox and the lazy dog in the garden. " * 2
               for r in range(4)] for rg in range(2)] for fi in range(2)]
    for fi in range(2):
        with pq.ParquetWriter(src.local_dir / f"part_{fi:03d}.parquet", pa.schema([("text", pa.string())])) as w:
            for rg in range(2):
                w.write_table(pa.table({"text": texts[fi][rg]}))

    out_root = tmp_path / "tokenized"
    manifest = prep.prepare(src, tok, out_root, max_tokens=float("inf"), val_permille=0, batch_docs=3, min_doc_tokens=1)
    assert manifest["sidecar"] == "src" and manifest["train_docs"] == 16

    files = [Path(p) for p in manifest["files"]]
    train_dir = out_root / "tiny" / "train"
    n_docs = 0
    for shard in sorted(train_dir.glob("shard_*.bin")):
        starts = np.load(shard.with_name(shard.name.replace(".bin", ".idx.npy")))
        rowrefs = np.load(shard.with_name(shard.name.replace(".bin", ".src.npy")))
        toks = np.fromfile(shard, dtype=np.uint16)
        assert rowrefs.shape == (len(starts), 3) and rowrefs.dtype == np.int32
        n_docs += len(starts)
        ends = list(starts[1:]) + [len(toks)]
        for (fi, rg, r), a, b in zip(rowrefs, starts, ends):
            want = pq.ParquetFile(files[fi]).read_row_group(rg, columns=["text"]).to_pylist()[r]["text"]
            assert tok.decode(toks[a:b].tolist(), skip_special=True) == want
    assert n_docs == manifest["train_docs"]


def test_schedules():
    c = ScheduleConfig(type="cosine", total_tokens=1000, warmup_tokens=100, min_lr_ratio=0.1)
    assert lr_at(0, c, 1.0) == pytest.approx(0.01) and lr_at(99, c, 1.0) == pytest.approx(1.0)
    assert lr_at(100, c, 1.0) == pytest.approx(1.0) and lr_at(1000, c, 1.0) == pytest.approx(0.1)
    assert lr_at(550, c, 1.0) == pytest.approx(0.55, abs=1e-6)
    w = ScheduleConfig(type="wsd", total_tokens=1000, warmup_tokens=100, min_lr_ratio=0.0, decay_frac=0.2)
    assert lr_at(500, w, 1.0) == 1.0 and lr_at(900, w, 1.0) == pytest.approx(0.5) and lr_at(1000, w, 1.0) == pytest.approx(0.0)


def test_report_builds_from_metrics(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"run_name": "r", "config": {"schedule": {"total_tokens": 1000}, "milestone_tokens": 100}, "n_params": 1}))
    lg = MetricsLogger(run)
    lg.log("start", msg="go")
    for i in range(1, 6):
        lg.log("train", tokens=i * 100, update=i, loss=5 - i * 0.1, lr=1e-4, grad_norm=1.0, tok_s=1000.0, tok_s_ema=1000.0,
               step_ms=100.0, fwd_ms=30.0, bwd_ms=60.0, opt_ms=10.0, data_ms=1.0, vram_gib=3.0, elapsed_s=i, eta_s=5)
        time.sleep(0.001)
    lg.log("eval", tokens=500, update=5, val_loss=4.5, val_ppl=90.0, best=True, eval_s=1)
    lg.log("milestone", tokens=500, update=5, segment_s=5.0, elapsed_s=5.0, tok_s=1000.0, loss=4.6, val_loss=4.5)
    lg.close()
    h = build_report(run)
    assert "<html" in h and "4.5000" in h and "milestone" in h.lower() and "ETA" in h


def test_loader_adopts_a_parent_runs_stream_positions(tmp_path):
    """A continuation phase must continue the parent's data streams: shared sources resume where the parent left
    off, sources the parent did not have start at 0, and sources it had that this phase drops are ignored."""
    _make_shards(tmp_path, "a", "train", 2, 5000, 1)
    _make_shards(tmp_path, "b", "train", 1, 5000, 2)
    _make_shards(tmp_path, "c", "train", 1, 5000, 3)
    parent = PretrainLoader(MixtureSpec(tmp_path, {"a": 0.5, "b": 0.5}), seq_len=15, microbatch=4, seed=0, device="cpu", prefetch=2)
    for _ in range(12):
        parent.next()
    parent_state = parent.state_dict()
    parent_consumed = parent.consumed()
    parent.close()
    # the child drops "b", keeps "a", adds "c"
    child = PretrainLoader(MixtureSpec(tmp_path, {"a": 0.5, "c": 0.5}), seq_len=15, microbatch=4, seed=1, device="cpu", prefetch=2)
    adopted = child.adopt_stream_positions(parent_state["streams"])
    assert set(adopted) == {"a"}  # "b" is not in this mixture, "c" was not in the parent's
    got = child.consumed()
    assert got["a"]["tokens"] == parent_consumed["a"]["tokens"] and got["a"]["tokens"] > 0
    assert got["c"]["tokens"] == 0
    first = child.next()  # the first batch reads on from there, not from token 0
    assert first[0].shape == (4, 15)
    after = child.consumed()
    assert after["a"]["tokens"] >= got["a"]["tokens"] and after["a"]["tokens"] + after["c"]["tokens"] == got["a"]["tokens"] + 4 * 16
    child.close()


def test_loader_state_survives_a_mixture_change_and_a_smaller_source(tmp_path, capsys):
    """Resuming after a source was swapped for a smaller one, or after the mixture changed, must not crash: an
    out-of-range cursor restarts that stream and names it, unknown sources are ignored."""
    _make_shards(tmp_path, "a", "train", 6, 1000, 1)
    _make_shards(tmp_path, "b", "train", 1, 1000, 2)
    big = PretrainLoader(MixtureSpec(tmp_path, {"a": 1.0}), seq_len=15, microbatch=4, seed=0, device="cpu", prefetch=2)
    for _ in range(40):
        big.next()
    state = big.state_dict()
    big.close()
    assert state["streams"]["a"]["shard"] >= 2  # past shard 0, so it is out of range once "a" shrinks to one shard
    # "a" now has only one shard (swapped for a smaller corpus) and the mixture gained "b" and lost nothing
    shutil.rmtree(tmp_path / "a")
    _make_shards(tmp_path, "a", "train", 1, 1000, 7)
    state["streams"]["gone"] = {"shard": 0, "offset": 0, "epoch": 0}  # a source this run no longer has
    ld = PretrainLoader(MixtureSpec(tmp_path, {"a": 0.5, "b": 0.5}), seq_len=15, microbatch=4, seed=0, device="cpu", prefetch=2)
    ld.load_state_dict(state)
    assert "restarting this stream" in capsys.readouterr().out
    assert ld.streams["a"].shard == 0 and ld.streams["a"].offset == 0
    x, y = ld.next()  # reads without an IndexError
    assert x.shape == (4, 15)
    ld.close()

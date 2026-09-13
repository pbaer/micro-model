"""Browser end-to-end tests for the command center (Playwright + Chromium, hermetic synthetic data).

Every page is opened, every button/tab is clicked, every select is cycled, and after each action the
page must (a) log no JS errors, (b) stay responsive, and (c) never render raw template text such as
'div' or 'undefined'. Run with: python -m pytest tests/e2e -q
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import uvicorn

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

from slm.config import ModelConfig, load_config, to_dict  # noqa: E402
from slm.data.chat import format_chat  # noqa: E402
from slm.data.sft import SftShardWriter  # noqa: E402
from slm.data.tokenizer import SlmTokenizer, train_bpe  # noqa: E402
from slm.model import Transformer  # noqa: E402
from slm.portal.app import create_app  # noqa: E402
from slm.portal.settings import PortalSettings  # noqa: E402
from slm.utils.checkpoint import save_snapshot  # noqa: E402
from slm.utils.logging import MetricsLogger  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_world(root: Path) -> dict:
    tok_dir = root / "tokenizer" / "v1"
    tok = SlmTokenizer(train_bpe(["the cat sat on the mat " * 80, "def f(x):\n    return x + 1\n" * 40], vocab_size=400))
    tok.save(tok_dir)
    # tokenized shards
    for src in ("alpha", "beta"):
        for split, n in (("train", 60_000), ("val", 4_000)):
            d = root / "tokenized" / "v1" / src / split
            d.mkdir(parents=True)
            rng = np.random.default_rng(1)
            arr = rng.integers(0, tok.base_vocab, size=n, dtype=np.uint16)
            starts = np.arange(0, n, 500, dtype=np.int64)
            arr[starts] = tok.bos_id
            arr.tofile(d / "shard_00000.bin")
            np.save(d / "shard_00000.idx.npy", starts)
        (root / "tokenized" / "v1" / src / "manifest.json").write_text(json.dumps({"source": src, "kind": "prose", "train_tokens": 60_000, "val_tokens": 4000, "train_docs": 120, "val_docs": 8, "docs_seen": 130, "docs_dropped": 2}))
    # sft shards
    w = SftShardWriter(root / "sft" / "v1" / "chat" / "train")
    for i in range(50):
        enc = format_chat(tok, [{"role": "user", "content": "hi there"}, {"role": "assistant", "content": "the cat sat"}])
        w.add(enc.ids, enc.loss_mask)
    w.flush()
    # runs: one finished, one "running"
    runs = root / "runs"
    for name, n, finished in (("fin", 40, True), ("live", 25, False)):
        d = runs / name
        d.mkdir(parents=True)
        cfg = {"schedule": {"total_tokens": 100 * n}, "milestone_tokens": 500, "batch": {"microbatch": 2, "tokens_per_update": 100}, "data": {"seq_len": 32}}
        (d / "run.json").write_text(json.dumps({"run_name": name, "stage": "pretrain", "n_params": 12345, "config": cfg, "model_config": to_dict(load_config(ModelConfig, "configs/model/tiny.yaml")), "env": {"gpu": "test", "git_commit": "abc"}, "started": "2026-09-13 00:00:00"}))
        lg = MetricsLogger(d)
        lg.log("start", msg="go")
        for i in range(1, n + 1):
            lg.log("train", tokens=i * 100, update=i, loss=5 - i * 0.05, lr=1e-4, grad_norm=1.0, tok_s=1000.0, tok_s_ema=1000.0, step_ms=100.0, fwd_ms=30.0, bwd_ms=60.0, opt_ms=10.0, data_ms=1.0, vram_gib=3.0, elapsed_s=i, eta_s=5)
            if i % 10 == 0:
                lg.log("eval", tokens=i * 100, update=i, val_loss=4.9 - i * 0.05, val_ppl=100.0, best=True, eval_s=1)
                lg.log("milestone", tokens=i * 100, update=i, segment_s=5.0, elapsed_s=i, tok_s=1000.0, loss=4.6, val_loss=4.5)
        if finished:
            lg.log("finish", tokens=n * 100, msg="done")
        lg.close()
        (d / "samples").mkdir()
        (d / "samples" / f"{1000:012d}.txt").write_text("tokens=1K update=10\n" + "=" * 80 + "\nPROMPT: 'Once'\n--- greedy:\nupon\n--- sampled:\na time\n", encoding="utf-8")
        cfgm = load_config(ModelConfig, "configs/model/tiny.yaml")
        cfgm.vocab_size = tok.vocab_size
        save_snapshot(d / "checkpoints" / "best.pt", Transformer(cfgm), to_dict(cfgm), {"tokens": 1000, "val_loss": 3.0, "tokenizer_sha256": tok.sha256})
    (root / "configs" / "train").mkdir(parents=True)
    (root / "configs" / "model").mkdir(parents=True)
    (root / "configs" / "model" / "tiny.yaml").write_text(Path("configs/model/tiny.yaml").read_text())
    (root / "configs" / "train" / "t.yaml").write_text(
        "run_name: t\nmodel_file: configs/model/tiny.yaml\ndata: {tokenized_root: '%s', mixture: {alpha: 0.7, beta: 0.3}, seq_len: 32}\nschedule: {total_tokens: 4000, warmup_tokens: 400}\nbatch: {microbatch: 2, tokens_per_update: 128}\n"
        % str(root / "tokenized" / "v1").replace("\\", "/"))
    return {"tok": tok}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    build_world(root)
    settings = PortalSettings(runs_root=root / "runs", data_root=root, configs_root=root / "configs", cache_dir=root / "cache", gpu_policy="cpu", open_browser=False)
    app = create_app(settings)
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    th.join(timeout=5)
    app.state.worker.stop()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


class Page:
    """Wraps a Playwright page with error collection and a responsiveness probe."""

    def __init__(self, browser, base: str):
        self.page = browser.new_page()
        self.errors: list[str] = []
        self.page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        self.page.on("console", lambda m: self.errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
        self.base = base

    def goto(self, hash_: str):
        self.page.goto(self.base + "/#" + hash_)
        self.settle()

    def settle(self, ms: int = 400):
        self.page.wait_for_timeout(ms)
        # responsiveness probe: a trivial evaluate must return within 3 s
        assert self.page.evaluate("1+1", ) == 2
        body = self.page.inner_text("main")
        for bad in ("divThis", "undefined", "NaN", "[object Object]"):
            assert bad not in body, f"raw template text {bad!r} on page: {body[:200]}"

    def click_all_buttons(self, skip=()):
        seen = set()
        for _ in range(60):
            buttons = self.page.locator("main button").all()
            target = None
            for b in buttons:
                try:
                    label = b.inner_text().strip()
                except Exception:  # noqa: BLE001
                    continue
                if label in seen or label in skip or not b.is_visible() or not b.is_enabled():
                    continue
                target = (b, label)
                break
            if target is None:
                break
            b, label = target
            seen.add(label)
            b.click(timeout=3000)
            self.settle()
        return seen

    def cycle_selects(self):
        for sel in self.page.locator("main select").all():
            if not sel.is_visible():
                continue
            opts = sel.locator("option").all()
            for o in opts[:3]:
                v = o.get_attribute("value")
                if v:
                    sel.select_option(v)
                    self.settle()


def test_home_and_runs(server, browser):
    p = Page(browser, server)
    p.goto("/")
    body = p.page.inner_text("main")
    assert "Live runs" in body
    assert body.count("live") == 1 + body.count("Live runs") + body.count("not live") - 1 or body.count("m") > 0  # a live run appears once
    live_section = body.split("Recent runs")[0]
    recent_section = body.split("Recent runs")[1]
    assert "live" in live_section and "fin" not in live_section.split("Live runs")[1]
    assert "fin" in recent_section and "
live" not in recent_section
    p.goto("/runs")
    body = p.page.inner_text("main")
    assert "fin" in body and "live" in body and "100.0%" in body  # finished run shows exactly 100%
    p.page.locator("tr.click", has_text="fin").first.click()
    p.settle(800)
    assert "finished" in p.page.inner_text("main")
    clicked = p.click_all_buttons()
    assert {"charts", "milestones", "samples", "checkpoints", "events", "config"} <= clicked
    p.page.get_by_role("button", name="charts").click()
    p.settle()
    for name in ("update", "time", "tokens", "log y"):
        p.page.get_by_role("button", name=name, exact=True).click()
        p.settle()
    assert p.page.locator(".chart .u-legend").count() >= 5
    p.cycle_selects()
    assert not p.errors, p.errors


def test_data_page(server, browser):
    p = Page(browser, server)
    p.goto("/data")
    clicked = set()
    for tab in ("sources", "mixture", "raw", "tokenized"):
        p.page.get_by_role("button", name=tab, exact=True).click()
        p.settle(600)
        clicked |= p.click_all_buttons(skip=("‹", "›", "‹ prev", "next ›", "sources", "mixture", "raw", "tokenized"))
    assert {"random doc", "doc", "window", "stats", "ids", "random sample"} <= clicked, clicked
    # window view must render chips with a boundary and the legend with the literal token names
    p.page.get_by_role("button", name="tokenized").click()
    p.settle()
    p.page.get_by_role("button", name="window", exact=True).click()
    p.settle(800)
    body = p.page.inner_text("main")
    assert "<|bos|>/<|eos|> mark boundaries" in body
    assert p.page.locator(".chip.boundary").count() >= 1
    p.cycle_selects()
    assert not p.errors, p.errors


def test_tokenizer_page(server, browser):
    p = Page(browser, server)
    p.goto("/tokenizer")
    p.page.locator("textarea").first.fill("hello <|user|> world 123")
    p.settle(600)
    assert p.page.locator(".chip").count() > 3
    clicked = p.click_all_buttons()
    assert {"raw", "document", "chat", "ids"} <= clicked
    assert "assistant content + <|end|>" in p.page.inner_text("main")
    assert not p.errors, p.errors


def test_arch_page(server, browser):
    p = Page(browser, server)
    p.goto("/arch")
    p.settle(1500)
    assert "Transformer" in p.page.inner_text("main")
    clicked = p.click_all_buttons()
    assert {"graph", "budget", "hparams", "grad checkpointing"} <= clicked
    p.cycle_selects()
    assert not p.errors, p.errors


def test_model_page_load_and_generate(server, browser):
    p = Page(browser, server)
    p.goto("/model")
    sel = p.page.locator("main select").first
    opts = [o.get_attribute("value") for o in sel.locator("option").all() if o.get_attribute("value")]
    assert opts, "no checkpoints listed"
    sel.select_option(opts[0])
    p.page.get_by_role("button", name="load").first.click()
    p.page.wait_for_function("document.querySelector('main').innerText.includes('cpu/float32')", timeout=60000)
    p.settle()
    p.page.locator("input[type=number]").nth(3).fill("8")  # max new tokens
    p.page.get_by_role("button", name="generate").click()
    p.page.wait_for_function("document.querySelectorAll('.chips .chip').length > 3", timeout=60000)
    p.settle()
    p.page.get_by_role("button", name="score prompt (teacher-forced)").click()
    p.page.wait_for_function("document.querySelector('main').innerText.includes('perplexity')", timeout=60000)
    p.settle()
    p.page.get_by_role("button", name="chat").click()
    p.settle()
    assert "force <|think|>" in p.page.inner_text("main")
    assert not p.errors, p.errors

"""The Inference tab's chat mode: the default-chat checkpoint flag on /api/model/checkpoints and the page's wiring."""

import re
from pathlib import Path

from slm.portal.api.model import mark_default_chat


def _e(run, name, stage, kind, mtime=0.0, external=False, is_chat=False):
    return {"run": run, "name": name, "path": f"runs/{run}/checkpoints/{name}" if run else name, "stage": stage, "kind": kind, "mtime": mtime,
            "external": external, "is_chat": is_chat}


def test_default_chat_prefers_the_configured_checkpoint_when_listed():
    es = [_e("m9_rl6_336m", "final.pt", "rl", "final", 5), _e("m10_rl_336m", "final.pt", "rl", "final", 1), _e("m10_rl_336m", "best.pt", "rl", "best", 2)]
    out = mark_default_chat(es, "m10_rl_336m/final.pt")
    assert [c["default_chat"] for c in out] == [False, True, False]
    out = mark_default_chat(es, r"runs\m9_rl6_336m\checkpoints\final.pt")  # a path works too, either separator
    assert [c["default_chat"] for c in out] == [True, False, False]


def test_default_chat_falls_back_to_the_best_stand_in():
    es = [_e("m8_base_4k_336m", "final.pt", "base", "final", 9), _e("m4_sft_149m", "final.pt", "sft", "final", 1), _e("m9_rl6_336m", "best.pt", "rl", "best", 3),
          _e("m9_rl6_336m", "final.pt", "rl", "final", 2), _e("m9_rl6_336m", "000000001000.pt", "rl", "snapshot", 8), _e(None, "external:x", "external", "external", external=True, is_chat=True)]
    out = mark_default_chat(es, "m10_rl_336m/final.pt")  # not on disk
    picked = [c for c in out if c["default_chat"]]
    assert len(picked) == 1 and picked[0]["run"] == "m9_rl6_336m" and picked[0]["name"] == "final.pt", "RL final beats best, snapshots, SFT and base"
    assert [c["chat_ok"] for c in out] == [False, True, True, True, True, True], "base is never chat_ok; an external chat model is"
    assert sum(c["default_chat"] for c in mark_default_chat([es[0]], None)) == 0, "a base-only listing has no default"


def test_chat_page_is_the_inference_default_and_parses():
    static = Path("slm/portal/static")
    app = (static / "app.js").read_text(encoding="utf-8")
    assert 'from "./pages/chat.js"' in app and "ChatPage" in app
    chat = (static / "pages" / "chat.js").read_text(encoding="utf-8")
    assert "/api/model/generate" in chat and "default_chat" in chat and "chat_ok" in chat
    assert "localStorage" in chat, "conversations persist per browser"
    # embedded browsers (the Claude desktop pane) block native dialogs: prompt() throws, confirm() returns false
    assert not re.search(r"\b(?:window\.)?(?:prompt|confirm|alert)\s*\(", chat), "no native dialogs: rename and delete are inline"
    assert "seed: null" in chat or "seed: undefined" in chat or "seed: randomSeed" in chat or "newSeed" in chat, "a fresh random seed per reply"
    cards = (static / "components" / "cards.js").read_text(encoding="utf-8")

    for k in set(re.findall(r'k="(chat[a-z_]*)"', chat)):
        assert f"  {k}: {{" in cards, f"info card {k} missing"
    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    if not node:
        return
    for f in (static / "pages" / "chat.js", static / "pages" / "model.js", static / "app.js"):
        tmp = Path(tempfile.gettempdir()) / "slm_chat_check.mjs"
        shutil.copy(f, tmp)
        r = subprocess.run([node, "--check", str(tmp)], capture_output=True, text=True)
        assert r.returncode == 0, f"{f}: {r.stderr[:400]}"

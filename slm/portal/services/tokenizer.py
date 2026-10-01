from __future__ import annotations

import json
import threading
from pathlib import Path

from slm.data.chat import format_chat
from slm.data.tokenizer import SlmTokenizer, apply_named


def text_runs(tok: SlmTokenizer, ids: list[int], mask: list | None = None) -> list[dict]:
    """Decoded text split at the reserved tokens, for the pages' text view.

    Returns `[{"text": str} | {"special": "<|bos|>", "i": token index}, ...]`; with a loss mask every run also
    carries `"loss"`. Identity comes from the ids, never from the text: a literal "<|user|>" typed into a
    document is ordinary BPE pieces and stays text. Consecutive ordinary ids with the same mask value are
    decoded together, so a character split across byte-level ids comes out whole.
    """
    out: list[dict] = []
    run: list[int] = []
    run_m = None

    def flush() -> None:
        if run:
            d = {"text": tok.decode(run)}
            if mask is not None:
                d["loss"] = bool(run_m)
            out.append(d)
            run.clear()

    for k, i in enumerate(ids):
        i = int(i)
        m = None if mask is None else bool(mask[k]) if k < len(mask) else False
        if i >= tok.base_vocab:
            flush()
            d = {"special": tok.id_to_special.get(i, f"<|unk_special_{i}|>"), "i": k}
            if mask is not None:
                d["loss"] = m
            out.append(d)
            continue
        if run and m != run_m:
            flush()
        run_m = m
        run.append(i)
    flush()
    return out


class TokenizerRegistry:
    def __init__(self, tokenizer_root: Path) -> None:
        self.root = Path(tokenizer_root)
        self._cache: dict[str, SlmTokenizer] = {}
        self.lock = threading.Lock()

    def tags(self) -> list[dict]:
        out = []
        if not self.root.exists():
            return out
        for d in sorted(self.root.iterdir()):
            if (d / "tokenizer.json").exists():
                meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
                out.append({"tag": d.name, "path": str(d), "vocab_size": meta.get("vocab_size"), "sha256": meta.get("sha256"), "n_special": len(meta.get("specials", [])),
                            "specials": apply_named(meta.get("specials", []))})
        return out

    def get(self, tag: str) -> SlmTokenizer:
        with self.lock:
            if tag not in self._cache:
                d = self.root / tag
                if not (d / "tokenizer.json").exists() or d.name != tag:
                    raise KeyError(tag)
                self._cache[tag] = SlmTokenizer.load(d)
            return self._cache[tag]

    def by_sha(self, sha: str) -> str | None:
        for t in self.tags():
            if t["sha256"] == sha:
                return t["tag"]
        return None

    def encode(self, tag: str, text: str = "", mode: str = "raw", messages: list[dict] | None = None, add_generation_prompt: bool = False) -> dict:
        tok = self.get(tag)
        if mode == "chat":
            enc = format_chat(tok, messages or [], add_generation_prompt=add_generation_prompt)
            pieces = [{"id": i, "piece": tok.token_str(i), "special": i >= tok.base_vocab, "loss": m} for i, m in zip(enc.ids, enc.loss_mask)]
            return {"ids": enc.ids, "pieces": pieces, "runs": text_runs(tok, enc.ids, enc.loss_mask), "segments": enc.segments,
                    "n_tokens": len(enc.ids), "n_target": sum(enc.loss_mask)}
        e = tok.tok.encode(text, add_special_tokens=False)
        pieces = [{"id": i, "piece": tok.token_str(i), "special": False, "start": s, "end": t} for i, (s, t) in zip(e.ids, e.offsets)]
        ids = list(e.ids)
        if mode == "document":
            pieces = [{"id": tok.bos_id, "piece": "<|bos|>", "special": True}, *pieces, {"id": tok.eos_id, "piece": "<|eos|>", "special": True}]
            ids = [tok.bos_id, *ids, tok.eos_id]
        return {"ids": ids, "pieces": pieces, "runs": text_runs(tok, ids), "n_tokens": len(ids), "n_chars": len(text),
                "chars_per_token": len(text) / max(1, len(e.ids))}

    def pieces(self, tag: str, ids: list[int]) -> list[dict]:
        tok = self.get(tag)
        return [{"id": int(i), "piece": tok.token_str(int(i)), "special": int(i) >= tok.base_vocab} for i in ids]

    def runs(self, tag: str, ids: list[int], mask: list | None = None) -> list[dict]:
        return text_runs(self.get(tag), ids, mask)

    def vocab_search(self, tag: str, q: str, limit: int = 100) -> list[dict]:
        tok = self.get(tag)
        vocab = tok.tok.get_vocab()
        q = q or ""
        out = []
        for piece, i in vocab.items():
            s = tok.tok.decode([i])
            if q in s or q in piece:
                out.append({"id": i, "piece": s, "raw": piece})
        out.sort(key=lambda x: x["id"])
        specials = [{"id": tok.special(s), "piece": s, "raw": s} for s in tok.specials if q in s]
        return (out + specials)[:limit]

    def token(self, tag: str, i: int) -> dict:
        tok = self.get(tag)
        s = tok.token_str(i)
        return {"id": i, "piece": s, "special": i >= tok.base_vocab, "bytes": list(s.encode("utf-8", errors="replace"))}

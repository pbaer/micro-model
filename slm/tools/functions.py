"""Declared functions: capabilities a conversation hands the model before its first turn.

A declaration is one block, emitted right after <|bos|> and before the first turn, one block per
function, every token of it loss-masked (environment-written, like <|python_result|>):

    <|python_def|>def unit_price(item: str) -> float<|python_comment|>Catalogue price of an item in dollars. Use it instead of guessing.<|/python_def|>

The signature is a real Python `def` line without a body; the comment is natural language (what it
does and when to use it). The function itself is an ordinary Python callable we provide: it is
registered in the conversation's `PySession`, so the model calls it inside <|python_call|> through
the normal call path, with the same result rendering and the same error hints. Declared functions are
for capabilities the model cannot write itself (a price list, a lookup, a physical measurement);
anything it can compute in the sandbox needs no declaration.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from slm.data.tokenizer import SlmTokenizer
from slm.tools.calculator import ToolError

SIG_RE = re.compile(r"^def\s+([A-Za-z_]\w*)\s*\(.*\)(\s*->\s*\S.*)?$")


@dataclass
class FunctionDecl:
    """One declared function: how it is written in the prompt, and (ours to provide) what it does."""

    name: str
    signature: str  # a `def` line without a body: "def unit_price(item: str) -> float"
    comment: str  # natural language, one line
    impl: Callable | None = None  # None = declared but not runnable here (e.g. the portal, or a dataset row)

    def __post_init__(self) -> None:
        self.signature = " ".join(self.signature.strip().rstrip(":").split())
        self.comment = " ".join(self.comment.split())
        m = SIG_RE.match(self.signature)
        if not m or m.group(1) != self.name:
            raise ValueError(f"signature must be a body-less 'def {self.name}(...)' line, got {self.signature!r}")

    def as_dict(self) -> dict:
        """JSON form (portal requests, dataset rows); `impl` is not serializable and is dropped."""
        return {"name": self.name, "signature": self.signature, "comment": self.comment}


def as_decls(items) -> list[FunctionDecl]:
    """Normalize FunctionDecls / dicts (JSON from the portal or a dataset row) to FunctionDecls."""
    out = []
    for it in items or []:
        out.append(it if isinstance(it, FunctionDecl) else FunctionDecl(it["name"], it["signature"], it.get("comment", ""), it.get("impl")))
    return out


def functions_env(decls) -> dict[str, Callable]:
    """{name: callable} for a PySession namespace. A declaration without an impl is still registered,
    so calling it says so instead of looking like a name the model invented."""
    env: dict[str, Callable] = {}
    for f in as_decls(decls):
        env[f.name] = f.impl if f.impl is not None else _unavailable(f.name)
    return env


def _unavailable(name: str) -> Callable:
    def fn(*args, **kwargs):
        raise ToolError(f"{name}() is declared but has no implementation in this session")

    return fn


def def_ids(tok: SlmTokenizer) -> dict[str, int]:
    return {"def_open": tok.special("<|python_def|>"), "comment": tok.special("<|python_comment|>"), "def_close": tok.special("<|/python_def|>")}


def render_defs(tok: SlmTokenizer, decls) -> list[int]:
    """Ids of the declaration blocks (all of them loss-masked by the caller), in order."""
    t = def_ids(tok)
    ids: list[int] = []
    for f in as_decls(decls):
        ids += [t["def_open"], *tok.encode(f.signature), t["comment"], *tok.encode(f.comment), t["def_close"]]
    return ids


def parse_defs(tok: SlmTokenizer, ids: list[int]) -> list[FunctionDecl]:
    """Inverse of `render_defs` (no impl): the declarations found in `ids`. Tokens outside a block and
    unterminated or unparsable blocks are ignored, so a whole prompt can be passed in."""
    t = def_ids(tok)
    out: list[FunctionDecl] = []
    state, sig, buf = "out", [], []
    for i in ids:
        if i == t["def_open"]:
            state, sig, buf = "sig", [], []
        elif i == t["comment"] and state == "sig":
            state, sig, buf = "comment", buf, []
        elif i == t["def_close"]:
            if state == "comment":
                signature = " ".join(tok.decode(sig, skip_special=True).rstrip(":").split())
                m = SIG_RE.match(signature)
                if m:
                    out.append(FunctionDecl(m.group(1), signature, tok.decode(buf, skip_special=True)))
            state = "out"
        elif state != "out":
            buf.append(i)
    return out

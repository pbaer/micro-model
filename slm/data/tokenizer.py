"""Our own byte-level BPE tokenizer (32,768 ids = 32,704 BPE + 64 reserved specials).

Design
- Byte-level BPE (GPT-2/Llama-3 lineage) with a GPT-4-style pre-tokenization regex, except that
  digits are always split one at a time (better arithmetic for a small model).
- Special tokens are NOT added tokens inside the HF tokenizer. They occupy the top 64 ids and are
  known only to `SlmTokenizer`, so raw text can never produce them; only the chat formatter can.
- The tokenizer is frozen after M1 sign-off; its sha256 is recorded in every checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

VOCAB_SIZE = 32768
N_SPECIAL = 64
BPE_VOCAB = VOCAB_SIZE - N_SPECIAL

# GPT-4 cl100k pattern with \p{N}{1,3} replaced by \p{N} (single-digit tokens).
PRETOK_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

NAMED_SPECIALS = [
    "<|bos|>",  # document start (every document, every conversation)
    "<|eos|>",  # document end / packing separator; masked in SFT loss
    "<|pad|>",  # batch padding, never a target
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",  # end of turn; loss target and generation stop token
    "<|think|>",
    "<|/think|>",
    "<|python_call|>",  # sandboxed Python call (body = code); see slm/tools
    "<|/python_call|>",
    "<|python_result|>",  # environment-written result; never a loss target
    "<|/python_result|>",
    "<|python_def|>",  # declared function: signature block before the first turn; never a loss target
    "<|python_comment|>",  # natural-language description of the declared function
    "<|/python_def|>",
]
SPECIAL_TOKENS = NAMED_SPECIALS + [f"<|reserved_{i}|>" for i in range(N_SPECIAL - len(NAMED_SPECIALS))]
assert len(SPECIAL_TOKENS) == N_SPECIAL


def apply_named(specials: list[str]) -> list[str]:
    """Give the reserved slots their current names.

    A saved tokenizer stores the specials list of the day it was trained, so names appended to
    NAMED_SPECIALS later would stay `<|reserved_i|>` on a loaded tokenizer. Naming a reserved slot
    changes no id and no vocab size (and not the sha256, which covers tokenizer.json only), so the
    names are re-applied by position at load time instead of retraining or re-saving.
    """
    out = list(specials)
    for i, name in enumerate(NAMED_SPECIALS):
        if i < len(out) and out[i] != name:
            assert out[i].startswith("<|reserved_"), f"slot {i} is {out[i]}, cannot be renamed to {name}"
            out[i] = name
    for i in range(len(NAMED_SPECIALS), len(out)):  # the unused tail keeps counting from 0, as SPECIAL_TOKENS does
        if out[i].startswith("<|reserved_"):
            out[i] = f"<|reserved_{i - len(NAMED_SPECIALS)}|>"
    return out


def build_untrained() -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(PRETOK_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    return tok


def train_bpe(texts: Iterable[str], vocab_size: int = BPE_VOCAB, min_frequency: int = 2) -> Tokenizer:
    tok = build_untrained()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
        max_token_length=32,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    if tok.get_vocab_size() != vocab_size:
        print(f"WARNING: BPE vocab {tok.get_vocab_size()} != requested {vocab_size} (corpus too small for that many merges)")
    return tok


class SlmTokenizer:
    """Wrapper that owns the special-token id space on top of a trained HF BPE tokenizer."""

    def __init__(self, tok: Tokenizer, specials: list[str] = SPECIAL_TOKENS, sha256: str = "") -> None:
        self.tok = tok
        self.base_vocab = tok.get_vocab_size()
        self.specials = apply_named(specials)
        self.vocab_size = self.base_vocab + len(self.specials)
        self.special_to_id = {s: self.base_vocab + i for i, s in enumerate(self.specials)}
        self.id_to_special = {v: k for k, v in self.special_to_id.items()}
        self.sha256 = sha256

    # ---- ids of the named specials
    def special(self, name: str) -> int:
        return self.special_to_id[name]

    @property
    def bos_id(self) -> int:
        return self.special("<|bos|>")

    @property
    def eos_id(self) -> int:
        return self.special("<|eos|>")

    @property
    def pad_id(self) -> int:
        return self.special("<|pad|>")

    @property
    def end_id(self) -> int:
        return self.special("<|end|>")

    # ---- encode / decode
    def encode(self, text: str) -> list[int]:
        """Plain text -> BPE ids. Never emits special ids."""
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(texts, add_special_tokens=False)]

    def encode_document(self, text: str) -> list[int]:
        """Pretraining document: <|bos|> text <|eos|>."""
        return [self.bos_id, *self.encode(text), self.eos_id]

    def decode(self, ids: list[int], skip_special: bool = False) -> str:
        out: list[str] = []
        run: list[int] = []
        for i in ids:
            if i >= self.base_vocab:
                if run:
                    out.append(self.tok.decode(run))
                    run = []
                if not skip_special:
                    out.append(self.id_to_special.get(i, f"<|unk_special_{i}|>"))
            else:
                run.append(i)
        if run:
            out.append(self.tok.decode(run))
        return "".join(out)

    def token_str(self, i: int) -> str:
        if i >= self.base_vocab:
            return self.id_to_special.get(i, f"<|unk_special_{i}|>")
        return self.tok.decode([i])

    # ---- persistence
    def save(self, d: str | Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        self.tok.save(str(d / "tokenizer.json"))
        sha = hashlib.sha256((d / "tokenizer.json").read_bytes()).hexdigest()
        self.sha256 = sha
        meta = {
            "vocab_size": self.vocab_size,
            "base_vocab": self.base_vocab,
            "specials": self.specials,
            "pretok_pattern": PRETOK_PATTERN,
            "sha256": sha,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, d: str | Path) -> SlmTokenizer:
        d = Path(d)
        tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        sha = hashlib.sha256((d / "tokenizer.json").read_bytes()).hexdigest()
        assert sha == meta["sha256"], "tokenizer.json does not match meta.json sha256"
        return cls(tok, meta["specials"], sha)

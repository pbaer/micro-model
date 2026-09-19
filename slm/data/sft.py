"""Supervised fine-tuning data: chat examples -> packed token/mask shards -> loaders.

Shard layout (per source, per split) under C:/slm-data/sft/<tag>/<name>/{train,val}/:
    tokens_00000.bin   uint16 token ids of examples laid back to back (each = <|bos|> ... <|eos|>)
    mask_00000.bin     uint8, 1 where the token is a loss target (assistant content + <|end|>)
    idx_00000.npy      int64 example start offsets
The loaders pack examples contiguously into fixed-length windows exactly like pretraining; the mask
travels with the tokens so the loss covers only assistant tokens. Cross-example attention inside a
window is the usual packing compromise (documented, measurable, and cheap).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from slm.data.chat import format_chat, rows_to_messages  # noqa: F401 (re-export)
from slm.data.loader import PretrainLoader, TokenStream
from slm.data.sources import DATA_ROOT, SOURCES, Source
from slm.data.tokenizer import SlmTokenizer
from slm.tools.protocol import hoist_calls
from slm.model.loss import IGNORE_INDEX

SFT_DIR = DATA_ROOT / "sft"
SHARD_TOKENS = 50_000_000


class SftShardWriter:
    def __init__(self, out_dir: Path, shard_tokens: int = SHARD_TOKENS) -> None:
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_tokens = shard_tokens
        self.tok = np.empty(shard_tokens, dtype=np.uint16)
        self.mask = np.empty(shard_tokens, dtype=np.uint8)
        self.n = 0
        self.starts: list[int] = []
        self.shard_idx = 0
        self.total_tokens = self.total_targets = self.total_examples = 0

    def add(self, ids: list[int], mask: list[int]) -> None:
        if self.n + len(ids) > self.shard_tokens:
            self.flush()
        self.starts.append(self.n)
        self.tok[self.n : self.n + len(ids)] = ids
        self.mask[self.n : self.n + len(ids)] = mask
        self.n += len(ids)
        self.total_tokens += len(ids)
        self.total_targets += sum(mask)
        self.total_examples += 1

    def flush(self) -> None:
        if self.n == 0:
            return
        self.tok[: self.n].tofile(self.out_dir / f"tokens_{self.shard_idx:05d}.bin")
        self.mask[: self.n].tofile(self.out_dir / f"mask_{self.shard_idx:05d}.bin")
        np.save(self.out_dir / f"idx_{self.shard_idx:05d}.npy", np.array(self.starts, dtype=np.int64))
        self.shard_idx += 1
        self.n = 0
        self.starts = []


def prepare_sft(src: Source, tok: SlmTokenizer, out_root: Path, max_len: int = 2048, val_permille: int = 10, max_examples: int | None = None,
                think_required: bool = False, name: str | None = None, tools: bool = False, marker_mix: float = 0.5) -> dict:
    """tools=True: <<expr=result>> annotations in assistant text become calculator calls (see slm.tools);
    rows whose assistant text has no such annotation are dropped, so the set teaches tool use consistently.
    marker_mix: for verifiable (math) rows, the share whose user turn asks for `#### <number>` and gets it; the rest
    keep the bare question and answer in a natural sentence (see slm.data.answers)."""
    import random

    from slm.data.answers import apply_style

    style_rng = random.Random(f"style-{src.name}-{name}")
    files = sorted(p for p in src.local_dir.rglob("*.parquet"))
    assert files, f"no raw files for {src.name}"
    out = out_root / (name or src.name)
    train, val = SftShardWriter(out / "train"), SftShardWriter(out / "val")
    n_seen = n_drop = n_trunc = n_notool = 0
    t0 = time.time()
    for f in files:
        is_test = "test" in f.name
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                n_seen += 1
                msgs = rows_to_messages(src, row)
                if msgs is None:
                    n_drop += 1
                    continue
                if src.kind in ("math_qa", "math_cot"):
                    apply_style(msgs, style_rng, marker_mix)
                if tools:
                    for m in msgs:  # the number must follow the tool result, never precede the call
                        if m["role"] == "assistant" and m.get("think"):
                            m["think"] = hoist_calls(m["think"])
                if tools and not any("<<" in (m.get("think") or "") + m.get("content", "") for m in msgs if m["role"] == "assistant"):
                    n_notool += 1
                    continue
                enc = format_chat(tok, msgs, think_required=think_required, tools=tools)
                if len(enc.ids) > max_len:
                    n_trunc += 1
                    continue  # drop rather than truncate: a cut-off answer teaches bad endings
                if sum(enc.loss_mask) == 0:
                    n_drop += 1
                    continue
                key = hashlib.sha1(msgs[0]["content"][:512].encode("utf-8", errors="ignore")).digest()
                to_val = is_test or int.from_bytes(key[:4], "little") % 1000 < val_permille
                (val if to_val else train).add(enc.ids, enc.loss_mask)
                if max_examples and train.total_examples >= max_examples:
                    break
            if max_examples and train.total_examples >= max_examples:
                break
        if max_examples and train.total_examples >= max_examples:
            break
    train.flush()
    val.flush()
    m = {"source": src.name, "name": name or src.name, "tokenizer_sha256": tok.sha256, "max_len": max_len, "think_required": think_required, "tools": tools, "no_tool_calls": n_notool,
         "marker_mix": marker_mix if src.kind in ("math_qa", "math_cot") else None,
         "train_examples": train.total_examples, "train_tokens": train.total_tokens, "train_targets": train.total_targets, "train_shards": train.shard_idx,
         "val_examples": val.total_examples, "val_tokens": val.total_tokens, "val_targets": val.total_targets, "val_shards": val.shard_idx,
         "seen": n_seen, "dropped": n_drop, "too_long": n_trunc, "seconds": time.time() - t0}
    (out / "manifest.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(f"[{src.name}] {m['train_examples']} train ex ({m['train_tokens'] / 1e6:.1f}M tok, {m['train_targets'] / max(1, m['train_tokens']) * 100:.0f}% targets), {m['val_examples']} val ex, dropped {n_drop}, too long {n_trunc}, {m['seconds']:.0f}s")
    return m


# --------------------------------------------------------------------------------- loaders
class SftStream(TokenStream):
    """Token stream plus a parallel mask stream."""

    def __init__(self, split_dir: Path) -> None:
        self.dir = Path(split_dir)
        self.paths = sorted(self.dir.glob("tokens_*.bin"))
        assert self.paths, f"no sft shards under {split_dir}"
        self.mm = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.paths]
        self.mask_mm = [np.memmap(p.with_name(p.name.replace("tokens_", "mask_")), dtype=np.uint8, mode="r") for p in self.paths]
        self.sizes = [len(m) for m in self.mm]
        self.total = sum(self.sizes)
        self.shard = self.offset = self.epoch = 0

    def next_window_masked(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        if self.offset + n > self.sizes[self.shard]:
            self.shard += 1
            self.offset = 0
            if self.shard >= len(self.mm):
                self.shard = 0
                self.epoch += 1
        t = np.asarray(self.mm[self.shard][self.offset : self.offset + n])
        m = np.asarray(self.mask_mm[self.shard][self.offset : self.offset + n])
        self.offset += n
        return t, m


class SftLoader(PretrainLoader):
    """Same interface as PretrainLoader; targets are IGNORE_INDEX wherever the mask is 0."""

    def __init__(self, root: Path, mixture: dict[str, float], seq_len: int, microbatch: int, seed: int = 0, device="cuda", prefetch: int = 4) -> None:
        from slm.data.loader import MixtureSpec

        spec = MixtureSpec(Path(root), mixture, "train")
        self.spec = spec
        self.streams = {name: SftStream(Path(root) / name / "train") for name in mixture}
        self.names = list(self.streams)
        w = np.array([mixture[n] for n in self.names], dtype=np.float64)
        self.probs = w / w.sum()
        self.seq_len, self.mb = seq_len, microbatch
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.n_batches = self.tokens_served = 0
        self.prefetch = prefetch
        self._q = self._thread = None
        import threading

        self._stop = threading.Event()
        self._state_after_last = None

    def _make_batch(self):
        n = self.seq_len + 1
        rows = np.empty((self.mb, n), dtype=np.int64)
        masks = np.empty((self.mb, n), dtype=np.int64)
        src_idx = self.rng.choice(len(self.names), size=self.mb, p=self.probs)
        for i, si in enumerate(src_idx):
            rows[i], masks[i] = self.streams[self.names[si]].next_window_masked(n)
        self.n_batches += 1
        self.tokens_served += self.mb * self.seq_len
        t = torch.from_numpy(rows)
        m = torch.from_numpy(masks)
        if self.device.type == "cuda":
            t, m = t.pin_memory(), m.pin_memory()
        return (t, m), self._snapshot()

    def next(self):
        if self._thread is None:
            self._start_thread()
        (rows, masks), state = self._q.get()
        self._state_after_last = state
        rows = rows.to(self.device, non_blocking=True)
        masks = masks.to(self.device, non_blocking=True)
        x = rows[:, :-1]
        y = torch.where(masks[:, 1:] > 0, rows[:, 1:], torch.full_like(rows[:, 1:], IGNORE_INDEX))
        return x, y


class SftValLoader:
    def __init__(self, root: Path, mixture: dict[str, float], seq_len: int, microbatch: int, n_tokens: int, device="cuda") -> None:
        n = seq_len + 1
        rows, masks = [], []
        w = np.array(list(mixture.values()), dtype=np.float64)
        w = w / w.sum()
        for name, frac in zip(mixture, w):
            s = SftStream(Path(root) / name / "val")
            k = max(1, int(n_tokens * frac / seq_len))
            for _ in range(k):
                t, m = s.next_window_masked(n)
                rows.append(t.astype(np.int64))
                masks.append(m.astype(np.int64))
        R = torch.from_numpy(np.stack(rows))
        M = torch.from_numpy(np.stack(masks))
        self.batches = [(R[i : i + microbatch], M[i : i + microbatch]) for i in range(0, len(R), microbatch)]
        self.device = torch.device(device)
        self.n_tokens = len(R) * seq_len

    def __iter__(self):
        for r, m in self.batches:
            r, m = r.to(self.device, non_blocking=True), m.to(self.device, non_blocking=True)
            yield r[:, :-1], torch.where(m[:, 1:] > 0, r[:, 1:], torch.full_like(r[:, 1:], IGNORE_INDEX))

    def __len__(self) -> int:
        return len(self.batches)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--val-permille", type=int, default=10)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--think-required", action="store_true", help="always emit a <|think|> span (reasoning SFT)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--tools", action="store_true", help="convert <<expr=result>> annotations to calculator calls; drop rows without any")
    ap.add_argument("--marker-mix", type=float, default=0.5, help="share of math rows that ask for and use the '#### <number>' marker")
    a = ap.parse_args()
    tok = SlmTokenizer.load(a.tokenizer)
    out_root = SFT_DIR / Path(a.tokenizer).name
    for s in a.sources:
        prepare_sft(SOURCES[s], tok, out_root, a.max_len, a.val_permille, a.max_examples, a.think_required, a.name if len(a.sources) == 1 else None, tools=a.tools, marker_mix=a.marker_mix)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()

"""Memory-mapped token-shard loaders for pretraining, with resumable state and prefetching.

Packing strategy: each source is a contiguous token stream (documents already wrapped in
<|bos|> ... <|eos|>); a training window is `seq_len + 1` consecutive tokens (inputs/targets are the
same window shifted by one). Windows may start mid-document; the separators let the model learn
document boundaries. Mixture sampling picks the source of every row from configured weights.

State (per-source cursor + RNG + batch counter) is captured *after* each batch is consumed, not
when it was prefetched, so a checkpoint resumes on exactly the next batch.
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


class TokenStream:
    """All shards of one split of one source, read as a single stream of uint16 tokens."""

    def __init__(self, split_dir: Path) -> None:
        self.dir = Path(split_dir)
        self.paths = sorted(self.dir.glob("shard_*.bin"))
        assert self.paths, f"no shards under {split_dir}"
        self.mm = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.paths]
        self.sizes = [len(m) for m in self.mm]
        self.total = sum(self.sizes)
        self.shard = 0
        self.offset = 0
        self.epoch = 0

    def next_window(self, n: int) -> np.ndarray:
        """Next `n` tokens; windows never straddle shards (tail < n tokens of a shard is skipped)."""
        if self.offset + n > self.sizes[self.shard]:
            self.shard += 1
            self.offset = 0
            if self.shard >= len(self.mm):
                self.shard = 0
                self.epoch += 1
        out = np.asarray(self.mm[self.shard][self.offset : self.offset + n])
        self.offset += n
        return out

    def state_dict(self) -> dict:
        return {"shard": self.shard, "offset": self.offset, "epoch": self.epoch}

    def load_state_dict(self, d: dict) -> None:
        self.shard, self.offset, self.epoch = d["shard"], d["offset"], d["epoch"]

    def doc_starts(self, shard: int) -> np.ndarray:
        return np.load(self.paths[shard].with_name(self.paths[shard].name.replace(".bin", ".idx.npy")))


@dataclass
class MixtureSpec:
    """name -> weight; dirs resolved under `root/<name>/<split>`."""

    root: Path
    weights: dict[str, float]
    split: str = "train"

    def streams(self) -> dict[str, TokenStream]:
        return {name: TokenStream(Path(self.root) / name / self.split) for name in self.weights}


@dataclass
class LoaderState:
    streams: dict[str, dict]
    rng: dict
    n_batches: int
    tokens_served: int


class PretrainLoader:
    def __init__(
        self,
        spec: MixtureSpec,
        seq_len: int,
        microbatch: int,
        seed: int = 0,
        device: str | torch.device = "cuda",
        prefetch: int = 4,
    ) -> None:
        self.spec = spec
        self.streams = spec.streams()
        self.names = list(self.streams)
        w = np.array([spec.weights[n] for n in self.names], dtype=np.float64)
        self.probs = w / w.sum()
        self.seq_len = seq_len
        self.mb = microbatch
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.n_batches = 0
        self.tokens_served = 0
        self.prefetch = prefetch
        self._q: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._state_after_last: LoaderState | None = None

    # ------------------------------------------------------------------ state
    def state_dict(self) -> dict:
        s = self._state_after_last or self._snapshot()
        return {"streams": s.streams, "rng": s.rng, "n_batches": s.n_batches, "tokens_served": s.tokens_served}

    def load_state_dict(self, d: dict) -> None:
        self._stop_thread()
        for n, st in d["streams"].items():
            self.streams[n].load_state_dict(st)
        self.rng.bit_generator.state = d["rng"]
        self.n_batches = d["n_batches"]
        self.tokens_served = d["tokens_served"]
        self._state_after_last = None

    def consumed(self) -> dict[str, dict]:
        """Per source: tokens taken from its stream since the run started (torch-free, for metrics).

        Read from the post-batch snapshot, so it matches the cursor a checkpoint would store rather
        than the prefetch thread's read-ahead. `epoch` is fractional (tokens / stream total); skipped
        shard tails count as consumed, because the cursor has passed them.
        """
        s = self._state_after_last or self._snapshot()
        out: dict[str, dict] = {}
        for name, st in s.streams.items():
            stream = self.streams[name]
            tokens = int(st["epoch"]) * stream.total + sum(stream.sizes[: st["shard"]]) + int(st["offset"])
            out[name] = {"tokens": int(tokens), "epoch": tokens / stream.total if stream.total else 0.0,
                         "shard": int(st["shard"]), "offset": int(st["offset"])}
        return out

    def _snapshot(self) -> LoaderState:
        return LoaderState(
            {n: s.state_dict() for n, s in self.streams.items()},
            json.loads(json.dumps(self.rng.bit_generator.state)),
            self.n_batches,
            self.tokens_served,
        )

    # --------------------------------------------------------------- batching
    def _make_batch(self) -> tuple[torch.Tensor, LoaderState]:
        n = self.seq_len + 1
        rows = np.empty((self.mb, n), dtype=np.int64)
        src_idx = self.rng.choice(len(self.names), size=self.mb, p=self.probs)
        for i, si in enumerate(src_idx):
            rows[i] = self.streams[self.names[si]].next_window(n)
        self.n_batches += 1
        self.tokens_served += self.mb * self.seq_len
        t = torch.from_numpy(rows)
        if self.device.type == "cuda":
            t = t.pin_memory()
        return t, self._snapshot()

    def _producer(self) -> None:
        while not self._stop.is_set():
            item = self._make_batch()
            while not self._stop.is_set():
                try:
                    self._q.put(item, timeout=0.5)
                    break
                except queue.Full:
                    continue

    def _start_thread(self) -> None:
        self._q = queue.Queue(maxsize=self.prefetch)
        self._stop.clear()
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

    def _stop_thread(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            self._thread = None
            self._q = None

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (inputs, targets), each [mb, seq_len] on `device`."""
        if self._thread is None:
            self._start_thread()
        rows, state = self._q.get()
        self._state_after_last = state
        rows = rows.to(self.device, non_blocking=True)
        return rows[:, :-1], rows[:, 1:]

    def close(self) -> None:
        self._stop_thread()

    @property
    def total_tokens(self) -> int:
        return sum(s.total for s in self.streams.values())


class ValLoader:
    """A fixed set of windows from each source's val split (same every evaluation)."""

    def __init__(self, spec: MixtureSpec, seq_len: int, microbatch: int, n_tokens: int, device="cuda") -> None:
        streams = spec.streams()
        names = list(streams)
        w = np.array([spec.weights[n] for n in names], dtype=np.float64)
        w = w / w.sum()
        n = seq_len + 1
        windows = []
        for name, frac in zip(names, w):
            s = streams[name]
            k = max(1, int(n_tokens * frac / seq_len))
            for _ in range(k):
                windows.append(s.next_window(n))
        rows = torch.from_numpy(np.stack(windows).astype(np.int64))
        self.batches = [rows[i : i + microbatch] for i in range(0, len(rows), microbatch)]
        self.device = torch.device(device)
        self.n_tokens = len(rows) * seq_len

    def __iter__(self):
        for b in self.batches:
            b = b.to(self.device, non_blocking=True)
            yield b[:, :-1], b[:, 1:]

    def __len__(self) -> int:
        return len(self.batches)

"""Pretraining loop: resumable, token-indexed, instrumented, with an HTML progress report.

    python -m slm.train.pretrain --config configs/train/tinystories_26m.yaml [key=value ...]
    python -m slm.train.pretrain --config ... --fresh        # ignore an existing latest.pt

Stopping: Ctrl-C, or create the file runs/<run>/STOP. Either finishes the current update, writes a
full checkpoint and the report, and exits. Rerunning the same command resumes.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path

import torch

from slm.config import to_dict
from slm.data.loader import MixtureSpec, PretrainLoader, ValLoader
from slm.data.tokenizer import SlmTokenizer
from slm.eval.generation import format_samples, sample_suite
from slm.model import IGNORE_INDEX, Transformer
from slm.train.config import TrainConfig, load_train_config
from slm.train.schedule import lr_at
from slm.utils import checkpoint as ckpt
from slm.utils.logging import MetricsLogger, console, fmt_duration, fmt_tokens
from slm.utils.report import write_report
from slm.utils.sdpa import sdpa_context


class Trainer:
    def __init__(self, cfg: TrainConfig, fresh: bool = False) -> None:
        self.cfg = cfg
        self.run_dir = cfg.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "samples").mkdir(exist_ok=True)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.latest_path = self.ckpt_dir / "latest.pt"
        torch.manual_seed(cfg.runtime.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        free, total = torch.cuda.mem_get_info()
        assert free / 2**30 >= cfg.runtime.min_free_vram_gib, f"only {free / 2**30:.1f} GiB VRAM free"

        self.tok = SlmTokenizer.load(cfg.tokenizer_dir)
        self.mcfg = cfg.model_config()
        assert self.mcfg.vocab_size >= self.tok.vocab_size, "model vocab smaller than tokenizer"
        assert self.mcfg.max_seq_len >= cfg.data.seq_len
        self.model = Transformer(self.mcfg).cuda()
        self.n_params = self.model.num_params()
        self.fwd = torch.compile(self.model, dynamic=False) if cfg.runtime.compile else self.model
        self.optimizer = self._build_optimizer()
        self.accum = cfg.grad_accum

        if cfg.data.kind == "sft":
            from slm.data.sft import SftLoader, SftValLoader

            self.loader = SftLoader(Path(cfg.data.sft_root), cfg.data.mixture, cfg.data.seq_len, cfg.batch.microbatch, cfg.runtime.seed, "cuda", cfg.data.prefetch)
            self.val_loader = SftValLoader(Path(cfg.data.sft_root), cfg.data.mixture, cfg.data.seq_len, cfg.batch.microbatch, cfg.data.val_tokens)
        else:
            spec = MixtureSpec(Path(cfg.data.tokenized_root), cfg.data.mixture, "train")
            self.loader = PretrainLoader(spec, cfg.data.seq_len, cfg.batch.microbatch, cfg.runtime.seed, "cuda", cfg.data.prefetch)
            self.val_loader = ValLoader(MixtureSpec(Path(cfg.data.tokenized_root), cfg.data.mixture, "val"), cfg.data.seq_len, cfg.batch.microbatch, cfg.data.val_tokens)
        self.extra_val = None
        if cfg.data.extra_val_mixture:
            self.extra_val = ValLoader(MixtureSpec(Path(cfg.data.tokenized_root), cfg.data.extra_val_mixture, "val"), cfg.data.seq_len, cfg.batch.microbatch, cfg.data.val_tokens)
        if cfg.schedule.epochs > 0:
            cfg.schedule.total_tokens = int(cfg.schedule.epochs * self.loader.total_tokens)
            cfg.milestone_tokens = int(self.loader.total_tokens)
            console(f"[{cfg.run_name}] {cfg.schedule.epochs} epochs x {fmt_tokens(self.loader.total_tokens)} tokens = {fmt_tokens(cfg.schedule.total_tokens)} total; milestone = 1 epoch")

        self.log = MetricsLogger(self.run_dir)
        self.counters = {
            "tokens": 0, "update": 0, "elapsed_s": 0.0,
            "next_eval_at": cfg.eval.every_tokens, "next_gen_at": cfg.eval.gen_every_tokens,
            "next_milestone_at": cfg.milestone_tokens, "elapsed_at_last_milestone": 0.0,
            "best_val": float("inf"), "last_val": float("nan"),
        }
        self.stop_requested = False
        self.session_start = time.time()
        self.tok_s_ema = 0.0
        self.last_extra_val = None

        if self.latest_path.exists() and not fresh:
            self._resume()
        else:
            if cfg.init_from:
                self._init_from(cfg.init_from, cfg.init_optimizer)
            self._start_fresh()
        signal.signal(signal.SIGINT, self._on_sigint)

    # ------------------------------------------------------------------ setup
    def _build_optimizer(self) -> torch.optim.Optimizer:
        o = self.cfg.optim
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            (decay if (p.dim() >= 2 or not o.decay_only_matrices) else no_decay).append(p)
        groups = [{"params": decay, "weight_decay": o.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=o.lr, betas=tuple(o.betas), eps=o.eps, fused=o.fused)

    def _meta(self) -> dict:
        return {
            "run_name": self.cfg.run_name, "stage": "sft" if self.cfg.data.kind == "sft" else "pretrain",
            "config": to_dict(self.cfg), "model_config": to_dict(self.mcfg),
            "n_params": self.n_params, "n_params_nonembed": self.model.num_params(non_embedding=True),
            "tokenizer_sha256": self.tok.sha256, "env": ckpt.env_info(),
            "grad_accum": self.accum, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _start_fresh(self) -> None:
        meta = self._meta()
        (self.run_dir / "run.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        self.log.log("start", msg=f"fresh start: {self.n_params:,} params, accum {self.accum}, seq {self.cfg.data.seq_len}, "
                     f"mb {self.cfg.batch.microbatch}, {fmt_tokens(self.cfg.schedule.total_tokens)} tokens planned"
                     + (f"; {self._init_note}" if getattr(self, "_init_note", "") else ""))
        console(f"[{self.cfg.run_name}] fresh start. model {self.n_params:,} params ({self.model.num_params(True):,} non-embed); "
                f"{self.accum}x{self.cfg.batch.microbatch}x{self.cfg.data.seq_len} = {self.cfg.batch.tokens_per_update:,} tokens/update; "
                f"data {self.loader.total_tokens / 1e6:.0f}M tokens available; val {self.val_loader.n_tokens / 1e6:.1f}M tokens")

    def _init_from(self, path: str, with_optimizer: bool) -> None:
        ck = torch.load(path, map_location="cuda", weights_only=False)
        sd = {k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected keys in init checkpoint: {unexpected[:5]}"
        if with_optimizer and "optimizer" in ck:
            self.optimizer.load_state_dict(ck["optimizer"])
        src_tokens = ck.get("meta", {}).get("tokens") or ck.get("counters", {}).get("tokens")
        console(f"[{self.cfg.run_name}] initialized weights from {path} (trained {fmt_tokens(src_tokens or 0)} tokens; missing keys: {len(missing)}; optimizer: {with_optimizer})")
        self._init_note = f"init_from {path} @ {fmt_tokens(src_tokens or 0)} tokens"

    def _resume(self) -> None:
        ck = ckpt.load_full(self.latest_path, self.model, self.optimizer)
        self.loader.load_state_dict(ck["loader"])
        self.counters.update(ck["counters"])
        self.log.log("resume", msg=f"resumed from {self.latest_path.name} at {fmt_tokens(self.counters['tokens'])} tokens, update {self.counters['update']}")
        console(f"[{self.cfg.run_name}] resumed at {fmt_tokens(self.counters['tokens'])} tokens (update {self.counters['update']}), elapsed so far {fmt_duration(self.counters['elapsed_s'])}")

    def _on_sigint(self, *_) -> None:
        console("Ctrl-C received: finishing current update, then checkpointing...")
        self.stop_requested = True

    # ---------------------------------------------------------------- helpers
    @property
    def elapsed(self) -> float:
        return self.counters["elapsed_s"] + (time.time() - self.session_start)

    def _save_latest(self) -> None:
        self.counters["elapsed_s"] = self.elapsed
        self.session_start = time.time()
        t0 = time.time()
        ckpt.save_full(self.latest_path, self.model, self.optimizer, self.loader.state_dict(), dict(self.counters),
                       to_dict(self.cfg), {"tokenizer_sha256": self.tok.sha256, "git_commit": ckpt.git_commit(), "model_config": to_dict(self.mcfg)},
                       keep_prev=self.cfg.ckpt.keep_prev_latest)
        ckpt.update_index(self.ckpt_dir, "latest.pt", kind="latest", tokens=self.counters["tokens"], update=self.counters["update"], val_loss=self.counters["last_val"] if self.counters["last_val"] == self.counters["last_val"] else None)
        self.log.log("checkpoint", tokens=self.counters["tokens"], msg=f"latest.pt saved at {fmt_tokens(self.counters['tokens'])} tokens ({time.time() - t0:.1f}s)")

    @torch.no_grad()
    def _eval_loader(self, loader) -> float:
        loss_sum = torch.zeros((), device="cuda")
        n_sum = torch.zeros((), device="cuda")
        with sdpa_context(self.cfg.runtime.sdpa_backend), torch.autocast("cuda", dtype=torch.bfloat16):
            for x, y in loader:
                ls, n = self.fwd(x, y)
                loss_sum += ls
                n_sum += n
        return (loss_sum / n_sum).item()

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        self.model.eval()
        loss = self._eval_loader(self.val_loader)
        self.last_extra_val = self._eval_loader(self.extra_val) if self.extra_val is not None else None
        self.model.train()
        return loss, math.exp(min(loss, 20))

    def generate_samples(self) -> None:
        if not self.cfg.eval.prompts:
            return
        e = self.cfg.eval
        with sdpa_context("decode"):
            samples = sample_suite(self.model, self.tok, e.prompts, e.gen_max_new_tokens, e.gen_temperature, e.gen_top_p)
        text = f"tokens={fmt_tokens(self.counters['tokens'])} update={self.counters['update']} {time.strftime('%Y-%m-%d %H:%M:%S')}\n" + format_samples(samples)
        (self.run_dir / "samples" / f"{self.counters['tokens']:012d}.txt").write_text(text, encoding="utf-8")
        (self.run_dir / "samples" / "latest.txt").write_text(text, encoding="utf-8")

    # ------------------------------------------------------------------ train
    def train(self) -> None:
        cfg, c = self.cfg, self.counters
        total = cfg.schedule.total_tokens
        tpu = cfg.batch.tokens_per_update
        params = [p for p in self.model.parameters()]
        self.model.train()
        t_last_ckpt = t_last_report = time.time()
        t_window = time.time()
        tokens_window = 0
        loss_window = torch.zeros((), device="cuda")
        n_window = torch.zeros((), device="cuda")
        fwd_ms = bwd_ms = opt_ms = data_ms = 0.0
        ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731
        stop_file = self.run_dir / "STOP"
        initial_estimate_done = (self.run_dir / "run.json").exists() and "initial_estimate_s" in json.loads((self.run_dir / "run.json").read_text(encoding="utf-8"))
        updates_this_session = 0
        write_report(self.run_dir, "running")

        with sdpa_context(cfg.runtime.sdpa_backend):
            while c["tokens"] < total and not self.stop_requested:
                td0 = time.perf_counter()
                batches = [self.loader.next() for _ in range(self.accum)]
                n_valid = sum((y != IGNORE_INDEX).sum() for _, y in batches)
                data_ms += (time.perf_counter() - td0) * 1000
                lr = lr_at(c["tokens"], cfg.schedule, cfg.optim.lr)
                for g in self.optimizer.param_groups:
                    g["lr"] = lr
                e_f0, e_f1 = ev(), ev()
                e_f0.record()
                loss_acc = torch.zeros((), device="cuda")
                for i, (x, y) in enumerate(batches):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        ls, _ = self.fwd(x, y)
                    (ls / n_valid).backward()
                    loss_acc += ls.detach()
                e_f1.record()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                e_o1 = ev()
                e_o1.record()

                c["tokens"] += tpu
                c["update"] += 1
                updates_this_session += 1
                tokens_window += tpu
                loss_window += loss_acc
                n_window += n_valid

                if c["update"] % cfg.runtime.log_every_updates == 0:
                    torch.cuda.synchronize()
                    fb = e_f0.elapsed_time(e_f1)
                    fwd_ms, bwd_ms = fb / 3, fb * 2 / 3  # fwd/bwd are interleaved per microbatch; report the 1:2 split
                    opt_ms = e_f1.elapsed_time(e_o1)
                    now = time.time()
                    tok_s = tokens_window / (now - t_window)
                    self.tok_s_ema = tok_s if self.tok_s_ema == 0 else 0.8 * self.tok_s_ema + 0.2 * tok_s
                    loss = (loss_window / n_window).item()
                    gn = float(grad_norm)
                    eta = (total - c["tokens"]) / self.tok_s_ema if self.tok_s_ema > 0 else float("nan")
                    rec = dict(tokens=c["tokens"], update=c["update"], loss=loss, lr=lr, grad_norm=gn, tok_s=tok_s, tok_s_ema=self.tok_s_ema,
                               step_ms=(now - t_window) / cfg.runtime.log_every_updates * 1000, fwd_ms=fwd_ms, bwd_ms=bwd_ms, opt_ms=opt_ms,
                               data_ms=data_ms / cfg.runtime.log_every_updates, vram_gib=torch.cuda.max_memory_allocated() / 2**30,
                               elapsed_s=self.elapsed, eta_s=eta)
                    self.log.log("train", **rec)
                    console(f"upd {c['update']} | {fmt_tokens(c['tokens'])} ({c['tokens'] / total * 100:.1f}%) | loss {loss:.4f} | lr {lr:.2e} | gn {gn:.2f} | "
                            f"{tok_s:,.0f} tok/s | {rec['step_ms']:.0f} ms/upd (data {rec['data_ms']:.0f}) | {rec['vram_gib']:.1f} GiB | ETA {fmt_duration(eta)}")
                    if not math.isfinite(loss):
                        self.log.log("stop", tokens=c["tokens"], msg="non-finite loss; stopping without checkpoint")
                        write_report(self.run_dir, "stopped")
                        raise RuntimeError("non-finite loss")
                    t_window, tokens_window = now, 0
                    loss_window.zero_()
                    n_window.zero_()
                    data_ms = 0.0
                    if not initial_estimate_done and updates_this_session >= 3 * cfg.runtime.log_every_updates:
                        est = (total - c["tokens"]) / tok_s + self.elapsed  # latest window: excludes compile warmup
                        meta = json.loads((self.run_dir / "run.json").read_text(encoding="utf-8"))
                        meta["initial_estimate_s"] = est
                        (self.run_dir / "run.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
                        self.log.log("start", tokens=c["tokens"], msg=f"initial estimate: {fmt_duration(est)} total wall-clock at {tok_s:,.0f} tok/s")
                        console(f"*** initial estimate: whole run ~{fmt_duration(est)} at {tok_s:,.0f} tok/s ***")
                        initial_estimate_done = True

                if c["tokens"] >= c["next_eval_at"]:
                    t0 = time.time()
                    vl, ppl = self.evaluate()
                    c["last_val"] = vl
                    improved = vl < c["best_val"]
                    if improved:
                        c["best_val"] = vl
                        ckpt.save_snapshot(self.ckpt_dir / "best.pt", self.model, to_dict(self.mcfg), {"tokens": c["tokens"], "val_loss": vl, "tokenizer_sha256": self.tok.sha256})
                        ckpt.update_index(self.ckpt_dir, "best.pt", kind="best", tokens=c["tokens"], update=c["update"], val_loss=vl)
                    self.log.log("eval", tokens=c["tokens"], update=c["update"], val_loss=vl, val_ppl=ppl, best=improved, eval_s=time.time() - t0, val_pt_loss=self.last_extra_val)
                    console(f"eval @ {fmt_tokens(c['tokens'])}: val loss {vl:.4f} ppl {ppl:.2f}{' (best)' if improved else ''}"
                            + (f" | pretrain-val {self.last_extra_val:.4f}" if self.last_extra_val is not None else "") + f" [{time.time() - t0:.0f}s]")
                    c["next_eval_at"] += cfg.eval.every_tokens
                if c["tokens"] >= c["next_gen_at"]:
                    t0 = time.time()
                    self.generate_samples()
                    console(f"samples written @ {fmt_tokens(c['tokens'])} [{time.time() - t0:.0f}s]")
                    c["next_gen_at"] += cfg.eval.gen_every_tokens
                if c["tokens"] >= c["next_milestone_at"]:
                    seg = self.elapsed - c["elapsed_at_last_milestone"]
                    self.log.log("milestone", tokens=c["tokens"], update=c["update"], segment_s=seg, elapsed_s=self.elapsed,
                                 tok_s=cfg.milestone_tokens / seg if seg > 0 else 0.0, loss=loss if c["update"] >= cfg.runtime.log_every_updates else float("nan"),
                                 val_loss=c["last_val"])
                    console(f"=== milestone {fmt_tokens(c['tokens'])}: segment {fmt_duration(seg)}, elapsed {fmt_duration(self.elapsed)} ===")
                    c["elapsed_at_last_milestone"] = self.elapsed
                    c["next_milestone_at"] += cfg.milestone_tokens
                    if cfg.ckpt.snapshot_at_milestones:
                        ckpt.save_snapshot(self.ckpt_dir / ckpt.snapshot_name(c["tokens"]), self.model, to_dict(self.mcfg), {"tokens": c["tokens"], "val_loss": c["last_val"], "tokenizer_sha256": self.tok.sha256})
                        ckpt.update_index(self.ckpt_dir, ckpt.snapshot_name(c["tokens"]), kind="snapshot", tokens=c["tokens"], update=c["update"], val_loss=c["last_val"] if c["last_val"] == c["last_val"] else None)
                    self._save_latest()
                    t_last_ckpt = time.time()
                    write_report(self.run_dir, "running")
                    t_last_report = time.time()
                if time.time() - t_last_ckpt > cfg.ckpt.every_minutes * 60:
                    self._save_latest()
                    t_last_ckpt = time.time()
                if time.time() - t_last_report > cfg.eval.report_every_minutes * 60:
                    write_report(self.run_dir, "running")
                    t_last_report = time.time()
                if stop_file.exists():
                    console("STOP file found: stopping after this update")
                    stop_file.unlink()
                    self.stop_requested = True

        # ---- wrap up
        finished = c["tokens"] >= total
        if finished:
            vl, ppl = self.evaluate()
            c["last_val"] = vl
            if vl < c["best_val"]:
                c["best_val"] = vl
                ckpt.save_snapshot(self.ckpt_dir / "best.pt", self.model, to_dict(self.mcfg), {"tokens": c["tokens"], "val_loss": vl, "tokenizer_sha256": self.tok.sha256})
            self.log.log("eval", tokens=c["tokens"], update=c["update"], val_loss=vl, val_ppl=ppl, best=vl <= c["best_val"], eval_s=0, val_pt_loss=self.last_extra_val)
            self.generate_samples()
            ckpt.save_snapshot(self.ckpt_dir / "final.pt", self.model, to_dict(self.mcfg), {"tokens": c["tokens"], "val_loss": vl, "tokenizer_sha256": self.tok.sha256})
            ckpt.update_index(self.ckpt_dir, "final.pt", kind="final", tokens=c["tokens"], update=c["update"], val_loss=vl)
            if vl <= c["best_val"]:
                ckpt.update_index(self.ckpt_dir, "best.pt", kind="best", tokens=c["tokens"], update=c["update"], val_loss=vl)
        self._save_latest()
        self.log.log("finish" if finished else "stop", tokens=c["tokens"], elapsed_s=self.elapsed, msg=f"{'finished' if finished else 'stopped'} at {fmt_tokens(c['tokens'])} tokens after {fmt_duration(self.elapsed)}")
        write_report(self.run_dir, "finished" if finished else "stopped")
        self.loader.close()
        console(f"{'FINISHED' if finished else 'STOPPED'} at {fmt_tokens(c['tokens'])} tokens, elapsed {fmt_duration(self.elapsed)}, best val {c['best_val']:.4f}. Report: {self.run_dir / 'report.html'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_train_config(a.config, a.overrides)
    Trainer(cfg, fresh=a.fresh).train()


if __name__ == "__main__":
    sys.exit(main())

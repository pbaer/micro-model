"""GRPO-style RL with programmatic rewards (no critic, no reward model).

    python -m slm.train.rl --config configs/train/m6_rl_arith_149m.yaml

Per step: sample P prompts, roll out G completions each with the current policy, compute rewards and
group-relative advantages, then optimize the clipped policy objective (+ KL to the frozen reference)
over all P*G rollouts with gradient accumulation. Every rollout is persisted to
runs/<run>/rollouts/step_<n>.jsonl for failure-mode and reward-hacking inspection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import signal
import sys
import time
from pathlib import Path

import torch

from slm.config import ModelConfig, apply_overrides, from_dict, load_yaml, to_dict
from slm.data.tokenizer import SlmTokenizer
from slm.model import Transformer
from slm.rl.advantages import group_advantages
from slm.rl.objectives import entropy_from_logits, kl_penalty, policy_loss, sequence_logprobs
from slm.rl.rollout import Rollout, greedy_accuracy, rollout_group, save_rollouts
from slm.rl.tasks import make_tasks, set_task_corpus
from slm.train.rl_config import RlConfig, load_rl_config  # noqa: F401 (re-export: torch-free config for the portal)
from slm.utils import checkpoint as ckpt
from slm.utils.logging import MetricsLogger, console, fmt_duration
from slm.utils.gpu import GpuSampler
from slm.utils.report import write_report
from slm.utils.sdpa import sdpa_context


class RlTrainer:
    def __init__(self, cfg: RlConfig) -> None:
        self.cfg = cfg
        self.run_dir = cfg.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "rollouts").mkdir(exist_ok=True)
        self.ckpt_dir = self.run_dir / "checkpoints"
        torch.manual_seed(cfg.seed)
        self.rng = random.Random(cfg.seed)
        self.tok = SlmTokenizer.load(cfg.tokenizer_dir)
        self.mcfg = cfg.model_config()
        self.model = Transformer(self.mcfg).cuda()
        assert cfg.init_from, "RL needs init_from (a reasoning-SFT checkpoint)"
        ck = torch.load(cfg.init_from, map_location="cuda", weights_only=False)
        self.model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
        self.ref = copy.deepcopy(self.model).to(torch.bfloat16).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        decay = [p for p in self.model.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.model.parameters() if p.dim() < 2]
        self.optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr, betas=tuple(cfg.betas), fused=True)
        set_task_corpus(self.tok, cfg.ingredients_corpus or None)  # real sentences for the pytool/constraint grammars
        self.train_tasks = make_tasks(cfg.tasks, cfg.n_train_prompts, "train", cfg.seed)
        self.heldout_tasks = make_tasks(cfg.tasks, cfg.n_heldout_prompts, "heldout", cfg.seed)
        assert not ({t.prompt for t in self.train_tasks} & {t.prompt for t in self.heldout_tasks}), "train/heldout leak"
        self.log = MetricsLogger(self.run_dir)
        self.step = 0
        self.tokens = 0  # completion tokens optimized so far
        self.stop_requested = False
        self.session_start = time.time()
        self.elapsed_before = 0.0
        latest = self.ckpt_dir / "latest.pt"
        idx_path = self.ckpt_dir / "index.json"
        self.best_acc = -1.0  # best held-out accuracy so far (best.pt); read back from the index on resume
        if idx_path.exists():
            try:
                self.best_acc = float(json.loads(idx_path.read_text(encoding="utf-8")).get("best.pt", {}).get("heldout_acc", -1.0))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        if latest.exists():
            st = ckpt.load_full(latest, self.model, self.optimizer)
            self.step, self.tokens, self.elapsed_before = st["counters"]["step"], st["counters"]["tokens"], st["counters"]["elapsed_s"]
            self.rng.setstate(st["counters"]["py_rng"])
            self.log.log("resume", tokens=self.tokens, msg=f"resumed at step {self.step}")
        else:
            meta = {"run_name": cfg.run_name, "stage": "grpo", "config": to_dict(cfg), "model_config": to_dict(self.mcfg), "n_params": self.model.num_params(),
                    "env": ckpt.env_info(), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "tokenizer_sha256": self.tok.sha256}
            (self.run_dir / "run.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
            self.log.log("start", tokens=0, msg=f"RL start from {cfg.init_from}; tasks {cfg.tasks}; G={cfg.group_size} P={cfg.prompts_per_step}")
        signal.signal(signal.SIGINT, self._sigint)

    def _sigint(self, *_):
        console("Ctrl-C: finishing step then checkpointing")
        self.stop_requested = True

    @property
    def elapsed(self) -> float:
        return self.elapsed_before + time.time() - self.session_start

    def _save(self, name: str = "latest.pt") -> None:
        ckpt.save_full(self.ckpt_dir / name, self.model, self.optimizer, {}, {"step": self.step, "tokens": self.tokens, "elapsed_s": self.elapsed, "py_rng": self.rng.getstate()},
                       to_dict(self.cfg), {"tokenizer_sha256": self.tok.sha256, "model_config": to_dict(self.mcfg), "git_commit": ckpt.git_commit(), "tokens": self.tokens})

    # ------------------------------------------------------------------ core
    def collect(self) -> tuple[list[Rollout], dict]:
        c = self.cfg
        tasks = self.rng.sample(self.train_tasks, c.prompts_per_step)
        rollouts: list[Rollout] = []
        group_stds, all_zero = [], 0
        for t in tasks:
            with sdpa_context("decode"):
                g = rollout_group(self.model, self.tok, t, c.group_size, c.max_new_tokens, c.temperature, c.top_p, c.top_k, seed=self.rng.randrange(2**31),
                                  think_required=c.think_required, reward_scheme=c.reward_scheme, ref_model=self.ref, checkpoint=c.init_from, step=self.step,
                                  tools=c.tools, max_tool_calls=c.max_tool_calls, reward_schemes=c.reward_schemes)
            r = torch.tensor([x.reward for x in g])
            adv = group_advantages(r, c.normalize_std)
            for x, a in zip(g, adv.tolist()):
                x.advantage = a
            group_stds.append(float(r.std(unbiased=False)))
            all_zero += int(float(r.std(unbiased=False)) == 0.0)
            rollouts.extend(g)
        rewards = [x.reward for x in rollouts]
        stats = {
            "reward_mean": sum(rewards) / len(rewards), "success_rate": sum(x.correct for x in rollouts) / len(rollouts),
            "score_mean": sum(x.fraction if x.fraction is not None else float(x.correct) for x in rollouts) / len(rollouts),
            "group_std_mean": sum(group_stds) / len(group_stds), "groups_no_signal": all_zero / len(tasks),
            "adv_abs_mean": sum(abs(x.advantage) for x in rollouts) / len(rollouts),
            "len_mean": sum(x.n_tokens for x in rollouts) / len(rollouts), "len_max": max(x.n_tokens for x in rollouts),
            "malformed_rate": sum(x.malformed for x in rollouts) / len(rollouts), "length_term_rate": sum(x.termination == "length" for x in rollouts) / len(rollouts),
            "len_correct": (sum(x.n_tokens for x in rollouts if x.correct) / max(1, sum(x.correct for x in rollouts))),
            "len_wrong": (sum(x.n_tokens for x in rollouts if not x.correct) / max(1, sum(not x.correct for x in rollouts))),
            "tool_calls_mean": sum(x.tool_calls for x in rollouts) / len(rollouts),
            "tool_error_rate": sum(x.tool_errors for x in rollouts) / max(1, sum(x.tool_calls for x in rollouts)),
            "tool_use_rate": sum(x.tool_calls > 0 for x in rollouts) / len(rollouts),
            "answer_from_tool_rate": sum(x.answer_from_tool for x in rollouts) / len(rollouts),
        }
        return rollouts, stats

    def optimize(self, rollouts: list[Rollout]) -> dict:
        c = self.cfg
        self.model.train()
        live = [r for r in rollouts if r.advantage != 0.0]
        n_tok_total = sum((sum(r.gen_mask) if r.gen_mask else r.n_tokens) for r in live)  # only model-sampled tokens are optimized
        agg = {"policy_loss": 0.0, "kl": 0.0, "entropy": 0.0, "clip_frac": 0.0, "ratio_mean": 0.0, "n_minibatches": 0}
        if not live:
            return {**agg, "grad_norm": 0.0, "skipped": True}
        gn = 0.0
        for _ in range(c.ppo_epochs):
            order = live[:]
            self.rng.shuffle(order)
            self.optimizer.zero_grad(set_to_none=True)
            for i in range(0, len(order), c.microbatch):
                mb = order[i : i + c.microbatch]
                L = max(len(r.prompt_ids) + r.n_tokens for r in mb)
                ids = torch.full((len(mb), L), self.tok.pad_id, dtype=torch.long, device="cuda")
                mask = torch.zeros((len(mb), L - 1), device="cuda")
                old = torch.zeros((len(mb), L - 1), device="cuda")
                ref = torch.zeros((len(mb), L - 1), device="cuda")
                adv = torch.tensor([r.advantage for r in mb], device="cuda")
                for j, r in enumerate(mb):
                    P = len(r.prompt_ids)
                    ids[j, : P + r.n_tokens] = torch.tensor(r.prompt_ids + r.completion_ids, device="cuda")
                    mask[j, P - 1 : P - 1 + r.n_tokens] = torch.tensor(r.gen_mask, dtype=torch.float32, device="cuda") if r.gen_mask else 1.0
                    old[j, P - 1 : P - 1 + r.n_tokens] = torch.tensor(r.old_logprobs, device="cuda")
                    ref[j, P - 1 : P - 1 + r.n_tokens] = torch.tensor(r.ref_logprobs, device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.model(ids[:, :-1])
                logp = sequence_logprobs(logits, ids[:, 1:])
                pl, st = policy_loss(logp, old, adv, mask, c.clip_eps, c.use_ratio)
                kl = kl_penalty(logp, ref, mask, c.kl_kind)
                ent = (entropy_from_logits(logits) * mask).sum()
                loss = (pl + c.kl_coef * kl) / n_tok_total
                loss.backward()
                agg["policy_loss"] += float(pl) / n_tok_total
                agg["kl"] += float(kl) / n_tok_total
                agg["entropy"] += float(ent) / n_tok_total
                agg["clip_frac"] += st["clip_frac"] * mask.sum().item() / n_tok_total
                agg["ratio_mean"] += st["ratio_mean"] * mask.sum().item() / n_tok_total
                agg["n_minibatches"] += 1
            gn = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip))
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        self.tokens += n_tok_total
        return {**agg, "grad_norm": gn, "skipped": False, "n_live": len(live)}

    def evaluate(self) -> dict:
        c = self.cfg
        self.model.eval()
        with sdpa_context("decode"):
            held = greedy_accuracy(self.model, self.tok, self.heldout_tasks, c.eval_max_new_tokens, c.think_required, tools=c.tools, max_tool_calls=c.max_tool_calls)
            train_sub = self.rng.sample(self.train_tasks, min(len(self.heldout_tasks), len(self.train_tasks)))
            tr = greedy_accuracy(self.model, self.tok, train_sub, c.eval_max_new_tokens, c.think_required, tools=c.tools, max_tool_calls=c.max_tool_calls)
        self.model.train()
        return {"heldout_acc": held["accuracy"], "heldout_score": held["score"], "heldout_malformed": held["malformed_rate"], "heldout_len": held["mean_len"],
                "train_acc": tr["accuracy"], "train_score": tr["score"], "train_len": tr["mean_len"]}

    def train(self) -> None:
        c = self.cfg
        t_ckpt = t_rep = time.time()
        gpu = GpuSampler(c.gpu_sample_s, c.gpu_warn_temp_c).start()
        with sdpa_context(c.sdpa_backend):
            if self.step == 0:
                ev = self.evaluate()
                self.log.log("eval", tokens=0, update=0, step=0, val_loss=1.0 - ev["heldout_acc"], val_ppl=0.0, best=True, **ev)
                console(f"pre-RL: heldout acc {ev['heldout_acc']:.3f} (malformed {ev['heldout_malformed']:.2f}, len {ev['heldout_len']:.0f}) | train acc {ev['train_acc']:.3f}")
                self.best_acc = ev["heldout_acc"]
                ckpt.save_snapshot(self.ckpt_dir / "best.pt", self.model, to_dict(self.mcfg), {"step": 0, "tokens": 0, "heldout_acc": ev["heldout_acc"], "tokenizer_sha256": self.tok.sha256})
                ckpt.update_index(self.ckpt_dir, "best.pt", kind="best", tokens=0, update=0, heldout_acc=ev["heldout_acc"])
            while self.step < c.total_steps and not self.stop_requested:
                t0 = time.time()
                rollouts, rs = self.collect()
                t1 = time.time()
                os_ = self.optimize(rollouts)
                self.step += 1
                save_rollouts(rollouts, self.run_dir / "rollouts" / f"step_{self.step:05d}.jsonl")
                rec = dict(tokens=self.tokens, update=self.step, step=self.step, loss=os_["policy_loss"], lr=c.lr, tok_s=0.0, tok_s_ema=0.0,
                           step_ms=(time.time() - t0) * 1000, fwd_ms=(t1 - t0) * 1000, bwd_ms=(time.time() - t1) * 1000, opt_ms=0.0, data_ms=0.0,
                           vram_gib=torch.cuda.max_memory_allocated() / 2**30, elapsed_s=self.elapsed, eta_s=(c.total_steps - self.step) * (time.time() - t0),
                           **rs, **{k: v for k, v in os_.items() if k != "skipped"}, **gpu.record())
                self.log.log("train", **rec)
                if (hot := gpu.hot_warning()) is not None:
                    console(f"*** WARNING: {hot} ***")
                    self.log.log("warn", tokens=self.tokens, update=self.step, msg=hot, gpu_temp_c=rec["gpu_temp_c"], gpu_power_w=rec["gpu_power_w"])
                console(f"step {self.step} | reward {rs['reward_mean']:.3f} succ {rs['success_rate']:.2f} | len {rs['len_mean']:.0f} | malformed {rs['malformed_rate']:.2f} | "
                        f"kl {os_['kl']:.4f} ent {os_['entropy']:.2f} clip {os_['clip_frac']:.2f} | gn {os_['grad_norm']:.2f} | no-signal groups {rs['groups_no_signal']:.2f} | "
                        f"rollout {t1 - t0:.0f}s opt {time.time() - t1:.0f}s")
                if (c.entropy_stop and os_["entropy"] > c.entropy_stop) or (c.kl_stop and os_["kl"] > c.kl_stop):
                    msg = f"collapse guard: entropy {os_['entropy']:.2f} (limit {c.entropy_stop}) kl {os_['kl']:.3f} (limit {c.kl_stop}); stopping, best.pt keeps the best held-out policy"
                    console(f"*** {msg} ***")
                    self.log.log("warn", tokens=self.tokens, update=self.step, msg=msg)
                    self.stop_requested = True
                if self.step % c.eval_every_steps == 0 or self.stop_requested:
                    ev = self.evaluate()
                    improved = ev["heldout_acc"] > self.best_acc
                    self.log.log("eval", tokens=self.tokens, update=self.step, step=self.step, val_loss=1.0 - ev["heldout_acc"], val_ppl=0.0, best=improved, **ev)
                    console(f"eval step {self.step}: heldout acc {ev['heldout_acc']:.3f} (malformed {ev['heldout_malformed']:.2f}, len {ev['heldout_len']:.0f}) | train acc {ev['train_acc']:.3f}{' (best)' if improved else ''}")
                    ckpt.save_snapshot(self.ckpt_dir / f"step_{self.step:05d}.pt", self.model, to_dict(self.mcfg), {"step": self.step, "tokens": self.tokens, "heldout_acc": ev["heldout_acc"], "tokenizer_sha256": self.tok.sha256})
                    ckpt.update_index(self.ckpt_dir, f"step_{self.step:05d}.pt", kind="snapshot", tokens=self.tokens, update=self.step, heldout_acc=ev["heldout_acc"])
                    if improved:
                        self.best_acc = ev["heldout_acc"]
                        ckpt.save_snapshot(self.ckpt_dir / "best.pt", self.model, to_dict(self.mcfg), {"step": self.step, "tokens": self.tokens, "heldout_acc": ev["heldout_acc"], "tokenizer_sha256": self.tok.sha256})
                        ckpt.update_index(self.ckpt_dir, "best.pt", kind="best", tokens=self.tokens, update=self.step, heldout_acc=ev["heldout_acc"])
                if time.time() - t_ckpt > c.ckpt_every_minutes * 60:
                    self._save()
                    t_ckpt = time.time()
                if time.time() - t_rep > c.report_every_minutes * 60:
                    write_report(self.run_dir, "running")
                    t_rep = time.time()
                if (self.run_dir / "STOP").exists():
                    (self.run_dir / "STOP").unlink()
                    self.stop_requested = True
        gpu.stop()
        finished = self.step >= c.total_steps
        if finished:
            ckpt.save_snapshot(self.ckpt_dir / "final.pt", self.model, to_dict(self.mcfg), {"step": self.step, "tokens": self.tokens, "tokenizer_sha256": self.tok.sha256})
            ckpt.update_index(self.ckpt_dir, "final.pt", kind="final", tokens=self.tokens, update=self.step)
        self._save()
        self.log.log("finish" if finished else "stop", tokens=self.tokens, elapsed_s=self.elapsed, msg=f"{'finished' if finished else 'stopped'} at step {self.step} after {fmt_duration(self.elapsed)}")
        write_report(self.run_dir, "finished" if finished else "stopped")
        console(f"{'FINISHED' if finished else 'STOPPED'} at step {self.step}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    RlTrainer(load_rl_config(a.config, a.overrides)).train()


if __name__ == "__main__":
    sys.exit(main())

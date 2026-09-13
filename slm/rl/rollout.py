"""Group rollouts: sample G completions for one prompt, score them, and record everything.

One prompt per batch means all rows share the prompt length, so generation needs no padding.
Old-policy and reference log-probs are recomputed teacher-forced after sampling (same numerics as
the training forward), which keeps the importance ratio honest.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from slm.data.chat import format_chat, parse_assistant
from slm.data.tokenizer import SlmTokenizer
from slm.eval.sampling import sample_next
from slm.model import KVCache, Transformer
from slm.rl.objectives import sequence_logprobs
from slm.rl.rewards import reward_from_verdict, verify_numeric
from slm.rl.tasks import Task, prompt_messages


@dataclass
class Rollout:
    prompt_id: str
    task: str
    prompt: str
    gold: str
    prompt_ids: list[int]
    completion_ids: list[int]
    text: str
    think: str | None
    answer: str
    parsed: str | None
    correct: bool
    reward: float
    verifier: str
    malformed: bool
    termination: str
    n_tokens: int
    old_logprobs: list[float]
    ref_logprobs: list[float] | None
    advantage: float = 0.0
    temperature: float = 1.0
    top_p: float = 1.0
    checkpoint: str = ""
    step: int = 0
    meta: dict = field(default_factory=dict)


@torch.no_grad()
def teacher_forced_logprobs(model: Transformer, prompt_ids: list[int], completions: list[list[int]], pad_id: int) -> list[list[float]]:
    """Per-token log-probs of each completion given the shared prompt (right-padded batch)."""
    device = next(model.parameters()).device
    P = len(prompt_ids)
    L = max(len(c) for c in completions)
    B = len(completions)
    ids = torch.full((B, P + L), pad_id, dtype=torch.long, device=device)
    for i, c in enumerate(completions):
        ids[i, :P] = torch.tensor(prompt_ids, device=device)
        ids[i, P : P + len(c)] = torch.tensor(c, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(ids[:, :-1])
    lp = sequence_logprobs(logits, ids[:, 1:])  # position t predicts token t+1
    out = []
    for i, c in enumerate(completions):
        out.append(lp[i, P - 1 : P - 1 + len(c)].tolist())
    return out


@torch.no_grad()
def rollout_group(
    model: Transformer,
    tok: SlmTokenizer,
    task: Task,
    group_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    top_p: float = 0.95,
    top_k: int = 0,
    seed: int | None = None,
    think_required: bool = True,
    reward_scheme: str = "binary",
    ref_model: Transformer | None = None,
    checkpoint: str = "",
    step: int = 0,
) -> list[Rollout]:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    prompt_ids = format_chat(tok, prompt_messages(task), add_generation_prompt=True, think_required=think_required).ids
    B, P = group_size, len(prompt_ids)
    max_new_tokens = max(1, min(max_new_tokens, model.cfg.max_seq_len - P))
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    cache = KVCache(model.cfg, B, P + max_new_tokens, device, model.output_weight.dtype)
    cur = torch.tensor([prompt_ids] * B, device=device)
    stop = {tok.end_id, tok.eos_id}
    done = torch.zeros(B, dtype=torch.bool, device=device)
    comps: list[list[int]] = [[] for _ in range(B)]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for _ in range(max_new_tokens):
            logits = model(cur, cache=cache, last_only=True)[:, -1, :].float()
            nxt = sample_next(logits, temperature, top_p, top_k, gen)
            for i in range(B):
                if not done[i]:
                    comps[i].append(int(nxt[i]))
                    if int(nxt[i]) in stop:
                        done[i] = True
            if bool(done.all()):
                break
            cur = nxt[:, None]
    old_lp = teacher_forced_logprobs(model, prompt_ids, comps, tok.pad_id)
    ref_lp = teacher_forced_logprobs(ref_model, prompt_ids, comps, tok.pad_id) if ref_model is not None else None
    if was_training:
        model.train()
    out = []
    for i, c in enumerate(comps):
        parsed = parse_assistant(tok, c)
        v = verify_numeric(parsed["answer"], task.answer)
        malformed = bool(parsed["malformed"]) or not parsed["terminated"]
        out.append(Rollout(
            prompt_id=task.id, task=task.task, prompt=task.prompt, gold=task.answer, prompt_ids=prompt_ids, completion_ids=c,
            text=tok.decode(c), think=parsed["think"], answer=parsed["answer"], parsed=v.parsed, correct=v.correct,
            reward=reward_from_verdict(v, malformed, reward_scheme), verifier=v.reason, malformed=malformed,
            termination="stop" if parsed["terminated"] else "length", n_tokens=len(c), old_logprobs=old_lp[i],
            ref_logprobs=ref_lp[i] if ref_lp is not None else None, temperature=temperature, top_p=top_p, checkpoint=checkpoint, step=step,
        ))
    return out


def save_rollouts(rollouts: list[Rollout], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rollouts:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


@torch.no_grad()
def greedy_accuracy(model: Transformer, tok: SlmTokenizer, tasks: list[Task], max_new_tokens: int = 256, think_required: bool = True, batch: int = 16) -> dict:
    """Greedy decode each task once (one prompt per row; prompts padded on the LEFT via grouping by length)."""
    correct = 0
    malformed = 0
    lengths = []
    for t in tasks:
        r = rollout_group(model, tok, t, group_size=1, max_new_tokens=max_new_tokens, temperature=0.0, think_required=think_required)[0]
        correct += int(r.correct)
        malformed += int(r.malformed)
        lengths.append(r.n_tokens)
    n = max(1, len(tasks))
    return {"accuracy": correct / n, "malformed_rate": malformed / n, "mean_len": sum(lengths) / n, "n": len(tasks)}

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
from slm.rl.rewards import answer_from_tool, reward_from_verdict, verify_numeric
from slm.rl.tasks import Task, prompt_messages
from slm.tools.loop import sample_with_tools
from slm.tools.protocol import render_tools


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
    gen_mask: list[int] = field(default_factory=list)  # 1 = model-sampled, 0 = inserted tool result (empty = all ones)
    tool_calls: int = 0
    tool_errors: int = 0
    tool_results: list[list[str]] = field(default_factory=list)  # [code, result] per call
    answer_from_tool: bool = False  # the final number was produced by a (non-trivial) call
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
def sample_completions(
    model: Transformer,
    tok: SlmTokenizer,
    prompts: list[list[int]],
    max_new_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
    generator: torch.Generator | None = None,
    stop: set[int] | None = None,
) -> list[list[int]]:
    """Sample one completion per prompt. All prompts must have the same length (no padding needed)."""
    device = next(model.parameters()).device
    B, P = len(prompts), len(prompts[0])
    assert all(len(p) == P for p in prompts), "prompts must share a length"
    max_new_tokens = max(1, min(max_new_tokens, model.cfg.max_seq_len - P))
    stop = stop or {tok.end_id, tok.eos_id}
    cache = KVCache(model.cfg, B, P + max_new_tokens, device, model.output_weight.dtype)
    cur = torch.tensor(prompts, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    comps: list[list[int]] = [[] for _ in range(B)]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for _ in range(max_new_tokens):
            logits = model(cur, cache=cache, last_only=True)[:, -1, :].float()
            nxt = sample_next(logits, temperature, top_p, top_k, generator)
            nl = nxt.tolist()
            dl = done.tolist()
            for i in range(B):
                if not dl[i]:
                    comps[i].append(nl[i])
                    if nl[i] in stop:
                        done[i] = True
            if bool(done.all()):
                break
            cur = nxt[:, None]
    return comps


def _make_rollout(tok, task, prompt_ids, c, old_lp, ref_lp, temperature, top_p, reward_scheme, checkpoint, step, tc=None) -> Rollout:
    parsed = parse_assistant(tok, c)
    v = verify_numeric(parsed["answer"], task.answer)
    malformed = bool(parsed["malformed"]) or not parsed["terminated"]
    calls = list(tc.calls) if tc is not None else []
    aft = answer_from_tool(v.parsed, calls)
    return Rollout(
        prompt_id=task.id, task=task.task, prompt=task.prompt, gold=task.answer, prompt_ids=prompt_ids, completion_ids=c,
        text=render_tools(tok, c), think=parsed["think"], answer=parsed["answer"], parsed=v.parsed, correct=v.correct,
        reward=reward_from_verdict(v, malformed, reward_scheme, aft), verifier=v.reason, malformed=malformed,
        termination="stop" if parsed["terminated"] else (tc.termination if tc is not None else "length"), n_tokens=len(c), old_logprobs=old_lp,
        ref_logprobs=ref_lp, gen_mask=(tc.gen_mask if tc is not None else []), tool_calls=(tc.n_calls if tc is not None else 0),
        tool_errors=(tc.n_errors if tc is not None else 0), tool_results=[list(x) for x in calls], answer_from_tool=aft,
        temperature=temperature, top_p=top_p, checkpoint=checkpoint, step=step,
    )


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
    tools: bool = False,
    max_tool_calls: int = 8,
) -> list[Rollout]:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    prompt_ids = format_chat(tok, prompt_messages(task), add_generation_prompt=True, think_required=think_required).ids
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    if tools:
        tcs = sample_with_tools(model, tok, [prompt_ids] * group_size, max_new_tokens, temperature, top_p, top_k, gen, max_calls=max_tool_calls)
        comps = [t.ids for t in tcs]
    else:
        tcs = [None] * group_size
        comps = sample_completions(model, tok, [prompt_ids] * group_size, max_new_tokens, temperature, top_p, top_k, gen)
    old_lp = teacher_forced_logprobs(model, prompt_ids, comps, tok.pad_id)
    ref_lp = teacher_forced_logprobs(ref_model, prompt_ids, comps, tok.pad_id) if ref_model is not None else None
    if was_training:
        model.train()
    return [_make_rollout(tok, task, prompt_ids, c, old_lp[i], ref_lp[i] if ref_lp is not None else None, temperature, top_p, reward_scheme, checkpoint, step, tcs[i]) for i, c in enumerate(comps)]


def save_rollouts(rollouts: list[Rollout], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rollouts:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


@torch.no_grad()
def greedy_accuracy(model: Transformer, tok: SlmTokenizer, tasks: list[Task], max_new_tokens: int = 256, think_required: bool = True, batch: int = 32,
                    tools: bool = False, max_tool_calls: int = 8, keep: list | None = None) -> dict:
    """Greedy decode every task once, batched over prompts of equal token length. `keep` collects the Rollouts."""
    was_training = model.training
    model.eval()
    enc = [(t, format_chat(tok, prompt_messages(t), add_generation_prompt=True, think_required=think_required).ids) for t in tasks]
    by_len: dict[int, list] = {}
    for t, ids in enc:
        by_len.setdefault(len(ids), []).append((t, ids))
    correct = malformed = calls = errors = used = from_tool = 0
    lengths = []
    for group in by_len.values():
        for b in range(0, len(group), batch):
            chunk = group[b : b + batch]
            if tools:
                tcs = sample_with_tools(model, tok, [ids for _, ids in chunk], max_new_tokens, 0.0, max_calls=max_tool_calls)
                comps = [t.ids for t in tcs]
            else:
                tcs = [None] * len(chunk)
                comps = sample_completions(model, tok, [ids for _, ids in chunk], max_new_tokens, 0.0)
            for (t, ids), c, tc in zip(chunk, comps, tcs):
                r = _make_rollout(tok, t, ids, c, [], None, 0.0, 1.0, "binary", "", 0, tc)
                correct += int(r.correct)
                malformed += int(r.malformed)
                calls += r.tool_calls
                errors += r.tool_errors
                used += int(r.tool_calls > 0)
                from_tool += int(r.answer_from_tool)
                lengths.append(r.n_tokens)
                if keep is not None:
                    keep.append(r)
    if was_training:
        model.train()
    n = max(1, len(tasks))
    return {"accuracy": correct / n, "malformed_rate": malformed / n, "mean_len": sum(lengths) / n, "n": len(tasks),
            "tool_calls_mean": calls / n, "tool_error_rate": errors / max(1, calls), "tool_use_rate": used / n, "answer_from_tool_rate": from_tool / n}

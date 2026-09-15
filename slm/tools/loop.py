"""Generation with tool calls, batched.

`sample_with_tools` samples completions for a batch of prompts, pausing each row at <|/python_call|>,
running the code in that row's session (state persists across calls), appending
<|python_result|>...<|/python_result|>, and resuming. Rows diverge in length
once results are inserted, so decoding runs in rounds: rows still active are grouped by their current
length and each group is decoded with the plain same-length sampler (the prefix is recomputed each
round, which is cheap at these lengths). Every returned token carries a gen_mask bit: 1 if the model
sampled it, 0 if the harness inserted it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from slm.data.tokenizer import SlmTokenizer
from slm.model import Transformer
from slm.tools.protocol import run_tool, tool_ids
from slm.tools.pysandbox import PySession


@dataclass
class ToolCompletion:
    ids: list[int] = field(default_factory=list)  # everything after the prompt: sampled tokens and inserted results
    gen_mask: list[int] = field(default_factory=list)  # 1 = sampled by the model, 0 = inserted tool result
    n_calls: int = 0
    n_errors: int = 0
    calls: list[tuple[str, str]] = field(default_factory=list)  # (code, result) per call, in order
    termination: str = "length"  # stop | length | max_calls


@torch.no_grad()
def sample_with_tools(
    model: Transformer,
    tok: SlmTokenizer,
    prompts: list[list[int]],
    max_new_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
    generator: torch.Generator | None = None,
    max_calls: int = 8,
    tools: bool = True,
    sessions: list[PySession] | None = None,
) -> list[ToolCompletion]:
    from slm.rl.rollout import sample_completions  # local import: rollout imports this module

    t = tool_ids(tok)
    sessions = sessions or [PySession() for _ in prompts]
    stop = {tok.end_id, tok.eos_id} | ({t["call_close"]} if tools else set())
    outs = [ToolCompletion() for _ in prompts]
    seqs = [list(p) for p in prompts]
    budget = [max_new_tokens] * len(prompts)  # sampled tokens still allowed per row
    active = set(range(len(prompts)))
    while active:
        by_len: dict[int, list[int]] = {}
        for i in active:
            by_len.setdefault(len(seqs[i]), []).append(i)
        for L, rows in by_len.items():
            room = model.cfg.max_seq_len - L - 1
            n = min(min(budget[i] for i in rows), room)
            if n <= 0:
                for i in rows:
                    outs[i].termination = "length"
                    active.discard(i)
                continue
            comps = sample_completions(model, tok, [seqs[i] for i in rows], n, temperature, top_p, top_k, generator, stop)
            for i, c in zip(rows, comps):
                seqs[i].extend(c)
                outs[i].ids.extend(c)
                outs[i].gen_mask.extend([1] * len(c))
                budget[i] -= len(c)
                last = c[-1] if c else None
                if last in (tok.end_id, tok.eos_id):
                    outs[i].termination = "stop"
                    active.discard(i)
                elif tools and last == t["call_close"]:
                    outs[i].n_calls += 1
                    if outs[i].n_calls > max_calls:
                        outs[i].termination = "max_calls"
                        active.discard(i)
                        continue
                    # the code is what follows the last <|python_call|> in this row's completion
                    comp = outs[i].ids
                    opens = [k for k, x in enumerate(comp) if x == t["call_open"]]
                    if opens and opens[-1] < len(comp) - 1:
                        call = tok.decode(comp[opens[-1] + 1 : -1], skip_special=True)
                        result, ok = run_tool(call, sessions[i])
                    else:
                        call, result, ok = "", "error: malformed call (no <|python_call|> opening tag)", False
                    outs[i].n_errors += int(not ok)
                    outs[i].calls.append((call, result))
                    ins = [t["result_open"], *tok.encode(result), t["result_close"]]
                    if len(seqs[i]) + len(ins) >= model.cfg.max_seq_len - 1:
                        outs[i].termination = "length"
                        active.discard(i)
                        continue
                    seqs[i].extend(ins)
                    outs[i].ids.extend(ins)
                    outs[i].gen_mask.extend([0] * len(ins))
                    if budget[i] <= 0:
                        outs[i].termination = "length"
                        active.discard(i)
                elif budget[i] <= 0:
                    outs[i].termination = "length"
                    active.discard(i)
                # else: this row hit the group's shared budget but still has room; next round continues it
    return outs

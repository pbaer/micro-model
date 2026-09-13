"""Fixed-seed completions from a static prompt suite, for qualitative tracking during training."""

from __future__ import annotations

import torch

from slm.data.tokenizer import SlmTokenizer
from slm.model import Transformer


@torch.no_grad()
def sample_suite(
    model: Transformer,
    tok: SlmTokenizer,
    prompts: list[str],
    max_new_tokens: int = 96,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 1234,
    greedy_too: bool = True,
) -> list[dict]:
    """Each prompt is encoded as <|bos|> + text (a document start). Returns decoded completions."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    out = []
    for p in prompts:
        ids = torch.tensor([[tok.bos_id, *tok.encode(p)]], device=device)
        rec = {"prompt": p}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if greedy_too:
                g = model.generate(ids, max_new_tokens, temperature=0.0, stop_ids=(tok.eos_id,))
                rec["greedy"] = tok.decode(g[0, ids.shape[1] :].tolist())
            gen = torch.Generator(device=device).manual_seed(seed)
            s = model.generate(ids, max_new_tokens, temperature=temperature, top_p=top_p, stop_ids=(tok.eos_id,), generator=gen)
            rec["sampled"] = tok.decode(s[0, ids.shape[1] :].tolist())
        out.append(rec)
    if was_training:
        model.train()
    return out


def format_samples(samples: list[dict]) -> str:
    lines = []
    for s in samples:
        lines.append("=" * 80)
        lines.append(f"PROMPT: {s['prompt']!r}")
        if "greedy" in s:
            lines.append(f"--- greedy:\n{s['greedy']}")
        lines.append(f"--- sampled:\n{s['sampled']}")
    return "\n".join(lines) + "\n"

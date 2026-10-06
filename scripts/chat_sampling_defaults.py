"""Which sampling settings should the portal's chat mode default to? The judged suite's chat prompts plus a few
conversational ones, sampled under several (temperature, top-p, top-k) settings with a few seeds each; reports the
share of replies that close properly, repetition, length, diversity across seeds and tool misfires on non-math prompts.

    .venv/Scripts/python.exe -u scripts/chat_sampling_defaults.py --checkpoint runs/m10_rl_336m/checkpoints/final.pt
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

CHATTY = [
    "Tell me a short story about a dragon who is afraid of heights.",
    "What should I cook tonight? I have eggs, spinach and rice.",
    "Explain how rain forms, for a ten-year-old.",
    "Write a haiku about autumn.",
    "Give me three tips for sleeping better.",
    "What is the difference between a lake and a pond?",
    "Why is the sky blue?",
    "Write a short poem about the sea.",
    "Suggest a good name for a grey cat and explain why.",
    "I feel nervous about a job interview tomorrow. Any advice?",
]

SETTINGS = [  # (temperature, top_p, top_k)
    (0.0, 1.0, 0),
    (0.5, 0.9, 40),
    (0.6, 0.9, 40),
    (0.7, 0.9, 40),
    (0.7, 0.95, 0),
    (0.8, 0.9, 40),
    (0.8, 0.95, 0),
    (1.0, 0.9, 40),
]


def repeated_ngram(text: str, n: int = 4) -> bool:
    w = text.split()
    seen = set()
    for i in range(len(w) - n + 1):
        g = tuple(w[i : i + n])
        if g in seen:
            return True
        seen.add(g)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=r"C:\slm-data\tokenizer\v1")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--out", default="runs/m10_rl_336m/chat_sampling_defaults.json")
    a = ap.parse_args()

    import torch

    from slm.data.chat import format_chat, parse_assistant
    from slm.data.tokenizer import SlmTokenizer
    from slm.eval.quality import load_model
    from slm.eval.quality_suite import SUITE
    from slm.tools.loop import sample_with_tools
    from slm.utils.sdpa import sdpa_context

    model, _ = load_model(Path(a.checkpoint), "cuda")
    tok = SlmTokenizer.load(a.tokenizer)
    prompts = [(p["id"], p["category"], p["chat"]) for p in SUITE if p["category"] != "bash"] + [(f"chat{i}", "chatty", t) for i, t in enumerate(CHATTY)]
    ids = [format_chat(tok, [{"role": "user", "content": t}], add_generation_prompt=True, think_required=True, tools=True).ids for _, _, t in prompts]
    rows = []
    print(f"{len(prompts)} prompts x {len(SETTINGS)} settings x {a.seeds} seeds", flush=True)
    with torch.no_grad(), sdpa_context("decode"):
        for temp, top_p, top_k in SETTINGS:
            answers_by_prompt: dict[str, list[str]] = {}
            stats = {"n": 0, "closed": 0, "repeat": 0, "len": 0, "tool_nonmath": 0, "nonmath": 0, "empty": 0}
            t0 = time.time()
            for seed in range(a.seeds if temp > 0 else 1):
                gen = torch.Generator(device="cuda"); gen.manual_seed(100 + seed)
                outs = sample_with_tools(model, tok, ids, a.max_new, temperature=temp, top_p=top_p, top_k=top_k, generator=gen, max_calls=8)
                for (pid, cat, _), o in zip(prompts, outs):
                    p = parse_assistant(tok, o.ids, think_expected=True)
                    ans = (p.get("answer") or "").strip()
                    closed = bool(p.get("terminated")) and not p.get("malformed")
                    stats["n"] += 1; stats["closed"] += closed; stats["repeat"] += repeated_ngram(ans); stats["len"] += len(ans.split()); stats["empty"] += not ans
                    if cat not in ("arithmetic", "python"):
                        stats["nonmath"] += 1; stats["tool_nonmath"] += o.n_calls > 0
                    answers_by_prompt.setdefault(pid, []).append(ans)
            # diversity: mean over prompts of distinct answers / seeds (1.0 = every seed different)
            div = sum(len(set(v)) / len(v) for v in answers_by_prompt.values()) / len(answers_by_prompt)
            row = {"temperature": temp, "top_p": top_p, "top_k": top_k, "n": stats["n"], "closed": round(stats["closed"] / stats["n"], 3), "repeat": round(stats["repeat"] / stats["n"], 3),
                   "empty": round(stats["empty"] / stats["n"], 3), "mean_words": round(stats["len"] / stats["n"], 1), "tool_misfire": round(stats["tool_nonmath"] / max(1, stats["nonmath"]), 3),
                   "diversity": round(div, 2), "seconds": round(time.time() - t0, 1)}
            rows.append(row)
            print(f"T={temp:<4} p={top_p:<5} k={top_k:<3} closed {row['closed']:.2f}  repeat {row['repeat']:.2f}  empty {row['empty']:.2f}  words {row['mean_words']:5.1f}  misfire {row['tool_misfire']:.2f}  diversity {row['diversity']:.2f}  [{row['seconds']}s]", flush=True)
            torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"checkpoint": a.checkpoint, "prompts": len(prompts), "seeds": a.seeds, "rows": rows}, indent=1), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()

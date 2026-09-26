"""pass@k on word problems: is the bottleneck capability or selection?

    .venv/Scripts/python.exe scripts/pass_at_k.py --checkpoint runs/m9_rl2_336m/checkpoints/best.pt \
        --gsm8k 100 --svamp 100 --k 64 --out runs/m9_rl2_336m/pass_at_k.json

A model that is right 6% of the time per sample (GSM8K, greedy) may still contain the right answer somewhere in
64 samples: 1 - 0.94^64 is 98% if its errors are diverse, and near 6% if they are systematic. That one number
decides what a multi-agent swarm on batched inference could do. If pass@k is far above pass@1, the problem is
SELECTION -- picking the right sample -- and verification, voting and the sandbox are the tools. If pass@k is
also low, the problem is CAPABILITY and no amount of parallelism helps.

Three numbers per task set, all from the same k samples at the RL run's sampling settings:
  pass@1      mean per-sample accuracy (what greedy roughly measures)
  pass@k      share of problems where at least one sample is right (the oracle-selection ceiling)
  majority    share where the most common parsed answer is right (the zero-cost selection baseline)
plus pass@k' for k' = 1, 2, 4, ... k, empirically (any correct among the first k' samples).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=r"C:\slm-data\tokenizer\v1")
    ap.add_argument("--gsm8k", type=int, default=100)
    ap.add_argument("--svamp", type=int, default=100)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tool-calls", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch

    from slm.data.chat import format_chat, parse_assistant
    from slm.data.tokenizer import SlmTokenizer
    from slm.eval.quality import load_model
    from slm.eval.reasoning import gsm8k_tasks, svamp_tasks
    from slm.rl.rewards import _to_number, verify_answer
    from slm.rl.tasks import prompt_messages
    from slm.tools.loop import sample_with_tools
    from slm.utils.sdpa import sdpa_context

    model, _ = load_model(Path(a.checkpoint), "cuda")
    tok = SlmTokenizer.load(a.tokenizer)
    sets = {}
    if a.gsm8k:
        sets["gsm8k"] = gsm8k_tasks(a.gsm8k)
    if a.svamp:
        sets["svamp"] = svamp_tasks(a.svamp)
    ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= a.k]
    if a.k not in ks:
        ks.append(a.k)

    results = {}
    with torch.no_grad(), sdpa_context("decode"):
        for name, tasks in sets.items():
            t0 = time.time()
            per_task = []
            for i, t in enumerate(tasks):
                prompt_ids = format_chat(tok, prompt_messages(t), add_generation_prompt=True, think_required=True).ids
                gen = torch.Generator(device="cuda")
                gen.manual_seed(a.seed * 100003 + i)
                tcs = sample_with_tools(model, tok, [prompt_ids] * a.k, a.max_new, a.temperature, a.top_p, 0, gen,
                                        max_calls=a.max_tool_calls)
                corrects, parsed_vals, n_calls = [], [], 0
                for tc in tcs:
                    p = parse_assistant(tok, tc.ids)
                    v = verify_answer(p["answer"], t.answer, "auto")
                    corrects.append(bool(v.correct))
                    parsed_vals.append(_to_number(v.parsed) if v.parsed is not None else None)
                    n_calls += int(bool(tc.n_calls))
                votes = Counter(x for x in parsed_vals if x is not None)
                mode = votes.most_common(1)[0][0] if votes else None
                gold = _to_number(t.answer)
                per_task.append({
                    "id": t.id, "gold": t.answer,
                    "n_correct": sum(corrects), "k": a.k,
                    "any_correct": any(corrects),
                    "majority_correct": (mode is not None and gold is not None and mode == gold),
                    "n_distinct_answers": len(votes),
                    "tool_use": n_calls / a.k,
                    "pass_at": {str(kk): any(corrects[:kk]) for kk in ks},
                })
                torch.cuda.empty_cache()
                if (i + 1) % 10 == 0:
                    done = per_task
                    print(f"  {name} {i + 1}/{len(tasks)}: pass@1 {statistics.fmean(d['n_correct'] / d['k'] for d in done):.3f}"
                          f"  pass@{a.k} {statistics.fmean(d['any_correct'] for d in done):.3f}"
                          f"  majority {statistics.fmean(d['majority_correct'] for d in done):.3f}  [{time.time() - t0:.0f}s]", flush=True)
            n = len(per_task)
            summary = {
                "n_problems": n, "k": a.k,
                "pass_at_1": round(statistics.fmean(d["n_correct"] / d["k"] for d in per_task), 4),
                "pass_at_k": round(statistics.fmean(d["any_correct"] for d in per_task), 4),
                "majority": round(statistics.fmean(d["majority_correct"] for d in per_task), 4),
                "tool_use": round(statistics.fmean(d["tool_use"] for d in per_task), 3),
                "mean_distinct_answers": round(statistics.fmean(d["n_distinct_answers"] for d in per_task), 1),
                "pass_at_curve": {str(kk): round(statistics.fmean(d["pass_at"][str(kk)] for d in per_task), 4) for kk in ks},
                "seconds": round(time.time() - t0, 1),
            }
            results[name] = {"summary": summary, "problems": per_task}
            print(f"\n{name}: pass@1 {summary['pass_at_1']:.3f}  pass@{a.k} {summary['pass_at_k']:.3f}  majority {summary['majority']:.3f}"
                  f"  tool use {summary['tool_use']:.2f}  distinct answers/problem {summary['mean_distinct_answers']}")
            print("  pass@k curve: " + "  ".join(f"k={kk}:{v:.3f}" for kk, v in summary["pass_at_curve"].items()), flush=True)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"checkpoint": a.checkpoint, "temperature": a.temperature, "top_p": a.top_p,
                                           "results": results}, indent=1), encoding="utf-8")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

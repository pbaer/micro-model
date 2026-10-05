"""Prompt sweep: the same checkpoint under several system prompts, through the evals a prompt could plausibly move
(GSM8K / SVAMP with the calculator tool, the judged quality suite, the multi-turn chat diagnostics). Inference only.

    .venv/Scripts/python.exe -u scripts/prompt_sweep.py --checkpoint runs/m10_rl_336m/checkpoints/final.pt \
        --run m10_psweep_336m > runs/m10_psweep_336m/sweep.log 2>&1

The judged suite's outputs land in runs/<run>/quality/outputs/<variant index>.jsonl in the usual shape, so
`python -m slm.eval.quality pack --run <run>` / `ingest` judge them like any run's checkpoints (one "checkpoint" per
variant; `tokens` is the variant index). The prompt is applied with `slm.data.chat.system_overlay`, so every eval
runs unchanged; the bare variant is the published eval setting and stays the headline number.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

TECHNIQUES = ("You are a careful assistant. When you solve a problem: restate what is being asked, break it into steps, "
              "use Python for any arithmetic, and verify the result before giving the final answer.")

# name -> (where, text); None = the eval as published
VARIANTS: dict[str, tuple[str, str | None]] = {
    "bare": ("system", None),
    "persona": ("system", "You are a careful, helpful assistant. Answer fully and precisely, in complete sentences."),
    "check": ("system", "You are a careful assistant. Work the problem out before answering, then check the result once for mistakes "
                        "before you give it."),
    "techniques": ("system", TECHNIQUES),
    "think_hard": ("system", "Think carefully and at length before you answer. Write out all of your reasoning, step by step, "
                             "before the final answer."),
    "user_techniques": ("user", TECHNIQUES),  # the same text inside the first user turn (the arena's lesson: no preamble in its own role)
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=r"C:\slm-data\tokenizer\v1")
    ap.add_argument("--run", default="m10_psweep_336m", help="runs/<run>/ holds the outputs (no run.json: the portal's run list ignores it)")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--variants", default=",".join(VARIANTS), help="comma-separated subset, in order")
    ap.add_argument("--gsm8k", type=int, default=200)
    ap.add_argument("--svamp", type=int, default=300)
    ap.add_argument("--max-new", type=int, default=256, help="reasoning budget (slm.eval.reasoning's default)")
    ap.add_argument("--mt-n", type=int, default=64, help="multi-turn conversations per kind (0 = skip)")
    ap.add_argument("--no-quality", action="store_true")
    a = ap.parse_args()

    import torch

    from slm.data.chat import system_overlay
    from slm.data.tokenizer import SlmTokenizer
    from slm.eval import multiturn
    from slm.eval.quality import generate_suite, load_model, write_outputs
    from slm.eval.reasoning import gsm8k_tasks, svamp_tasks
    from slm.rl.rollout import greedy_accuracy
    from slm.utils.sdpa import sdpa_context

    run_dir = Path(a.runs_root) / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    names = [v.strip() for v in a.variants.split(",") if v.strip()]
    unknown = [v for v in names if v not in VARIANTS]
    if unknown:
        ap.error(f"unknown variants {unknown}; known: {list(VARIANTS)}")
    model, _ = load_model(Path(a.checkpoint), "cuda")
    tok = SlmTokenizer.load(a.tokenizer)
    out_path = run_dir / "sweep.json"
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {"checkpoint": a.checkpoint, "variants": {}}
    print(f"[{a.run}] {a.checkpoint}: {len(names)} variants, gsm8k {a.gsm8k} svamp {a.svamp} mt {a.mt_n}/kind", flush=True)

    for idx, name in enumerate(list(VARIANTS)):
        if name not in names:
            continue
        where, text = VARIANTS[name]
        if name in results["variants"] and results["variants"][name].get("done"):
            print(f"== {name}: already measured, skipping", flush=True)
            continue
        print(f"== {name} ({where}): {text!r}", flush=True)
        row: dict = {"index": idx, "where": where, "text": text}
        t0 = time.time()
        with system_overlay(text, where), torch.no_grad(), sdpa_context("decode"):
            keep: list = []
            for key, tasks in (("gsm8k", gsm8k_tasks(a.gsm8k) if a.gsm8k else []), ("svamp", svamp_tasks(a.svamp) if a.svamp else [])):
                if not tasks:
                    continue
                r = greedy_accuracy(model, tok, tasks, a.max_new, tools=True, max_tool_calls=8, keep=keep)
                row[key] = {k: r[k] for k in ("accuracy", "malformed_rate", "mean_len", "tool_use_rate", "answer_from_tool_rate", "n")}
                print(f"   {key:6s} acc {r['accuracy']:.3f}  malformed {r['malformed_rate']:.2f}  len {r['mean_len']:.0f}  tool use {r['tool_use_rate']:.2f}  (n={r['n']})", flush=True)
            with (run_dir / f"reasoning_{name}.jsonl").open("w", encoding="utf-8") as f:
                for r in keep:
                    f.write(json.dumps({"task": r.task, "id": r.prompt_id, "gold": r.gold, "text": r.text, "answer": r.parsed, "correct": r.correct,
                                        "malformed": r.malformed, "tool_calls": r.tool_calls, "n_tokens": r.n_tokens}, ensure_ascii=False) + "\n")
            torch.cuda.empty_cache()
            if not a.no_quality:
                tq = time.time()
                items = generate_suite(model, tok, "rl", "cuda", tools=True)
                write_outputs(run_dir, a.run, idx, name, "rl", "cuda", items, time.time() - tq)
                row["quality"] = {"n_items": len(items), "malformed": sum(1 for i in items if i.get("malformed")), "tool_calls": sum(i.get("tool_calls") or 0 for i in items),
                                  "mean_n_new": round(sum(i["n_new"] for i in items) / max(1, len(items)), 1)}
                print(f"   quality: {len(items)} outputs written (malformed {row['quality']['malformed']}, tool calls {row['quality']['tool_calls']}, mean new {row['quality']['mean_n_new']})", flush=True)
                torch.cuda.empty_cache()
        if a.mt_n:
            with system_overlay(text, where):  # run() loads its own copy of the model; small beside the suite
                mt = multiturn.run(a.checkpoint, a.tokenizer, a.mt_n, 0, 128, 4, 32)
            s = mt["summary"]
            row["multiturn"] = {k: s.get(k) for k in s if not isinstance(s[k], (dict, list))}
            (run_dir / f"multiturn_{name}.json").write_text(json.dumps(mt, indent=1, ensure_ascii=False), encoding="utf-8")
            print("   multiturn: " + "  ".join(f"{k} {v:.3f}" for k, v in row["multiturn"].items() if isinstance(v, float)), flush=True)
            torch.cuda.empty_cache()
        row["seconds"] = round(time.time() - t0, 1)
        row["done"] = True
        results["variants"][name] = row
        out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"   [{row['seconds']}s]", flush=True)
    print(f"[{a.run}] done; outputs under {run_dir}. Judge the suite with: python -m slm.eval.quality pack --run {a.run}", flush=True)


if __name__ == "__main__":
    main()

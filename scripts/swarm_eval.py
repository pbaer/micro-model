"""Swarm inference against its ceilings, on the word-problem tests.

    .venv/Scripts/python.exe scripts/swarm_eval.py --checkpoint runs/m9_rl5_336m/checkpoints/step_00200.pt \
        --gsm8k 50 --svamp 50 --k 16 --out runs/m9_rl5_336m/swarm_eval.json

Per test set, six numbers from the same k samples:
  greedy            one greedy answer (what the model scores without a swarm)
  majority          most-supported answer across the k samples (the zero-cost baseline)
  verified_majority most-supported answer among sandbox-verified candidates, else majority
  selector          the same model's pick from the selection prompt (slm.swarm.selector_messages)
  oracle            a correct answer exists among the k samples (pass@k: the ceiling of any selector)
  oracle_verified   a correct answer exists among the VERIFIED candidates (the ceiling after the sandbox filter)
  in_prompt         the correct answer's group survived into the selector prompt (the ceiling of the selector)
  tournament        (--mode both/tournament) the pairwise bracket's champion (slm.swarm.tournament)

The gaps between these say where to work: oracle - oracle_verified is what verification throws away,
oracle_verified - in_prompt what the prompt budget throws away, in_prompt - selector what the selector still
gets wrong. An untrained selector is expected to sit near majority; the RL `select` family is meant to move it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=r"C:\slm-data\tokenizer\v1")
    ap.add_argument("--gsm8k", type=int, default=50)
    ap.add_argument("--svamp", type=int, default=50)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", default="both", choices=["select", "tournament", "both"], help="which selection paths to run on the same samples")
    a = ap.parse_args()

    import torch

    from slm.data.answers import SUFFIX
    from slm.data.chat import format_chat, parse_assistant
    from slm.data.tokenizer import SlmTokenizer
    from slm.eval.quality import load_model
    from slm.eval.reasoning import gsm8k_tasks, svamp_tasks
    from slm.rl.rewards import verify_answer
    from slm.rl.rewards import parse_final_span
    from slm.swarm import display_answer, swarm_answer
    from slm.tools.loop import sample_with_tools
    from slm.utils.sdpa import sdpa_context

    model, _ = load_model(Path(a.checkpoint), "cuda")
    tok = SlmTokenizer.load(a.tokenizer)
    sets = {}
    if a.gsm8k:
        sets["gsm8k"] = gsm8k_tasks(a.gsm8k)
    if a.svamp:
        sets["svamp"] = svamp_tasks(a.svamp)

    def ok(answer: str | None, gold: str) -> bool:
        return answer is not None and bool(verify_answer(f"#### {answer}", gold, "auto").correct)

    results = {}
    with torch.no_grad(), sdpa_context("decode"):
        for name, tasks in sets.items():
            t0 = time.time()
            rows = []
            for i, t in enumerate(tasks):
                # greedy, the reference point
                pid = format_chat(tok, [{"role": "user", "content": t.prompt + SUFFIX}], add_generation_prompt=True, think_required=True).ids
                gen = torch.Generator(device="cuda"); gen.manual_seed(0)
                tc = sample_with_tools(model, tok, [pid], a.max_new, 1.0, 1.0, 1, gen, max_calls=6)[0]
                greedy = parse_final_span(parse_assistant(tok, tc.ids)["answer"])
                # the swarm
                res = swarm_answer(model, tok, t.prompt, k=a.k, temperature=a.temperature, max_new_tokens=a.max_new,
                                   seed=a.seed * 100003 + i, answer_suffix=SUFFIX, mode=a.mode)
                sel_final = display_answer(parse_final_span(res.selector_answer)) if parse_final_span(res.selector_answer) is not None else res.verified_majority
                correct_keys = {g.key for g in res.groups if ok(g.answer, t.answer)}
                verified_correct = any(c.verified and ok(c.parsed, t.answer) for c in res.candidates)
                in_prompt = any(f"- Answer: {g.answer} (" in (res.selector_messages[0]["content"] if res.selector_messages else "") for g in res.groups if g.key in correct_keys)
                rows.append({
                    "id": t.id, "gold": t.answer, "seconds": res.seconds,
                    "greedy": ok(greedy, t.answer),
                    "majority": ok(res.majority, t.answer),
                    "verified_majority": ok(res.verified_majority, t.answer),
                    "selector": ok(sel_final, t.answer),
                    "tournament": ok(res.tournament, t.answer) if res.tournament is not None else ok(res.verified_majority, t.answer),
                    "tournament_rounds": len(res.rounds), "tournament_answer": res.tournament,
                    "selector_called_tool": res.selector_calls > 0,
                    "oracle": any(ok(c.parsed, t.answer) for c in res.candidates),
                    "oracle_verified": verified_correct,
                    "in_prompt": in_prompt,
                    "n_groups": len(res.groups), "n_verified": sum(c.verified for c in res.candidates),
                    "selector_final": res.final, "majority_answer": res.majority,
                })
                torch.cuda.empty_cache()
                if (i + 1) % 10 == 0:
                    m = lambda key: statistics.fmean(r[key] for r in rows)
                    print(f"  {name} {i + 1}/{len(tasks)}: greedy {m('greedy'):.2f} majority {m('majority'):.2f} vmaj {m('verified_majority'):.2f} "
                          f"selector {m('selector'):.2f} tournament {m('tournament'):.2f} | oracle {m('oracle'):.2f} verified {m('oracle_verified'):.2f} in-prompt {m('in_prompt'):.2f} [{time.time() - t0:.0f}s]", flush=True)
            keys = ("greedy", "majority", "verified_majority", "selector", "tournament", "oracle", "oracle_verified", "in_prompt", "selector_called_tool")
            summary = {k: round(statistics.fmean(r[k] for r in rows), 3) for k in keys}
            summary.update({"n": len(rows), "k": a.k, "mean_groups": round(statistics.fmean(r["n_groups"] for r in rows), 1),
                            "mean_verified": round(statistics.fmean(r["n_verified"] for r in rows), 1), "seconds": round(time.time() - t0, 1)})
            results[name] = {"summary": summary, "problems": rows}
            print(f"\n{name} (n={len(rows)}, k={a.k}): " + "  ".join(f"{k} {v}" for k, v in summary.items() if k in keys), flush=True)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"checkpoint": a.checkpoint, "k": a.k, "temperature": a.temperature, "mode": a.mode, "results": results}, indent=1), encoding="utf-8")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

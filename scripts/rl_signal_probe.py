"""How much gradient will GRPO actually get from each task family?

    .venv/Scripts/python.exe scripts/rl_signal_probe.py --checkpoint runs/m9_tool4_336m/checkpoints/final.pt \
        --config configs/train/m9_rl_336m.yaml --prompts 8 --out runs/m9_rl_336m/signal_probe.json

GRPO advantages are group-relative: a group whose rollouts all score the same has zero std, zero advantage and
contributes nothing to the update. Pass rate alone does not say whether a family is useful -- a family the
policy always fails and one it always passes are equally worthless, and the informative band is the middle.
With group_size 6, a family at 4% yields signal in 22% of groups and one at 35% in 92%.

That matters here because M9 stage B v4 scores 0.04 on GSM8K, which the stage C config hands 30% of its
prompts. This samples each family exactly as the RL run will (same temperature, group size, tool loop and
reward scheme) and reports, per family: mean reward, pass rate, and the share of groups with any spread.
Multiply the last by the family's share to get the fraction of the run's compute that does anything.
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
    ap.add_argument("--config", required=True, help="the RL config whose sampling settings to mirror")
    ap.add_argument("--prompts", type=int, default=8, help="prompts per family")
    ap.add_argument("--families", nargs="*", default=None, help="default: every family named in the config's tasks, plus the pytool subfamilies")
    ap.add_argument("--temperature", type=float, default=None, help="override the config's rollout temperature")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    import yaml

    from slm.eval.quality import load_model
    from slm.data.tokenizer import SlmTokenizer
    from slm.rl.rollout import rollout_group
    from slm.rl.tasks import make_tasks
    from slm.utils.sdpa import sdpa_context

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    fams = a.families or sorted({*cfg["tasks"], "pytool_pipeline", "pytool_strings", "pytool_numbers",
                                 "pytool_sim", "pytool_runcode", "pytool_declared"})
    model, _ = load_model(Path(a.checkpoint), "cuda")
    tok = SlmTokenizer.load(cfg["tokenizer_dir"])
    gs = int(cfg.get("group_size", 6))
    temp = a.temperature if a.temperature is not None else cfg.get("temperature", 0.8)
    print(f"group_size {gs}, temperature {temp}, {a.prompts} prompts/family", flush=True)

    rows = []
    with torch.no_grad(), sdpa_context("decode"):
        for fam in fams:
            tasks = make_tasks([fam], a.prompts, "train", seed=int(cfg.get("seed", 0)))
            if not tasks:
                print(f"{fam}: no tasks generated, skipping", flush=True)
                continue
            t0 = time.time()
            rewards, groups_with_spread, n_correct, n_tool = [], 0, 0, 0
            for t in tasks:
                rs = rollout_group(
                    model, tok, t, group_size=gs,
                    max_new_tokens=int(cfg.get("max_new_tokens", 512)),
                    temperature=float(a.temperature if a.temperature is not None else cfg.get("temperature", 0.8)),
                    top_p=float(cfg.get("top_p", 0.95)),
                    think_required=bool(cfg.get("think_required", True)),
                    reward_scheme=cfg.get("reward_scheme", "tool"),
                    reward_schemes=cfg.get("reward_schemes"),
                    tools=bool(cfg.get("tools", False)),
                    max_tool_calls=int(cfg.get("max_tool_calls", 6)),
                    seed=hash(t.id) % (2**31),
                )
                g = [r.reward for r in rs]
                rewards += g
                groups_with_spread += int(max(g) - min(g) > 1e-9)
                n_correct += sum(r.correct for r in rs)
                n_tool += sum(bool(r.tool_calls) for r in rs)
            n = len(rewards)
            rows.append({"family": fam, "prompts": len(tasks), "group_size": gs,
                         "reward_mean": round(statistics.fmean(rewards), 3),
                         "pass_rate": round(n_correct / n, 3),
                         "tool_use_rate": round(n_tool / n, 3),
                         "signal_frac": round(groups_with_spread / len(tasks), 3),
                         "seconds": round(time.time() - t0, 1)})
            r = rows[-1]
            print(f"{fam:<18} reward {r['reward_mean']:.3f}  pass {r['pass_rate']:.3f}  tool {r['tool_use_rate']:.3f}"
                  f"  groups with spread {r['signal_frac']:.2f}  [{r['seconds']:.0f}s]", flush=True)
            torch.cuda.empty_cache()

    print(f"\n{'family':<18}{'pass':>7}{'signal':>8}   verdict")
    for r in sorted(rows, key=lambda x: -x["signal_frac"]):
        v = "informative" if r["signal_frac"] >= 0.5 else ("weak" if r["signal_frac"] >= 0.25 else "near-useless")
        print(f"{r['family']:<18}{r['pass_rate']:>7.2f}{r['signal_frac']:>8.2f}   {v}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"checkpoint": a.checkpoint, "config": a.config, "rows": rows}, indent=1), encoding="utf-8")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

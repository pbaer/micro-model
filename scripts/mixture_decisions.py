"""Conversation counts, by the decision each set teaches, for a candidate SFT mixture.

    .venv/Scripts/python.exe scripts/mixture_decisions.py configs/train/m9_tool3_336m.yaml

Mixture weights are token shares, but every behaviour this stage teaches is chosen ONCE PER CONVERSATION, and
conversation lengths run from 70 to 1470 tokens across these sets. M9 stage B v2 cut tool data from 50% to
33.5% of tokens and the tool-misfire rate barely moved (0.464 -> 0.429), because in conversations the mixture
was still 51:1 in favour of calling the tool on a short question-shaped prompt. This prints the ratio that
actually governs the model's routing, so a mixture can be argued about before it is trained for an hour.

CLASS says what decision each set demonstrates. The pairs that must be balanced are the ones where the prompt
shapes overlap: plain_no_tool against short_tool (a short question, data or no data), and math_tool against
math_no_tool (a word problem, reach for the sandbox or reason in prose).
"""

import json
import sys
from pathlib import Path

import yaml

SFT_ROOT = Path(r"C:\slm-data\sft\v1")

CLASS = {
    "synthetic-python-tools": "short_tool",
    "synthetic-multiturn-tools": "short_tool",
    "synthetic-reasoning-tools": "math_tool",
    "gsm8k-tools": "math_tool",
    "metamathqa-tools": "math_tool",
    "metamathqa-reasoning": "math_no_tool",
    "synthetic-reasoning": "math_no_tool",
    "gsm8k-reasoning": "math_no_tool",
}
PAIRS = [("short_tool", "plain_no_tool"), ("math_tool", "math_no_tool")]

# Share of a set's conversations that actually demonstrate the contested decision. A tool set demonstrates it
# every time. A `-short` set does too: every conversation in it is a short, non-computational prompt answered
# with a prose think span and no call. A `-direct` set mostly does not -- only ~16% of its conversations have a
# short first user turn, and only ~38% of those drew a think line (measured on shard 0, 2026-09-21), so it
# contributes about a fourteenth of what its conversation count suggests. Long prompts are not contested: the
# model already routes those correctly.
EFFECTIVE = {"smoltalk-smol-magpie-ultra-4k-direct": 0.076, "smoltalk-openhermes-100k-4k-direct": 0.055}


def class_of(name: str) -> str:
    if name in CLASS:
        return CLASS[name]
    return "plain_no_tool" if name.endswith(("-short", "-direct")) else "plain_other"


def main(path: str) -> None:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    total = cfg["schedule"]["total_tokens"]
    rows, by_class = [], {}
    for name, w in sorted(cfg["data"]["mixture"].items(), key=lambda kv: -kv[1]):
        m = json.loads((SFT_ROOT / name / "manifest.json").read_text(encoding="utf-8"))
        tpc = m["train_tokens"] / m["train_examples"]
        convs = w * total / tpc
        cls = class_of(name)
        eff = convs * EFFECTIVE.get(name, 1.0)
        by_class[cls] = by_class.get(cls, 0) + eff
        rows.append((name, w, tpc, convs, eff, w * total / m["train_tokens"], cls))
    print(f"{'set':<40}{'weight':>8}{'tok/conv':>10}{'convs':>10}{'teaching':>10}{'epochs':>8}  class")
    for name, w, tpc, convs, eff, ep, cls in rows:
        flag = "  <-- epochs" if ep > 3 else ""
        print(f"{name:<40}{w:>8}{tpc:>10.0f}{convs:>10,.0f}{eff:>10,.0f}{ep:>8.2f}  {cls}{flag}")
    print(f"\n{'class':<20}{'conversations that teach the decision':>40}")
    for cls, n in sorted(by_class.items(), key=lambda kv: -kv[1]):
        print(f"{cls:<20}{n:>15,.0f}")
    print()
    for a, b in PAIRS:
        na, nb = by_class.get(a, 0), by_class.get(b, 0)
        print(f"{a} : {b} = {na / max(1, nb):.1f} : 1   ({na:,.0f} vs {nb:,.0f})")
    s = sum(cfg["data"]["mixture"].values())
    print(f"\nweights sum to {s:.4f}" + ("" if abs(s - 1) < 1e-6 else "   <-- NOT 1.0"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/train/m9_tool2_336m.yaml")

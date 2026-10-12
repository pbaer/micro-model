"""SFT warm-up for arena robot turns (2026-10-12): the turn shape under a fair prompt, written out by a template.

RL alone barely moved (m11 try 3: arena reward 0.07 after 50 steps; half the samples called no tool, gold rush and tag
earned nothing), so this set shows the shape once: the situation under a fair prompt (slm.rl.arena.STYLES), then a turn
that reasons in one or two sentences, makes the call the scaffold would have spelled out, and answers. For a move the
reasoning works the coordinates out ("R2 is at (5, 5) and I am at (3, 4): it is 2 east and 1 south of me, so east
closes the distance"), the decision every model was at chance on. No teacher: the oracle's call and the template's
words, every target checked to score 1.0 under `arena_verdict` before it is written. Built from the training tasks only
(triangulate and team_relay are held out); worked-example turns in a prompt are masked, never targets.

    python -m slm.rl.synth_arena --tokenizer C:\\slm-data\\tokenizer\\v1 --n 20000 --out C:\\slm-data\\sft\\v1
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from slm.rl.arena import TRAIN_TASKS, arena_verdict, make_arena_task, replay

COMPASS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}


def _where(me, other) -> str:
    """'2 east and 1 south of me' for `other` seen from `me` (y grows south)."""
    dx, dy = other[0] - me[0], other[1] - me[1]
    parts = []
    if dx:
        parts.append(f"{abs(dx)} {'east' if dx > 0 else 'west'}")
    if dy:
        parts.append(f"{abs(dy)} {'south' if dy > 0 else 'north'}")
    return " and ".join(parts) + " of me" if parts else "on my cell"


def _move_turn(goal, n: int, rng: random.Random):
    """(direction, reasoning) for a move towards / away from the goal cells, or None when no step helps."""
    from slm.arena.world import _step, dist

    pos, cells, sense = tuple(goal[0]), [tuple(c) for c in goal[1]], goal[2]
    near = min(cells, key=lambda c: (dist(pos, c), c))
    before = dist(pos, near)
    good = []
    for d in COMPASS:
        new = _step(pos, d, n)
        if new == pos:
            continue
        after = min(dist(new, c) for c in cells)
        if (after < before) if sense == "towards" else (after > before):
            good.append(d)
    if not good:
        return None
    dx, dy = near[0] - pos[0], near[1] - pos[1]
    if sense == "towards":  # step along the bigger gap
        best = [d for d in good if (d in ("east", "west") and abs(dx) >= abs(dy)) or (d in ("north", "south") and abs(dy) > abs(dx))] or good
        d = rng.choice(best)
        why = (f"{near} is {_where(pos, near)}, I am at {pos}. "
               f"{'Going ' + d + ' closes the bigger gap' if len(good) > 1 else 'Only going ' + d + ' gets me closer'}, so I step {d}.")
    else:
        d = rng.choice(good)
        why = f"The chaser at {near} is {_where(pos, near)}, I am at {pos}. Going {d} takes me further from it, so I step {d}."
    return d, why


def target_turn(task, rng: random.Random):
    """(think text with the call markup, answer) for a task whose oracle call scores 1.0, else None."""
    from slm.arena.world import ALIASES

    j = task.meta["judge"]
    style = task.meta["arena"]["style"]
    alias = ALIASES if style.get("names") == "domain" else {}
    exp = task.answer  # the scaffold's call (one of them for a decision)
    name = re.match(r"\w+", exp).group(0)
    shown = alias.get(name, name)
    if name == "move":
        if not j.get("goal"):
            return None
        mv = _move_turn(j["goal"], j["n"], rng)
        if mv is None:
            return None
        d, why = mv
        call = f'move("{d}")'
        return f"{why} <<<{call}>>>", f"#### stepping {d}"
    args = re.findall(r'"([^"]*)"', exp)
    call = f"{shown}({', '.join(json.dumps(a) for a in args)})"
    if name in ("read_key", "read_code"):
        why = rng.choice(["I don't know the code yet, and the declared function tells me.", "The code is not in the question; I read it."])
    elif name == "distance_to_target":
        why = "I need my distance to the target; the declared function measures it."
    elif name == "look":
        why = rng.choice(["I have nothing to pass on yet, so I look who is around.", "Let me see who is near me."])
    elif name == "say":
        why = f"The robot that needs it can hear me, so I say it: {args[0]}."
    elif name in ("submit",):
        why = f"I am the last robot and I heard the code {args[0]}, so I hand it in."
    elif name == "open_door":
        why = f"I am on the door's cell and I was told the code {args[0]}, so I open the door with it."
    elif name == "tag":
        why = f"{args[0]} is right next to me, so I tag it."
    elif name == "grab":
        why = "I am standing on a coin, so I pick it up."
    else:
        return None
    answers = {"say": "#### said it", "look": "#### looked around", "tag": f"#### tagged {args[0] if args else ''}", "grab": "#### picked up the coin",
               "open_door": "#### opened the door", "submit": "#### handed in the code"}
    return f"{why} <<<{call}>>>", answers.get(name, "#### done")


def build(tok, n: int, out_root: Path, name: str = "arena-sft", seed: int = 0) -> dict:
    from slm.arena.world import native_messages
    from slm.data.chat import format_chat
    from slm.data.sft import SftShardWriter
    from slm.tools.functions import functions_env
    from slm.tools.pysandbox import PySession

    rng = random.Random(seed)
    out = out_root / name
    writers = {"train": SftShardWriter(out / "train"), "val": SftShardWriter(out / "val")}
    counts, tools, styles, skipped = Counter(), Counter(), Counter(), Counter()
    seen: Counter = Counter()  # identical conversations, capped: a turn whose question is the same in every world
    # (reading the code, picking up a coin) is word for word identical; deduplicated to one copy, the first build had 3-12
    # of them against 10k moves, and reading the code is the first link of every chain
    while sum(counts.values()) < n:
        task_name = rng.choice(TRAIN_TASKS)
        t = make_arena_task(rng, task_name)
        if t is None:
            skipped["no call"] += 1
            continue
        if t.answer.startswith("move(") and rng.random() < 0.5:  # moves are most robot-turns: kept to about half the set
            skipped["move thinned"] += 1
            continue
        tt = target_turn(t, rng)
        if tt is None:
            skipped["no template"] += 1
            continue
        think, answer = tt
        call = re.search(r"<<<(.*)>>>", think).group(1)
        if arena_verdict(t, [[call, ""]]).fraction != 1.0:  # every target must earn the full reward
            skipped["not 1.0"] += 1
            continue
        w, a = replay(t.meta["arena"])
        msgs = w.messages(a)
        key = (msgs[-1]["content"], call)
        if seen[key] >= 40:
            skipped["duplicate"] += 1
            continue
        seen[key] += 1
        decls = t.meta["functions"]
        session = PySession()
        session.register(functions_env(decls))
        conv = native_messages(tok, msgs) + [{"role": "assistant", "think": think, "content": answer}]
        enc = format_chat(tok, conv, think_required=True, tools=True, session=session, functions=decls)
        mask = list(enc.loss_mask)
        for s0, s1, label in enc.segments:  # worked examples are part of the prompt, never targets
            if label == "assistant_ids":
                mask[s0:s1] = [0] * (s1 - s0)
        r_open, r_close = tok.special("<|python_result|>"), tok.special("<|/python_result|>")
        inside = False
        for i, x in enumerate(enc.ids):  # a tool result is never a loss target
            if x == r_open:
                inside = True
            if inside:
                mask[i] = 0
            if x == r_close:
                inside = False
        if tok.special("<|python_call|>") not in enc.ids:
            skipped["markup did not run"] += 1
            continue
        split = "val" if rng.random() < 0.02 else "train"
        writers[split].add(enc.ids, mask)
        counts[t.task] += 1
        tools[re.match(r"\w+", call).group(0)] += 1
        styles[f"{t.meta['arena']['style']['question']}{'+examples' if t.meta['arena']['style'].get('demos') else ''}{'+names' if t.meta['arena']['style'].get('names') else ''}"] += 1
    for wr in writers.values():
        wr.flush()
    m = {"name": name, "tokenizer_sha256": tok.sha256, "think_required": True, "tools": True, "functions": True, "seed": seed,
         "conversations": sum(counts.values()), "task_counts": dict(counts), "tool_counts": dict(tools), "style_counts": dict(styles),
         "skipped": dict(skipped), "heldout_tasks": ["triangulate", "team_relay"],
         "train_examples": writers["train"].total_examples, "train_tokens": writers["train"].total_tokens, "train_targets": writers["train"].total_targets,
         "val_examples": writers["val"].total_examples, "val_tokens": writers["val"].total_tokens, "val_targets": writers["val"].total_targets}
    (out / "manifest.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    return m


def main() -> None:
    from slm.data.tokenizer import SlmTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="arena-sft")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    print(json.dumps(build(SlmTokenizer.load(a.tokenizer), a.n, Path(a.out), a.name, a.seed), indent=1))


if __name__ == "__main__":
    main()

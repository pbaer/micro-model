"""Hill-climb how arena robots are prompted, one decision at a time (2026-10-11).

Whole episodes are a slow, noisy way to compare prompts: one skipped call derails a chain. This probe measures the atomic
decision instead. It plays scaffolded episodes with an oracle that makes the call each question spells out, and at every
robot-turn renders the prompt that robot would get under every prompt style (slm.arena.world PROMPTS, or the variants
below). Each style's prompts are then generated once (greedy, our checkpoint's tool protocol, stub tools so nothing
changes the world) and judged like the Runner judges a turn: did the robot call anything (`acted`), one of the right
tools (`tool_ok`, against `tool_chance`), and with the right fact in the argument (`call_ok`: the code, the name, the
distance; moves are scored by the episodes). The winner is then checked with whole episodes (scripts/arena_eval.py).

    python -u scripts/arena_prompt_probe.py --checkpoint runs/m10_rl_336m/checkpoints/final.pt --out runs/arena/prompt_climb/round1.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from pathlib import Path

ENDINGS = {"what": " What do you do?", "tool": " Use one of your tools.", "which": " Which tool do you call, and with what argument?", "none": ""}
VARIANTS = {
    "scaffold": {"tools": "turn", "brief": "none", "question": "call", "ending": ""},
    "all_tools": {"tools": "all", "brief": "none", "question": "call", "ending": ""},
    "sit_turn": {"tools": "turn", "brief": "none", "question": "situation", "ending": ENDINGS["what"]},
    "sit": {"tools": "all", "brief": "none", "question": "situation", "ending": ENDINGS["what"]},
    "sit_sys": {"tools": "all", "brief": "system", "question": "situation", "ending": ENDINGS["what"]},
    "sit_user": {"tools": "all", "brief": "user", "question": "situation", "ending": ENDINGS["what"]},
    "sit_tool": {"tools": "all", "brief": "none", "question": "situation", "ending": ENDINGS["tool"]},
    "sit_which": {"tools": "all", "brief": "none", "question": "situation", "ending": ENDINGS["which"]},
    "sit_none": {"tools": "all", "brief": "none", "question": "situation", "ending": ENDINGS["none"]},
    "sit_demo1": {"tools": "all", "brief": "none", "question": "situation", "ending": ENDINGS["what"], "demo": 1},
    "sit_demo2": {"tools": "all", "brief": "none", "question": "situation", "ending": ENDINGS["what"], "demo": 2},
    "sit_sys_demo2": {"tools": "all", "brief": "system", "question": "situation", "ending": ENDINGS["what"], "demo": 2},
    "scaffold_demo2": {"tools": "turn", "brief": "none", "question": "call", "ending": "", "demo": 2},
    # round 2: the question the right tool's result answers, worded like its description (the declared-function training shape)
    "ask": {"tools": "all", "brief": "none", "question": "ask", "ending": ""},
    "ask_turn": {"tools": "turn", "brief": "none", "question": "ask", "ending": ""},
    "ask_demo1": {"tools": "all", "brief": "none", "question": "ask", "ending": "", "demo": 1},
    "ask_demo2": {"tools": "all", "brief": "none", "question": "ask", "ending": "", "demo": 2},
    # round 3: a third worked example (an information tool, look()), and tool descriptions that say what they return
    "ask_demo3": {"tools": "all", "brief": "none", "question": "ask", "ending": "", "demo": 3},
    "ask_demo2_rich": {"tools": "all", "brief": "none", "question": "ask", "ending": "", "demo": 2, "docs": "rich"},
    "ask_demo3_rich": {"tools": "all", "brief": "none", "question": "ask", "ending": "", "demo": 3, "docs": "rich"},
    "sit_demo2_rich": {"tools": "all", "brief": "none", "question": "situation", "ending": " What do you do?", "demo": 2, "docs": "rich"},
    "sit_demo3_rich": {"tools": "all", "brief": "none", "question": "situation", "ending": " What do you do?", "demo": 3, "docs": "rich"},
    # round 4: the order of the worked examples (0 move to a pad, 1 pass a password on, 2 look around); the last weighs most
    "ask_d10": {"tools": "all", "brief": "none", "question": "ask", "ending": "", "demos": [1, 0]},
    "ask_d210": {"tools": "all", "brief": "none", "question": "ask", "ending": "", "demos": [2, 1, 0]},
    "ask_d201": {"tools": "all", "brief": "none", "question": "ask", "ending": "", "demos": [2, 0, 1]},
    "ask_d120": {"tools": "all", "brief": "none", "question": "ask", "ending": "", "demos": [1, 2, 0]},
    # round 5: tools named with the words their questions use (key_code, secret_code, pick_up_coin, hand_in)
    "ask_d201_names": {"tools": "all", "brief": "none", "question": "ask", "ending": "", "demos": [2, 0, 1], "names": "domain"},
    "ask_names": {"tools": "all", "brief": "none", "question": "ask", "ending": "", "names": "domain"},
    "sit_d201_names": {"tools": "all", "brief": "none", "question": "situation", "ending": " What do you do?", "demos": [2, 0, 1], "names": "domain"},
}
SIZES = {"key_door": (8, 6), "relay": (14, 6), "triangulate": (8, 6), "team_relay": (8, 6), "tag": (8, 6), "gold_rush": (8, 6)}


def collect(tasks: list[str], seeds: int, turns: int, styles: dict) -> list[dict]:
    """Robot-turns from oracle-played scaffold episodes: per turn and robot, the scaffolded question and, per style, the
    messages and declared tools it would have got."""
    from slm.arena.world import Runner, World, expected_calls

    items = []
    for task in tasks:
        n, k = SIZES[task]
        for seed in range(seeds):
            w = World(task, n, k, seed, max_history=0, prompt="scaffold")  # no history: the question alone, as the probe judges it
            rng = random.Random(seed)
            r = None

            def oracle(prompts, sessions):
                out = []
                for sess in sessions:
                    a = next(x for x in w.agents if r.sessions[x.id] is sess)
                    q, names = w.turn_question(a)
                    item = {"task": task, "seed": seed, "turn": w.turn + 1, "agent": a.name, "question": q, "names": names, "styles": {}}
                    for key, st in styles.items():
                        w.prompt = st
                        item["styles"][key] = {"messages": w.messages(a), "tools": w.turn_tools(a)}
                    w.prompt = styles["scaffold"]
                    items.append(item)
                    exp = expected_calls(q)
                    call = rng.choice(exp) if exp else None  # a decision question offers several: any keeps the episode going
                    calls = []
                    if call:
                        name = re.match(r"\w+", call).group(0)
                        arg = re.findall(r'"([^"]*)"', call)
                        if name in sess.functions:
                            calls.append([call, str(sess.functions[name](*arg))])
                    out.append({"calls": calls, "answer": "", "think": ""})
                return out
            r = Runner(w, generate=oracle, retry_temperature=None)
            r.run(turns)
    return items


def generate(model, tok, items: list[dict], key: str, max_new: int, batch: int) -> list[dict]:
    """One greedy turn per item under style `key`, the declared tools stubbed (nothing reaches the world)."""
    import torch

    from slm.arena.world import native_messages
    from slm.data.chat import format_chat, parse_assistant
    from slm.tools.functions import functions_env
    from slm.tools.loop import sample_with_tools
    from slm.tools.pysandbox import PySession
    from slm.utils.sdpa import sdpa_context

    def stub(d):
        return lambda *a, **kw: f"({d.name} done)"
    outs = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        prompts, sessions = [], []
        for it in chunk:
            st = it["styles"][key]
            prompts.append(format_chat(tok, native_messages(tok, st["messages"]), add_generation_prompt=True, think_required=True, functions=st["tools"]).ids)
            sess = PySession()
            env = functions_env(st["tools"])
            sess.register({name: stub(d) for d, name in zip(st["tools"], env)})
            sessions.append(sess)
        gen = torch.Generator(device="cuda"); gen.manual_seed(0)
        with torch.no_grad(), sdpa_context("decode"):
            tcs = sample_with_tools(model, tok, prompts, max_new, 1.0, 1.0, 1, gen, max_calls=2, sessions=sessions)
        for it, tc, p in zip(chunk, tcs, prompts):
            outs.append({"calls": [list(c) for c in tc.calls], "answer": parse_assistant(tok, tc.ids)["answer"], "prompt_tokens": len(p)})
        torch.cuda.empty_cache()
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=r"C:\slm-data\tokenizer\v1")
    ap.add_argument("--tasks", default="key_door,relay,triangulate,team_relay,tag,gold_rush")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--show", type=int, default=2, help="print this many sample outputs per variant")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from slm.arena.world import _judge, load_player

    styles = {k: VARIANTS[k] for k in a.variants.split(",")}
    styles.setdefault("scaffold", VARIANTS["scaffold"])
    t0 = time.time()
    items = collect(a.tasks.split(","), a.seeds, a.turns, styles)
    print(f"{len(items)} robot-turns collected in {time.time() - t0:.0f}s", flush=True)
    player = load_player(a.checkpoint, a.tokenizer)
    rows = {}
    for key in styles:
        t1 = time.time()
        outs = generate(player.model, player.tok, items, key, a.max_new, a.batch)
        per_task = {}
        for it, o in zip(items, outs):
            j = _judge(it["question"], o["calls"], it["styles"][key]["tools"])
            o.update(j, acted=bool(o["calls"]))
            per_task.setdefault(it["task"], []).append(o)

        def summ(os_):
            judged = [o for o in os_ if o["tool_ok"] is not None]
            valued = [o for o in os_ if o["call_ok"] is not None]
            return {"n": len(os_), "acted": round(statistics.fmean(o["acted"] for o in os_), 3),
                    "tool_ok": round(statistics.fmean(o["tool_ok"] for o in judged), 3) if judged else None,
                    "tool_chance": round(statistics.fmean(o["tool_chance"] for o in judged), 3) if judged else None,
                    "call_ok": round(statistics.fmean(o["call_ok"] for o in valued), 3) if valued else None, "n_valued": len(valued),
                    "prompt_tokens": round(statistics.fmean(o["prompt_tokens"] for o in os_))}
        allo = [o for v in per_task.values() for o in v]
        rows[key] = {"style": styles[key], "all": summ(allo), "tasks": {t: summ(v) for t, v in per_task.items()}, "seconds": round(time.time() - t1, 1),
                     "samples": [{"task": it["task"], "agent": it["agent"], "user": it["styles"][key]["messages"][-1]["content"], **o}
                                 for it, o in list(zip(items, outs))[:: max(1, len(items) // 24)]]}
        s = rows[key]["all"]
        tm = statistics.fmean(v["tool_ok"] for v in rows[key]["tasks"].values() if v["tool_ok"] is not None)
        cm = [v["call_ok"] for v in rows[key]["tasks"].values() if v["call_ok"] is not None]
        rows[key]["task_mean"] = {"tool_ok": round(tm, 3), "call_ok": round(statistics.fmean(cm), 3) if cm else None}
        print(f"== {key:10s} acted {s['acted']:.2f}  tool_ok {s['tool_ok']:.2f} (chance {s['tool_chance']:.2f})  call_ok {s['call_ok']}  "
              f"task-mean tool {tm:.2f} call {rows[key]['task_mean']['call_ok']}  "
              f"prompt {s['prompt_tokens']} tok  [{rows[key]['seconds']}s]  | " + " ".join(f"{t[:6]} {v['tool_ok']:.2f}/{v['call_ok'] if v['call_ok'] is not None else '-'}"
                                                                                   for t, v in rows[key]["tasks"].items()), flush=True)
        for smp in rows[key]["samples"][: a.show]:
            print(f"   [{smp['task']} {smp['agent']}] {smp['user'][:160]!r}\n      calls={[c[0][:80] for c in smp['calls']]} answer={str(smp['answer'])[:100]!r}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"checkpoint": a.checkpoint, "n_items": len(items), "rows": rows}, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()

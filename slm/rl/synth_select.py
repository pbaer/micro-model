"""Selection data: the model's own candidate pools, collapsed and verified, with the right pick as the target.

    python -m slm.rl.synth_select --checkpoint runs/m9_rl5_336m/checkpoints/step_00200.pt \
        --n-gsm8k 800 --n-svamp 300 --n-synth 300 --k 8 --name select-sft

Writes two things under C:\\slm-data\\sft\\v1\\<name>\\:
  train/ val/          SFT shards: the selection prompt (slm.swarm.selector_messages) as the user turn, and an
                       assistant turn with a short templated think and `#### <gold>` -- the format the selector
                       is expected to answer in. Only problems where a correct candidate exists are included:
                       SFT teaches the format of picking, and there is nothing to pick when no attempt was right.
  select_pool.jsonl    every built prompt with its gold, including the no-correct-candidate ones, for the RL
                       `select` family (slm.rl.tasks.select_pool). RL rewards the emitted answer against gold, so
                       there the model can also learn to override a pool that is wrong.

No teacher: candidates are sampled from our own checkpoint, the collapse and evidence are mechanical, the
think line is a template stating what the pool showed, and the target is the dataset's gold. Sources are the
GSM8K and SVAMP *train* splits and the synthetic families (`algebra`, `arith2`, `word`); the test splits are
never touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

from slm.data.sft import SFT_DIR, SftShardWriter


def svamp_train_tasks(n: int, seed: int):
    import pyarrow.parquet as pq

    from slm.data.sources import SOURCES
    from slm.rl.tasks import Task

    files = [p for p in SOURCES["svamp"].local_dir.rglob("*.parquet") if "train" in p.name]
    rows = pq.read_table(files[0]).to_pylist() if files else []
    random.Random(seed).shuffle(rows)
    out = []
    for i, r in enumerate(rows[:n]):
        gold = str(r["Answer"]).strip()
        gold = gold[:-2] if gold.endswith(".0") else gold
        out.append(Task(id=f"svamp-train-{i}", prompt=r["question_concat"].strip(), answer=gold, task="svamp"))
    return out


THINK_PREFIX = "Looking at the attempts: "  # every think opens with the same tokens: see think_line


def think_line(groups, gold_key: str, k: int) -> str:
    """What the pool showed, stated plainly. Deterministic; it is the answer's justification, not a rationale.

    Every line opens with the same fixed prefix. The first version opened with the support count ("2 of 8
    attempts...") and greedy decoding never produced it: the probability of the first think token was spread
    over the digits while the empty think span -- one token, `<|/think|>` -- stayed the argmax, so the model
    learned the set (target loss 2.95 -> 1.08) and still answered in its old style. A fixed opening token is
    the argmax after very little training, and the rest follows it."""
    g = next((g for g in groups if g.key == gold_key), None)
    if g is None:
        return THINK_PREFIX + f"none of the {k} attempts reached the right answer; I will work it out myself."
    top = groups[0]
    if g is top:
        why = f"computed with code in {g.verified} of them" if g.verified else "the most agreed on"
        return THINK_PREFIX + f"{g.support} of {k} attempts reached {g.answer}, {why}. I will go with that."
    if g.verified and not top.verified:
        return THINK_PREFIX + (f"{top.support} attempts said {top.answer} but none computed it; {g.support} reached {g.answer} and "
                               f"{g.verified} of those computed it with code. The computed one is more trustworthy. I will go with {g.answer}.")
    return THINK_PREFIX + f"{g.support} of {k} attempts reached {g.answer}; checking it, that is the one that holds. I will go with {g.answer}."


_ANSWER_LINE = re.compile(r"^- Answer: (.+?) \(agreed by (\d+) attempts?; (?:computed with code in (\d+)|not computed with code)\)$")


def groups_from_prompt(prompt: str):
    """Recover the groups (answer, support, verified) from a rendered selection prompt, in prompt order --
    enough to rebuild the SFT targets from `select_pool.jsonl` without re-sampling."""
    from slm.swarm import Group, answer_key

    out = []
    for line in prompt.splitlines():
        m = _ANSWER_LINE.match(line)
        if m:
            ans, sup, ver = m.group(1), int(m.group(2)), int(m.group(3) or 0)
            out.append(Group(key=answer_key(ans), answer=ans, support=sup, verified=ver, members=[], rationale=""))
    return out


def rebuild(pool_path: str, name: str, tokenizer: str, k: int) -> dict:
    """Write a fresh SFT set (same prompts, current think template) from an existing pool file. No model."""
    from slm.data.chat import format_chat
    from slm.data.tokenizer import SlmTokenizer
    from slm.swarm import answer_key

    tok = SlmTokenizer.load(tokenizer)
    out = SFT_DIR / Path(tokenizer).name / name
    writers = {"train": SftShardWriter(out / "train"), "val": SftShardWriter(out / "val")}
    stats = Counter()
    for line in Path(pool_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        gold_key = answer_key(str(r["gold"]))
        groups = groups_from_prompt(r["prompt"])
        if not any(g.key == gold_key for g in groups):
            stats["no_correct"] += 1
            continue
        assistant = {"role": "assistant", "think": think_line(groups, gold_key, k), "content": f"#### {r['gold']}"}
        enc = format_chat(tok, [{"role": "user", "content": r["prompt"]}, assistant], think_required=True, tools=True)
        writers[r["split"]].add(enc.ids, enc.loss_mask)
        stats[f"sft:{r['split']}"] += 1
    for w in writers.values():
        w.flush()
    (out / "select_pool.jsonl").write_bytes(Path(pool_path).read_bytes())
    m = {"name": name, "source": "synth_select.rebuild", "pool": str(pool_path), "tokenizer_sha256": tok.sha256, "k": k, "think_prefix": THINK_PREFIX,
         "think_required": True, "tools": True, "counts": dict(stats),
         "train_examples": writers["train"].total_examples, "train_tokens": writers["train"].total_tokens, "train_targets": writers["train"].total_targets,
         "val_examples": writers["val"].total_examples, "val_tokens": writers["val"].total_tokens, "val_targets": writers["val"].total_targets}
    (out / "manifest.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    return m


def build(checkpoint: str, tokenizer: str, name: str, n_gsm8k: int, n_svamp: int, n_synth: int, k: int, seed: int,
          budget: int, temperature: float, max_new: int, val_permille: int = 50, device: str = "cuda") -> dict:
    import torch

    from slm.data.answers import SUFFIX
    from slm.data.chat import format_chat
    from slm.data.tokenizer import SlmTokenizer
    from slm.eval.quality import load_model
    from slm.rl.tasks import GENERATORS, gsm8k_pool
    from slm.swarm import answer_key, collapse, sample_candidates, selector_messages
    from slm.utils.sdpa import sdpa_context

    tok = SlmTokenizer.load(tokenizer)
    model, _ = load_model(Path(checkpoint), device)
    rng = random.Random(seed)
    tasks = []
    pool = list(gsm8k_pool()); rng.shuffle(pool); tasks += pool[:n_gsm8k]
    tasks += svamp_train_tasks(n_svamp, seed)
    fams = ["algebra", "arith2", "word"]
    seen = set()
    while sum(1 for t in tasks if t.task in fams) < n_synth:
        t = GENERATORS[rng.choice(fams)](rng)
        if t is not None and t.prompt not in seen:
            seen.add(t.prompt); tasks.append(t)
    rng.shuffle(tasks)

    out = SFT_DIR / Path(tokenizer).name / name
    writers = {"train": SftShardWriter(out / "train"), "val": SftShardWriter(out / "val")}
    out.mkdir(parents=True, exist_ok=True)
    pool_f = open(out / "select_pool.jsonl", "w", encoding="utf-8")
    stats = Counter()
    t0 = time.time()
    with torch.no_grad(), sdpa_context("decode"):
        for i, t in enumerate(tasks):
            gold_key = answer_key(t.answer)
            cands = sample_candidates(model, tok, [{"role": "user", "content": t.prompt + SUFFIX}], k, temperature, 0.95, max_new, 6,
                                      seed=seed * 7919 + i)
            groups = collapse(cands)
            if not groups:
                stats["no_groups"] += 1
                continue
            msgs = selector_messages(t.prompt, groups, tok, budget)
            has_correct = any(g.key == gold_key for g in groups)
            split = "val" if int.from_bytes(hashlib.sha1(t.prompt.encode()).digest()[:4], "little") % 1000 < val_permille else "train"
            pool_f.write(json.dumps({"prompt": msgs[0]["content"], "gold": t.answer, "source": t.task, "has_correct": has_correct,
                                     "n_groups": len(groups), "split": split}, ensure_ascii=False) + "\n")
            stats[f"pool:{t.task}"] += 1
            if not has_correct:
                stats["no_correct"] += 1
                continue
            assistant = {"role": "assistant", "think": think_line(groups, gold_key, k), "content": f"#### {t.answer}"}
            enc = format_chat(tok, msgs + [assistant], think_required=True, tools=True)
            writers[split].add(enc.ids, enc.loss_mask)
            stats[f"sft:{split}"] += 1
            if device != "cpu":
                torch.cuda.empty_cache()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(tasks)}: sft {stats['sft:train']}+{stats['sft:val']}, no correct candidate {stats['no_correct']} [{time.time() - t0:.0f}s]", flush=True)
    for w in writers.values():
        w.flush()
    pool_f.close()
    m = {"name": name, "source": "synth_select", "checkpoint": checkpoint, "tokenizer_sha256": tok.sha256, "k": k, "budget_tokens": budget,
         "think_required": True, "tools": True, "counts": dict(stats),
         "train_examples": writers["train"].total_examples, "train_tokens": writers["train"].total_tokens, "train_targets": writers["train"].total_targets,
         "val_examples": writers["val"].total_examples, "val_tokens": writers["val"].total_tokens, "val_targets": writers["val"].total_targets,
         "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--rebuild-from", default=None, help="an existing select_pool.jsonl: rewrite the SFT set from it with the current think template (no model)")
    ap.add_argument("--tokenizer", default=r"C:\slm-data\tokenizer\v1")
    ap.add_argument("--name", default="select-sft")
    ap.add_argument("--n-gsm8k", type=int, default=800)
    ap.add_argument("--n-svamp", type=int, default=300)
    ap.add_argument("--n-synth", type=int, default=300)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=2400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.rebuild_from:
        print(json.dumps(rebuild(a.rebuild_from, a.name, a.tokenizer, a.k), indent=1))
        return
    if not a.checkpoint:
        ap.error("--checkpoint is required unless --rebuild-from is given")
    m = build(a.checkpoint, a.tokenizer, a.name, a.n_gsm8k, a.n_svamp, a.n_synth, a.k, a.seed, a.budget, a.temperature, a.max_new, device=a.device)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()

"""Judged quality over training: run the prompt suite on saved checkpoints, hand the outputs to a judge
model blind, ingest the scores, and summarise them per checkpoint for the portal.

    python -m slm.eval.quality generate --run m8_base_stable_336m --device cpu --threads 8   # every snapshot
    python -m slm.eval.quality pack --run m8_base_stable_336m            # blind packets for the judge
    python -m slm.eval.quality ingest --run m8_base_stable_336m --scores scores.json --judge claude-sonnet
    python -m slm.eval.quality status --run m8_base_stable_336m

Layout under runs/<run>/quality/:
    outputs/<tokens:012d>.jsonl   header line + one line per prompt (the model's output, greedy)
    packets/<run>-NN.json         what the judge sees: shuffled items with opaque ids, no checkpoint identity
    scores.jsonl                  one line per judged item (item_id, scores, note, judge, versions)
    summary.json                  per-checkpoint means (overall, per rubric, per category) for the charts

The judge is an LLM driven from outside this module (a Claude subagent in practice; docs/quality_eval.md has the
protocol). Everything it needs is in the packet, including the rubric, and everything it returns is validated
here. Item ids are hashes of (run, tokens, prompt id, suite version): stable, and meaningless to the judge.

CPU generation is the default so it can run beside a training job: it pins a thread count and lowers the
process priority, and a 336M model does about 35 tokens/s that way (~1 minute per checkpoint).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

from slm.eval.quality_suite import CATEGORIES, JUDGE_INSTRUCTIONS, RUBRIC_VERSION, RUBRICS, SUITE, SUITE_VERSION
from slm.utils.stage import run_meta, run_stage, run_tools

LEGACY3 = ("rome", "fib", "cap_france")  # the three prompts the trainer has sampled since M1: the long-history subset
DEFAULT_TOKENIZER = r"C:\slm-data\tokenizer\v1"


# ----------------------------------------------------------------------------------------------- helpers
def item_id(run: str, tokens: int, prompt_id: str, suite: str = SUITE_VERSION) -> str:
    return hashlib.sha1(f"{run}|{tokens}|{prompt_id}|{suite}".encode()).hexdigest()[:12]


def qdir(run_dir: Path) -> Path:
    return Path(run_dir) / "quality"


def outputs_path(run_dir: Path, tokens: int) -> Path:
    return qdir(run_dir) / "outputs" / f"{tokens:012d}.jsonl"


def read_outputs(path: Path) -> tuple[dict, list[dict]]:
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header = lines[0] if lines and lines[0].get("header") else {}
    return header, [line for line in lines if not line.get("header")]


def all_outputs(run_dir: Path) -> list[tuple[dict, list[dict]]]:
    d = qdir(run_dir) / "outputs"
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.jsonl")):
        h, items = read_outputs(p)
        if h:
            out.append((h, items))
    return out


def read_scores(run_dir: Path) -> dict[str, dict]:
    p = qdir(run_dir) / "scores.jsonl"
    if not p.exists():
        return {}
    scores: dict[str, dict] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            scores[r["item_id"]] = r  # last write wins
    return scores


def selected_checkpoints(run_dir: Path, names: list[str] | None, every: int, include_final: bool = True) -> list[tuple[str, int]]:
    """(checkpoint file name, tokens) for the run's snapshots (every k-th) plus final.pt, from checkpoints/index.json."""
    idx_p = Path(run_dir) / "checkpoints" / "index.json"
    idx = json.loads(idx_p.read_text(encoding="utf-8")) if idx_p.exists() else {}
    if names:
        out = []
        for n in names:
            e = idx.get(n)
            if e is None or e.get("tokens") is None:
                raise SystemExit(f"{n}: not in {idx_p} (or no token count)")
            out.append((n, int(e["tokens"])))
        return out
    snaps = sorted(((int(e["tokens"]), n) for n, e in idx.items() if e.get("kind") == "snapshot" and e.get("tokens") is not None))
    picked = [(n, t) for i, (t, n) in enumerate(snaps) if i % max(1, every) == 0 or i == len(snaps) - 1]
    if include_final and "final.pt" in idx and idx["final.pt"].get("tokens") is not None:
        ft = int(idx["final.pt"]["tokens"])
        if all(t != ft for _, t in picked):
            picked.append(("final.pt", ft))
    return [(n, t) for n, t in picked if (Path(run_dir) / "checkpoints" / n).exists()]


def _lower_priority() -> None:
    """Below-normal process priority on Windows so a training job's data loader keeps the cores it needs."""
    try:
        import ctypes

        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
    except Exception:  # noqa: BLE001 (non-Windows or no permission: fine)
        pass


# ----------------------------------------------------------------------------------------------- generate
def load_model(path: Path, device: str):
    import torch

    from slm.config import ModelConfig, from_dict
    from slm.model import Transformer

    ck = torch.load(path, map_location=device, weights_only=False)
    mcfg = from_dict(ModelConfig, ck.get("meta", {}).get("model_config") or ck["config"])
    m = Transformer(mcfg).to(device)
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
    if device != "cpu":
        m = m.to(torch.bfloat16)
    return m.eval(), ck.get("meta", {})


def load_tokenizer(run_dir: Path):
    from slm.data.tokenizer import SlmTokenizer

    cfg = run_meta(run_dir).get("config") or {}
    return SlmTokenizer.load(cfg.get("tokenizer_dir") or DEFAULT_TOKENIZER)


def generate_suite(model, tok, stage: str, device: str, max_new_cap: int | None = None, tools: bool = False) -> list[dict]:
    """Greedy output for every suite prompt. Base checkpoints continue the completion prompt; SFT/RL checkpoints
    answer the chat prompt as an assistant (think span parsed off). tools=True runs the answer through the tool
    loop (one PySession per prompt), so a model trained to call Python sees real results instead of derailing on
    an empty one."""
    import torch

    from slm.data.chat import format_chat, parse_assistant
    from slm.utils.sdpa import sdpa_context

    with torch.no_grad(), sdpa_context("decode"):
        return _generate_suite(torch, model, tok, stage, device, max_new_cap, tools, format_chat, parse_assistant)


def _budget(p: dict, chat: bool, think_required: bool, max_new_cap: int | None) -> int:
    max_new = min(p["max_new_tokens"], max_new_cap) if max_new_cap else p["max_new_tokens"]
    if chat:
        max_new += 32  # an assistant answer carries a preamble and code fences the completion form does not
    if chat and think_required:
        max_new += 64  # room for a think span before the answer
    return max_new


def _generate_suite(torch, model, tok, stage: str, device: str, max_new_cap, tools: bool, format_chat, parse_assistant) -> list[dict]:
    chat = stage != "base"
    think_required = stage in ("reasoning", "rl")
    items = []
    if chat and tools:
        from slm.tools.loop import sample_with_tools

        prompts = [format_chat(tok, [{"role": "user", "content": p["chat"]}], add_generation_prompt=True, think_required=think_required, tools=True).ids for p in SUITE]
        budget = max(_budget(p, True, think_required, max_new_cap) for p in SUITE)
        t0 = time.time()
        outs = sample_with_tools(model, tok, prompts, budget, temperature=0.0, max_calls=8)
        per = round((time.time() - t0) / len(SUITE), 2)
        for p, o in zip(SUITE, outs):
            parsed = parse_assistant(tok, o.ids, think_expected=think_required)
            items.append({"id": p["id"], "category": p["category"], "mode": "chat", "prompt": p["chat"], "expect": p["expect"], "n_new": len(o.ids), "max_new": budget,
                          "seconds": per, "output": parsed["answer"], "think": parsed.get("think"), "malformed": bool(parsed.get("malformed")),
                          "stopped": o.termination == "stop", "tool_calls": o.n_calls, "tool_errors": o.n_errors, "termination": o.termination})
        return items
    for p in SUITE:
        if chat:
            ids = format_chat(tok, [{"role": "user", "content": p["chat"]}], add_generation_prompt=True, think_required=think_required).ids
            shown = p["chat"]
        else:
            ids = [tok.bos_id, *tok.encode(p["completion"])]
            shown = p["completion"]
        max_new = _budget(p, chat, think_required, max_new_cap)
        x = torch.tensor([ids], device=device)
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            out = model.generate(x, max_new, temperature=0.0, stop_ids=(tok.eos_id, tok.end_id) if chat else (tok.eos_id,))
        gen = out[0, len(ids) :].tolist()
        rec = {"id": p["id"], "category": p["category"], "mode": "chat" if chat else "completion", "prompt": shown, "expect": p["expect"],
               "n_new": len(gen), "max_new": max_new, "seconds": round(time.time() - t0, 2)}
        if chat:
            parsed = parse_assistant(tok, gen, think_expected=think_required)
            rec.update(output=parsed["answer"], think=parsed.get("think"), malformed=bool(parsed.get("malformed")), stopped=bool(parsed.get("terminated", True)))
        else:
            stopped = bool(gen) and gen[-1] == tok.eos_id
            rec.update(output=tok.decode([g for g in gen if g != tok.eos_id]), stopped=stopped)
        items.append(rec)
    return items


def write_outputs(run_dir: Path, run: str, tokens: int, checkpoint: str, stage: str, device: str, items: list[dict], seconds: float) -> Path:
    """Write quality/outputs/<tokens>.jsonl (header line + one line per prompt) and refresh summary.json.
    Shared by the CLI (saved checkpoints) and the trainer's milestone hook (the live model)."""
    header = {"header": True, "run": run, "tokens": tokens, "checkpoint": checkpoint, "stage": stage, "suite": SUITE_VERSION, "device": device,
              "n_items": len(items), "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(seconds, 1), "n_new_total": sum(i["n_new"] for i in items)}
    p = outputs_path(run_dir, tokens)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(header) + "\n")
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    write_summary(run_dir)
    return p


def cmd_generate(a: argparse.Namespace) -> None:
    run_dir = Path(a.runs_root) / a.run
    meta = run_meta(run_dir)
    stage, tools = run_stage(meta), run_tools(meta)
    picks = selected_checkpoints(run_dir, a.checkpoints, a.every, include_final=not a.no_final)
    if not picks:
        raise SystemExit(f"{a.run}: no checkpoints to score")
    todo = [(n, t) for n, t in picks if a.force or not outputs_path(run_dir, t).exists()]
    print(f"[{a.run}] stage={stage}{' +tools' if tools else ''} device={a.device} checkpoints={len(picks)} to generate={len(todo)} (suite {SUITE_VERSION}, {len(SUITE)} prompts)", flush=True)
    if a.device == "cpu":
        import torch

        torch.set_num_threads(a.threads)
        _lower_priority()
    tok = load_tokenizer(run_dir)
    for name, tokens in todo:
        t0 = time.time()
        model, meta = load_model(run_dir / "checkpoints" / name, a.device)
        if meta.get("tokenizer_sha256") and meta["tokenizer_sha256"] != tok.sha256:
            print(f"  WARNING {name}: checkpoint tokenizer {meta['tokenizer_sha256'][:8]} != loaded {tok.sha256[:8]}", flush=True)
        items = generate_suite(model, tok, stage, a.device, a.max_new, tools=tools)
        del model
        p = write_outputs(run_dir, a.run, tokens, name, stage + (" +tools" if tools else ""), a.device, items, time.time() - t0)
        secs = time.time() - t0
        print(f"  {name:16s} {tokens / 1e9:6.2f}B  {secs:6.1f}s  {sum(i['n_new'] for i in items) / max(1e-6, secs):5.1f} tok/s  -> {p.name}", flush=True)


# ----------------------------------------------------------------------------------------------- pack / ingest
def unjudged_items(run_dir: Path) -> list[dict]:
    """Items with no score yet. A byte-identical (prompt, output) pair is packed once: adjacent checkpoints often
    produce the same text, and two judge instances scoring the same text a point apart is pure noise (measured:
    15 of 28 identical pairs in M8 round 1). `propagate_scores` copies the score to the twins on ingest."""
    scores = read_scores(run_dir)
    out, seen = [], set()
    for h, items in all_outputs(run_dir):
        for it in items:
            iid = item_id(h["run"], h["tokens"], it["id"], h.get("suite", SUITE_VERSION))
            key = (it["id"], it["output"])
            if iid in scores or key in seen:
                continue
            seen.add(key)
            out.append({"item_id": iid, "category": it["category"], "mode": it["mode"], "prompt": it["prompt"], "output": it["output"], "expect": it["expect"]})
    return out


def propagate_scores(run_dir: Path, judge: str, now: str) -> int:
    """Give every unjudged item whose (prompt id, output) matches a judged one that item's score. Returns the count."""
    scores = read_scores(run_dir)
    by_key: dict[tuple[str, str], dict] = {}
    pending: list[tuple[str, int, str, dict]] = []
    for h, items in all_outputs(run_dir):
        for it in items:
            iid = item_id(h["run"], h["tokens"], it["id"], h.get("suite", SUITE_VERSION))
            key = (it["id"], it["output"])
            if iid in scores:
                by_key.setdefault(key, scores[iid])
            else:
                pending.append((iid, h["tokens"], it["id"], key))
    n = 0
    with (qdir(run_dir) / "scores.jsonl").open("a", encoding="utf-8", newline="\n") as out:
        for iid, tokens, pid, key in pending:
            src = by_key.get(key)
            if src is None:
                continue
            rec = {"item_id": iid, "tokens": tokens, "prompt_id": pid, "scores": dict(src["scores"]), "note": src.get("note", ""), "run": Path(run_dir).name,
                   "judge": src.get("judge", judge), "suite": SUITE_VERSION, "rubric": RUBRIC_VERSION, "judged_at": now, "copied_from": src["item_id"]}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def cmd_pack(a: argparse.Namespace) -> None:
    run_dir = Path(a.runs_root) / a.run
    items = unjudged_items(run_dir)
    if not items:
        print(f"[{a.run}] nothing to judge")
        return
    rng = random.Random(a.seed)
    rng.shuffle(items)  # the judge never sees two checkpoints' answers to one prompt side by side in order
    d = qdir(run_dir) / "packets"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob(f"{a.run}-*.json"):
        old.unlink()
    n_batches = (len(items) + a.batch - 1) // a.batch
    for k in range(n_batches):
        batch = items[k * a.batch : (k + 1) * a.batch]
        p = d / f"{a.run}-{k + 1:02d}.json"
        p.write_text(json.dumps({"run": a.run, "packet": k + 1, "of": n_batches, "suite": SUITE_VERSION, "rubric": RUBRIC_VERSION, "rubrics": list(RUBRICS),
                                 "instructions": JUDGE_INSTRUCTIONS, "items": batch}, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {p} ({len(batch)} items)")
    print(f"[{a.run}] {len(items)} unjudged items in {n_batches} packets under {d}")


def validate_scores(rows: list[dict], known: dict[str, tuple[int, str]]) -> list[dict]:
    good, bad = [], []
    for r in rows:
        iid = r.get("item_id")
        if iid not in known:
            bad.append((iid, "unknown item_id"))
            continue
        try:
            sc = {k: int(r[k]) for k in RUBRICS}
        except (KeyError, TypeError, ValueError) as e:
            bad.append((iid, f"missing/invalid rubric: {e}"))
            continue
        if any(not 1 <= v <= 5 for v in sc.values()):
            bad.append((iid, f"out of range: {sc}"))
            continue
        good.append({"item_id": iid, "tokens": known[iid][0], "prompt_id": known[iid][1], "scores": sc, "note": str(r.get("note", ""))[:200]})
    if bad:
        print(f"  rejected {len(bad)}: " + "; ".join(f"{i}: {why}" for i, why in bad[:8]))
    return good


def cmd_ingest(a: argparse.Namespace) -> None:
    run_dir = Path(a.runs_root) / a.run
    known: dict[str, tuple[int, str]] = {}
    for h, items in all_outputs(run_dir):
        for it in items:
            known[item_id(h["run"], h["tokens"], it["id"], h.get("suite", SUITE_VERSION))] = (h["tokens"], it["id"])
    rows: list[dict] = []
    for f in a.scores:
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        rows.extend(data if isinstance(data, list) else data.get("scores", []))
    good = validate_scores(rows, known)
    existing = read_scores(run_dir)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    n_new = n_dup = 0
    with (qdir(run_dir) / "scores.jsonl").open("a", encoding="utf-8", newline="\n") as out:
        for g in good:
            if g["item_id"] in existing and not a.replace:
                n_dup += 1
                continue
            g.update(run=a.run, judge=a.judge, suite=SUITE_VERSION, rubric=RUBRIC_VERSION, judged_at=now)
            out.write(json.dumps(g, ensure_ascii=False) + "\n")
            n_new += 1
    n_copy = propagate_scores(run_dir, a.judge, now)
    print(f"[{a.run}] ingested {n_new} scores ({n_dup} already judged, skipped; {n_copy} copied to identical outputs at other checkpoints)")
    write_summary(run_dir)


# ----------------------------------------------------------------------------------------------- summary
def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(run_dir: Path) -> dict:
    scores = read_scores(run_dir)
    ckpts = []
    for h, items in all_outputs(run_dir):
        rows = []
        for it in items:
            s = scores.get(item_id(h["run"], h["tokens"], it["id"], h.get("suite", SUITE_VERSION)))
            if s:
                sc = s["scores"]
                rows.append((it, sc, sum(sc[r] for r in RUBRICS) / len(RUBRICS)))
        entry = {"tokens": h["tokens"], "checkpoint": h["checkpoint"], "stage": h["stage"], "n_items": len(items), "n_scored": len(rows),
                 "overall": _mean([o for _, _, o in rows]), "legacy3": _mean([o for it, _, o in rows if it["id"] in LEGACY3]),
                 "legacy3_n": sum(1 for it, _, _ in rows if it["id"] in LEGACY3), "stopped_frac": round(sum(1 for it in items if it.get("stopped")) / max(1, len(items)), 3)}
        for r in RUBRICS:
            entry[r] = _mean([sc[r] for _, sc, _ in rows])
        entry["categories"] = {c: {"overall": _mean([o for it, _, o in rows if it["category"] == c]), "n": sum(1 for it, _, _ in rows if it["category"] == c),
                                   **{r: _mean([sc[r] for it, sc, _ in rows if it["category"] == c]) for r in RUBRICS}} for c in CATEGORIES}
        ckpts.append(entry)
    ckpts.sort(key=lambda e: e["tokens"])
    judges = sorted({s.get("judge", "?") for s in scores.values()})
    return {"run": Path(run_dir).name, "suite": SUITE_VERSION, "rubric": RUBRIC_VERSION, "rubrics": list(RUBRICS), "categories": CATEGORIES, "n_prompts": len(SUITE),
            "judges": judges, "checkpoints": ckpts, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def write_summary(run_dir: Path) -> dict:
    s = summarize(run_dir)
    if s["checkpoints"]:
        (qdir(run_dir) / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    return s


def checkpoint_detail(run_dir: Path, tokens: int) -> dict:
    """Outputs joined with scores for one checkpoint (the portal's per-checkpoint table)."""
    p = outputs_path(run_dir, tokens)
    if not p.exists():
        raise FileNotFoundError(p)
    h, items = read_outputs(p)
    scores = read_scores(run_dir)
    for it in items:
        s = scores.get(item_id(h["run"], h["tokens"], it["id"], h.get("suite", SUITE_VERSION)))
        it["scores"] = s["scores"] if s else None
        it["note"] = s.get("note") if s else None
        it["judge"] = s.get("judge") if s else None
    return {"header": h, "items": items}


def cmd_status(a: argparse.Namespace) -> None:
    run_dir = Path(a.runs_root) / a.run
    s = summarize(run_dir)
    if not s["checkpoints"]:
        print(f"[{a.run}] no outputs yet")
        return
    print(f"[{a.run}] suite {s['suite']} rubric {s['rubric']} judges {s['judges']}")
    print(f"{'tokens':>9} {'checkpoint':16s} {'scored':>7} {'overall':>7} " + " ".join(f"{r[:6]:>6}" for r in RUBRICS))
    for c in s["checkpoints"]:
        print(f"{c['tokens'] / 1e9:8.2f}B {c['checkpoint']:16s} {c['n_scored']:3d}/{c['n_items']:<3d} {str(c['overall']):>7} " + " ".join(f"{str(c[r]):>6}" for r in RUBRICS))


# ----------------------------------------------------------------------------------------------- cli
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default="runs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="run the suite on checkpoints and write outputs")
    g.add_argument("--run", required=True)
    g.add_argument("--checkpoints", nargs="*", default=None, help="checkpoint file names (default: every snapshot + final.pt)")
    g.add_argument("--every", type=int, default=1, help="use every k-th snapshot")
    g.add_argument("--no-final", action="store_true")
    g.add_argument("--device", default="cpu")
    g.add_argument("--threads", type=int, default=8, help="CPU threads (leave cores for a running trainer)")
    g.add_argument("--max-new", type=int, default=None, help="cap max_new_tokens for every prompt")
    g.add_argument("--force", action="store_true", help="regenerate existing outputs")
    g.set_defaults(fn=cmd_generate)
    p = sub.add_parser("pack", help="write blind judge packets for unjudged items")
    p.add_argument("--run", required=True)
    p.add_argument("--batch", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_pack)
    i = sub.add_parser("ingest", help="validate and store judge scores, rebuild the summary")
    i.add_argument("--run", required=True)
    i.add_argument("--scores", nargs="+", required=True, help="json files: a list of {item_id, correctness, coherence, task, note}")
    i.add_argument("--judge", required=True, help="judge model name recorded with every score")
    i.add_argument("--replace", action="store_true", help="re-score items that already have scores")
    i.set_defaults(fn=cmd_ingest)
    s = sub.add_parser("status")
    s.add_argument("--run", required=True)
    s.set_defaults(fn=cmd_status)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()

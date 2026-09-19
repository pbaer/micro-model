"""Replay every stored tool call in the SFT tool sets and check the stored <|python_result|> is byte-for-byte what
the sandbox produces now (one PySession per conversation, declared functions registered from the generator's
service registry, calls replayed in order so REPL state carries).

    python scripts/verify_tool_results.py                 # every *tools* set under C:\\slm-data\\sft\\v1
    python scripts/verify_tool_results.py gsm8k-tools --max-conversations 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from slm.data.tokenizer import SlmTokenizer
from slm.tools.functions import functions_env, parse_defs
from slm.tools.protocol import run_tool
from slm.tools.pysandbox import PySession


def service_impls() -> dict:
    """name -> impl for every declared-function service the generator knows (tables are deterministic)."""
    from slm.rl.synth_python import DECL_SERVICES

    impls = {}
    for make in DECL_SERVICES.values():
        for d in make():
            impls[d.name] = d.impl
    return impls


def verify(set_dir: Path, tok: SlmTokenizer, impls: dict, max_conversations: int | None) -> dict:
    sp = {n: tok.special(n) for n in ("<|bos|>", "<|python_call|>", "<|/python_call|>", "<|python_result|>", "<|/python_result|>", "<|python_def|>", "<|/python_def|>")}
    bos, co, cc, ro, rc = sp["<|bos|>"], sp["<|python_call|>"], sp["<|/python_call|>"], sp["<|python_result|>"], sp["<|/python_result|>"]
    stats = {"conversations": 0, "calls": 0, "errors_stored": 0, "mismatches": 0, "examples": []}
    for shard in sorted((set_dir / "train").glob("tokens_*.bin")):
        ids = np.fromfile(shard, dtype=np.uint16)
        starts = np.flatnonzero(ids == bos)
        bounds = list(starts) + [len(ids)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if max_conversations and stats["conversations"] >= max_conversations:
                return stats
            conv = ids[a:b].tolist()
            decls = parse_defs(tok, conv)
            session = PySession(functions={d.name: impls[d.name] for d in decls if d.name in impls}) if decls else PySession()
            missing = [d.name for d in decls if d.name not in impls]
            if missing:
                stats["examples"].append((shard.name, a, f"no impl for declared {missing}")); stats["mismatches"] += 1
                continue
            stats["conversations"] += 1
            i = 0
            while True:
                try:
                    c0 = conv.index(co, i)
                except ValueError:
                    break
                c1 = conv.index(cc, c0); r0 = conv.index(ro, c1); r1 = conv.index(rc, r0)
                code = tok.decode(conv[c0 + 1:c1]); stored = tok.decode(conv[r0 + 1:r1])
                res, ok = run_tool(code, session)
                stats["calls"] += 1
                stats["errors_stored"] += stored.startswith("error:")
                if res != stored:
                    stats["mismatches"] += 1
                    if len(stats["examples"]) < 6:
                        stats["examples"].append((shard.name, a, f"code={code[:60]!r} stored={stored[:60]!r} replay={res[:60]!r}"))
                i = r1 + 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="*")
    ap.add_argument("--sft-root", default=r"C:\slm-data\sft\v1")
    ap.add_argument("--tokenizer", default=r"C:\slm-data\tokenizer\v1")
    ap.add_argument("--max-conversations", type=int, default=None)
    a = ap.parse_args()
    root = Path(a.sft_root)
    sets = a.sets or sorted(p.name for p in root.iterdir() if p.is_dir() and "tools" in p.name)
    tok = SlmTokenizer.load(a.tokenizer)
    impls = service_impls()
    total_calls = total_mis = 0
    for name in sets:
        s = verify(root / name, tok, impls, a.max_conversations)
        total_calls += s["calls"]; total_mis += s["mismatches"]
        print(f"{name:28s} conversations={s['conversations']:7d} calls={s['calls']:7d} stored-errors={s['errors_stored']:6d} mismatches={s['mismatches']}", flush=True)
        for ex in s["examples"]:
            print("   ", ex)
    print(f"TOTAL calls={total_calls} mismatches={total_mis}")
    sys.exit(1 if total_mis else 0)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()

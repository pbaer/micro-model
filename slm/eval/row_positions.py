"""Per-position loss inside packed training rows: does a row that starts mid-document hurt?

Rows are `seq_len` consecutive tokens of a source's stream, so most rows open with the tail of one document (a
truncated prefix), then `<|bos|>` and further documents. This measures, on validation rows packed exactly the way
training packs them:

  tail      loss of tokens BEFORE the first <|bos|> in the row, by position in the row (context = truncated prefix)
  in_doc    loss of tokens AFTER a <|bos|>, by distance from that <|bos|> (context = full document prefix)
  in_doc split by whether the document started at row position 0 (nothing before it) or later (earlier
            documents sit in the context): the cross-document-attention question

    python -m slm.eval.row_positions --run m8_base_stable_336m --checkpoint final.pt --rows 256 --device cpu --threads 24

Writes runs/<run>/row_position_loss.json and prints a binned table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from slm.eval.quality import _lower_priority, load_model, load_tokenizer

BINS = [(0, 1), (1, 4), (4, 16), (16, 64), (64, 256), (256, 1024), (1024, 2048)]


def _next_aligned_window(stream, bos: int, n: int):
    """Advance the stream to the next <|bos|> and return a window starting exactly there (control condition:
    every row begins a document, so nothing from another document precedes it)."""
    import numpy as np

    while True:
        w = stream.next_window(n)
        hits = np.flatnonzero(w == bos)
        if len(hits) == 0:
            continue
        j = int(hits[0])
        # the model predicts w[1:], so "a document at row start" means w[1] == bos: the window must begin one
        # token before the bos (the previous document's <|eos|>)
        if j == 1:
            return w
        if j == 0:
            continue  # no token before this bos in the window; take the next one
        stream.offset -= n - (j - 1)
        if stream.offset < 0:  # crossed a shard boundary inside next_window; just take the next window
            stream.offset = 0
            continue
        return stream.next_window(n)


def collect(model, tok, stream, seq_len: int, n_rows: int, device: str, log=print, align_bos: bool = False) -> dict:
    import torch
    import torch.nn.functional as F

    bos = tok.bos_id
    tail_sum = np.zeros(seq_len); tail_n = np.zeros(seq_len)
    doc_sum = np.zeros(seq_len); doc_n = np.zeros(seq_len)            # by distance from the token's own <|bos|>
    doc0_sum = np.zeros(seq_len); doc0_n = np.zeros(seq_len)          # documents that start at row position 0
    docx_sum = np.zeros(seq_len); docx_n = np.zeros(seq_len)          # documents that start later in the row
    pos_sum = np.zeros(seq_len); pos_n = np.zeros(seq_len)            # everything, by row position
    n_tail_rows = 0
    t0 = time.time()
    with torch.no_grad():
        for r in range(n_rows):
            w = _next_aligned_window(stream, bos, seq_len + 1) if align_bos else stream.next_window(seq_len + 1)
            x = torch.tensor(w[:-1], dtype=torch.long, device=device)[None]
            y = torch.tensor(w[1:], dtype=torch.long, device=device)[None]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                logits = model(x)
            loss = F.cross_entropy(logits[0].float(), y[0], reduction="none").cpu().numpy()  # loss[i] predicts token i+1 from x[:i+1]
            ids = w[1:]  # the predicted tokens; position i in the row means the model saw x[0..i]
            # which document does each predicted token belong to? a <|bos|> at index j starts a document at j
            bos_pos = np.flatnonzero(ids == bos)
            first = bos_pos[0] if len(bos_pos) else seq_len
            if first > 0:
                n_tail_rows += 1
            pos_sum += loss; pos_n += 1
            tail_sum[:first] += loss[:first]; tail_n[:first] += 1
            starts = list(bos_pos) + [seq_len]
            for a, b in zip(starts[:-1], starts[1:]):
                d = np.arange(b - a)  # distance from this document's <|bos|> (0 = the bos token itself, predicted from the previous doc)
                seg = loss[a:b]
                doc_sum[d] += seg; doc_n[d] += 1
                if a == 0:
                    doc0_sum[d] += seg; doc0_n[d] += 1
                else:
                    docx_sum[d] += seg; docx_n[d] += 1
            if (r + 1) % 32 == 0:
                log(f"  {r + 1}/{n_rows} rows, {time.time() - t0:.0f}s")
    mean = lambda s, n: np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return {"seq_len": seq_len, "rows": n_rows, "align_bos": align_bos, "rows_starting_mid_document": n_tail_rows, "tail_token_share": float(tail_n.sum() / pos_n.sum()),
            "by_position": {"all": mean(pos_sum, pos_n).tolist(), "tail": mean(tail_sum, tail_n).tolist(), "tail_n": tail_n.tolist()},
            "by_doc_distance": {"all": mean(doc_sum, doc_n).tolist(), "doc_at_row_start": mean(doc0_sum, doc0_n).tolist(), "doc_after_other_text": mean(docx_sum, docx_n).tolist(),
                                "n_at_row_start": doc0_n.tolist(), "n_after_other_text": docx_n.tolist()}}


def binned(values: list[float], counts: list[float] | None = None) -> list[tuple[str, float, int]]:
    v = np.array(values, dtype=float); c = np.array(counts if counts is not None else [1] * len(values), dtype=float)
    out = []
    for lo, hi in BINS:
        seg, cnt = v[lo:hi], c[lo:hi]
        ok = ~np.isnan(seg) & (cnt > 0)
        out.append((f"{lo}-{hi - 1}" if hi - lo > 1 else str(lo), float(np.average(seg[ok], weights=cnt[ok])) if ok.any() else float("nan"), int(cnt[ok].sum())))
    return out


def table(res: dict) -> str:
    bp, bd = res["by_position"], res["by_doc_distance"]
    lines = [f"{res['rows']} rows x {res['seq_len']} tokens; {res['rows_starting_mid_document']} rows start mid-document; {res['tail_token_share'] * 100:.1f}% of tokens sit before the row's first <|bos|>",
             "", f"{'position / distance':>20} | {'tail (truncated prefix)':>24} | {'in-doc (full prefix)':>22} | {'doc at row start':>17} | {'doc after other text':>21}"]
    tail = binned(bp["tail"], bp["tail_n"]); doc = binned(bd["all"], [a + b for a, b in zip(bd["n_at_row_start"], bd["n_after_other_text"])])
    d0 = binned(bd["doc_at_row_start"], bd["n_at_row_start"]); dx = binned(bd["doc_after_other_text"], bd["n_after_other_text"])
    for (name, t, tn), (_, dv, dn), (_, z, zn), (_, x, xn) in zip(tail, doc, d0, dx):
        f = lambda v, n: f"{v:6.3f} (n={n})" if n else "      -"
        lines.append(f"{name:>20} | {f(t, tn):>24} | {f(dv, dn):>22} | {f(z, zn):>17} | {f(x, xn):>21}")
    overall = float(np.nanmean(bp["all"]))
    lines.append("")
    lines.append(f"mean loss, all tokens: {overall:.4f}; tail tokens: {np.nansum(np.array(bp['tail']) * np.array(bp['tail_n'])) / max(1, sum(bp['tail_n'])):.4f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--checkpoint", default="final.pt")
    ap.add_argument("--source", default="fineweb-edu-10bt")
    ap.add_argument("--tokenized-root", default=r"C:\slm-data\tokenized\v1")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--align-bos", action="store_true", help="control: every row starts at a document boundary")
    ap.add_argument("--out", default=None, help="json path (default runs/<run>/row_position_loss[_aligned].json)")
    a = ap.parse_args()
    from slm.data.loader import TokenStream
    from slm.utils.sdpa import sdpa_context

    if a.device == "cpu" and a.threads:
        import torch

        torch.set_num_threads(a.threads)
        _lower_priority()
    run_dir = Path(a.runs_root) / a.run
    tok = load_tokenizer(run_dir)
    model, _ = load_model(run_dir / "checkpoints" / a.checkpoint, a.device)
    stream = TokenStream(Path(a.tokenized_root) / a.source / "val")
    with sdpa_context(None if a.device == "cpu" else "auto"):
        res = collect(model, tok, stream, a.seq_len, a.rows, a.device, align_bos=a.align_bos)
    res.update(run=a.run, checkpoint=a.checkpoint, source=a.source)
    out = Path(a.out) if a.out else run_dir / ("row_position_loss_aligned.json" if a.align_bos else "row_position_loss.json")
    out.write_text(json.dumps(res), encoding="utf-8")
    print(table(res))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()

"""lm-evaluation-harness adapter for our checkpoints (HellaSwag, ARC, PIQA, MMLU subsets, ...).

    python -m slm.eval.lm_eval_wrapper --checkpoint runs/<run>/checkpoints/final.pt --tasks hellaswag,arc_easy,piqa --limit 500

Only what the multiple-choice tasks need is implemented carefully (loglikelihood with a
context/continuation split that respects BPE merges); generate_until is greedy and simple.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from slm.config import ModelConfig, from_dict
from slm.data.tokenizer import SlmTokenizer
from slm.model import Transformer
from slm.utils.sdpa import sdpa_context


@register_model("slm")
class SlmLM(LM):
    def __init__(self, checkpoint: str, tokenizer: str = r"C:\slm-data\tokenizer\v1", batch_size: int = 16, device: str = "cuda", max_len: int | None = None) -> None:
        super().__init__()
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        mcfg = from_dict(ModelConfig, ck.get("meta", {}).get("model_config") or ck["config"])
        self.model = Transformer(mcfg).to(device)
        self.model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
        self.model.eval()
        self.tok = SlmTokenizer.load(tokenizer)
        self._device = torch.device(device)
        self.bs = int(batch_size)
        self.max_len = max_len or mcfg.max_seq_len

    # ---------------------------------------------------------------- helpers
    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        whole = self.tok.encode(context + continuation)
        ctx = self.tok.encode(context) if context else []
        if whole[: len(ctx)] == ctx:
            cont = whole[len(ctx) :]
        else:  # merge across the boundary: fall back to separate encodings
            cont = self.tok.encode(continuation)
        return [self.tok.bos_id, *ctx], cont

    @torch.no_grad()
    def _score_batch(self, items: list[tuple[list[int], list[int]]]) -> list[tuple[float, bool]]:
        seqs = [(c + k)[-self.max_len :] for c, k in items]
        L = max(len(s) for s in seqs)
        x = torch.full((len(seqs), L), self.tok.pad_id, dtype=torch.long, device=self._device)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s, device=self._device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._device.type == "cuda"):
            logits = self.model(x)
        logp = F.log_softmax(logits.float(), dim=-1)
        out = []
        for i, (c, k) in enumerate(items):
            s = seqs[i]
            n_k = len(k)
            start = len(s) - n_k  # first continuation position in the (possibly truncated) sequence
            lp = logp[i, start - 1 : len(s) - 1]
            tgt = x[i, start : len(s)]
            tok_lp = lp.gather(1, tgt[:, None]).squeeze(1)
            greedy = bool((lp.argmax(-1) == tgt).all())
            out.append((float(tok_lp.sum()), greedy))
        return out

    # ---------------------------------------------------------------- LM API
    def loglikelihood(self, requests, disable_tqdm: bool = False):
        pairs = [self._encode_pair(*r.args) for r in requests]
        order = sorted(range(len(pairs)), key=lambda i: -(len(pairs[i][0]) + len(pairs[i][1])))
        res = [None] * len(pairs)
        with sdpa_context("auto"):
            for b in range(0, len(order), self.bs):
                idx = order[b : b + self.bs]
                for i, r in zip(idx, self._score_batch([pairs[i] for i in idx])):
                    res[i] = r
        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        out = []
        with sdpa_context("auto"):
            for r in requests:
                ids = [self.tok.bos_id, *self.tok.encode(r.args[0])]
                total = 0.0
                for s in range(1, len(ids), self.max_len - 1):
                    chunk = ids[max(0, s - 1) : s - 1 + self.max_len]
                    ctx, cont = chunk[:1], chunk[1:]
                    if not cont:
                        break
                    total += self._score_batch([(ctx, cont)])[0][0]
                out.append(total)
        return out

    def generate_until(self, requests, disable_tqdm: bool = False):
        out = []
        with sdpa_context("decode"), torch.no_grad():
            for r in requests:
                context, gen_kwargs = r.args
                until = gen_kwargs.get("until", []) or []
                max_new = int(gen_kwargs.get("max_gen_toks", 128))
                ids = [self.tok.bos_id, *self.tok.encode(context)][-(self.max_len - max_new) :]
                x = torch.tensor([ids], device=self._device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._device.type == "cuda"):
                    y = self.model.generate(x, max_new, temperature=0.0, stop_ids=(self.tok.eos_id,))
                text = self.tok.decode(y[0, len(ids) :].tolist(), skip_special=True)
                for u in until:
                    if u in text:
                        text = text.split(u)[0]
                out.append(text)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=r"C:\slm-data\tokenizer\v1")
    ap.add_argument("--tasks", default="hellaswag,arc_easy,piqa")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.environ.setdefault("HF_HOME", r"C:\slm-data\hf-cache")
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
    import lm_eval

    lm = SlmLM(a.checkpoint, a.tokenizer, a.batch_size)
    res = lm_eval.simple_evaluate(model=lm, tasks=a.tasks.split(","), limit=a.limit, log_samples=False)
    summary = {t: {k: v for k, v in m.items() if not k.endswith("stderr") and isinstance(v, (int, float))} for t, m in res["results"].items()}
    for t, m in summary.items():
        print(t, {k: round(v, 4) for k, v in m.items()})
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"checkpoint": a.checkpoint, "limit": a.limit, "results": summary}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())

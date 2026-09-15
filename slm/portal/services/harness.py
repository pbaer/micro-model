"""Model harness that runs INSIDE the worker subprocess: checkpoint slots, streaming generation
with per-token log-probs and top-k alternatives, teacher-forced scoring, and diagnostics."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from slm.config import ModelConfig, from_dict
from slm.data.chat import format_chat, parse_assistant
from slm.data.tokenizer import SlmTokenizer
from slm.eval.sampling import sample_next
from slm.model import KVCache, Transformer
from slm.tools.protocol import run_tool, tool_ids
from slm.tools.pysandbox import PySession

MAX_SESSIONS = 32
TOOL_RESULT_ROOM = 48  # cache slots reserved per allowed tool call for inserted result tokens


class Slot:
    def __init__(self, name: str) -> None:
        self.name = name
        self.model: Transformer | None = None
        self.tok: SlmTokenizer | None = None
        self.info: dict = {}


class Harness:
    def __init__(self, tokenizer_root: Path) -> None:
        self.tokenizer_root = Path(tokenizer_root)
        self.slots: dict[str, Slot] = {"A": Slot("A"), "B": Slot("B")}
        self._toks: dict[str, SlmTokenizer] = {}
        self.sessions: dict[str, PySession] = {}  # conversation id -> Python session (REPL state)

    # --------------------------------------------------------------- sessions
    def session(self, session_id: str | None) -> PySession:
        sid = session_id or "default"
        if sid not in self.sessions:
            if len(self.sessions) >= MAX_SESSIONS:
                self.sessions.pop(next(iter(self.sessions)))
            self.sessions[sid] = PySession()
        return self.sessions[sid]

    def reset_session(self, session_id: str | None) -> dict:
        sid = session_id or "default"
        self.sessions.pop(sid, None)
        return {"session_id": sid, "reset": True}

    # ------------------------------------------------------------------ slots
    def _tokenizer(self, sha: str | None, tag: str | None) -> tuple[SlmTokenizer, str, bool]:
        tags = [d for d in sorted(self.tokenizer_root.iterdir()) if (d / "tokenizer.json").exists()] if self.tokenizer_root.exists() else []
        chosen, matched = None, False
        for d in tags:
            t = self._toks.get(d.name) or SlmTokenizer.load(d)
            self._toks[d.name] = t
            if sha and t.sha256 == sha:
                chosen, matched = d.name, True
                break
        if chosen is None:
            chosen = tag if tag and (self.tokenizer_root / tag / "tokenizer.json").exists() else (tags[-1].name if tags else None)
        if chosen is None:
            raise RuntimeError("no tokenizer found")
        return self._toks[chosen], chosen, matched

    def load(self, slot: str, checkpoint: str, device: str = "cuda", dtype: str = "bf16", tokenizer_tag: str | None = None) -> dict:
        s = self.slots[slot]
        self.unload(slot)
        t0 = time.time()
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=False)
        mcfg_d = ck.get("meta", {}).get("model_config") or (ck.get("config") if "n_layers" in ck.get("config", {}) else None)
        if mcfg_d is None:
            raise RuntimeError("checkpoint has no model config")
        mcfg = from_dict(ModelConfig, mcfg_d)
        model = Transformer(mcfg)
        sd = {k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()}
        model.load_state_dict(sd)
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        if device == "cpu" and dt == torch.float16:
            dt = torch.float32
        model = model.to(device=device, dtype=dt if device == "cuda" else torch.float32).eval()
        meta = ck.get("meta", {})
        tok, tag, matched = self._tokenizer(meta.get("tokenizer_sha256"), tokenizer_tag)
        s.model, s.tok = model, tok
        s.info = {
            "slot": slot, "checkpoint": str(checkpoint), "name": Path(checkpoint).name, "device": device, "dtype": str(model.output_weight.dtype).replace("torch.", ""),
            "params": model.num_params(), "tokens": meta.get("tokens") or ck.get("counters", {}).get("tokens"), "val_loss": meta.get("val_loss"),
            "tokenizer": tag, "tokenizer_matched": matched, "model_config": mcfg_d, "load_s": time.time() - t0,
            "vram_gib": torch.cuda.memory_allocated() / 2**30 if device == "cuda" else 0.0,
        }
        del ck, sd
        return s.info

    def unload(self, slot: str) -> dict:
        s = self.slots[slot]
        s.model, s.tok, s.info = None, None, {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"slot": slot, "loaded": False}

    def status(self) -> dict:
        return {"slots": {k: (v.info if v.model is not None else {"slot": k, "loaded": False}) for k, v in self.slots.items()},
                "cuda": torch.cuda.is_available(), "vram_gib": torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0}

    # --------------------------------------------------------------- prompts
    def _prompt_ids(self, s: Slot, mode: str, text: str = "", messages: list[dict] | None = None, think_required: bool = False,
                    tools: bool = False, session: PySession | None = None) -> list[int]:
        tok = s.tok
        if mode == "chat":
            return format_chat(tok, messages or [], add_generation_prompt=True, think_required=think_required, tools=tools, session=session).ids
        return [tok.bos_id, *tok.encode(text)]

    # ------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, slot: str, mode: str = "completion", text: str = "", messages: list[dict] | None = None, temperature: float = 0.8, top_p: float = 0.95,
                 top_k: int = 0, max_new_tokens: int = 128, seed: int | None = 1234, logprobs_topk: int = 5, think_required: bool = False,
                 tools: bool = False, session_id: str | None = None, max_tool_calls: int = 8, should_stop=None):
        """Yields {'event': 'prompt'|'token'|'tool'|'done', ...}.

        With tools (chat mode) generation pauses at <|/python_call|>: the code runs in the conversation's
        session, a 'tool' event reports it, the result tokens are fed through the model (marked inserted:
        true, no log-prob) and sampling resumes. A call after <|/think|> ends the turn as malformed. The
        'done' event carries the parsed assistant turn (think, answer, ids, well_formed) so a UI can append
        it to the conversation verbatim."""
        s = self.slots[slot]
        if s.model is None:
            raise RuntimeError(f"slot {slot} is empty")
        model, tok = s.model, s.tok
        device = next(model.parameters()).device
        tools = bool(tools) and mode == "chat"
        sess = self.session(session_id) if tools else None
        ids = self._prompt_ids(s, mode, text, messages, think_required, tools, sess)
        if len(ids) + max_new_tokens > model.cfg.max_seq_len:
            max_new_tokens = max(1, model.cfg.max_seq_len - len(ids))
        stop = {tok.eos_id, tok.end_id} if mode == "chat" else {tok.eos_id}
        t = tool_ids(tok) if tools else None
        think_close = tok.special("<|/think|>")
        if tools:
            stop = stop | {t["call_close"]}
        yield {"event": "prompt", "ids": ids, "pieces": [tok.token_str(i) for i in ids], "n": len(ids)}
        gen = torch.Generator(device=device)
        gen.manual_seed(seed if seed is not None else int(time.time() * 1000) % 2**31)
        capacity = min(model.cfg.max_seq_len, len(ids) + max_new_tokens + (max_tool_calls * TOOL_RESULT_ROOM if tools else 0))
        cache = KVCache(model.cfg, 1, capacity, device, model.output_weight.dtype)
        cur = torch.tensor([ids], device=device)
        t0 = time.time()
        n = 0  # sampled tokens
        total = len(ids)  # tokens in the cache
        gen_ids: list[int] = []  # everything after the prompt, sampled and inserted
        calls: list[dict] = []
        reason = "length"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            while n < max_new_tokens:
                logits = model(cur, cache=cache, last_only=True)[:, -1, :].float()
                logp = torch.log_softmax(logits, dim=-1)
                nxt = sample_next(logits, temperature, top_p, top_k, gen)
                k = max(1, min(logprobs_topk, logits.shape[-1]))
                top = torch.topk(logp[0], k)
                nid = int(nxt[0])
                rank = int((logp[0] > logp[0, nid]).sum())
                n += 1
                total += 1
                gen_ids.append(nid)
                yield {"event": "token", "id": nid, "piece": tok.token_str(nid), "logprob": float(logp[0, nid]), "rank": rank,
                       "entropy": float(-(logp[0].exp() * logp[0]).sum()),
                       "topk": [{"id": int(i), "piece": tok.token_str(int(i)), "logprob": float(v)} for v, i in zip(top.values, top.indices)],
                       "special": nid >= tok.base_vocab, "inserted": False}
                if tools and nid == t["call_close"]:
                    if think_close in gen_ids:
                        reason = "tool_outside_think"
                        break
                    if len(calls) >= max_tool_calls:
                        reason = "max_calls"
                        break
                    opens = [i for i, x in enumerate(gen_ids) if x == t["call_open"]]
                    if opens and opens[-1] < len(gen_ids) - 1:
                        code = tok.decode(gen_ids[opens[-1] + 1 : -1], skip_special=True)
                        result, ok = run_tool(code, sess)
                    else:
                        code, result, ok = "", "error: malformed call (no <|python_call|> opening tag)", False
                    calls.append({"code": code, "result": result, "ok": ok})
                    ins = [t["result_open"], *tok.encode(result), t["result_close"]]
                    if total + len(ins) + 1 > capacity:
                        reason = "length"
                        break
                    yield {"event": "tool", "code": code, "result": result, "ok": ok, "n_calls": len(calls)}
                    for tid in ins:
                        yield {"event": "token", "id": tid, "piece": tok.token_str(tid), "logprob": None, "rank": None, "entropy": None, "topk": [],
                               "special": tid >= tok.base_vocab, "inserted": True}
                    gen_ids.extend(ins)
                    total += len(ins)
                    cur = torch.tensor([ins], device=device)
                    continue
                if nid in stop:
                    reason = "stop"
                    break
                if should_stop is not None and should_stop():
                    reason = "cancelled"
                    break
                if total + 1 > capacity:
                    reason = "length"
                    break
                cur = nxt[:, None]
        dt = time.time() - t0
        done = {"event": "done", "n": n, "n_total": len(gen_ids), "seconds": dt, "tok_s": n / dt if dt > 0 else 0.0, "reason": reason, "calls": calls}
        if mode == "chat":
            parsed = parse_assistant(tok, gen_ids, think_expected=think_required)
            malformed = bool(parsed["malformed"]) or reason not in ("stop",)
            done["assistant"] = {"think": parsed["think"], "answer": parsed["answer"], "terminated": parsed["terminated"], "malformed": malformed,
                                 "well_formed": parsed["terminated"] and not malformed, "ids": gen_ids, "n_calls": len(calls),
                                 "tool_errors": sum(not c["ok"] for c in calls)}
        yield done

    @torch.no_grad()
    def score(self, slot: str, mode: str = "completion", text: str = "", messages: list[dict] | None = None) -> dict:
        """Teacher-forced per-token log-probs of a text/conversation under the slot's model."""
        s = self.slots[slot]
        if s.model is None:
            raise RuntimeError(f"slot {slot} is empty")
        model, tok = s.model, s.tok
        device = next(model.parameters()).device
        if mode == "chat":
            enc = format_chat(tok, messages or [], add_generation_prompt=False)
            ids, mask = enc.ids, enc.loss_mask
        else:
            ids = tok.encode_document(text)
            mask = [1] * len(ids)
        ids = ids[: model.cfg.max_seq_len]
        x = torch.tensor([ids], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)[0].float()
        logp = torch.log_softmax(logits[:-1], dim=-1)
        tgt = x[0, 1:]
        lp = logp.gather(1, tgt[:, None]).squeeze(1)
        rank = (logp > lp[:, None]).sum(1)
        out = [{"id": int(ids[0]), "piece": tok.token_str(ids[0]), "logprob": None, "rank": None, "target": False}]
        for i in range(1, len(ids)):
            out.append({"id": int(ids[i]), "piece": tok.token_str(ids[i]), "logprob": float(lp[i - 1]), "rank": int(rank[i - 1]), "target": bool(mask[i]) if i < len(mask) else True})
        valid = [t["logprob"] for t in out[1:] if t["target"]]
        mean = sum(valid) / len(valid) if valid else 0.0
        return {"tokens": out, "n": len(ids), "mean_logprob": mean, "ppl": float(torch.exp(torch.tensor(-mean))) if valid else None}

    def diagnostics(self, slot: str, tokenized_root: str, source: str, seq_len: int = 1024, n_batches: int = 4, mb: int = 2, ablations: bool = True) -> dict:
        from slm.data.loader import MixtureSpec, ValLoader
        from slm.eval.diagnostics import run_diagnostics

        s = self.slots[slot]
        if s.model is None:
            raise RuntimeError(f"slot {slot} is empty")
        device = next(s.model.parameters()).device
        vl = ValLoader(MixtureSpec(Path(tokenized_root), {source: 1.0}, "val"), seq_len, mb, seq_len * mb * n_batches, device=device)
        batches = list(vl)[:n_batches]
        return run_diagnostics(s.model, batches, do_ablations=ablations, attn_max_len=min(512, seq_len))

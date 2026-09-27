"""Model harness that runs INSIDE the worker subprocess: checkpoint slots, streaming generation
with per-token log-probs and top-k alternatives, teacher-forced scoring, and diagnostics.

A slot can also hold an external comparison model (`external:<name>`, slm.eval.external): an `HfChatModel` with its
own tokenizer and chat template. It streams completions (any model) and chat replies (chat models only: a base model
is never given a template), and runs swarm sampling with majority, selector and tournament through its own template.
Our protocol (the Python tool, declared functions, the forced think span) is refused for it, there is no sandbox
verification (n/a, not approximated), and teacher-forced scoring and diagnostics are n/a."""

from __future__ import annotations

import gc
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
EXTERNAL_PREFIX = "external:"  # LoadRequest.checkpoint of an external comparison model (slm.eval.external)


class Slot:
    def __init__(self, name: str) -> None:
        self.name = name
        self.model: Transformer | None = None
        self.tok: SlmTokenizer | None = None
        self.ext = None  # slm.eval.external.HfChatModel when the slot holds an external comparison model
        self.info: dict = {}

    @property
    def loaded(self) -> bool:
        return self.model is not None or self.ext is not None

    def require(self, what: str | None = None) -> None:
        """Raise unless something is loaded; with `what`, also unless it is one of our checkpoints."""
        if not self.loaded:
            raise RuntimeError(f"slot {self.name} is empty")
        if what and self.ext is not None:
            raise RuntimeError(f"{what} is n/a for an external slot (slot {self.name} holds {self.ext.name}): it needs one of our checkpoints")


class _Pieces:
    """Incremental detokenizer for an HF tokenizer: the text each new id adds. Byte-level BPE splits a character
    across ids, so a piece is emitted only once the text decodes cleanly (the ids before it fold into that piece and
    show as empty chips)."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.prefix = self.read = 0

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def push(self, i: int) -> str:
        self.ids.append(int(i))
        prev, new = self.decode(self.ids[self.prefix : self.read]), self.decode(self.ids[self.prefix :])
        if len(new) > len(prev) and not new.endswith("�"):
            self.prefix, self.read = self.read, len(self.ids)
            return new[len(prev) :]
        return ""


class _PlainEncoder:
    """`tok.encode` for the swarm prompt budgets, measured with the external model's own tokenizer."""

    def __init__(self, ext) -> None:
        self.encode = ext.encode_plain


def _special_ids(ext) -> set[int]:
    return set(getattr(ext.tokenizer, "all_special_ids", None) or []) | set(ext.eos_ids)


def _external_candidates(ext, msgs: list[dict], k: int, temperature: float, top_p: float, max_new_tokens: int, seed: int) -> list:
    """k chat replies in one batch through the model's own template, as swarm Candidates: the reply is the answer
    (no think span), `#### <answer>` parsed as for ours, never from a tool, so never verified."""
    from slm import swarm as S
    from slm.rl.rewards import parse_final_span

    gens = ext.batch_generate_chat([msgs] * k, max_new_tokens, temperature=temperature, top_p=top_p, top_k=0, seed=seed, batch_size=max(1, k))
    out = []
    for j, g in enumerate(gens):
        parsed = parse_final_span(g.text)
        out.append(S.Candidate(idx=j, think=None, answer=g.text, parsed=parsed, key=S.answer_key(parsed), terminated=g.terminated,
                               n_calls=0, n_errors=0, calls=[], from_tool=False, n_tokens=g.n_tokens))
    return out


def _fill_rationales(groups: list, cands: list, max_chars: int = 240) -> None:
    """An external reply has no think span, so `collapse` leaves every rationale empty: use the shortest member's reply
    up to its '####' line, so the selector and pairwise prompts carry the reasoning as they do for ours."""
    from slm import swarm as S

    for g in groups:
        if g.rationale:
            continue
        rep = min((cands[i] for i in g.members), key=lambda c: len(c.answer))
        g.rationale = S._trim(rep.answer.split("####")[0].strip(), max_chars)


def _external_select(ext, sel_msgs: list[dict]):
    """The selector pass through the model's own template, greedy: (think, answer, parsed, n_calls) like slm.swarm.select."""
    from slm.rl.rewards import parse_final_span

    answer = ext.generate_chat(sel_msgs, max_new_tokens=384, temperature=0.0)
    return None, answer, parse_final_span(answer), 0


def _external_compare(ext, text: str, pairs: list, enc, pair_budget_tokens: int, max_new_tokens: int = 96) -> list[int | None]:
    """One greedy batch of pairwise prompts (slm.swarm.pair_messages) through the model's own template."""
    from slm import swarm as S

    msgs = [S.pair_messages(text, a, b, enc, pair_budget_tokens) for a, b in pairs]
    gens = ext.batch_generate_chat(msgs, max_new_tokens, temperature=0.0, batch_size=max(1, len(msgs)))
    return [S.parse_pick(g.text) for g in gens]


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
        if checkpoint.startswith(EXTERNAL_PREFIX):
            return self._load_external(slot, checkpoint[len(EXTERNAL_PREFIX) :], device, dtype)
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

    def _load_external(self, slot: str, name: str, device: str, dtype: str) -> dict:
        """An external comparison model from its local directory (never the network): bf16 (or the requested dtype)
        on cuda, fp32 on cpu. `load_external` is looked up on the module at call time (tests stub it)."""
        from slm.eval import external as E

        s = self.slots[slot]
        self.unload(slot)
        if name not in E.EXTERNAL_MODELS:
            raise RuntimeError(f"unknown external model {name!r}; known: {', '.join(E.EXTERNAL_MODELS)}")
        entry = E.EXTERNAL_MODELS[name]
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype] if device == "cuda" else torch.float32
        t0 = time.time()
        try:
            m = E.load_external(name, device=device, dtype=dt)
        except SystemExit as e:  # the registry's clean exits (weights missing) must not end the worker loop
            raise RuntimeError(str(e)) from None
        s.ext = m
        s.info = {
            "slot": slot, "checkpoint": EXTERNAL_PREFIX + name, "name": name, "external": True, "stage": "external", "run": None,
            "hf_id": entry.hf_id, "params": entry.params, "license": entry.license, "is_chat": entry.is_chat, "context": entry.max_positions,
            "train_tokens": entry.train_tokens, "notes": entry.notes, "device": getattr(m, "device", device),
            "dtype": str(dt).replace("torch.", ""), "tokens": None, "val_loss": None, "tokenizer": f"own ({entry.hf_id})", "tokenizer_matched": True,
            "load_s": time.time() - t0, "vram_gib": torch.cuda.memory_allocated() / 2**30 if device == "cuda" else 0.0,
        }
        return s.info

    def unload(self, slot: str) -> dict:
        s = self.slots[slot]
        had_ext = s.ext is not None
        s.model, s.tok, s.ext, s.info = None, None, None, {}
        if had_ext:
            gc.collect()  # the HF module graph holds reference cycles: free its weights now, not at the next collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"slot": slot, "loaded": False}

    def status(self) -> dict:
        return {"slots": {k: (v.info if v.loaded else {"slot": k, "loaded": False}) for k, v in self.slots.items()},
                "cuda": torch.cuda.is_available(), "vram_gib": torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0}

    # --------------------------------------------------------------- prompts
    def _prompt_ids(self, s: Slot, mode: str, text: str = "", messages: list[dict] | None = None, think_required: bool = False,
                    tools: bool = False, session: PySession | None = None, functions: list | None = None) -> tuple[list[int], list]:
        """(prompt ids, segments); the segments let the UI colour the declared-function blocks."""
        tok = s.tok
        if mode == "chat":
            enc = format_chat(tok, messages or [], add_generation_prompt=True, think_required=think_required, tools=tools, session=session, functions=functions)
            return enc.ids, enc.segments
        return [tok.bos_id, *tok.encode(text)], []

    # ------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, slot: str, mode: str = "completion", text: str = "", messages: list[dict] | None = None, temperature: float = 0.8, top_p: float = 0.95,
                 top_k: int = 0, max_new_tokens: int = 128, seed: int | None = 1234, logprobs_topk: int = 5, think_required: bool = False,
                 tools: bool = False, session_id: str | None = None, max_tool_calls: int = 8, functions: list | None = None, should_stop=None):
        """Yields {'event': 'prompt'|'token'|'tool'|'done', ...}.

        With tools (chat mode) generation pauses at <|/python_call|>: the code runs in the conversation's
        session, a 'tool' event reports it, the result tokens are fed through the model (marked inserted:
        true, no log-prob) and sampling resumes. A call after <|/think|> ends the turn as malformed. The
        'done' event carries the parsed assistant turn (think, answer, ids, well_formed) so a UI can append
        it to the conversation verbatim.

        `functions` (chat mode) are declared functions as dicts ({name, signature, comment}): their masked
        <|python_def|> blocks open the prompt and the 'prompt' event carries the segments that mark them.

        An external slot streams through `_generate_external` (the same events, no tool calls)."""
        s = self.slots[slot]
        s.require()
        if s.ext is not None:
            yield from self._generate_external(s, mode, text, messages, temperature, top_p, top_k, max_new_tokens, seed, logprobs_topk,
                                               think_required, tools, functions, should_stop)
            return
        model, tok = s.model, s.tok
        device = next(model.parameters()).device
        tools = bool(tools) and mode == "chat"
        sess = self.session(session_id) if tools else None
        ids, segments = self._prompt_ids(s, mode, text, messages, think_required, tools, sess, functions if mode == "chat" else None)
        if len(ids) + max_new_tokens > model.cfg.max_seq_len:
            max_new_tokens = max(1, model.cfg.max_seq_len - len(ids))
        stop = {tok.eos_id, tok.end_id} if mode == "chat" else {tok.eos_id}
        t = tool_ids(tok) if tools else None
        think_close = tok.special("<|/think|>")
        if tools:
            stop = stop | {t["call_close"]}
        yield {"event": "prompt", "ids": ids, "pieces": [tok.token_str(i) for i in ids], "n": len(ids), "segments": segments}
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
            think_open = tok.special("<|think|>")
            # with think_required the opening tag was part of the prompt: put it back so the ids are the whole turn
            turn_ids = ([think_open] if think_required and (not gen_ids or gen_ids[0] != think_open) else []) + gen_ids
            done["assistant"] = {"think": parsed["think"], "answer": parsed["answer"], "terminated": parsed["terminated"], "malformed": malformed,
                                 "well_formed": parsed["terminated"] and not malformed, "ids": turn_ids, "n_calls": len(calls),
                                 "tool_errors": sum(not c["ok"] for c in calls)}
        yield done

    def _generate_external(self, s: Slot, mode: str, text: str, messages: list[dict] | None, temperature: float, top_p: float, top_k: int,
                           max_new_tokens: int, seed: int | None, logprobs_topk: int, think_required: bool, tools: bool, functions: list | None,
                           should_stop):
        """`generate` for an external slot, with the same prompt / token / done events. Completion mode encodes the text
        with the model's tokenizer (no BOS: what its tokenizer does by default); chat mode renders the messages with the
        model's own chat template and is refused for a base model. Log-prob, rank, entropy and top-k are the model's own,
        over its own vocabulary. The Python tool, declared functions and the forced think span are our protocol: refused."""
        m = s.ext
        bad = [n for n, v in (("the Python tool (tools)", tools), ("declared functions (functions)", functions),
                              ("forced <|think|> (think_required)", think_required)) if v]
        if bad:
            one = len(bad) == 1
            raise RuntimeError(f"slot {s.name} holds an external model ({m.name}): {', '.join(bad)} {'is' if one else 'are'} part of our "
                               f"chat/tool protocol and {'does' if one else 'do'} not apply to it. Turn {'it' if one else 'them'} off for this slot.")
        if mode == "chat":
            if not m.is_chat:
                raise RuntimeError(f"chat mode is n/a for {m.name}: it is a base model, and a base model is never given a chat template "
                                   "(none is invented for it). Use completion mode.")
            ids = m.encode_chat([{"role": x.get("role", "user"), "content": x.get("content") or ""} for x in (messages or [])])
        elif mode == "completion":
            ids = m.encode_text(text)
        else:
            raise RuntimeError(f"mode {mode!r}: expected completion or chat")
        if not ids:
            raise RuntimeError("empty prompt: an external model's tokenizer adds no BOS, so there is nothing to continue from")
        if len(ids) >= m.max_positions:
            raise RuntimeError(f"the prompt ({len(ids)} tokens) fills {m.name}'s context of {m.max_positions}")
        special = _special_ids(m)
        pp = _Pieces(m.tokenizer)
        yield {"event": "prompt", "ids": ids, "pieces": [pp.push(i) for i in ids], "n": len(ids), "segments": [], "external": True}
        dec = _Pieces(m.tokenizer)
        seed = seed if seed is not None else int(time.time() * 1000) % 2**31
        stream = m.stream_ids(ids, max_new_tokens, temperature, top_p, top_k, seed, stop_ids=m.eos_ids, logprobs_topk=logprobs_topk)
        gen_ids: list[int] = []
        reason = "length"
        t0 = time.time()
        try:
            for ev in stream:
                nid = ev["id"]
                gen_ids.append(nid)
                yield {"event": "token", "id": nid, "piece": dec.push(nid), "logprob": ev["logprob"], "rank": ev["rank"], "entropy": ev["entropy"],
                       "topk": [{"id": i, "piece": dec.decode([i]), "logprob": lp} for i, lp in ev["topk"]], "special": nid in special,
                       "inserted": False}
                if nid in m.eos_ids:
                    reason = "stop"
                    break
                if should_stop is not None and should_stop():
                    reason = "cancelled"
                    break
        finally:
            stream.close()
        dt = time.time() - t0
        n = len(gen_ids)
        done = {"event": "done", "n": n, "n_total": n, "seconds": dt, "tok_s": n / dt if dt > 0 else 0.0, "reason": reason, "calls": [],
                "external": True}
        if mode == "chat":
            body = gen_ids[:-1] if gen_ids and gen_ids[-1] in m.eos_ids else gen_ids
            ok = reason == "stop"
            done["assistant"] = {"think": None, "answer": m.decode(body), "terminated": ok, "malformed": False, "well_formed": ok, "ids": None,
                                 "n_calls": 0, "tool_errors": 0}
        yield done

    @torch.no_grad()
    def swarm(self, slot: str, text: str, k: int = 16, temperature: float = 0.8, top_p: float = 0.95, max_new_tokens: int = 512,
              max_calls: int = 6, seed: int | None = None, budget_tokens: int = 2400, max_groups: int = 12,
              answer_suffix: bool = True, mode: str = "both", pair_budget_tokens: int = 1200, max_entrants: int = 16,
              should_stop=None):
        """Swarm inference (slm.swarm) with a progress event per stage. Yields {'event': 'stage', 'stage':
        'sampling'|'collapsed'|'selecting'|'tournament', ...} and finally {'event': 'done', 'result': SwarmResult.to_dict()}.

        The stages are the calls `swarm_answer` makes, in the same order and with the same generator seeding, so a
        seeded run here reproduces it. `mode` is the library's: "select" (one selector prompt over all groups),
        "tournament" (pairwise single-elimination bracket), "both" (selector, then bracket; `final` follows the bracket).

        The bracket is `slm.swarm.tournament`'s loop replicated round by round so each round can be streamed: one
        'tournament' event with `round` 0 when the bracket is seeded (entrants, no matches), then one per decided round
        (`round` 1..n) with its matches ({a, b, swapped, pick, winner}, exactly the library's round log) and byes.

        The k samples are one batch (splitting it would change both the throughput and the random stream), so a cancel
        takes effect at the next stage boundary or bracket round: the selector / the remaining rounds are skipped, the
        rounds already decided are kept, and `final` falls back to the verified majority in the modes where it follows
        the bracket (the library's rule for a missing champion). With no seed a fresh one is drawn and reported in
        `result.meta.seed` (an unseeded torch.Generator always starts from the same state).

        An external slot (chat models only) runs the same stages through the model's own chat template: k replies in one
        batch (`batch_generate_chat`), `#### <answer>` parsed and collapsed as for ours, the selector and the pairwise
        prompts' content as the user message, greedy. There is no sandbox, so nothing is verified: `verified_majority` is
        null, `n_verified` and every candidate's `verified` are null, `meta.verification` is "n/a", and the fallbacks
        that use the verified majority for ours use the plain majority. The tool budget (`max_calls`) does not apply."""
        from dataclasses import asdict

        from slm import swarm as S
        from slm.data.answers import SUFFIX
        from slm.utils.sdpa import sdpa_context

        if mode not in ("select", "tournament", "both"):
            raise RuntimeError(f"swarm mode {mode!r}: expected select, tournament or both")
        s = self.slots[slot]
        s.require()
        if not text.strip():
            raise RuntimeError("empty task prompt")
        ext, model, tok = s.ext, s.model, s.tok
        if ext is not None and not ext.is_chat:
            raise RuntimeError(f"swarm is n/a for {ext.name}: it is a base model, swarm samples chat replies, and a base model is never "
                               "given a chat template (none is invented for it)")
        seed = int(seed) if seed is not None else int(time.time() * 1000) % 2**31
        suffix = SUFFIX if answer_suffix else None
        meta = {"slot": slot, "checkpoint": s.info.get("name"), "device": s.info.get("device"), "seed": seed, "temperature": temperature,
                "top_p": top_p, "max_new_tokens": max_new_tokens, "max_calls": max_calls if ext is None else 0, "budget_tokens": budget_tokens,
                "max_groups": max_groups, "answer_suffix": suffix, "mode": mode, "pair_budget_tokens": pair_budget_tokens,
                "max_entrants": max_entrants, "cancelled": False, "external": ext is not None, "verification": "sandbox" if ext is None else "n/a"}
        stop = should_stop or (lambda: False)
        if ext is None:  # module attributes looked up at call time (tests script them)
            enc = tok
            sample = lambda m: S.sample_candidates(model, tok, m, k, temperature, top_p, max_new_tokens, max_calls, seed)  # noqa: E731
            select = lambda m: S.select(model, tok, m)  # noqa: E731
            compare = None
        else:
            enc = _PlainEncoder(ext)
            sample = lambda m: _external_candidates(ext, m, k, temperature, top_p, max_new_tokens, seed)  # noqa: E731
            select = lambda m: _external_select(ext, m)  # noqa: E731
            compare = lambda pairs: _external_compare(ext, text, pairs, enc, pair_budget_tokens)  # noqa: E731
        na = {"verification": "n/a", "external": True} if ext is not None else {}

        def n_in_prompt(msgs: list[dict]) -> int:
            return msgs[0]["content"].count("\n- Answer: ") if msgs else 0

        t0 = time.time()
        try:
            with sdpa_context("decode"):
                msgs = [{"role": "user", "content": text + (suffix or "")}]
                yield {"event": "stage", "stage": "sampling", "k": k, "seconds": 0.0, **na}
                cands = sample(msgs)
                groups = S.collapse(cands)
                if ext is not None:
                    _fill_rationales(groups, cands)
                maj = S.majority(groups)
                vm = S.verified_majority(groups) if ext is None else None
                fallback = vm if ext is None else maj
                yield {"event": "stage", "stage": "collapsed", "seconds": round(time.time() - t0, 2), "n_candidates": len(cands),
                       "n_parsed": sum(c.key is not None for c in cands), "n_verified": sum(c.verified for c in cands) if ext is None else None,
                       "groups": [asdict(g) for g in groups], "majority": maj, "verified_majority": vm, **na}
                sel_msgs = S.selector_messages(text, groups, enc, budget_tokens, max_groups) if groups and mode != "tournament" else []
                think, answer, parsed, n_calls = (None, "", None, 0)
                if sel_msgs and stop():
                    meta["cancelled"] = True
                elif sel_msgs:
                    yield {"event": "stage", "stage": "selecting", "seconds": round(time.time() - t0, 2),
                           "prompt_tokens": len(enc.encode(sel_msgs[0]["content"])), "groups_in_prompt": n_in_prompt(sel_msgs), **na}
                    think, answer, parsed, n_calls = select(sel_msgs)
                champion, rounds = None, []
                if groups and mode != "select" and not meta["cancelled"]:
                    champion, rounds = yield from self._tournament_rounds(model, tok, text, groups, pair_budget_tokens, max_entrants, stop, meta, t0,
                                                                          compare=compare)
            if mode == "select":
                final = S.display_answer(parsed) if parsed is not None else fallback
            else:
                final = champion.answer if champion is not None else fallback
            res = S.SwarmResult(prompt=text, k=k, candidates=cands, groups=groups, majority=maj, verified_majority=vm, selector_messages=sel_msgs,
                                selector_think=think, selector_answer=answer, selector_calls=n_calls, final=final,
                                seconds=round(time.time() - t0, 2), meta=meta,
                                tournament=champion.answer if champion is not None else None, rounds=rounds)
            out = res.to_dict()
            for c, cd in zip(cands, out["candidates"]):
                cd["verified"] = c.verified if ext is None else None  # a property, so asdict leaves it out; null = n/a (no sandbox)
            out["meta"]["selector_parsed"] = parsed is not None  # False: no '####' line from the selector (or no selector ran)
            out["meta"]["selector_final"] = S.display_answer(parsed) if parsed is not None else None
            out["meta"]["groups_in_prompt"] = n_in_prompt(sel_msgs)
            out["meta"]["prompt_tokens"] = len(enc.encode(sel_msgs[0]["content"])) if sel_msgs else 0
            yield {"event": "done", "stage": "done", "result": out}
        finally:
            if s.info.get("device") == "cuda":
                torch.cuda.empty_cache()  # generation KV caches leave reserved segments behind (CLAUDE.md)

    @staticmethod
    def _tournament_rounds(model, tok, text: str, groups: list, pair_budget_tokens: int, max_entrants: int, stop, meta: dict, t0: float,
                           compare=None):
        """`slm.swarm.tournament` with one yielded event per round (a generator that returns (champion | None, rounds)).
        The loop is the library's line for line -- first-vs-last seeding with a bye for the odd one out, every odd-indexed
        pair presented swapped, the evidence order (verified, then support) when a comparison gives no parsable pick -- so
        a bracket streamed here and one run by `swarm_answer` agree. `compare_batch` is looked up on the module at call
        time (tests script it); `compare(pairs) -> picks` replaces it (an external slot). A cancel is honoured between
        rounds: champion None, the decided rounds kept."""
        from slm import swarm as S

        entrants = list(groups[:max_entrants])
        n, n_expected = len(entrants), 0
        while n > 1:
            n, n_expected = n // 2 + n % 2, n_expected + 1
        meta.update(n_entrants=len(entrants), n_rounds_expected=n_expected, tournament_complete=False)
        rounds: list[list[dict]] = []
        yield {"event": "stage", "stage": "tournament", "round": 0, "n_rounds_expected": n_expected, "seconds": round(time.time() - t0, 2),
               "entrants": [g.answer for g in entrants], "byes": [], "matches": []}
        while len(entrants) > 1:
            if stop():
                meta["cancelled"] = True
                return None, rounds
            pairs, byes = S.seed_pairs(entrants)
            oriented = [(b, a) if j % 2 else (a, b) for j, (a, b) in enumerate(pairs)]
            picks = compare(oriented) if compare is not None else S.compare_batch(model, tok, text, oriented, pair_budget_tokens, 96)
            winners, log = [], []
            for j, ((x, y), pick) in enumerate(zip(oriented, picks)):
                if pick is None:
                    w = x if (x.verified, x.support) >= (y.verified, y.support) else y
                else:
                    w = x if pick == 0 else y
                winners.append(w)
                log.append({"a": x.answer, "b": y.answer, "swapped": bool(j % 2), "pick": pick, "winner": w.answer})
            rounds.append(log)
            yield {"event": "stage", "stage": "tournament", "round": len(rounds), "n_rounds_expected": n_expected,
                   "seconds": round(time.time() - t0, 2), "entrants": [g.answer for g in entrants], "byes": [g.answer for g in byes],
                   "matches": log}
            entrants = winners + byes
        meta["tournament_complete"] = True
        return entrants[0], rounds

    @torch.no_grad()
    def score(self, slot: str, mode: str = "completion", text: str = "", messages: list[dict] | None = None) -> dict:
        """Teacher-forced per-token log-probs of a text/conversation under the slot's model (n/a for an external slot)."""
        s = self.slots[slot]
        s.require("teacher-forced scoring")
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
        s.require("diagnostics")
        device = next(s.model.parameters()).device
        vl = ValLoader(MixtureSpec(Path(tokenized_root), {source: 1.0}, "val"), seq_len, mb, seq_len * mb * n_batches, device=device)
        batches = list(vl)[:n_batches]
        return run_diagnostics(s.model, batches, do_ablations=ablations, attn_max_len=min(512, seq_len))

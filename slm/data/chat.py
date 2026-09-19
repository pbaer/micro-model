"""Chat formatting with our reserved special tokens, plus the SFT loss mask.

Format (see CLAUDE.md):
    <|bos|><|system|>...<|end|><|user|>...<|end|><|assistant|><|think|>...<|/think|>answer<|end|><|eos|>

Loss mask: 1 on assistant-turn content tokens and their closing <|end|> (the stop signal), 0 on
everything else, including <|eos|> (the model cannot know whether another turn follows).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import re

from slm.data.tokenizer import SlmTokenizer

ROLES = ("system", "user", "assistant")
TOOL_MARK = "<<"  # assistant text may carry <<expr=result>> tool markup (slm.tools.protocol)


@dataclass
class ChatEncoding:
    ids: list[int]
    loss_mask: list[int]
    # (start, end, label) spans over ids, for visualization: role tokens, content, think, specials
    segments: list[tuple[int, int, str]] = field(default_factory=list)


def format_chat(
    tok: SlmTokenizer,
    messages: list[dict],
    add_generation_prompt: bool = False,
    bos: bool = True,
    eos: bool = True,
    think_required: bool = False,
    tools: bool = False,
    session=None,
    functions: list | None = None,
) -> ChatEncoding:
    """tools=True: <<expr=result>> markup inside assistant think text becomes a tool call (loss target) followed
    by the tool's result (masked), run in `session` (a PySession; one is created if None). Without it the markup
    is encoded literally. An assistant message may carry "ids": the exact tokens of a generated turn (think,
    tool spans, answer, <|end|>), which are used verbatim so a live conversation never re-runs its tool calls.

    `functions` are FunctionDecls (or their dicts) declared to the model: one masked
    <|python_def|>signature<|python_comment|>comment<|/python_def|> block per function right after <|bos|>,
    before the first turn. With tools=True their impls are registered in the conversation's session, so
    markup that calls them resolves here exactly as it would at inference. A conversation stored as plain
    data can carry them instead as a leading {"role": "functions", "decls": [...]} message (the parameter
    wins if both are given); that keeps one serialized conversation self-contained."""
    ids: list[int] = []
    mask: list[int] = []
    segs: list[tuple[int, int, str]] = []
    if messages and messages[0].get("role") == "functions":
        functions = functions or messages[0].get("decls")
        messages = messages[1:]

    def push(seq: list[int], m: int, label: str) -> None:
        segs.append((len(ids), len(ids) + len(seq), label))
        ids.extend(seq)
        mask.extend([m] * len(seq))

    # one sandbox session per conversation: state carries across calls and turns

    def push_text(text: str, m: int, label: str) -> None:
        """Tool markup is honoured only in the think span: tool calls are part of thinking, never of the answer."""
        nonlocal session
        if not (tools and m == 1 and label == "think" and TOOL_MARK in text):
            push(tok.encode(text), m, label)
            return
        from slm.tools.protocol import encode_tool_span, split_markup
        from slm.tools.pysandbox import PySession

        session = session or PySession()
        for span in split_markup(text, session):
            if span.kind == "text":
                push(tok.encode(span.text), m, label)
            else:
                call, result, _ = encode_tool_span(tok, span)
                push(call, 1, "python_call")
                push(result, 0, "python_result")

    if bos:
        push([tok.bos_id], 0, "bos")
    if functions:
        from slm.tools.functions import as_decls, functions_env, render_defs
        from slm.tools.pysandbox import PySession

        decls = as_decls(functions)
        for f in decls:  # environment-written, like a tool result: every token of the block is masked
            push(render_defs(tok, [f]), 0, "python_def")
        if tools:
            session = session or PySession()
            session.register(functions_env(decls))
    for msg in messages:
        role = msg["role"]
        assert role in ROLES, f"unknown role {role}"
        push([tok.special(f"<|{role}|>")], 0, f"role:{role}")
        target = 1 if role == "assistant" else 0
        if role == "assistant" and msg.get("ids"):
            gen_ids = [int(i) for i in msg["ids"]]
            t_open, t_close = tok.special("<|think|>"), tok.special("<|/think|>")
            if t_close in gen_ids and t_open not in gen_ids[: gen_ids.index(t_close)]:
                gen_ids = [t_open, *gen_ids]  # generated under a forced <|think|> prompt: the opening tag was not sampled
            push(gen_ids, 1, "assistant_ids")
            if not gen_ids or gen_ids[-1] not in (tok.end_id, tok.eos_id):
                push([tok.end_id], 1, "end")
            continue
        if role == "assistant":
            think = msg.get("think")
            if think is not None or think_required:
                push([tok.special("<|think|>")], target, "think_open")
                push_text(think or "", target, "think")
                push([tok.special("<|/think|>")], target, "think_close")
        push_text(msg.get("content", ""), target, f"content:{role}")
        push([tok.end_id], target, "end")
    if add_generation_prompt:
        push([tok.special("<|assistant|>")], 0, "role:assistant")
        if think_required:
            push([tok.special("<|think|>")], 0, "think_open")
    elif eos:
        push([tok.eos_id], 0, "eos")
    return ChatEncoding(ids, mask, segs)


def parse_assistant(tok: SlmTokenizer, ids: list[int], think_expected: bool = True) -> dict:
    """Split a generated assistant turn into think / answer text.

    `ids` are the tokens generated after the assistant role token. When the generation prompt already
    ended with <|think|> (think_required prompts), the completion starts inside the think span, so
    the opening tag may legitimately be absent; the closing <|/think|> is what separates the spans.
    """
    think_open, think_close = tok.special("<|think|>"), tok.special("<|/think|>")
    stop = {tok.end_id, tok.eos_id}
    body: list[int] = []
    terminated = False
    for i in ids:
        if i in stop:
            terminated = True
            break
        body.append(i)
    if body and body[0] == think_open:
        body = body[1:]
        had_open = True
    else:
        had_open = False
    tool_ids = {tok.special(s) for s in ("<|python_call|>", "<|/python_call|>", "<|python_result|>", "<|/python_result|>")}
    if think_close in body:
        j = body.index(think_close)
        think, answer = body[:j], body[j + 1 :]
        malformed = think_close in answer or think_open in answer or any(i in tool_ids for i in answer)  # nested tags / tool use outside think
    elif think_expected:
        think, answer = None, body  # no closing tag: treat everything as the answer, flag it
        malformed = True
    else:
        think, answer = None, body
        malformed = had_open or any(i in tool_ids for i in body)  # opened a think span without closing it / tool call with no think span
    from slm.tools.protocol import render_tools  # tool spans render as <<expr=result>> markup

    return {
        "think": render_tools(tok, think) if think is not None else None,
        "answer": render_tools(tok, answer),
        "terminated": terminated,
        "malformed": bool(malformed) or not terminated,
    }


def rows_to_messages(src, row: dict) -> list[dict] | None:
    """Normalize a raw row to [{role, content, think?}] or None to drop it."""
    if src.kind == "chat":
        msgs = row.get("messages")
        if not msgs:
            return None
        out = []
        for m in msgs:
            role = m.get("role")
            if role not in ("system", "user", "assistant") or not m.get("content"):
                return None
            out.append({"role": role, "content": m["content"]})
        return out if any(m["role"] == "assistant" for m in out) else None
    if src.kind == "math_cot":  # metamathqa-style chat: solution text ending in "The answer is: X"
        msgs = row.get("messages") or []
        if len(msgs) < 2 or msgs[-1].get("role") != "assistant":
            return None
        sol = msgs[-1]["content"]
        m = re.search(r"The answer is:\s*(.+?)\s*$", sol.strip(), flags=re.S)
        if not m:
            return None
        final = m.group(1).strip().rstrip(".")
        if not re.fullmatch(r"-?[\d,]+(?:\.\d+)?(?:/\d+)?", final):
            return None  # numeric answers only (verifiable)
        think = sol[: m.start()].strip()
        think = re.sub(r"\n?####\s*[^\n]*$", "", think).strip()  # drop a trailing gsm8k-style marker inside the trace
        if not think:
            return None
        return [{"role": "user", "content": msgs[0]["content"].strip()}, {"role": "assistant", "think": think, "content": "#### " + final}]
    if src.kind == "math_qa":  # gsm8k: question / answer ("reasoning\n#### 42")
        q, a = row.get("question"), row.get("answer")
        if not q or not a or "####" not in a:
            return None
        reasoning, _, final = a.rpartition("####")
        return [{"role": "user", "content": q.strip()}, {"role": "assistant", "think": reasoning.strip(), "content": "#### " + final.strip()}]
    return None

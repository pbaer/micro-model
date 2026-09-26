"""The Python-tool protocol on the reserved special tokens.

Generated form (what the model emits and sees):
    ... 120 - 36 = <|python_call|>120-36<|/python_call|><|python_result|>84<|/python_result|>84 pages left ...
The call body is plain Python for the sandboxed interpreter (slm.tools.pysandbox). The model generates
through <|/python_call|>; the harness runs the code and appends the result span; generation resumes.
Result tokens are environment-written: loss mask 0 in SFT, excluded from the policy gradient in RL.

REPL semantics: one `PySession` per conversation keeps variables and functions across calls, and across
turns, so a later call can reuse `total` from an earlier one. Future tools are Python functions exposed in
the session namespace (provided in-context or trained in), not new token types.

Text form (datasets, synthetic traces, display):
    <<expr=result>>        one expression, GSM8K's own annotation syntax -> the bare expression (REPL echo)
    <<<code>>>             a short program (may span lines) -> code; its output is the result
`split_markup` turns text into (text | tool) spans, running the code in a session so the recorded result
is exactly what the harness would insert; `render_tools` turns generated ids back into the same markup.
Dataset annotations echo the number right after the markup ("<<12*52=624>>624 pages"); the echo is dropped
because the result span already carries it, so the model is not trained to repeat tool output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from slm.data.tokenizer import SlmTokenizer
from slm.tools.calculator import ToolError
from slm.tools.pysandbox import PySession

TOOL_MARK_RE = re.compile(r"<<<(.+?)>>>|<<([^<>]+?)=([^<>=]*?)>>", re.S)
MAX_RESULT_CHARS = 300  # results and error messages are inserted into the model's context; keep them short
NO_OUTPUT = "(no output; use print(...) to show a value, or end with a bare expression)"


@dataclass
class ToolSpan:
    kind: str  # "text" | "tool"
    text: str = ""  # for text spans
    code: str = ""  # for tool spans: the Python code
    result: str = ""  # sandbox output


def run_tool(code: str, session: PySession | None = None) -> tuple[str, bool]:
    """Run code in the session (a fresh one if None); returns (result text, ok). Errors come back as
    'error: ...' so the model can read them and try again."""
    session = session or PySession()
    try:
        out = session.run(code)
        return (out if out != "" else NO_OUTPUT)[:MAX_RESULT_CHARS], True
    except ToolError as e:
        return f"error: {str(e)[:MAX_RESULT_CHARS]}", False
    except Exception as e:  # noqa: BLE001 - the harness boundary: a program the model wrote must never crash the caller
        # M9 stage C run 5 died at step 60 on a float slice index the sandbox had not anticipated (TypeError
        # straight out of Python). The specific case is now a ToolError, but the class of bug is "the sandbox
        # missed a case", so anything else that escapes is reported to the model the same way.
        return f"error: {type(e).__name__}: {str(e)[:MAX_RESULT_CHARS]}", False


_HOIST_RE = re.compile(r"^(?P<lead>.*?\S)\s*(?:because|since|as|so|,|:)?\s+(?P<expr>[-\d.,$%]+(?:\s*[-+*/x×÷]\s*[-\d.,$%]+)+)\s*=\s*(?P<call><<[^>]*>>)(?P<rest>.*)$")


def hoist_calls(text: str) -> str:
    """GSM8K-style traces often state a result and then justify it: "He eats 32 pieces because 2 x 16 =
    <<2*16=32>>32". Converted to tool calls as-is, that teaches the model to produce the number BEFORE calling
    the tool, i.e. that the call is decorative (measured: 8% of GSM8K annotations). Move the computation in front
    of such a sentence so the number only appears after the result: "2 x 16 = <<2*16=32>>. He eats 32 pieces."
    Lines whose prose does not already contain the result are left alone."""
    out = []
    for line in text.split("\n"):
        m = _HOIST_RE.match(line)
        if m:
            result = m.group("call").split("=")[-1].rstrip(">").strip()
            lead = m.group("lead")
            if result and re.search(rf"(?<![\d.]){re.escape(result)}(?![\d.])", lead):
                rest = m.group("rest").strip()
                rest = rest[len(result):].lstrip() if rest.startswith(result) else rest  # drop the echoed result
                lead = lead.rstrip(" ,:").rstrip()
                line = f"{m.group('expr')} = {m.group('call')}. {lead}{'.' if not lead.endswith(('.', '!', '?')) else ''}{(' ' + rest) if rest else ''}"
        out.append(line)
    return "\n".join(out)


def split_markup(text: str, session: PySession | None = None) -> list[ToolSpan]:
    """Split text on tool markup. The code is run on each span (in `session`, so state carries across
    spans of one conversation): when it succeeds, the sandbox output is the result; when it fails, the span
    stays plain text (the dataset's annotated result is kept for <<expr=result>>)."""
    session = session or PySession()
    out: list[ToolSpan] = []
    pos = 0
    for m in TOOL_MARK_RE.finditer(text):
        if m.start() > pos:
            out.append(ToolSpan("text", text[pos : m.start()]))
        pos = m.end()
        if m.group(1) is not None:  # <<<code>>>
            code = m.group(1).strip("\n")
            res, ok = run_tool(code, session)
            out.append(ToolSpan("tool", code=code, result=res) if ok else ToolSpan("text", code))
            echo = res if ok else ""
        else:  # <<expr=result>>
            expr, annotated = m.group(2).strip(), m.group(3).strip()
            code = expr  # a bare expression echoes its value, like a REPL; no print() needed
            res, ok = run_tool(code, session)
            out.append(ToolSpan("tool", code=code, result=res) if ok else ToolSpan("text", f"{expr} = {annotated}" if annotated else expr))
            echo = annotated if ok else ""
        if echo:  # drop the dataset's echoed number ("<<a*b=c>>c pages" -> "<<a*b=c>> pages"); "$c" and "c." forms too
            rest = text[pos:]
            for form in (echo, res if ok else ""):
                if form and rest.startswith(form):
                    pos += len(form)
                    break
    if pos < len(text):
        out.append(ToolSpan("text", text[pos:]))
    return out


def tool_ids(tok: SlmTokenizer) -> dict[str, int]:
    return {"call_open": tok.special("<|python_call|>"), "call_close": tok.special("<|/python_call|>"),
            "result_open": tok.special("<|python_result|>"), "result_close": tok.special("<|/python_result|>")}


def encode_tool_span(tok: SlmTokenizer, span: ToolSpan) -> tuple[list[int], list[int], list[int]]:
    """(call ids, result ids, loss mask over call+result): the call is a model target, the result is not."""
    t = tool_ids(tok)
    call = [t["call_open"], *tok.encode(span.code), t["call_close"]]
    result = [t["result_open"], *tok.encode(span.result), t["result_close"]]
    return call, result, [1] * len(call) + [0] * len(result)


def markup_for(code: str, result: str | None) -> str:
    """Text markup for a generated call: one line -> <<code=result>>, a program -> <<<code>>>=result."""
    c = code.strip()
    if "\n" not in c and "<" not in c and ">" not in c and "=" not in c:
        return f"<<{c}={result}>>" if result is not None else f"<<{c}=>>"
    return f"<<<{code}>>>" + (f"={result}" if result is not None else "=")


def render_tools(tok: SlmTokenizer, ids: list[int]) -> str:
    """Decode ids to text, turning call/result spans back into markup and dropping other specials.
    Unclosed spans render as far as they go."""
    t = tool_ids(tok)
    out: list[str] = []
    buf: list[int] = []
    state = "text"
    code = ""
    for i in ids:
        if state == "text" and i == t["call_open"]:
            out.append(tok.decode(buf, skip_special=True)); buf = []; state = "call"
        elif state == "call" and i == t["call_close"]:
            code = tok.decode(buf, skip_special=True); buf = []; state = "after_call"
        elif state == "after_call" and i == t["result_open"]:
            state = "result"
        elif state == "result" and i == t["result_close"]:
            out.append(markup_for(code, tok.decode(buf, skip_special=True))); buf = []; state = "text"
        elif state == "after_call":  # call without a result (e.g. cut off): render the call, keep going
            out.append(markup_for(code, None)); state = "text"; buf = [i]
        else:
            buf.append(i)
    if state == "call":
        out.append("<<<" + tok.decode(buf, skip_special=True))
    elif state == "after_call":
        out.append(markup_for(code, None))
    elif state == "result":
        out.append(markup_for(code, tok.decode(buf, skip_special=True)))
    else:
        out.append(tok.decode(buf, skip_special=True))
    return "".join(out)

"""The tool protocol on the reserved special tokens.

Generated form (what the model emits and sees):
    ... 198+209 = <|tool_call|>python: print(198+209)<|/tool_call|><|tool_result|>407<|/tool_result|>407 newspapers ...
The model generates up to and including <|/tool_call|>; the harness runs the tool and appends
<|tool_result|>...<|/tool_result|>, then generation resumes. Result tokens are environment-written:
loss mask 0 in SFT, excluded from the policy gradient in RL.

Tools: `python` (sandboxed subset interpreter, slm.tools.pysandbox; the default) and `calc` (arithmetic
expressions only). Call syntax is "<name>: <args>"; a bare expression means the calculator.

Text form (datasets, synthetic traces, display):
    <<expr=result>>        one expression, GSM8K's own annotation syntax -> python: print(expr)
    <<<code>>>             a short program (may span lines) -> python: code; its output is the result
`split_markup` turns text into (text | tool) spans, running the tool so the recorded result is exactly
what the harness would insert; `render_tools` turns generated ids back into the same markup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from slm.data.tokenizer import SlmTokenizer
from slm.tools.calculator import ToolError, calc
from slm.tools.pysandbox import run_python

TOOL_MARK_RE = re.compile(r"<<<(.+?)>>>|<<([^<>]+?)=([^<>=]*?)>>", re.S)
TOOLS = {"python": run_python, "calc": calc}
DEFAULT_TOOL = "python"
_PRINT_RE = re.compile(r"^print\((.*)\)$", re.S)


@dataclass
class ToolSpan:
    kind: str  # "text" | "tool"
    text: str = ""  # for text spans
    args: str = ""  # for tool spans: the code / expression
    result: str = ""  # tool output
    call: str = ""  # "<tool>: <args>"


def parse_tool_call(text: str) -> tuple[str, str]:
    """'python: print(12*3)' -> ('python', 'print(12*3)'); a bare expression -> ('calc', expr)."""
    name, sep, args = text.partition(":")
    if sep and name.strip() in TOOLS:
        return name.strip(), args.strip()
    return "calc", text.strip()


def run_tool(text: str) -> tuple[str, bool]:
    """Run a call string; returns (result text, ok). Errors come back as 'error: ...' so the model sees them."""
    name, args = parse_tool_call(text)
    fn = TOOLS.get(name)
    if fn is None:
        return f"error: unknown tool {name}", False
    try:
        out = fn(args)
        return (out if out != "" else "(no output)"), True
    except ToolError as e:
        return f"error: {e}", False


def split_markup(text: str) -> list[ToolSpan]:
    """Split text on tool markup. The tool is run on each span: when it succeeds, its own output is the
    result (so training matches inference); when it fails, the span stays plain text."""
    out: list[ToolSpan] = []
    pos = 0
    for m in TOOL_MARK_RE.finditer(text):
        if m.start() > pos:
            out.append(ToolSpan("text", text[pos : m.start()]))
        if m.group(1) is not None:  # <<<code>>>
            code = m.group(1).strip("\n")
            call = f"{DEFAULT_TOOL}: {code}"
            res, ok = run_tool(call)
            out.append(ToolSpan("tool", args=code, result=res, call=call) if ok else ToolSpan("text", code))
        else:  # <<expr=result>>
            expr, annotated = m.group(2).strip(), m.group(3).strip()
            call = f"{DEFAULT_TOOL}: print({expr})"
            res, ok = run_tool(call)
            out.append(ToolSpan("tool", args=f"print({expr})", result=res, call=call) if ok else ToolSpan("text", f"{expr} = {annotated}" if annotated else expr))
        pos = m.end()
    if pos < len(text):
        out.append(ToolSpan("text", text[pos:]))
    return out


def tool_ids(tok: SlmTokenizer) -> dict[str, int]:
    return {"call_open": tok.special("<|tool_call|>"), "call_close": tok.special("<|/tool_call|>"),
            "result_open": tok.special("<|tool_result|>"), "result_close": tok.special("<|/tool_result|>")}


def encode_tool_span(tok: SlmTokenizer, span: ToolSpan) -> tuple[list[int], list[int], list[int]]:
    """(call ids, result ids, loss mask over call+result): the call is a model target, the result is not."""
    t = tool_ids(tok)
    call = [t["call_open"], *tok.encode(span.call), t["call_close"]]
    result = [t["result_open"], *tok.encode(span.result), t["result_close"]]
    return call, result, [1] * len(call) + [0] * len(result)


def markup_for(call_text: str, result: str | None) -> str:
    """Text markup for a generated call: single print(expr) -> <<expr=result>>, anything else -> <<<code>>>=result."""
    name, args = parse_tool_call(call_text)
    m = _PRINT_RE.match(args) if name == "python" else None
    if name == "calc" or (m and "\n" not in args):
        expr = m.group(1) if m else args
        return f"<<{expr}={result}>>" if result is not None else f"<<{expr}=>>"
    return f"<<<{args}>>>" + (f"={result}" if result is not None else "=")


def render_tools(tok: SlmTokenizer, ids: list[int]) -> str:
    """Decode ids to text, turning tool call/result spans back into markup and dropping other specials.
    Unclosed spans render as far as they go."""
    t = tool_ids(tok)
    out: list[str] = []
    buf: list[int] = []
    state = "text"
    call_text = ""
    for i in ids:
        if state == "text" and i == t["call_open"]:
            out.append(tok.decode(buf, skip_special=True)); buf = []; state = "call"
        elif state == "call" and i == t["call_close"]:
            call_text = tok.decode(buf, skip_special=True); buf = []; state = "after_call"
        elif state == "after_call" and i == t["result_open"]:
            state = "result"
        elif state == "result" and i == t["result_close"]:
            out.append(markup_for(call_text, tok.decode(buf, skip_special=True))); buf = []; state = "text"
        elif state == "after_call":  # call without a result (e.g. cut off): render the call, keep going
            out.append(markup_for(call_text, None)); state = "text"; buf = [i]
        else:
            buf.append(i)
    if state == "call":
        out.append("<<" + tok.decode(buf, skip_special=True))
    elif state == "after_call":
        out.append(markup_for(call_text, None))
    elif state == "result":
        out.append(markup_for(call_text, tok.decode(buf, skip_special=True)))
    else:
        out.append(tok.decode(buf, skip_special=True))
    return "".join(out)

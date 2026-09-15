"""Tool use: a sandboxed Python subset interpreter with REPL-style sessions, the call/result protocol on the
reserved tokens (<|python_call|> code <|/python_call|><|python_result|> out <|/python_result|>), and the
generation loop that pauses at calls, runs the code, and resumes with the result appended (masked from the
loss in SFT and from the policy gradient in RL, since the environment wrote it)."""

from slm.tools.calculator import ToolError, calc
from slm.tools.protocol import TOOL_MARK_RE, ToolSpan, render_tools, run_tool, split_markup
from slm.tools.pysandbox import PySession, run_python

__all__ = ["ToolError", "calc", "TOOL_MARK_RE", "ToolSpan", "render_tools", "run_tool", "split_markup", "PySession", "run_python"]

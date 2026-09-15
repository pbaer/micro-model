"""Tool use: a calculator tool, the call/result protocol on the reserved tool tokens, and the generation
loop that pauses at tool calls, runs the tool, and resumes with the result appended (masked from the
loss in SFT and from the policy gradient in RL, since the environment wrote it)."""

from slm.tools.calculator import ToolError, calc
from slm.tools.protocol import TOOL_MARK_RE, ToolSpan, parse_tool_call, render_tools, run_tool, split_markup
from slm.tools.pysandbox import run_python

__all__ = ["ToolError", "calc", "TOOL_MARK_RE", "ToolSpan", "parse_tool_call", "render_tools", "run_tool", "split_markup", "run_python"]

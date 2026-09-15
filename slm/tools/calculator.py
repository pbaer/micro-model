"""A safe arithmetic evaluator: the only tool for now.

Accepts + - * / % ** (also ^ × ÷), parentheses, unary minus, integers and decimals; $ and thousands
commas are stripped. Evaluation is exact (fractions), the result is printed as an integer when it is
one and otherwise as a decimal with up to 6 fractional digits (trailing zeros removed), so "2.50" from a
dataset annotation comes back as "2.5" and the model learns the tool's own format.
"""

from __future__ import annotations

import ast
import operator
from fractions import Fraction

MAX_EXPR_CHARS = 200
MAX_ABS = Fraction(10**30)
MAX_POW = 64


class ToolError(ValueError):
    pass


_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: None}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _num(node: ast.AST, src: str) -> Fraction:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        # exact from the literal text, not from the float
        seg = ast.get_source_segment(src, node) or str(node.value)
        try:
            return Fraction(seg)
        except ValueError as e:  # e.g. "1e5"
            raise ToolError(f"bad number {seg!r}") from e
    raise ToolError("only numbers and + - * / % ** ( ) are allowed")


def _eval(node: ast.AST, src: str) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval(node.body, src)
    if isinstance(node, ast.Constant):
        return _num(node, src)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _UN[type(node.op)](_eval(node.operand, src))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        a, b = _eval(node.left, src), _eval(node.right, src)
        if isinstance(node.op, ast.Pow):
            if b.denominator != 1 or abs(b) > MAX_POW:
                raise ToolError("exponent must be an integer with |e| <= 64")
            if a == 0 and b < 0:
                raise ToolError("division by zero")
            return a ** int(b)
        if isinstance(node.op, (ast.Div, ast.Mod)) and b == 0:
            raise ToolError("division by zero")
        return _BIN[type(node.op)](a, b)
    raise ToolError("only numbers and + - * / % ** ( ) are allowed")


def format_number(v: Fraction, max_decimals: int = 6) -> str:
    if v.denominator == 1:
        return str(v.numerator)
    s = f"{float(v):.{max_decimals}f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def calc(expr: str) -> str:
    """Evaluate an arithmetic expression; raises ToolError with a short message on anything else."""
    e = expr.strip().replace("$", "").replace(",", "").replace("×", "*").replace("÷", "/").replace("^", "**").replace("−", "-")
    if not e:
        raise ToolError("empty expression")
    if len(e) > MAX_EXPR_CHARS:
        raise ToolError("expression too long")
    try:
        tree = ast.parse(e, mode="eval")
    except SyntaxError as ex:
        raise ToolError("cannot parse expression") from ex
    v = _eval(tree, e)
    if abs(v) > MAX_ABS:
        raise ToolError("result too large")
    return format_number(v)

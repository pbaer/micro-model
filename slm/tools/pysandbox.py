"""A sandboxed Python subset for model-generated calculations.

Safety model (err on the side of safety; capability is deliberately small):
  * The code is parsed with `ast.parse` (parsing never executes anything) and then run by the tree-walking
    interpreter below. CPython's exec/eval/compile are never called on model output, so there is no
    builtins escape, no __subclasses__ walk, no import system, no frames to reach.
  * No imports, no attribute access on user values (only `math.<whitelisted function>`), no dunder names,
    no classes, no lambdas/closures over mutable state, no with/try/global/yield/async, no comprehension
    other than list comprehensions.
  * Everything is pure computation in-process with no I/O primitive available at all: the only "output" is
    the text captured from print() and the value of a final bare expression.
  * Resource limits enforced by the interpreter itself: an operation budget (stops infinite loops),
    caps on integer size, sequence and string length, output length, call depth and loop iterations,
    and a wall-clock backstop. A limit violation raises ToolError like any other failure.

Supported: int/float/bool/str/None literals, lists, tuples, dicts, arithmetic, comparisons, boolean ops,
if/elif/else, for (over range/list/tuple/str/dict), while, break/continue, def with positional args and
return, list comprehensions, indexing and slicing, f-strings, augmented assignment, tuple unpacking, and a
whitelist of builtins (abs, min, max, round, sum, len, range, int, float, str, bool, divmod, sorted,
reversed, enumerate, zip, list, tuple, dict, print, math.*).
"""

from __future__ import annotations

import ast
import math
import operator
import time
from dataclasses import dataclass, field

from slm.tools.calculator import ToolError, format_number

MAX_STEPS = 200_000
MAX_INT_BITS = 4096
MAX_SEQ_LEN = 10_000
MAX_STR_LEN = 10_000
MAX_OUTPUT = 400
MAX_DEPTH = 40
MAX_LOOP_ITERS = 100_000
MAX_CODE_CHARS = 2000
MAX_POW = 512
TIME_LIMIT_S = 2.0
MAX_NAMES = 200  # variables/functions a session may hold

def _factorial(n):
    if not isinstance(n, int) or isinstance(n, bool) or n < 0 or n > 2000:
        raise ToolError("factorial argument too large or invalid (integer in [0, 2000])")
    return math.factorial(n)


_SAFE_MATH = {n: getattr(math, n) for n in ("sqrt", "floor", "ceil", "log", "log2", "log10", "exp", "sin", "cos", "tan", "pi", "e", "gcd", "isqrt", "fabs", "trunc")}
_SAFE_MATH["factorial"] = _factorial
_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: None, ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor,
}
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_}
# Methods callable on values the sandbox itself created (bound methods of builtin types; results are checked).
# str.format is deliberately absent: format strings can reach attributes ("{0.__class__}").
_METHODS = {
    list: ("append", "pop", "extend", "insert", "remove", "index", "count", "sort", "reverse", "copy"),
    tuple: ("index", "count"),
    str: ("upper", "lower", "strip", "lstrip", "rstrip", "split", "join", "replace", "startswith", "endswith", "isdigit", "isalpha", "count", "find", "zfill", "title", "capitalize"),
    dict: ("keys", "values", "items", "get", "pop", "update", "copy"),
}
_VIEW_TYPES = (type({}.keys()), type({}.values()), type({}.items()))
BUILTIN_NAMES = "abs, min, max, round, sum, len, range, int, float, str, bool, divmod, sorted, reversed, enumerate, zip, list, tuple, dict, print, pow"
MATH_NAMES = "math.sqrt, math.floor, math.ceil, math.log, math.exp, math.pi, math.gcd, math.factorial, ..."
_MODULE_NAMES = {"np", "numpy", "pandas", "pd", "re", "os", "sys", "random", "datetime", "sympy", "fractions", "decimal", "itertools", "collections", "json"}
_HINTS = {
    "Import": "imports are not available; math is built in (use math.sqrt(x) etc.), everything else must be written with basic Python",
    "ImportFrom": "imports are not available; math is built in (use math.sqrt(x) etc.), everything else must be written with basic Python",
    "Lambda": "lambda is not supported; define a function with 'def name(x):' and 'return' instead",
    "ClassDef": "classes are not supported; use plain variables, lists and functions",
    "With": "'with' is not supported; there are no files or resources to open",
    "Try": "try/except is not supported; check values with if/else instead",
    "Global": "'global' is not supported; pass values as function arguments",
    "Yield": "generators are not supported; build a list instead",
    "Await": "async code is not supported",
    "AsyncFunctionDef": "async code is not supported",
    "Delete": "'del' is not supported; assign a new value instead",
    "Assert": "'assert' is not supported; use if/else and print",
    "Raise": "'raise' is not supported; print a message instead",
    "DictComp": "dict comprehensions are not supported; build the dict in a for loop",
    "SetComp": "sets are not supported; use a list",
    "Set": "sets are not supported; use a list",
    "Starred": "* unpacking is not supported; pass arguments one by one",
    "NamedExpr": "':=' is not supported; use a normal assignment",
    "Match": "match statements are not supported; use if/elif",
}


class _Return(Exception):
    def __init__(self, value):
        self.value = value


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


@dataclass
class _Func:
    name: str
    args: list[str]
    body: list[ast.stmt]


@dataclass
class Sandbox:
    max_steps: int = MAX_STEPS
    steps: int = 0
    depth: int = 0
    out: list[str] = field(default_factory=list)
    out_len: int = 0
    t0: float = field(default_factory=time.time)
    functions: dict = field(default_factory=dict)  # declared functions (name -> callable); see slm/tools/functions.py

    def __post_init__(self) -> None:
        for name in self.functions:
            _check_function_name(name)

    # ----------------------------------------------------------------- guards
    def tick(self, n: int = 1) -> None:
        self.steps += n
        if self.steps > self.max_steps:
            raise ToolError("operation budget exceeded")
        if (self.steps & 1023) == 0 and time.time() - self.t0 > TIME_LIMIT_S:
            raise ToolError("time limit exceeded")

    def check(self, v):
        if isinstance(v, bool) or v is None:
            return v
        if isinstance(v, int):
            if v.bit_length() > MAX_INT_BITS:
                raise ToolError("integer too large")
        elif isinstance(v, float):
            if math.isinf(v) or math.isnan(v):
                raise ToolError("non-finite float")
        elif isinstance(v, str):
            if len(v) > MAX_STR_LEN:
                raise ToolError("string too long")
        elif isinstance(v, (list, tuple, dict)):
            if len(v) > MAX_SEQ_LEN:
                raise ToolError("sequence too long")
        elif isinstance(v, (_Func, range)) or callable(v):
            pass
        else:
            raise ToolError(f"unsupported value type {type(v).__name__}")
        return v

    def emit(self, text: str) -> None:
        self.out_len += len(text)
        if self.out_len > MAX_OUTPUT * 4:
            raise ToolError("too much output")
        self.out.append(text)

    # ----------------------------------------------------------------- builtins
    def builtins(self) -> dict:
        def _range(*a):
            r = range(*a)
            if len(r) > MAX_LOOP_ITERS:
                raise ToolError("range too long")
            return r

        def _pow(a, b, *rest):
            return _power(a, b)

        def _round(x, nd=None):
            return round(x, nd) if nd is not None else round(x)

        return {
            "abs": abs, "min": min, "max": max, "round": _round, "sum": sum, "len": len, "range": _range, "int": int, "float": float, "str": str,
            "bool": bool, "divmod": divmod, "sorted": sorted, "reversed": lambda x: list(reversed(x)), "enumerate": lambda x, s=0: list(enumerate(x, s)),
            "zip": lambda *a: list(zip(*a)), "list": list, "tuple": tuple, "dict": dict, "print": _PRINT, "pow": _pow, "math": _SAFE_MATH,
            "True": True, "False": False, "None": None,
            **self.functions,  # declared functions are part of the namespace, so a def in the sandbox can call them too
        }

    # ----------------------------------------------------------------- statements
    def run(self, code: str, env: dict | None = None) -> str:
        """Run code in `env` (a persistent session namespace, or a fresh one). Budgets are per call."""
        if len(code) > MAX_CODE_CHARS:
            raise ToolError("code too long")
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as e:
            lines = code.splitlines()
            src = ("  |  " + lines[e.lineno - 1].strip()[:80]) if e.lineno and 0 < e.lineno <= len(lines) else ""
            raise ToolError(f"line {e.lineno}: SyntaxError: {e.msg}{src}") from e
        env = self.builtins() if env is None else env
        last = None
        for i, st in enumerate(tree.body):
            try:
                if i == len(tree.body) - 1 and isinstance(st, ast.Expr):
                    last = self.expr(st.value, env)
                else:
                    self.stmt(st, env)
            except ToolError as e:
                raise _with_line(e, st, code) from None
        text = "".join(self.out)
        if last is not None:
            text += _fmt(last)
        text = text.strip()
        if len(text) > MAX_OUTPUT:
            text = text[: MAX_OUTPUT - 3] + "..."
        return text

    def block(self, body: list[ast.stmt], env: dict) -> None:
        for st in body:
            try:
                self.stmt(st, env)
            except ToolError as e:
                raise _with_line(e, st) from None

    def stmt(self, st: ast.stmt, env: dict) -> None:
        self.tick()
        if isinstance(st, ast.Expr):
            self.expr(st.value, env)
        elif isinstance(st, ast.Assign):
            v = self.expr(st.value, env)
            for tgt in st.targets:
                self.assign(tgt, v, env)
        elif isinstance(st, ast.AugAssign):
            cur = self.expr(st.target, env)
            v = self.binop(st.op, cur, self.expr(st.value, env))
            self.assign(st.target, v, env)
        elif isinstance(st, ast.AnnAssign) and st.value is not None:
            self.assign(st.target, self.expr(st.value, env), env)
        elif isinstance(st, ast.If):
            self.block(st.body if self.expr(st.test, env) else st.orelse, env)
        elif isinstance(st, ast.While):
            n = 0
            while self.expr(st.test, env):
                n += 1
                if n > MAX_LOOP_ITERS:
                    raise ToolError("loop iteration limit exceeded")
                try:
                    self.block(st.body, env)
                except _Break:
                    break
                except _Continue:
                    continue
            else:
                self.block(st.orelse, env)
        elif isinstance(st, ast.For):
            it = self.expr(st.iter, env)
            if not isinstance(it, (range, list, tuple, str, dict)):
                raise ToolError(f"cannot loop over a {type(it).__name__}; loop over range(n), a list, a tuple, a string or a dict")
            n = 0
            for item in list(it):
                n += 1
                if n > MAX_LOOP_ITERS:
                    raise ToolError("loop iteration limit exceeded")
                self.assign(st.target, item, env)
                try:
                    self.block(st.body, env)
                except _Break:
                    break
                except _Continue:
                    continue
            else:
                self.block(st.orelse, env)
        elif isinstance(st, ast.Break):
            raise _Break()
        elif isinstance(st, ast.Continue):
            raise _Continue()
        elif isinstance(st, ast.Return):
            raise _Return(self.expr(st.value, env) if st.value is not None else None)
        elif isinstance(st, ast.FunctionDef):
            a = st.args
            if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs or a.defaults or st.decorator_list:
                raise ToolError("function parameters must be plain positional names (no default values, *args, **kwargs or decorators); pass every argument explicitly")
            _check_name(st.name)
            env[st.name] = _Func(st.name, [x.arg for x in a.args], st.body)
        elif isinstance(st, ast.Pass):
            pass
        else:
            raise ToolError(_HINTS.get(type(st).__name__, f"{type(st).__name__} is not supported"))

    def assign(self, tgt: ast.expr, v, env: dict) -> None:
        if isinstance(tgt, ast.Name):
            _check_name(tgt.id)
            env[tgt.id] = self.check(v)
        elif isinstance(tgt, (ast.Tuple, ast.List)):
            if not isinstance(v, (list, tuple)) or len(v) != len(tgt.elts):
                raise ToolError("cannot unpack")
            for t, x in zip(tgt.elts, v):
                self.assign(t, x, env)
        elif isinstance(tgt, ast.Subscript):
            container = self.expr(tgt.value, env)
            key = self.expr(tgt.slice, env)
            if not isinstance(container, (list, dict)):
                raise ToolError("item assignment only on lists and dicts")
            try:
                container[key] = self.check(v)
            except (IndexError, KeyError, TypeError) as e:
                raise ToolError(f"{type(e).__name__}: {e}") from e
            self.check(container)
        else:
            raise ToolError("unsupported assignment target")

    # ----------------------------------------------------------------- expressions
    def binop(self, op, a, b):
        self.tick()
        if isinstance(op, ast.Pow):
            return self.check(_power(a, b))
        fn = _BIN.get(type(op))
        if fn is None:
            raise ToolError(f"operator {type(op).__name__} is not allowed")
        if isinstance(op, ast.Mult) and (isinstance(a, (str, list, tuple)) or isinstance(b, (str, list, tuple))):
            n = b if isinstance(a, (str, list, tuple)) else a
            if isinstance(n, int) and n * len(a if isinstance(a, (str, list, tuple)) else b) > MAX_SEQ_LEN:
                raise ToolError("sequence too long")
        try:
            return self.check(fn(a, b))
        except ZeroDivisionError as e:
            raise ToolError("division by zero") from e
        except (TypeError, ValueError, OverflowError) as e:
            raise ToolError(_friendly(e)) from e

    def expr(self, e: ast.expr, env: dict):
        self.tick()
        if isinstance(e, ast.Constant):
            if isinstance(e.value, (bytes, complex)) or type(e.value) not in (int, float, str, bool, type(None)):
                raise ToolError("unsupported literal")
            return self.check(e.value)
        if isinstance(e, ast.Name):
            _check_name(e.id)
            if e.id not in env:
                if e.id in _SAFE_MATH:
                    raise ToolError(f"NameError: {e.id} is not defined; use math.{e.id}(...)")
                if e.id in _MODULE_NAMES:
                    raise ToolError(f"NameError: {e.id} is not available (no modules can be imported); use basic Python and math.*")
                raise ToolError(f"NameError: {e.id} is not defined; assign it first, or use a builtin ({BUILTIN_NAMES}) or math.<function>")
            return env[e.id]
        if isinstance(e, ast.BinOp):
            return self.binop(e.op, self.expr(e.left, env), self.expr(e.right, env))
        if isinstance(e, ast.UnaryOp) and type(e.op) in _UN:
            return self.check(_UN[type(e.op)](self.expr(e.operand, env)))
        if isinstance(e, ast.BoolOp):
            vals = e.values
            if isinstance(e.op, ast.And):
                v = True
                for x in vals:
                    v = self.expr(x, env)
                    if not v:
                        return v
                return v
            for x in vals:
                v = self.expr(x, env)
                if v:
                    return v
            return v
        if isinstance(e, ast.Compare):
            left = self.expr(e.left, env)
            for op, right_e in zip(e.ops, e.comparators):
                right = self.expr(right_e, env)
                fn = _CMP.get(type(op))
                if fn is None:
                    raise ToolError("comparison not allowed")
                try:
                    if not fn(left, right):
                        return False
                except TypeError as ex:
                    raise ToolError(f"TypeError: {ex}") from ex
                left = right
            return True
        if isinstance(e, ast.IfExp):
            return self.expr(e.body if self.expr(e.test, env) else e.orelse, env)
        if isinstance(e, (ast.List, ast.Tuple)):
            vals = [self.expr(x, env) for x in e.elts]
            return self.check(vals if isinstance(e, ast.List) else tuple(vals))
        if isinstance(e, ast.Dict):
            d = {}
            for k, v in zip(e.keys, e.values):
                if k is None:
                    raise ToolError("dict unpacking is not allowed")
                kk = self.expr(k, env)
                if not isinstance(kk, (int, str, float, bool, tuple)):
                    raise ToolError("dict keys must be numbers, strings or tuples")
                d[kk] = self.expr(v, env)
            return self.check(d)
        if isinstance(e, ast.Subscript):
            c = self.expr(e.value, env)
            if not isinstance(c, (list, tuple, str, dict, range)):
                raise ToolError("indexing only on lists, tuples, strings, dicts")
            if isinstance(e.slice, ast.Slice):
                lo = self.expr(e.slice.lower, env) if e.slice.lower else None
                hi = self.expr(e.slice.upper, env) if e.slice.upper else None
                step = self.expr(e.slice.step, env) if e.slice.step else None
                if isinstance(c, dict):
                    raise ToolError("cannot slice a dict")
                return self.check(c[lo:hi:step])
            key = self.expr(e.slice, env)
            try:
                return self.check(c[key])
            except (IndexError, KeyError, TypeError) as ex:
                raise ToolError(f"{type(ex).__name__}: {ex}") from ex
        if isinstance(e, ast.Attribute):
            base = self.expr(e.value, env)
            if base is _SAFE_MATH:
                if e.attr in _SAFE_MATH:
                    return _SAFE_MATH[e.attr]
                raise ToolError(f"math.{e.attr} is not available; available: {', '.join(sorted(_SAFE_MATH))}")
            _check_name(e.attr)
            for typ, names in _METHODS.items():
                if type(base) is typ:
                    if e.attr in names:
                        return getattr(base, e.attr)
                    raise ToolError(f"{typ.__name__}.{e.attr}() is not supported; supported {typ.__name__} methods: {', '.join(names)}")
            raise ToolError(f"'.{e.attr}' is not supported on a {type(base).__name__}; only list/str/dict methods and math.<function> can be used")
        if isinstance(e, ast.Call):
            return self.call(e, env)
        if isinstance(e, (ast.ListComp, ast.GeneratorExp)):
            if len(e.generators) != 1 or e.generators[0].is_async:
                raise ToolError("only one 'for' per comprehension is supported; nest loops instead")
            g = e.generators[0]
            it = self.expr(g.iter, env)
            if not isinstance(it, (range, list, tuple, str, dict)):
                raise ToolError(f"cannot loop over a {type(it).__name__} in a comprehension; use range(n), a list, a tuple, a string or a dict")
            local = dict(env)
            out = []
            for item in list(it):
                self.tick()
                self.assign(g.target, item, local)
                if all(self.expr(c, local) for c in g.ifs):
                    out.append(self.expr(e.elt, local))
                    if len(out) > MAX_SEQ_LEN:
                        raise ToolError("sequence too long")
            return out
        if isinstance(e, ast.JoinedStr):
            parts = []
            for v in e.values:
                if isinstance(v, ast.Constant):
                    parts.append(str(v.value))
                elif isinstance(v, ast.FormattedValue):
                    val = self.expr(v.value, env)
                    spec = ""
                    if v.format_spec is not None:
                        spec = "".join(str(x.value) if isinstance(x, ast.Constant) else _fmt(self.expr(x.value, env)) for x in v.format_spec.values)
                    try:
                        parts.append(format(val, spec) if spec else _fmt(val))
                    except (ValueError, TypeError) as ex:
                        raise ToolError(f"format error: {ex}") from ex
            return self.check("".join(parts))
        raise ToolError(_HINTS.get(type(e).__name__, f"{type(e).__name__} is not supported"))

    def call(self, e: ast.Call, env: dict):
        if e.keywords and not (isinstance(e.func, ast.Name) and e.func.id == "print" and all(k.arg in ("sep", "end") for k in e.keywords)):
            raise ToolError("keyword arguments are not supported (only print(sep=..., end=...)); pass arguments positionally, e.g. round(x, 2)")
        fn = self.expr(e.func, env)
        args = [self.expr(a, env) for a in e.args]
        kwargs = {k.arg: self.expr(k.value, env) for k in e.keywords if k.arg in ("sep", "end")}
        if isinstance(fn, _Func):
            if len(args) != len(fn.args):
                raise ToolError(f"{fn.name}() takes {len(fn.args)} arguments")
            self.depth += 1
            if self.depth > MAX_DEPTH:
                raise ToolError("recursion too deep")
            local = {k: v for k, v in env.items() if isinstance(v, _Func) or k in self.builtins()}
            local.update(zip(fn.args, args))
            try:
                self.block(fn.body, local)
                return None
            except _Return as r:
                return r.value
            finally:
                self.depth -= 1
        if fn is _PRINT:
            self.emit(kwargs.get("sep", " ").join(_fmt(a) for a in args) + kwargs.get("end", "\n"))
            return None
        if not callable(fn) or isinstance(fn, dict):
            raise ToolError(f"a {type(fn).__name__} is not callable; did you mean to index it with [...]?")
        try:
            res = fn(*args, **kwargs)
            if isinstance(res, _VIEW_TYPES):
                res = list(res)
            return self.check(res)
        except ToolError:
            raise
        except (TypeError, ValueError, ZeroDivisionError, OverflowError, KeyError, IndexError, AttributeError) as ex:
            raise ToolError(_friendly(ex)) from ex
        except Exception as ex:  # noqa: BLE001 - a declared function raised: report it like any other tool error
            raise ToolError(f"{type(ex).__name__}: {str(ex)[:200]}") from ex


_BLOCKED = {"exec", "eval", "compile", "open", "__import__", "globals", "locals", "getattr", "setattr", "delattr", "vars", "dir", "type", "object", "input", "breakpoint", "exit", "quit"}


class _Print:
    """Sentinel for print(): handled inside Sandbox.call so output goes to the sandbox running the call."""

    def __call__(self, *a, **k):  # only reached if someone bypasses call(); route nowhere
        raise ToolError("print is only callable as a statement")


_PRINT = _Print()


class PySession:
    """A REPL-like session: variables and functions persist across calls (and across turns of one
    conversation). Each call gets fresh budgets; the namespace is capped at MAX_NAMES entries.

    `functions` are declared functions (name -> callable) provided by us and part of the namespace from
    the start, like a builtin; see slm/tools/functions.py."""

    def __init__(self, functions: dict | None = None) -> None:
        self.env: dict | None = None
        self.n_calls = 0
        self.functions: dict = {}
        if functions:
            self.register(functions)

    def register(self, functions: dict) -> None:
        """Add declared functions (name -> callable); an already running session picks them up too."""
        for name in functions:
            _check_function_name(name)
        self.functions.update(functions)
        if self.env is not None:
            self.env.update(functions)

    def run(self, code: str) -> str:
        sb = Sandbox(functions=self.functions)
        if self.env is None:
            self.env = sb.builtins()
        self.n_calls += 1
        try:
            return sb.run(code, self.env)
        finally:
            if len(self.env) > MAX_NAMES + len(sb.builtins()):
                self.env = None  # a runaway namespace is dropped rather than kept growing
                raise ToolError("too many variables in the session; it has been reset")

    def reset(self) -> None:
        self.env = None


_BUILTIN_KEYS = frozenset(Sandbox().builtins())


def _check_function_name(name: str) -> None:
    """A declared function's name must be a plain identifier that shadows nothing in the namespace."""
    if not name.isidentifier() or name.startswith("_"):
        raise ToolError(f"declared function name must be a plain identifier not starting with _ ({name})")
    if name in _BLOCKED or name in _BUILTIN_KEYS:
        raise ToolError(f"declared function name {name} shadows a builtin")


def _check_name(name: str) -> None:
    if name.startswith("__"):
        raise ToolError(f"names starting with __ are not allowed ({name})")
    if name in _BLOCKED:
        if name == "input":
            raise ToolError("input() is not available; write the problem's numbers directly in the code")
        if name == "open":
            raise ToolError("files are not available; the sandbox only computes")
        raise ToolError(f"{name}() is not available in the sandbox")


def _friendly(ex: BaseException) -> str:
    msg = f"{type(ex).__name__}: {ex}"
    text = str(ex)
    if "concatenate" in text or "unsupported operand" in text or "can't multiply sequence" in text:
        msg += " (convert with int(x), float(x) or str(x) first)"
    elif "not subscriptable" in text:
        msg += " (only lists, tuples, strings and dicts can be indexed)"
    elif "has no len" in text:
        msg += " (len() works on lists, tuples, strings and dicts)"
    elif "invalid literal for int()" in text:
        msg += " (int() needs a whole number; use float() for decimals)"
    return msg


def _with_line(e: ToolError, st: ast.AST, code: str | None = None) -> ToolError:
    text = str(e)
    if text.startswith("line "):
        return e
    ln = getattr(st, "lineno", None)
    src = ""
    if code is not None and ln is not None:
        lines = code.splitlines()
        if 0 < ln <= len(lines):
            src = "  |  " + lines[ln - 1].strip()[:80]
    return ToolError(f"line {ln}: {text}{src}" if ln is not None else text)


def _power(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or isinstance(a, bool) or isinstance(b, bool):
        raise ToolError("** needs numbers")
    if isinstance(b, int) and abs(b) > MAX_POW:
        raise ToolError(f"exponent too large (max {MAX_POW})")
    if isinstance(b, float) and abs(b) > 64:
        raise ToolError("exponent too large (max 512 for integers)")
    if isinstance(a, int) and isinstance(b, int) and b >= 0 and a.bit_length() * b > MAX_INT_BITS:
        raise ToolError("integer too large")
    try:
        return a ** b
    except (OverflowError, ZeroDivisionError, ValueError) as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e


def _fmt(v, nested: bool = False) -> str:
    """print()-style formatting: floats without binary noise, strings quoted inside containers like Python does."""
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, str):
        return repr(v) if nested else v
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        from fractions import Fraction

        return format_number(Fraction(v).limit_denominator(10**9)) if abs(v) < 1e15 else repr(v)
    if isinstance(v, (list, tuple)):
        inner = ", ".join(_fmt(x, True) for x in v)
        return f"[{inner}]" if isinstance(v, list) else f"({inner}{',' if len(v) == 1 else ''})"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_fmt(k, True)}: {_fmt(x, True)}" for k, x in v.items()) + "}"
    return str(v)


def run_python(code: str, session: "PySession | None" = None) -> str:
    """Run sandboxed code (in `session` if given, else a fresh namespace); returns captured print output
    plus the value of a final bare expression. Raises ToolError on any violation or runtime error."""
    try:
        return (session or PySession()).run(code)
    except ToolError:
        raise
    except RecursionError as e:
        raise ToolError("recursion too deep") from e
    except MemoryError as e:
        raise ToolError("out of memory") from e
    except Exception as e:  # noqa: BLE001 - anything else is an interpreter bug; still never crash the trainer
        raise ToolError(f"{type(e).__name__}: {str(e)[:80]}") from e

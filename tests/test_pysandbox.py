"""Sandboxed Python subset: what it can do, and (mostly) what it must refuse. Every refusal is a ToolError,
never an exception that could escape into the trainer, and never a side effect."""

import os
import time

import pytest

from slm.tools import ToolError
from slm.tools.pysandbox import MAX_OUTPUT, run_python


def test_basic_calculation_semantics():
    assert run_python("print(12*2)") == "24"
    assert run_python("a = 120 - 36\nprint(a / 2)") == "42"
    assert run_python("x = 5 * .50\nx") == "2.5"
    assert run_python("total = 0\nfor i in range(1, 11):\n    total += i\nprint(total)") == "55"
    assert run_python("def f(n):\n    return n * (n + 1) // 2\nprint(f(100))") == "5050"
    assert run_python("xs = [3, 1, 2]\nprint(sorted(xs), len(xs), max(xs), sum(xs))") == "[1, 2, 3] 3 3 6"
    assert run_python("d = {'a': 2}\nd['b'] = 3\nprint(d['a'] * d['b'])") == "6"
    assert run_python("print([i * i for i in range(5) if i % 2 == 0])") == "[0, 4, 16]"
    assert run_python("n = 7\nprint(f'{n} items cost {n * 1.5} dollars')") == "7 items cost 10.5 dollars"
    assert run_python("print(math.sqrt(16), round(10 / 3, 2), divmod(17, 5))") == "4 3.33 (3, 2)"
    assert run_python("a, b = 3, 4\nprint(a ** 2 + b ** 2)") == "25"
    assert run_python("s = 'abc'\nprint(s[1], s[::-1], 'b' in s)") == "b cba True"
    assert run_python("x = 3\nif x > 2:\n    print('big')\nelse:\n    print('small')") == "big"
    assert run_python("i = 0\nwhile i < 3:\n    i += 1\nprint(i)") == "3"
    assert run_python("print('a', 'b', sep='-', end='!')") == "a-b!"
    assert run_python("") == ""


@pytest.mark.parametrize("code", [
    "import os",
    "from os import system",
    "__import__('os')",
    "open('x.txt', 'w')",
    "exec('print(1)')",
    "eval('1+1')",
    "compile('1', 'x', 'eval')",
    "().__class__.__bases__[0].__subclasses__()",
    "(1).__class__",
    "x = 1\nx.__class__",
    "getattr(1, '__class__')",
    "globals()",
    "vars()",
    "type(1)",
    "input()",
    "class A: pass",
    "lambda x: x",
    "with open('f') as f: pass",
    "try:\n    pass\nexcept Exception:\n    pass",
    "global g",
    "def f(*a): pass",
    "def f(a=1): pass",
    "import math",  # math is available without import; the import statement itself is refused
    "math.__dict__",
    "math.os",
    "'a'.upper()",
    "[].append(1)",
    "print.__name__",
    "x = [1]\nx.__len__()",
    "async def f(): pass",
    "yield 1",
    "del x",
    "assert 1 == 2",
    "raise ValueError('x')",
    "print(1 if True else 2, sep='x', file=None)",
])
def test_refused_constructs(code):
    with pytest.raises(ToolError):
        run_python(code)


@pytest.mark.parametrize("code, needle", [
    ("while True:\n    pass", "budget"),
    ("for i in range(10**9):\n    pass", "range too long"),
    ("x = 2 ** 100000", "too large"),
    ("x = 10 ** 600\ny = x * x * x * x * x * x * x * x", "too large"),
    ("s = 'a' * 10 ** 9", "too long"),
    ("xs = [0] * 10 ** 7", "too long"),
    ("def f(n):\n    return f(n + 1)\nf(0)", "recursion"),
    ("print(1 / 0)", "division by zero"),
    ("x = undefined_name + 1", "NameError"),
    ("print(math.factorial(10 ** 6))", "too large"),
    ("xs = []\nfor i in range(20000):\n    xs = xs + [i]", ""),  # sequence cap or budget, either way refused
    ("print('x' * 5000)", "too much output"),
])
def test_resource_limits(code, needle):
    if code == "print(1)":
        return
    with pytest.raises(ToolError) as ei:
        run_python(code)
    assert needle in str(ei.value)


def test_limits_are_fast_and_side_effect_free(tmp_path):
    marker = tmp_path / "should_not_exist.txt"
    t0 = time.time()
    for code in ("while True:\n    x = 1", "for i in range(100000):\n    for j in range(100000):\n        pass", f"open({str(marker)!r}, 'w').write('x')"):
        with pytest.raises(ToolError):
            run_python(code)
    assert time.time() - t0 < 6.0
    assert not marker.exists() and not os.path.exists("should_not_exist.txt")


def test_output_is_truncated_not_unbounded():
    out = run_python("for i in range(200):\n    print(i)")
    assert len(out) <= MAX_OUTPUT and out.endswith("...")


def test_run_tool_wraps_errors_for_the_model():
    from slm.tools.protocol import run_tool

    assert run_tool("python: print(6*7)") == ("42", True)
    assert run_tool("python: import os") == ("error: ImportError: import is not allowed", False) or run_tool("python: import os")[0].startswith("error:")
    assert run_tool("python: x = 1")[0] == "(no output)"
    assert run_tool("calc: 6*7") == ("42", True)

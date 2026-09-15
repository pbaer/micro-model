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
    "'a'.format(1)",
    "'{0.__class__}'.format(1)",
    "[].__len__()",
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


def test_supported_methods_and_generators():
    assert run_python("xs = [1, 2]\nxs.append(3)\nprint(xs, xs.index(3), xs.pop())") == "[1, 2] 2 3"
    assert run_python("s = 'a b'\nprint(s.split(), ', '.join(['x', 'y']), s.upper(), s.replace('a', 'c'))") == "['a', 'b'] x, y A B c b"
    assert run_python("d = {'a': 1}\nprint(d.get('a'), d.get('z', 0), list(d.keys()), sorted(d.items()))") == "1 0 ['a'] [('a', 1)]"
    assert run_python("print(sum(x * x for x in range(4)))") == "14"


def test_error_messages_help_the_model():
    def err(code):
        with pytest.raises(ToolError) as ei:
            run_python(code)
        return str(ei.value)

    assert "math is built in" in err("import math\nprint(math.sqrt(16))") and err("import math").startswith("line 1:")
    assert "use math.sqrt(...)" in err("print(sqrt(16))")
    assert "no modules can be imported" in err("print(np.sqrt(16))")
    assert "assign it first" in err("total = 12 * 3\nprint(totl)") and "line 2" in err("total = 12 * 3\nprint(totl)")
    assert "supported list methods" in err("xs = [1]\nxs.appendd(2)")
    assert "only list/str/dict methods" in err("x = 5\nprint(x.real)")
    assert "define a function with 'def" in err("f = lambda x: x * 2")
    assert "pass every argument explicitly" in err("def f(a=1):\n    return a")
    assert "pass arguments positionally" in err("print(max(3, 4, key=abs))")
    assert "convert with int(x)" in err("print('5' + 3)")
    assert "int() needs a whole number" in err("print(int('3.5'))")
    assert "write the problem's numbers directly" in err("x = input()")
    assert "SyntaxError" in err("print(12 * 3") and "|  print(12 * 3" in err("print(12 * 3")
    assert "loop over range(n)" in err("for i in 5:\n    print(i)")
    assert "is not callable" in err("x = [1, 2]\nprint(x(0))")
    assert "max 512" in err("print(3 ** 2 ** 40)")


def test_output_is_truncated_not_unbounded():
    out = run_python("for i in range(200):\n    print(i)")
    assert len(out) <= MAX_OUTPUT and out.endswith("...")


def test_run_tool_wraps_errors_for_the_model():
    from slm.tools.protocol import run_tool

    assert run_tool("python: print(6*7)") == ("42", True)
    assert run_tool("python: import os") == ("error: ImportError: import is not allowed", False) or run_tool("python: import os")[0].startswith("error:")
    assert run_tool("python: x = 1")[0].startswith("(no output")
    assert run_tool("calc: 6*7") == ("42", True)

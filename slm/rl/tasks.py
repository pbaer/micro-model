"""Programmatic, verifiable task generators for RL (and for reasoning-SFT synthetic data).

Every task yields {id, prompt, answer, meta}. Train/held-out splits use disjoint seeds AND a
hash-based partition of the canonical problem text, so identical problems can never appear on
both sides even if the generators overlap.

Curriculum (from the brief):
  A  arith1/arith2      1-2 digit addition/subtraction/multiplication, exact integer answer
  B  arith_multi/algebra multi-step arithmetic expressions, linear equations, formatted answers
  C  gsm8k              real word problems (train split only)
  D  pytool_*           the Python-tool grammars of `slm.rl.synth_python` (`slm.rl.pytool`): gold answers
                        are whatever the sandbox printed, often a list, a word or a boolean
  E  constraints        writing prompts with 1-3 machine-checkable instructions (`slm.rl.constraints`);
                        the reward is the fraction satisfied, not a right answer
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Task:
    id: str
    prompt: str
    answer: str  # the gold: a number, an exact string, or (constraints) the JSON spec list
    task: str  # family name, and the key per-family reward schemes resolve against
    meta: dict = field(default_factory=dict)
    # meta keys used by the rollout path:
    #   functions     list[FunctionDecl] declared to the model (blocks in the prompt, impls in the session)
    #   verifier      "auto" (default) | "constraints"
    #   answer_style  "auto" (append the '#### <answer>' instruction) | "free" (ask nothing)


_CORPUS = None


def task_corpus():
    """Real-text ingredients (sentences, words) shared by the pytool and constraint generators.

    Cached for the process. Without `set_task_corpus` this is `synth_python`'s built-in fallback list, so
    tasks generate on any machine; a trainer with a tokenizer calls `set_task_corpus` to draw from real text."""
    global _CORPUS
    if _CORPUS is None:
        from slm.rl.synth_python import load_corpus

        _CORPUS = load_corpus()
    return _CORPUS


def set_task_corpus(tok=None, path=None) -> None:
    """Point the generators at a tokenized split (default: `synth_python.DEFAULT_CORPUS`). Never fails:
    `load_corpus` falls back to the built-in sentences when the data root is not on this machine."""
    global _CORPUS
    from slm.rl.synth_python import DEFAULT_CORPUS, load_corpus

    _CORPUS = load_corpus(tok, DEFAULT_CORPUS if path is None else Path(path))


def _split_of(canonical: str, holdout_permille: int = 100) -> str:
    h = int.from_bytes(hashlib.sha1(canonical.encode()).digest()[:4], "little") % 1000
    return "heldout" if h < holdout_permille else "train"


def gen_arith(rng: random.Random, digits: int = 1, ops: str = "+-") -> Task:
    lo, hi = 0, 10**digits - 1
    a, b = rng.randint(lo, hi), rng.randint(lo, hi)
    op = rng.choice(ops)
    if op == "-" and b > a:
        a, b = b, a
    val = {"+": a + b, "-": a - b, "*": a * b}[op]
    expr = f"{a} {op} {b}"
    return Task(id="", prompt=f"What is {expr}?", answer=str(val), task=f"arith{digits}", meta={"expr": expr})


def gen_arith_multi(rng: random.Random, terms: int = 3, digits: int = 2) -> Task:
    nums = [rng.randint(1, 10**digits - 1) for _ in range(terms)]
    ops = [rng.choice("+-*") for _ in range(terms - 1)]
    expr = str(nums[0])
    for o, n in zip(ops, nums[1:]):
        expr += f" {o} {n}"
    val = eval(expr)  # noqa: S307 - trusted generator output
    return Task(id="", prompt=f"Compute {expr}.", answer=str(val), task="arith_multi", meta={"expr": expr})


def gen_linear(rng: random.Random) -> Task:
    x = rng.randint(-20, 20)
    a = rng.choice([i for i in range(-9, 10) if i != 0])
    b = rng.randint(-30, 30)
    c = a * x + b
    eq = f"{a}x {'+' if b >= 0 else '-'} {abs(b)} = {c}"
    return Task(id="", prompt=f"Solve for x: {eq}", answer=str(x), task="algebra", meta={"eq": eq})


def gen_word_problem(rng: random.Random) -> Task:
    names = ["Ada", "Ben", "Cleo", "Dev", "Eli", "Fay"]
    n = rng.choice(names)
    a, b = rng.randint(2, 40), rng.randint(2, 40)
    kind = rng.choice(["apples", "books", "coins", "marbles"])
    v = rng.randint(0, 2)
    if v == 0:
        p, ans = f"{n} has {a} {kind} and buys {b} more. How many {kind} does {n} have now?", a + b
    elif v == 1:
        a = max(a, b)
        p, ans = f"{n} has {a} {kind} and gives away {b}. How many {kind} are left?", a - b
    else:
        k = rng.randint(2, 6)
        p, ans = f"{n} has {k} boxes with {a} {kind} each. How many {kind} in total?", k * a
    return Task(id="", prompt=p, answer=str(ans), task="word", meta={})


_GSM8K_POOL: list[Task] | None = None


def gsm8k_pool() -> list[Task]:
    """GSM8K *train* problems as verifiable prompts (gold = the number after ####). The test split is
    reserved for the benchmark and never used here."""
    global _GSM8K_POOL
    if _GSM8K_POOL is None:
        import pyarrow.parquet as pq

        from slm.data.sources import SOURCES

        files = [p for p in SOURCES["gsm8k"].local_dir.rglob("*.parquet") if "train" in p.name]
        pool = []
        for p in files:
            for i, r in enumerate(pq.read_table(p).to_pylist()):
                gold = r["answer"].rpartition("####")[2].strip().replace(",", "")
                pool.append(Task(id=f"gsm8k-train-{i}", prompt=r["question"].strip(), answer=gold, task="gsm8k"))
        _GSM8K_POOL = pool
    return _GSM8K_POOL


def _pytool(rng: random.Random, family: str | None = None):
    from slm.rl.pytool import gen_pytool  # lazy: pytool imports this module

    return gen_pytool(rng, family)


def _constraints(rng: random.Random):
    from slm.rl.constraints import gen_constraints  # lazy: constraints imports this module

    return gen_constraints(rng)


GENERATORS = {
    "arith1": lambda r: gen_arith(r, 1, "+-"),
    "arith2": lambda r: gen_arith(r, 2, "+-"),
    "arith2mul": lambda r: gen_arith(r, 2, "+-*"),
    "arith_multi": lambda r: gen_arith_multi(r, 3, 2),
    "algebra": gen_linear,
    "word": gen_word_problem,
    # Python-tool grammars (slm.rl.pytool): a generator may return None when its draw failed
    "pytool": _pytool,
    "pytool_pipeline": lambda r: _pytool(r, "pytool_pipeline"),
    "pytool_strings": lambda r: _pytool(r, "pytool_strings"),
    "pytool_numbers": lambda r: _pytool(r, "pytool_numbers"),
    "pytool_sim": lambda r: _pytool(r, "pytool_sim"),
    "pytool_runcode": lambda r: _pytool(r, "pytool_runcode"),
    "pytool_declared": lambda r: _pytool(r, "pytool_declared"),
    # instruction following (slm.rl.constraints)
    "constraints": _constraints,
}


def make_tasks(names: list[str], n: int, split: str, seed: int = 0, holdout_permille: int = 100) -> list[Task]:
    """Deterministic task list for a split; train/heldout use different seeds and disjoint hash buckets."""
    rng = random.Random(f"{seed}-{split}-{','.join(names)}")
    out: list[Task] = []
    seen: set[str] = set()
    attempts = 0
    pool = [t for t in gsm8k_pool() if _split_of(t.prompt, holdout_permille) == split] if "gsm8k" in names else []
    rng.shuffle(pool)
    while len(out) < n and attempts < n * 50:
        attempts += 1
        name = rng.choice(names)
        t = None
        # Retry the family that was drawn instead of redrawing one. A generated prompt falls in the held-out
        # bucket about one time in ten, while the gsm8k pool is pre-filtered and always lands: redrawing would
        # make the held-out set almost all gsm8k and hide every other family from the eval that picks best.pt.
        for _ in range(60):
            if name == "gsm8k":
                if not pool:
                    break
                t = pool.pop()
            else:
                t = GENERATORS[name](rng)
                if t is None:  # a grammar-based generator whose draw did not work out
                    continue
            canon = t.prompt  # split by prompt text alone: generators overlap (arith1 vs arith2), and the
            # same question must never be in train for one and held-out for another
            if canon in seen or _split_of(canon, holdout_permille) != split:
                t = None
                continue
            break
        if t is None:
            continue
        seen.add(t.prompt)
        t.id = f"{t.task}-{hashlib.sha1(t.prompt.encode()).hexdigest()[:10]}"
        out.append(t)
    return out


# `slm.data.answers.SUFFIX` asks for '#### <number>'; a pytool gold is often a word, a list or a boolean.
SUFFIX_TEXT = "\nThink step by step, then give the final answer on its own line as '#### <answer>'."


def prompt_messages(t: Task) -> list[dict]:
    """The user turn for a task. Verifiable tasks ask for the `#### ` marker (the verifier is strict about
    it); `answer_style="free"` tasks (constraints) must not, since their answer *is* the writing."""
    from slm.data.answers import SUFFIX, is_numeric_answer

    if t.meta.get("answer_style") == "free":
        return [{"role": "user", "content": t.prompt}]
    return [{"role": "user", "content": t.prompt + (SUFFIX if is_numeric_answer(t.answer) else SUFFIX_TEXT)}]

"""RL tasks drawn from the Python-tool grammars of `slm.rl.synth_python`.

RL used to see arithmetic only, so it optimized one corner of a sandbox that runs loops, `def`s, lists,
strings and dicts. These families are the *same* grammars the tool SFT set is written from — this module
only turns a generated conversation into a verifiable prompt (`synth_python.sample_question`), so there is
exactly one copy of every grammar:

    pytool_pipeline   produce data -> 1-3 transforms -> aggregate (ints, floats, ranges, words, text, dicts)
    pytool_strings    reverse / palindrome / counts / capitalize / replace / longest word / acronym
    pytool_numbers    gcd, lcm, primes, digit sums, factorials, powers, fibonacci, collatz, bases
    pytool_sim        loops with state: doubling, interest, inventories, scoring rules
    pytool_runcode    "run this and tell me what it prints": the user's program, executed verbatim
    pytool_declared   1-3 declared `FunctionDecl` capabilities the model must call to get the answer
    pytool            the umbrella: samples across all of the above

The gold answer is produced by the sandbox while the task is generated (the generators drop anything the
sandbox does not reproduce), so every task is correct by construction. Gold answers are not always numbers
here — a list, a word or a boolean is common — which is what `slm.rl.rewards.verify_exact` is for.

Declared-function tasks carry their `FunctionDecl`s (with impls) in `Task.meta["functions"]`;
`slm.rl.rollout` renders the declaration blocks into the prompt and registers the impls in each rollout's
`PySession`, so the model can actually call them.
"""

from __future__ import annotations

import random

from slm.data.answers import is_numeric_answer
from slm.rl.tasks import Task, task_corpus

# RL family name -> synth_python family. `multiturn`/`error`/`arith` are left out: multiturn has no single
# prompt, and error/arith add no prompt shape the others (and the arith families) do not already cover.
PYTOOL_FAMILIES = {
    "pytool_pipeline": "pipeline",
    "pytool_strings": "strings",
    "pytool_numbers": "numbers",
    "pytool_sim": "simulation",
    "pytool_runcode": "runcode",
    "pytool_declared": "declared",
}
# umbrella mix: the SFT shares, renormalized over the families we keep
PYTOOL_WEIGHTS = {"pytool_pipeline": 0.30, "pytool_strings": 0.15, "pytool_numbers": 0.15, "pytool_sim": 0.10,
                  "pytool_runcode": 0.10, "pytool_declared": 0.20}


def gen_pytool(rng: random.Random, family: str | None = None) -> Task | None:
    """One task from a pytool family (None = the umbrella mix). None when the grammar's draw failed
    (an empty pipeline, a program the sandbox refused): the caller draws again."""
    from slm.rl import synth_python as sp

    name = family or rng.choices(list(PYTOOL_WEIGHTS), weights=list(PYTOOL_WEIGHTS.values()))[0]
    if name not in PYTOOL_FAMILIES:
        raise ValueError(name)
    s = sp.generate_sample(PYTOOL_FAMILIES[name], rng, task_corpus())
    q = sp.sample_question(s)
    if q is None:
        return None
    prompt, gold, decls = q
    # `codes` is the reference program the gold came out of: not used for training (the model writes its own),
    # kept so a task can be re-verified against the sandbox and so failures are readable.
    meta: dict = {"family": s.family, "numeric": is_numeric_answer(gold), "codes": list(s.codes)}
    if decls:
        meta["functions"] = decls
    return Task(id="", prompt=prompt, answer=gold, task=name, meta=meta)

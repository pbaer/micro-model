"""Procedural multi-turn chat families for RL: the conversation before the final turn is *given* (environment-written),
the model writes the final turn, and a checker scores it. No model writes any training text.

    recall    turn 1 states a fact ("My neighbour's parrot is named Pistachio. Please keep that in mind."), turn 2 is a
              real SmolTalk exchange (an unrelated question and its stored answer), turn 3 asks for the fact. The
              answer must contain it, as plain chat (no tool call). One in four asks for something that was NEVER
              stated ("What is my neighbour's dog called?"): the right answer says so and names nothing -- the
              negative case the recall eval cannot see, so the model learns *when* to recall, not to always emit a
              name from the history.
    revise    turn 1 is a real SmolTalk exchange; turn 2 asks to rewrite that answer under 1-3 machine-checkable
              instructions (slm.rl.constraints: bullets, sentence count, forbidden word, ending phrase, ...) while
              keeping one content word from the original. Checked by the constraints checker (fraction reward).
    sysrule   a system prompt sets a rule every reply must keep (lowercase only, at most N sentences, end with a
              phrase, never a given word); the given assistant turn already obeys it (a real SmolTalk answer,
              mechanically transformed), and the final user turn is a new real question. The reply is checked
              against the rule.

The earlier assistant turns are real SmolTalk answers (public data, allowed), the fact tables are disjoint from
`slm.eval.multiturn`'s (which stay the eval's), and the rules are the constraint types that make sense as
standing rules. Tasks carry the earlier turns in `meta["history"]`; `slm.rl.tasks.prompt_messages` prepends them.
"""

from __future__ import annotations

import json
import random
import re

from slm.rl.tasks import Task

# ------------------------------------------------------------------------------------------------ tables (disjoint from the eval's)
PET_NAMES = ["Pistachio", "Waffles", "Sprocket", "Dumpling", "Mochi", "Turnip", "Biscotti", "Noodle", "Pickle", "Truffle",
             "Ziggy", "Pumpernickel", "Gadget", "Marmalade", "Tinsel", "Crumpet"]
PEOPLE = ["Ingrid", "Tobias", "Marisol", "Kenji", "Priya", "Lukas", "Amara", "Sven", "Yusuf", "Beatrix", "Dario", "Nadia"]
TOWNS = ["Bruges", "Dunedin", "Tromso", "Salta", "Galway", "Luang Prabang", "Rovaniemi", "Zadar", "Kandy", "Cuenca", "Hobart", "Trieste"]
JOBS = ["ferrier", "bookbinder", "tugboat pilot", "clockmaker", "apiary apprentice", "radio operator", "stonemason",
        "sail maker", "map engraver", "cheesemonger", "dispatcher", "orchard manager"]
DISHES = ["shakshuka", "bibimbap", "ratatouille", "pierogi", "laksa", "moussaka", "jollof rice", "okonomiyaki", "cassoulet", "feijoada"]
FLOORS = [str(n) for n in (2, 3, 4, 5, 6, 8, 9, 11, 12, 14, 15, 16)]  # no 7: it is in the eval's NUMBERS table

# (statement, question, table, subject key, sibling question about a subject that was not stated)
RECALL_FACTS = [
    ("My neighbour's parrot is named {x}.", "What is my neighbour's parrot named?", PET_NAMES, "parrot", "What is my neighbour's dog called?"),
    ("My youngest brother is called {x}.", "What is my youngest brother called?", PEOPLE, "brother", "What is my oldest sister called?"),
    ("I was born in {x}.", "Where was I born?", TOWNS, "birthplace", "Which city do I work in?"),
    ("My aunt works as a {x}.", "What does my aunt do for a living?", JOBS, "aunt's job", "What does my uncle do for a living?"),
    ("My favourite dish is {x}.", "What is my favourite dish?", DISHES, "dish", "What is my favourite drink?"),
    ("I live on floor {x}.", "Which floor do I live on?", FLOORS, "floor", "Which street do I live on?"),
    ("My hamster is called {x}.", "What is my hamster called?", PET_NAMES, "hamster", "What is my goldfish called?"),
    ("My manager is called {x}.", "What is my manager called?", PEOPLE, "manager", "What is my dentist called?"),
]
REMEMBER = [" Please keep that in mind.", " Remember that for later.", " Just so you know.", ""]
ACKS = ["Got it -- I'll remember that.", "Noted.", "Thanks, I'll keep that in mind.", "Understood.", "Okay, noted!"]

# standing rules for `sysrule`: (spec, system sentence, transform that makes a real answer obey it)
def _rule_lowercase(rng):
    return {"type": "all_lowercase"}, "Always write in lowercase letters only, never any capital letters.", lambda t: t.lower()


def _rule_end_with(rng):
    phrase = rng.choice(["Over and out.", "That is all.", "Hope this helps!", "End of message."])
    return ({"type": "end_with", "phrase": phrase}, f'End every reply with exactly this phrase: "{phrase}"',
            lambda t: t.rstrip() + " " + phrase)


def _rule_sentences_max(rng):
    n = rng.choice([1, 2, 3])
    from slm.rl.constraints import sentences

    def cut(t):
        s = sentences(t)
        return " ".join(s[:n]) if s else t
    return {"type": "sentences_max", "n": n}, f"Never use more than {n} sentence{'s' if n > 1 else ''} in a reply.", cut


def _rule_forbid_word(rng):
    word = rng.choice(["very", "really", "just", "thing", "basically", "actually"])
    pat = re.compile(rf"\b{word}\b\s*", re.I)
    return {"type": "forbid_word", "word": word}, f'Never use the word "{word}".', lambda t: pat.sub("", t)


def _rule_no_commas(rng):
    return {"type": "no_commas"}, "Do not use any commas in your replies.", lambda t: t.replace(",", "")


RULES = [_rule_lowercase, _rule_end_with, _rule_sentences_max, _rule_forbid_word, _rule_no_commas]

# ------------------------------------------------------------------------------------------------ real exchanges
_PAIRS: list[tuple[str, str]] | None = None
_FALLBACK_PAIRS = [
    ("Why do leaves change colour in autumn?", "As days shorten, trees stop making chlorophyll, the green pigment. With it gone, the yellow and orange pigments that were there all along show through, and some trees make red ones from sugars trapped in the leaf."),
    ("Give me one tip for sleeping better.", "Keep the same wake-up time every day, including weekends. A steady rhythm does more for sleep quality than any single evening routine."),
    ("What is a haiku?", "A haiku is a short Japanese poem of three lines with five, seven and five syllables. It usually captures one moment, often in nature, and leaves the feeling for the reader to complete."),
    ("How does a thermostat work?", "A thermostat compares the room temperature with the setting you chose. When the room drifts below it, the thermostat closes a switch that turns the heating on, and opens it again once the target is reached."),
    ("Suggest a name for a bakery.", "How about \"Second Rise\"? It nods to the proving step in bread making and to fresh starts, and it is easy to say and remember."),
    ("What does a compiler do?", "A compiler translates a program from the language a person wrote it in into instructions a machine can run. Along the way it checks the program for errors and often rearranges it to run faster."),
]


def smoltalk_pairs(min_len: int = 40, max_len: int = 600) -> list[tuple[str, str]]:
    """First (user, assistant) exchanges from the SmolTalk chat sets: one-paragraph prose answers without code,
    the material the given turns are made of. Falls back to a built-in handful when the data root is absent."""
    global _PAIRS
    if _PAIRS is None:
        pairs: list[tuple[str, str]] = []
        try:
            import pyarrow.parquet as pq

            from slm.data.direct_think import is_computational
            from slm.data.sources import SOURCES

            for name in ("smoltalk-everyday-conversations", "smoltalk-openhermes-100k", "smoltalk-smol-magpie-ultra"):
                src = SOURCES.get(name)
                if src is None or not src.local_dir.exists():
                    continue
                for p in sorted(src.local_dir.rglob("*.parquet"))[:2]:
                    for r in pq.read_table(p, columns=["messages"]).to_pylist():
                        ms = r["messages"]
                        if len(ms) < 2 or ms[0]["role"] != "user" or ms[1]["role"] != "assistant":
                            continue
                        u, a = " ".join(ms[0]["content"].split()), " ".join(ms[1]["content"].split())
                        if not (15 <= len(u) <= 200 and min_len <= len(a) <= max_len) or "```" in a or "```" in u or "\n" in ms[1]["content"].strip()[:0]:
                            continue
                        if is_computational(u) or "#" in a or "http" in a:
                            continue
                        pairs.append((u, a))
                    if len(pairs) >= 20000:
                        break
        except Exception:  # noqa: BLE001 - no data root (tests): the fallback below
            pairs = []
        _PAIRS = pairs or list(_FALLBACK_PAIRS)
    return _PAIRS


def _pair(rng: random.Random) -> tuple[str, str]:
    return rng.choice(smoltalk_pairs())


# ------------------------------------------------------------------------------------------------ generators
def gen_recall(rng: random.Random, absent_share: float = 0.25) -> Task:
    stmt, q, table, subject, sibling_q = rng.choice(RECALL_FACTS)
    x = rng.choice(table)
    du, da = _pair(rng)
    history = [
        {"role": "user", "content": stmt.format(x=x) + rng.choice(REMEMBER)},
        {"role": "assistant", "content": rng.choice(ACKS)},
        {"role": "user", "content": du},
        {"role": "assistant", "content": da},
    ]
    if rng.random() < absent_share:
        return Task(id=f"recall-absent-{rng.getrandbits(40):010x}", prompt=sibling_q, answer=json.dumps({"absent": True, "not_these": table}),
                    task="recall", meta={"verifier": "recall", "answer_style": "free", "history": history, "subject": subject, "absent": True})
    return Task(id=f"recall-{rng.getrandbits(40):010x}", prompt=q, answer=json.dumps({"fact": x}), task="recall",
                meta={"verifier": "recall", "answer_style": "free", "history": history, "subject": subject, "absent": False})


_REVISE_ASKS = ["Rewrite your answer so that it follows these rules:", "Please redo that answer with these constraints:",
                "Give me the same answer again, but:", "Revise your previous reply as follows:"]


def gen_revise(rng: random.Random) -> Task | None:
    from slm.rl.constraints import describe, gen_constraints

    u, a = _pair(rng)
    base = gen_constraints(rng, rng.choice([1, 2, 2, 3]))
    if base is None:
        return None
    specs = json.loads(base.answer)
    keep = [w for w in re.findall(r"[A-Za-z]{6,}", a) if w.lower() not in ("really", "because", "should", "through", "between")]
    if keep:
        specs.append({"type": "contains_word", "word": rng.choice(keep).lower()})
    lines = "\n".join(f"- {describe(s)}" for s in specs)
    history = [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return Task(id=f"revise-{rng.getrandbits(40):010x}", prompt=f"{rng.choice(_REVISE_ASKS)}\n{lines}", answer=json.dumps(specs),
                task="revise", meta={"verifier": "constraints", "answer_style": "free", "history": history})


def gen_sysrule(rng: random.Random) -> Task:
    spec, sentence, transform = rng.choice(RULES)(rng)
    u1, a1 = _pair(rng)
    u2, _ = _pair(rng)
    history = [{"role": "system", "content": f"You are a helpful assistant. {sentence}"},
               {"role": "user", "content": u1}, {"role": "assistant", "content": transform(a1)}]
    return Task(id=f"sysrule-{rng.getrandbits(40):010x}", prompt=u2, answer=json.dumps([spec]), task="sysrule",
                meta={"verifier": "constraints", "answer_style": "free", "history": history, "rule": spec["type"]})


GENERATORS = {"recall": gen_recall, "revise": gen_revise, "sysrule": gen_sysrule}

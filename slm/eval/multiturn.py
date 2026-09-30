"""Multi-turn chat: does the model use the conversation it is in, and does it hold the format?

    python -m slm.eval.multiturn --checkpoint runs/m9_rl2_336m/checkpoints/best.pt --n 64
    python -m slm.eval.multiturn --checkpoint ... --kinds recall,sysrule --n 32

Every other eval in this project is single-turn. This one runs scripted conversations of four kinds, `--n` of each:

    recall         turn 1 the user states a fact    "My cat is called Biscuit."
                   turn 2 an unrelated distractor   "Why is the sky blue?"
                   turn 3 a question that needs it  "What is my cat called?"          -> the answer contains the fact
    recall_absent  the same shape, but turn 3 asks about something that was never stated ("What is my canoe called?"):
                   the right answer names none of the stated fact's table *and* says it was not told. Scored by
                   `slm.rl.rewards.verify_recall`, the RL reward's own rule, so eval and reward agree.
    revise         turn 1 a question with a *given* answer (not generated: one of this module's own paragraphs), turn 2
                   "rewrite it so that ..." with 1-3 machine-checkable constraints (`slm.rl.constraints` types). Scored
                   by `verify_constraints`: the share satisfied (`revise`) and all satisfied (`revise_all`).
    sysrule        a system prompt sets a rule every reply must keep (lowercase only, at most N sentences, end with a
                   phrase, never a word, no commas; sometimes two of them), turn 1 is a question whose given answer
                   already obeys it (a paragraph transformed mechanically), turn 2 a new question. Scored the same
                   way: share of rule parts kept (`sysrule`) and every part kept (`sysrule_all`).

and scores, deterministically and without a judge, over every kind:

    format     every generated assistant turn terminated properly (<|end|> reached, not cut off)
    misfire    share of generated assistant turns that called the Python tool -- these are chat turns, so any call is one
    templated  share of recall turn-3 answers that are a bare verifier template ("So the answer is Biscuit."); per kind
               in `by_kind`

All material is drawn from this module's own tables with a seeded RNG, one RNG per kind, so a run is reproducible, the
`recall` conversations are the same ones every earlier version of this eval ran for a given seed, and adding a kind
never changes another kind's conversations. The tables of the three newer kinds are held out from the RL families in
`slm.rl.synth_chat` (different facts, different paragraphs, different rule wording, phrases and words; the tests
assert it on the constants). Decoding is greedy (top_k=1). The tool loop is used for generation so a model that does
call the tool gets a real result back, exactly as it would in the harness.

    python -m slm.eval.multiturn --external smollm2-360m-instruct --n 64     # slm.eval.external (chat models only)

An external chat model runs the same conversations through its own chat template, greedy, with the same per-turn
budget; given and generated replies go into the history as plain assistant messages, and the `sysrule` system prompt
goes through the template's system role (it replaces the template's default system prompt). A template without a
system role gets the rule prepended to the first user turn instead, and `summary.system_prompt` says which. It has no
tool, so `misfire` is null.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from pathlib import Path

# ------------------------------------------------------------------------------------------------ recall (the original eval)
NAMES = ["Biscuit", "Marlowe", "Pepper", "Juniper", "Atlas", "Clementine", "Bramble", "Nimbus", "Saffron", "Wilbur",
         "Odette", "Fennel", "Rowan", "Thistle", "Quincy", "Maple"]
CITIES = ["Oslo", "Lisbon", "Nairobi", "Kyoto", "Bogota", "Tallinn", "Auckland", "Marrakesh", "Ljubljana", "Hanoi",
          "Reykjavik", "Valparaiso", "Tbilisi", "Porto", "Adelaide", "Cusco"]
JOBS = ["beekeeper", "glassblower", "cartographer", "lighthouse keeper", "piano tuner", "florist", "locksmith",
        "sommelier", "falconer", "typesetter", "ferry captain", "archivist"]
COLOURS = ["teal", "maroon", "mustard", "lavender", "olive", "crimson", "turquoise", "amber", "indigo", "coral"]
NUMBERS = [str(n) for n in (7, 13, 21, 34, 42, 58, 63, 77, 81, 96, 108, 117, 144, 233)]

# (statement template, question template, answer table, hint for matching)
FACTS = [
    ("My cat is called {x}.", "What is my cat called?", NAMES),
    ("My dog's name is {x}.", "What is my dog's name?", NAMES),
    ("I live in {x}.", "Which city do I live in?", CITIES),
    ("I grew up in {x}.", "Where did I grow up?", CITIES),
    ("I work as a {x}.", "What do I do for a living?", JOBS),
    ("My car is {x}.", "What colour is my car?", COLOURS),
    ("My favourite number is {x}.", "What is my favourite number?", NUMBERS),
    ("My sister is called {x}.", "What is my sister called?", NAMES),
]
DISTRACTORS = [
    "Why is the sky blue?", "Write one sentence about rain.", "Name a primary colour.",
    "What is the capital of France?", "Give me a word that rhymes with cat.", "Describe a forest in one sentence.",
    "What do bees make?", "Is a tomato a fruit or a vegetable?", "Name a musical instrument.",
    "What season comes after winter?",
]
_TEMPLATE_OPENERS = ("so the answer is", "the answer is", "that would be", "it is", "so it's", "that makes", "that gives")

KINDS = ("recall", "recall_absent", "revise", "sysrule")

# ------------------------------------------------------------------------------------------------ recall_absent (held out)
BOATS = ["Halcyon", "Seraphina", "Driftwood", "Kestrel", "Wanderlust", "Albatross", "Moonraker", "Periwinkle", "Stormrider", "Solstice"]
GIVEN_NAMES = ["Ottoline", "Casimir", "Leocadia", "Evander", "Philippa", "Barnaby", "Ximena", "Radomir", "Cosima", "Alaric",
               "Henrike", "Thaddeus"]
PLACES = ["Hallstatt", "Gdansk", "Arequipa", "Innsbruck", "Matera", "Stellenbosch", "Kotor", "Cordoba", "Mostar", "Nafplio",
          "Colmar", "Bergen"]
INSTRUMENTS = ["oboe", "bassoon", "harpsichord", "ukulele", "cello", "accordion", "banjo", "marimba", "mandolin", "tuba"]
HOBBIES = ["origami", "birdwatching", "rock climbing", "pottery", "fencing", "kayaking", "calligraphy", "orienteering", "archery",
           "embroidery"]
TORTOISE_NAMES = ["Methuselah", "Gingersnap", "Pebbles", "Chestnut", "Sherbet", "Tapioca", "Winifred", "Bartholomew", "Custard",
                  "Hazelnut"]

# (statement, a question about something that was never stated, the stated fact's table)
ABSENT_FACTS = [
    ("My sailboat is called {x}.", "What is my canoe called?", BOATS),
    ("My colleague's name is {x}.", "What is my landlord's name?", GIVEN_NAMES),
    ("Last summer I stayed in {x}.", "Where did I stay last winter?", PLACES),
    ("I play the {x} in an amateur orchestra.", "Which instrument does my father play?", INSTRUMENTS),
    ("My weekend hobby is {x}.", "What is my partner's hobby?", HOBBIES),
    ("My grandmother was born in {x}.", "Where was my grandfather born?", PLACES),
    ("My tortoise is named {x}.", "What is my ferret named?", TORTOISE_NAMES),
    ("My best friend is called {x}.", "What is my cousin called?", GIVEN_NAMES),
]
ABSENT_REMEMBER = [" Keep it in mind, please.", " I might bring it up again.", " Worth noting."]

# ------------------------------------------------------------------------------------------------ paragraphs (held out)
# Written for this eval, not taken from SmolTalk: the given answers of `revise` and `sysrule`. Neutral, 3 sentences each,
# with commas and capitals so the rewrite constraints bite.
PARAGRAPHS: list[tuple[str, str]] = [
    ("In a few lines, how do tides work?",
     "Tides are caused mostly by the Moon's gravity pulling on the oceans. The water bulges toward the Moon on one side of the Earth, and a second bulge forms on the far side. As the Earth turns, a coastline passes through both bulges, so most places see two high tides a day."),
    ("Why do cats purr, as far as anyone knows?",
     "Cats purr when they are relaxed, but also when they are hurt or nervous. The sound comes from rapid twitches of the muscles in the voice box. Some researchers think the low vibration may even help bones and tissue heal."),
    ("Briefly, what happens during a solar eclipse?",
     "A solar eclipse happens when the Moon passes between the Sun and the Earth. For a few minutes the Moon blocks the Sun's light, and the sky can turn dark in the middle of the day. Looking at it without proper filters can damage your eyes."),
    ("What is the best way to keep fresh herbs from wilting?",
     "Soft herbs such as basil and parsley keep best standing in a glass of water, like cut flowers. Woody herbs, for example rosemary and thyme, prefer to be wrapped in a damp paper towel and kept in the fridge. Either way, wash them only right before you use them."),
    ("What exactly does a referee do in a football match?",
     "A referee enforces the laws of the game during a match. They decide on fouls, award free kicks and penalties, and can show yellow or red cards. Two assistants on the touchlines help with offside calls and with the ball leaving the pitch."),
    ("How did the sea end up salty?",
     "Rain slowly wears down rocks on land, and rivers carry the dissolved minerals out to sea. When seawater evaporates, the salt stays behind, so over millions of years it has built up. Vents on the ocean floor add some minerals as well."),
    ("What's a safe way to get a splinter out of my finger?",
     "Wash the area with soap and water first. If the end is sticking out, grip it with clean tweezers and pull gently in the same direction it went in. For a splinter under the skin, soaking in warm water for a few minutes often brings it closer to the surface."),
    ("Could you describe what a volcano is?",
     "A volcano is an opening in the Earth's crust where molten rock, gas and ash escape. The molten rock, called magma underground, becomes lava once it reaches the surface. Over many eruptions, the cooled material can build a tall mountain."),
    ("How is weather different from climate?",
     "Weather describes the conditions outside on a particular day, such as rain, wind or sunshine. Climate is the average pattern of weather in a region over many years. A cold week in summer is weather, while a trend of warmer winters over decades is climate."),
    ("In simple terms, how does a vaccine protect you?",
     "A vaccine shows the immune system a harmless piece or a weakened form of a germ. The body learns to recognise it and makes antibodies and memory cells. If the real germ arrives later, the defences are ready and respond much faster."),
    ("Is there a known reason why people yawn?",
     "Nobody knows for certain why we yawn. One idea is that it helps cool the brain, and another is that it signals a change in alertness, for example when we are tired or bored. Yawning is also contagious, and seeing someone else yawn often sets us off."),
    ("I want to take up running. How should a beginner start?",
     "Begin with short sessions that mix walking and gentle jogging, three times a week. Increase the running time a little each week rather than all at once. Comfortable shoes and a slow pace matter more than speed in the first months."),
    ("What makes a rainbow appear in the sky?",
     "A rainbow appears when sunlight shines through raindrops while the Sun is behind you. Each drop bends the light, splits it into colours and reflects it back toward your eyes. Because every colour leaves at a slightly different angle, they spread into bands across the sky."),
    ("What does an editor at a publishing house actually do?",
     "An editor helps turn a writer's draft into a finished piece. They check that the argument is clear, cut repetition, and suggest changes to structure and tone. Some editors also correct spelling and grammar, although that job is often done separately by a proofreader."),
    ("Why does chopping onions make my eyes water?",
     "Cutting an onion breaks its cells and releases enzymes that produce a sulphur gas. When the gas reaches your eyes, it forms a mild irritant, and your tear glands try to wash it away. Chilling the onion first or using a sharp knife reduces the effect."),
    ("Where does the noise of thunder come from?",
     "Thunder is the sound made by lightning. A lightning bolt heats the air around it extremely quickly, and the air expands with a sudden crack. Because light travels faster than sound, you see the flash before you hear the rumble."),
    ("How do I keep a potted houseplant healthy?",
     "Most houseplants need bright indirect light and water only when the top of the soil feels dry. Overwatering kills more plants than neglect, so check before you pour. Turning the pot now and then helps the plant grow evenly."),
    ("Can you explain inflation without jargon?",
     "Inflation is the general rise in prices over time, which means the same amount of money buys less. A little inflation is normal in a growing economy. Central banks usually try to keep it low and steady, often around two percent a year."),
    ("What is the point of adding a leap day to the calendar?",
     "The Earth takes about 365 and a quarter days to go around the Sun. Adding one extra day every four years keeps the calendar in step with the seasons. Without it, the dates would drift by almost a month every century."),
    ("How would you describe a glacier to a child?",
     "A glacier is a large, slow river of ice that forms where more snow falls than melts. Over many years the snow is pressed into dense ice, which moves downhill under its own weight. Glaciers carve valleys and leave behind rocks and gravel when they retreat."),
    ("How do noise-cancelling headphones block sound?",
     "Small microphones on the headphones listen to the sound around you. The electronics then create a second sound wave that is the exact opposite of the noise. When the two waves meet, they largely cancel out, which works best for steady sounds like engines."),
    ("What should I keep in mind when choosing a password?",
     "A good password is long, unique and hard to guess. A phrase of four or five unrelated words is easier to remember than a short jumble of symbols, and it is often stronger. Using a different password for every account limits the damage if one leaks."),
    ("Why do some birds fly south every year?",
     "Many birds migrate to follow food and good weather. In spring they fly toward regions with long days and plenty of insects for raising their chicks. When autumn comes, they head back to warmer places where food is still easy to find."),
    ("How do cacao beans become a chocolate bar?",
     "Chocolate starts as the seeds of the cacao tree, which grow in large pods. The seeds are fermented, dried and roasted, then ground into a thick paste. Sugar, milk and cocoa butter are mixed in, and the mixture is stirred for hours to make it smooth."),
    ("Give me a short explanation of photosynthesis.",
     "Photosynthesis is how plants make their own food. Using energy from sunlight, they turn water and carbon dioxide into sugar, and release oxygen as a by-product. It takes place mainly in the leaves, inside tiny structures called chloroplasts."),
    ("How do I write a complaint that is polite but firm?",
     "State the problem clearly and briefly, with dates and any order numbers. Explain what you would like to happen, for example a refund or a replacement. Keeping a calm tone usually gets a faster and friendlier response."),
    ("Why does ice float instead of sinking?",
     "Water is unusual because it expands when it freezes. The molecules lock into an open, regular pattern, which makes ice less dense than liquid water. That is why ponds freeze from the top down, and fish can survive the winter underneath."),
    ("What is a museum curator responsible for?",
     "A curator looks after a museum's collection and decides how it is shown. They research objects, plan exhibitions and write the labels that visitors read. In smaller museums the same person may also handle loans, storage and conservation."),
    ("What actually causes an earthquake?",
     "The Earth's outer shell is broken into large plates that slowly move. Where plates meet, stress builds up along faults until the rock suddenly slips. That release of energy sends waves through the ground, which we feel as shaking."),
    ("What is a good method for memorising vocabulary in a new language?",
     "Learn words in context, such as short sentences, rather than as isolated lists. Review them a few days later and then at growing intervals, because spaced repetition helps memory. Using a new word in conversation as soon as possible makes it stick."),
]

# ------------------------------------------------------------------------------------------------ revise (held out)
REVISE_TYPES = ("bullets_exactly", "sentences_exactly", "sentences_max", "forbid_word", "end_with", "all_lowercase", "no_commas",
                "contains_word")
REVISE_ASKS = ["Could you rewrite it so that:", "Please write that again so that:", "Rework your reply so that:"]
EVAL_END_PHRASES = ["and that covers it.", "thanks for reading.", "hope that is clearer.", "that sums it up."]
# our own wording of each constraint (the RL tasks use `slm.rl.constraints.describe`); the checker reads the spec, never this
REVISE_PHRASES = {
    "bullets_exactly": lambda s: f'it is a list of exactly {s["n"]} points, one per line, each line starting with "- "',
    "sentences_exactly": lambda s: f"it is exactly {s['n']} sentence{'s' if s['n'] > 1 else ''} long",
    "sentences_max": lambda s: f"it has no more than {s['n']} sentence{'s' if s['n'] > 1 else ''}",
    "forbid_word": lambda s: f'the word "{s["word"]}" is left out entirely',
    "end_with": lambda s: f'it ends with exactly "{s["phrase"]}" and nothing after that',
    "all_lowercase": lambda s: "no capital letters appear anywhere in it",
    "no_commas": lambda s: "there is not a single comma in it",
    "contains_word": lambda s: f'it still uses the word "{s["word"]}"',
}
_STOP = {"which", "their", "there", "about", "these", "those", "other", "while", "where", "through", "would", "could", "should",
         "every", "after", "before", "under", "because", "until", "being", "rather"}

# ------------------------------------------------------------------------------------------------ sysrule (held out)
SYS_PERSONAS = ["You are a friendly assistant.", "You are an assistant that answers everyday questions.",
                "You are a patient and knowledgeable assistant."]
SYS_END_PHRASES = ["Take care.", "Cheers for now.", "Best wishes.", "Glad to help."]
SYS_FORBID_WORDS = ["also", "quite", "many", "often", "usually", "important", "simply"]
SYS_SENTENCES = {
    "all_lowercase": lambda s: "Write every reply entirely in small letters: no capital letters anywhere, not even at the start of a sentence.",
    "sentences_max": lambda s: f"Your replies must be short: {s['n']} sentence{'s' if s['n'] > 1 else ''} at most.",
    "end_with": lambda s: f'Close every reply with the words "{s["phrase"]}"',
    "forbid_word": lambda s: f'The word "{s["word"]}" is banned: do not use it in any reply.',
    "no_commas": lambda s: "Write your replies without a single comma.",
}
# rule pairs that can be kept together (sentences_max + end_with is left out: the closing words would be one more sentence)
SYS_PAIRS = [("all_lowercase", "no_commas"), ("all_lowercase", "sentences_max"), ("all_lowercase", "forbid_word"),
             ("all_lowercase", "end_with"), ("no_commas", "sentences_max"), ("no_commas", "end_with"),
             ("forbid_word", "sentences_max"), ("forbid_word", "end_with")]
_SYS_ORDER = ("forbid_word", "no_commas", "sentences_max", "end_with", "all_lowercase")  # transform order for the given answer


# ------------------------------------------------------------------------------------------------ building
def max_conversations() -> int:
    """Distinct (statement, value, distractor) triples the tables allow: the hard cap on `n` for `recall`."""
    return sum(len(table) for _, _, table in FACTS) * len(DISTRACTORS)


def max_absent_conversations() -> int:
    return sum(len(table) for _, _, table in ABSENT_FACTS) * len(DISTRACTORS)


def make_conversations(n: int, seed: int) -> list[dict]:
    """`n` distinct `recall` conversations. Distinct on the (statement, value, distractor) triple, and bounded: asking
    for more than the tables can supply raises instead of spinning -- the first version of this deduped on
    (statement, value), 116 combinations, and a test asking for 400 looped forever on two CPUs."""
    cap = max_conversations()
    if n > cap:
        raise ValueError(f"only {cap} distinct conversations are possible from the tables; asked for {n}")
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        stmt, q, table = rng.choice(FACTS)
        x = rng.choice(table)
        d = rng.choice(DISTRACTORS)
        key = (stmt, x, d)
        if key in seen:
            continue
        seen.add(key)
        out.append({"fact": x, "turns": [stmt.format(x=x) + " Please remember that.", d, q]})
    return out


def _kind_rng(kind: str, seed: int) -> random.Random:
    return random.Random(f"multiturn:{kind}:{seed}")  # str seeds hash with sha512: stable across processes


def make_absent(n: int, seed: int) -> list[dict]:
    cap = max_absent_conversations()
    if n > cap:
        raise ValueError(f"only {cap} distinct recall_absent conversations are possible; asked for {n}")
    rng = _kind_rng("recall_absent", seed)
    out, seen = [], set()
    while len(out) < n:
        stmt, q, table = rng.choice(ABSENT_FACTS)
        x = rng.choice(table)
        d = rng.choice(DISTRACTORS)
        if (stmt, x, d) in seen:
            continue
        seen.add((stmt, x, d))
        out.append({"kind": "recall_absent", "stated": x, "not_these": list(table),
                    "turns": [stmt.format(x=x) + rng.choice(ABSENT_REMEMBER), d, q], "given": []})
    return out


def _content_words(text: str, min_len: int) -> list[str]:
    seen, out = set(), []
    for w in re.findall(rf"[A-Za-z]{{{min_len},}}", text):
        lw = w.lower()
        if lw not in _STOP and lw not in seen:
            seen.add(lw)
            out.append(lw)
    return out


def _revise_specs(rng: random.Random, answer: str) -> list[dict]:
    from slm.rl.constraints import CONFLICTS, TYPES, sentences

    k = rng.choice([1, 2, 2, 3, 3])
    names = list(REVISE_TYPES)
    rng.shuffle(names)
    specs: list[dict] = []
    groups: set[str] = set()
    n_sent = len(sentences(answer))
    for name in names:
        if len(specs) >= k:
            break
        g = TYPES[name].group
        if g in groups or any(frozenset((g, h)) in CONFLICTS for h in groups):
            continue
        taken = [s.get("word") or s.get("phrase") for s in specs if s.get("word") or s.get("phrase")]
        if name == "bullets_exactly":
            spec = {"type": name, "n": rng.choice([2, 3, 4])}
        elif name == "sentences_exactly":
            spec = {"type": name, "n": rng.choice([n for n in (1, 2, 4) if n != n_sent])}
        elif name == "sentences_max":
            spec = {"type": name, "n": rng.choice([n for n in (1, 2) if n < n_sent] or [1])}
        elif name in ("forbid_word", "contains_word"):
            pool = [w for w in _content_words(answer, 5 if name == "forbid_word" else 6)
                    if all(w not in t.lower() and t.lower() not in w for t in taken)]
            if not pool:
                continue
            spec = {"type": name, "word": rng.choice(pool)}
        elif name == "end_with":
            forbidden = [s["word"] for s in specs if s["type"] == "forbid_word"]
            pool = [p for p in EVAL_END_PHRASES if not any(re.search(rf"\b{w}\b", p) for w in forbidden)]
            spec = {"type": name, "phrase": rng.choice(pool)}
        else:
            spec = {"type": name}
        if spec["type"] == "forbid_word" and any(re.search(rf"\b{spec['word']}\b", s.get("phrase", "")) for s in specs):
            continue
        specs.append(spec)
        groups.add(g)
    return specs


def make_revise(n: int, seed: int) -> list[dict]:
    rng = _kind_rng("revise", seed)
    out, seen, tries = [], set(), 0
    while len(out) < n:
        tries += 1
        if tries > 200 * max(n, 1):
            raise ValueError(f"could not build {n} distinct revise conversations")
        i = rng.randrange(len(PARAGRAPHS))
        q, a = PARAGRAPHS[i]
        specs = _revise_specs(rng, a)
        key = (i, json.dumps(specs, sort_keys=True))
        if not specs or key in seen:
            continue
        seen.add(key)
        ask = rng.choice(REVISE_ASKS) + "\n" + "\n".join(f"{j + 1}) {REVISE_PHRASES[s['type']](s)}" for j, s in enumerate(specs))
        out.append({"kind": "revise", "turns": [q, ask], "given": [a], "specs": specs})
    return out


def _sys_part(rng: random.Random, name: str, lower: bool) -> dict:
    if name == "sentences_max":
        return {"type": name, "n": rng.choice([1, 2])}
    if name == "end_with":
        p = rng.choice(SYS_END_PHRASES)
        return {"type": name, "phrase": p.lower() if lower else p}
    if name == "forbid_word":
        return {"type": name, "word": rng.choice(SYS_FORBID_WORDS)}
    return {"type": name}


def sysrule_transform(text: str, specs: list[dict]) -> str:
    """Make a paragraph obey the rule mechanically (the given answer of a `sysrule` conversation)."""
    from slm.rl.constraints import sentences

    by = {s["type"]: s for s in specs}
    for name in _SYS_ORDER:
        s = by.get(name)
        if s is None:
            continue
        if name == "forbid_word":
            text = re.sub(rf"\b{re.escape(s['word'])}\b\s*", "", text, flags=re.I)
        elif name == "no_commas":
            text = text.replace(",", "")
        elif name == "sentences_max":
            text = " ".join(sentences(text)[: s["n"]])
        elif name == "end_with":
            text = text.rstrip() + " " + s["phrase"]
        elif name == "all_lowercase":
            text = text.lower()
    return text


def make_sysrule(n: int, seed: int) -> list[dict]:
    from slm.rl.constraints import check_constraints

    rng = _kind_rng("sysrule", seed)
    out, seen, tries = [], set(), 0
    while len(out) < n:
        tries += 1
        if tries > 200 * max(n, 1):
            raise ValueError(f"could not build {n} distinct sysrule conversations")
        names = list(rng.choice(SYS_PAIRS)) if rng.random() < 0.4 else [rng.choice(list(SYS_SENTENCES))]
        lower = "all_lowercase" in names
        specs = [_sys_part(rng, name, lower) for name in names]
        i1, i2 = rng.sample(range(len(PARAGRAPHS)), 2)
        key = (json.dumps(specs, sort_keys=True), i1, i2)
        if key in seen:
            continue
        seen.add(key)
        given = sysrule_transform(PARAGRAPHS[i1][1], specs)
        ok, failed = check_constraints(given, specs)
        assert not failed, (specs, given, failed)  # the given turn must itself keep the rule
        system = rng.choice(SYS_PERSONAS) + " " + " ".join(SYS_SENTENCES[s["type"]](s) for s in specs)
        out.append({"kind": "sysrule", "system": system, "turns": [PARAGRAPHS[i1][0], PARAGRAPHS[i2][0]], "given": [given],
                    "specs": specs, "rule": "+".join(s["type"] for s in specs)})
    return out


def build_conversations(kinds: list[str], n: int, seed: int) -> list[dict]:
    """`n` conversations of every kind in `kinds`, in KINDS order. `recall` rows are exactly `make_conversations(n, seed)`."""
    bad = [k for k in kinds if k not in KINDS]
    if bad:
        raise ValueError(f"unknown kinds {bad}; known: {', '.join(KINDS)}")
    out: list[dict] = []
    for k in KINDS:
        if k not in kinds:
            continue
        if k == "recall":
            out += [{**c, "kind": "recall"} for c in make_conversations(n, seed)]
        else:
            out += {"recall_absent": make_absent, "revise": make_revise, "sysrule": make_sysrule}[k](n, seed)
    return out


def _n_given(c: dict) -> int:
    return len(c.get("given") or [])


def messages_for(c: dict, j: int, generated: list[dict], system_in_template: bool = True) -> list[dict]:
    """The chat messages that precede the reply to user turn `j`: the system prompt, every earlier user turn with its
    given or generated reply, then turn `j`. `generated[i]` is the message for the i-th *generated* reply."""
    given = c.get("given") or []
    msgs: list[dict] = []
    system = c.get("system")
    if system and system_in_template:
        msgs.append({"role": "system", "content": system})
    for i in range(j + 1):
        content = c["turns"][i]
        if i == 0 and system and not system_in_template:
            content = f"{system}\n\n{content}"
        msgs.append({"role": "user", "content": content})
        if i < j:
            msgs.append({"role": "assistant", "content": given[i]} if i < len(given) else generated[i - len(given)])
    return msgs


# ------------------------------------------------------------------------------------------------ scoring
def score_absent(answer: str, not_these: list[str]):
    """The RL reward's rule (slm.rl.rewards.verify_recall, absent form): names none of the table and says so."""
    from slm.rl.rewards import verify_recall

    return verify_recall(answer, json.dumps({"absent": True, "not_these": not_these}))


def score_constraints(answer: str, specs: list[dict]):
    """The RL reward's rule for revise / sysrule (verify_constraints): `.fraction` satisfied, `.correct` = all."""
    from slm.rl.rewards import verify_constraints

    return verify_constraints(answer, json.dumps(specs))


def _templated(final: str) -> bool:
    return final.lower().startswith(_TEMPLATE_OPENERS) and len(final.split()) <= 8


def _mean(xs) -> float | None:
    xs = list(xs)
    return round(statistics.fmean(xs), 3) if xs else None


def _score(convs: list[dict], n: int, seed: int, t0: float, tools: bool = True) -> dict:
    from slm.rl.constraints import check_constraints

    for c in convs:
        kind = c.get("kind", "recall")
        final = c["assistant"][-1]["answer"].strip()
        if kind == "recall":
            c["recall"] = c["fact"].lower() in final.lower()
        elif kind == "recall_absent":
            v = score_absent(final, c["not_these"])
            c["recall_absent"], c["reason"] = bool(v.correct), v.reason
        else:
            v = score_constraints(final, c["specs"])
            c[kind], c[f"{kind}_all"] = round(v.fraction or 0.0, 3), bool(v.correct)
            c["failed"] = check_constraints(final, c["specs"])[1]
        c["format_ok"] = all(a["terminated"] for a in c["assistant"])
        c["misfires"] = sum(1 for a in c["assistant"] if a["tool_calls"]) if tools else None
        c["templated"] = _templated(final)
    by: dict[str, list[dict]] = {}
    for c in convs:
        by.setdefault(c.get("kind", "recall"), []).append(c)
    rec = by.get("recall", [])
    turns = sum(len(c["assistant"]) for c in convs)
    s = {
        "n": n, "seed": seed,
        "recall": _mean(c["recall"] for c in rec),
        "format": _mean(c["format_ok"] for c in convs),
        "misfire": (round(sum(c["misfires"] for c in convs) / turns, 3) if turns else None) if tools else None,
        "templated": _mean(c["templated"] for c in rec),
        "mean_answer_tokens": round(statistics.fmean(a["n_tokens"] for c in convs for a in c["assistant"]), 1) if turns else None,
        "seconds": round(time.time() - t0, 1),
        "kinds": [k for k in KINDS if k in by],
        "counts": {k: len(by[k]) for k in KINDS if k in by},
        "recall_absent": _mean(c["recall_absent"] for c in by.get("recall_absent", [])),
        "revise": _mean(c["revise"] for c in by.get("revise", [])),
        "revise_all": _mean(c["revise_all"] for c in by.get("revise", [])),
        "sysrule": _mean(c["sysrule"] for c in by.get("sysrule", [])),
        "sysrule_all": _mean(c["sysrule_all"] for c in by.get("sysrule", [])),
    }
    s["by_kind"] = {}
    for k in s["kinds"]:
        cs = by[k]
        t = sum(len(c["assistant"]) for c in cs)
        s["by_kind"][k] = {"n": len(cs), "score": s[k], "format": _mean(c["format_ok"] for c in cs),
                           "misfire": round(sum(c["misfires"] for c in cs) / t, 3) if tools and t else None,
                           "templated": _mean(c["templated"] for c in cs), "generated_turns": t}
    return s


# ------------------------------------------------------------------------------------------------ running
def _generate_turn(model, tok, histories, max_new, max_calls, device="cuda"):
    """One assistant turn for every conversation, batched; returns the tool-loop results in order."""
    import torch

    from slm.data.chat import format_chat
    from slm.tools.loop import sample_with_tools

    prompts = [format_chat(tok, h, add_generation_prompt=True, think_required=True).ids for h in histories]
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    return sample_with_tools(model, tok, prompts, max_new, 1.0, 1.0, 1, gen, max_calls=max_calls)  # top_k=1: greedy


def _chunks(convs: list[dict], batch: int):
    """Batches of one kind each (every conversation in a batch has the same turn structure)."""
    for k in KINDS:
        cs = [c for c in convs if c.get("kind", "recall") == k]
        for lo in range(0, len(cs), batch):
            yield cs[lo:lo + batch]


def run(checkpoint: str, tokenizer: str, n: int, seed: int, max_new: int, max_calls: int, batch: int,
        kinds: list[str] | None = None, device: str = "cuda") -> dict:
    import torch

    from slm.data.chat import parse_assistant
    from slm.data.tokenizer import SlmTokenizer
    from slm.eval.quality import load_model
    from slm.utils.sdpa import sdpa_context

    model, _ = load_model(Path(checkpoint), device)
    tok = SlmTokenizer.load(tokenizer)
    convs = build_conversations(list(kinds or KINDS), n, seed)
    t0 = time.time()
    with torch.no_grad(), sdpa_context("decode"):
        for chunk in _chunks(convs, batch):
            gen_msgs: list[list[dict]] = [[] for _ in chunk]
            for j in range(_n_given(chunk[0]), len(chunk[0]["turns"])):
                histories = [messages_for(c, j, g) for c, g in zip(chunk, gen_msgs)]
                tcs = _generate_turn(model, tok, histories, max_new, max_calls, device)
                for c, g, tc in zip(chunk, gen_msgs, tcs):
                    p = parse_assistant(tok, tc.ids)
                    c.setdefault("assistant", []).append({
                        "answer": p["answer"], "think": p["think"], "terminated": bool(p["terminated"]) and not p["malformed"],
                        "tool_calls": int(tc.n_calls), "n_tokens": len(tc.ids)})
                    g.append({"role": "assistant", "ids": list(tc.ids)})
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return {"checkpoint": checkpoint, "summary": _score(convs, n, seed, t0), "conversations": convs}


def system_role_supported(model) -> bool:
    """Does the external model's chat template render a system message (rather than drop it or raise)?"""
    probe = "QZXSYSPROBE"
    try:
        return probe in model.render_chat([{"role": "system", "content": probe}, {"role": "user", "content": "hi"}])
    except Exception:  # noqa: BLE001 - a template that raises on a system role has none
        return False


def run_external(name: str, n: int, seed: int, max_new: int, batch: int, kinds: list[str] | None = None) -> dict:
    """`run` for an external chat model (slm.eval.external): same conversations, greedy, same budget, no tool."""
    from slm.eval.external import chat_only, load_external, result_header

    chat_only(name, "multiturn")
    model = load_external(name)
    sys_ok = system_role_supported(model)
    convs = build_conversations(list(kinds or KINDS), n, seed)
    t0 = time.time()
    for chunk in _chunks(convs, batch):
        gen_msgs: list[list[dict]] = [[] for _ in chunk]
        for j in range(_n_given(chunk[0]), len(chunk[0]["turns"])):
            histories = [messages_for(c, j, g, system_in_template=sys_ok) for c, g in zip(chunk, gen_msgs)]
            gens = model.batch_generate_chat(histories, max_new, 0.0, batch_size=batch)
            for c, g, gn in zip(chunk, gen_msgs, gens):
                c.setdefault("assistant", []).append({"answer": gn.text, "think": None, "terminated": gn.terminated, "tool_calls": None,
                                                      "n_tokens": gn.n_tokens})
                g.append({"role": "assistant", "content": gn.text})
    summary = _score(convs, n, seed, t0, tools=False)
    if any(c.get("system") for c in convs):
        summary["system_prompt"] = "template system role" if sys_ok else "prepended to the first user turn (the template has no system role)"
    return {**result_header(name), "summary": summary, "conversations": convs}


def _f(x) -> str:
    return "n/a" if x is None else format(x, ".3f")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--external", default=None, help="a registered external chat model (slm.eval.external) instead of a checkpoint")
    ap.add_argument("--tokenizer", default=r"C:\slm-data\tokenizer\v1")
    ap.add_argument("--kinds", default=",".join(KINDS), help=f"comma-separated subset of {','.join(KINDS)}")
    ap.add_argument("--n", type=int, default=64, help="conversations per kind")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--max-tool-calls", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda", help="cuda, or cpu for a smoke run beside a training job")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]
    if bad := [k for k in kinds if k not in KINDS]:
        ap.error(f"unknown kinds {bad}; known: {', '.join(KINDS)}")
    if a.external:
        res = run_external(a.external, a.n, a.seed, a.max_new, a.batch, kinds)
    elif not a.checkpoint:
        ap.error("--checkpoint is required (or --external)")
    else:
        res = run(a.checkpoint, a.tokenizer, a.n, a.seed, a.max_new, a.max_tool_calls, a.batch, kinds, a.device)
    s = res["summary"]
    print(f"multiturn n={s['n']}/kind {s['counts']}: recall {_f(s['recall'])}  recall_absent {_f(s['recall_absent'])}  "
          f"revise {_f(s['revise'])} (all {_f(s['revise_all'])})  sysrule {_f(s['sysrule'])} (all {_f(s['sysrule_all'])})  "
          f"format {_f(s['format'])}  misfire {_f(s['misfire'])}  templated {_f(s['templated'])}  "
          f"mean tokens/turn {s['mean_answer_tokens']}  [{s['seconds']:.0f}s]")
    if s.get("system_prompt"):
        print(f"  system prompt: {s['system_prompt']}")
    for k in s["kinds"]:
        for c in [c for c in res["conversations"] if c.get("kind", "recall") == k][:2]:
            verdict = c.get(k)
            print(f"  {k:<13} {str(verdict):<6} last={c['assistant'][-1]['answer'][:80]!r}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

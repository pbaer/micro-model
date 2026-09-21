"""The judged-quality prompt suite and the rubric the judge scores it with.

The suite is what a model of this size can realistically be expected to do well by the end of training but has
not saturated early: short facts, encyclopedic prose, small Python and bash tasks, arithmetic, list and pattern
continuation, definitions, a story opening, and Q/A. Every prompt has a `completion` form for base checkpoints
(the model continues the text) and a `chat` form for SFT/RL checkpoints (a user turn). `expect` is what the
judge is told a good answer contains; it is guidance, not a string the output must match.

Versioned: changing a prompt, an expectation or the rubric text means a new SUITE_VERSION / RUBRIC_VERSION, so
scores from different versions are never mixed on one chart.
"""

from __future__ import annotations

SUITE_VERSION = "v1"
RUBRIC_VERSION = "v1"

RUBRICS = ("correctness", "coherence", "task")
# Categories that stay in the suite (so old and new checkpoints are scored on the same items) but are excluded from
# `overall`: bash was dropped from every mixture after the first base on 2026-09-19 and is allowed to fade.
EXCLUDED_FROM_OVERALL = frozenset({"bash"})
# No prompt in this suite REQUIRES a tool call, so there is no "correct tool use" rate to read here (that lives
# in slm.eval.reasoning --tools, on GSM8K). What the suite can measure is the opposite: a call on a prompt where
# running code cannot help. Only arithmetic is exempt -- a call there is redundant but defensible. The `python`
# prompts ask the model to WRITE a function, so running one is a misfire like any other.
#
# M9 stage B v1 (2026-09-21) misfired on ~50% of the eligible prompts at every checkpoint, and every category
# that misfired regressed against stage A while every category that did not held or improved. The rate is
# reported per checkpoint so the next run's routing is visible without re-deriving it from raw outputs.
TOOL_DEFENSIBLE = frozenset({"arithmetic"})

# id, category, completion prompt, chat prompt, expectation, max_new_tokens
_P = [
    # ---- facts: one short answer, then whatever follows
    ("cap_france", "facts", "The capital of France is", "What is the capital of France?", "Paris.", 32),
    ("boil_water", "facts", "Water boils at a temperature of", "At what temperature does water boil?", "100 degrees Celsius (212 F) at sea level / standard pressure.", 32),
    ("largest_planet", "facts", "The largest planet in our solar system is", "Which is the largest planet in our solar system?", "Jupiter.", 32),
    ("hamlet_author", "facts", "The play Hamlet was written by", "Who wrote the play Hamlet?", "William Shakespeare.", 32),
    ("gold_symbol", "facts", "The chemical symbol for gold is", "What is the chemical symbol for gold?", "Au.", 24),
    ("continents", "facts", "The seven continents of the world are", "Name the seven continents.", "Africa, Antarctica, Asia, Australia (Oceania), Europe, North America, South America.", 48),
    ("moon_landing", "facts", "The first humans landed on the Moon in the year", "In what year did humans first land on the Moon?", "1969 (Apollo 11).", 24),
    ("spider_legs", "facts", "Question: How many legs does a spider have?\nAnswer:", "How many legs does a spider have?", "Eight.", 24),
    # ---- encyclopedic prose: a few accurate, non-repetitive sentences
    ("rome", "prose", "The history of the Roman Empire begins", "Give a short overview of the history of the Roman Empire.",
     "Rome grew from a city-state/kingdom to a republic (509 BC) to an empire under Augustus (27 BC); expansion around the Mediterranean; the western empire fell in 476 AD, the eastern (Byzantine) continued. Any accurate, non-repetitive account earns credit; wrong founders/dates lose it.", 96),
    ("industrial_rev", "prose", "The Industrial Revolution was a period", "What was the Industrial Revolution?",
     "Late 18th to 19th century, began in Britain; shift from hand production to machines, steam power, factories, urbanisation.", 96),
    ("photosynthesis", "prose", "Photosynthesis is the process by which", "Explain photosynthesis in a few sentences.",
     "Plants (and algae) use light energy, water and carbon dioxide to make glucose and release oxygen; chlorophyll in chloroplasts.", 80),
    ("water_cycle", "prose", "The water cycle describes how water", "Describe the water cycle.",
     "Evaporation, condensation (clouds), precipitation, collection/runoff; driven by the sun; continuous.", 80),
    # ---- python: a working function body
    ("fib", "python", "def fibonacci(n):", "Write a Python function fibonacci(n) that returns the nth Fibonacci number.",
     "A correct recursive or iterative implementation with base cases (0/1). Syntactically valid Python. Score correctness on whether it actually computes Fibonacci numbers.", 96),
    ("is_prime", "python", "def is_prime(n):", "Write a Python function is_prime(n) that returns True if n is prime.",
     "Handles n < 2 (False), loops over divisors (to sqrt(n) or n-1), returns True/False. Valid Python.", 96),
    ("reverse_str", "python", "def reverse_string(s):", "Write a Python function reverse_string(s) that returns the string reversed.",
     "return s[::-1] or an equivalent loop. Valid Python.", 48),
    ("count_lines", "python", "def count_lines(path):\n    \"\"\"Return the number of lines in the text file at path.\"\"\"", "Write a Python function count_lines(path) that returns the number of lines in a text file.",
     "Opens the file, counts lines (e.g. sum(1 for _ in f) or len(f.readlines())), returns the count. Valid Python.", 80),
    ("find_max", "python", "def find_max(numbers):\n    \"\"\"Return the largest number in the list without using max().\"\"\"", "Write a Python function find_max(numbers) that returns the largest number in a list without using max().",
     "Initialises with the first element, loops, compares, returns. Valid Python.", 80),
    ("class_point", "python", "class Point:\n    def __init__(self, x, y):", "Write a Python class Point that stores x and y and has a method distance_to(other).",
     "self.x = x; self.y = y; ideally a distance method using math.sqrt or ** 0.5. Valid Python.", 96),
    # ---- bash
    ("bash_largest", "bash", "#!/bin/bash\n# Print the five largest files in the current directory\n", "Write a bash command that prints the five largest files in the current directory.",
     "e.g. ls -lS | head -6, du -a | sort -rn | head -5, or find ... -printf | sort. Must be plausible bash that does the job.", 64),
    ("bash_loop", "bash", "#!/bin/bash\n# For every .txt file in this directory, print its name and its line count\n", "Write a bash script that loops over every .txt file in the current directory and prints the file name and its line count.",
     "for f in *.txt; do echo \"$f\"; wc -l < \"$f\"; done (or wc -l \"$f\"). Valid bash.", 64),
    ("bash_grep", "bash", "# Count how many lines in app.log contain the word ERROR\n$ ", "Write a shell command that counts how many lines in app.log contain the word ERROR.",
     "grep -c ERROR app.log (or grep ERROR app.log | wc -l).", 32),
    # ---- arithmetic (Q/A form for the base model)
    ("add", "arithmetic", "Question: What is 17 + 25?\nAnswer:", "What is 17 + 25?", "42.", 24),
    ("mul", "arithmetic", "Question: What is 12 times 12?\nAnswer:", "What is 12 times 12?", "144.", 24),
    ("sub", "arithmetic", "Question: What is 100 minus 37?\nAnswer:", "What is 100 minus 37?", "63.", 24),
    ("train", "arithmetic", "Question: A train travels at 60 miles per hour for 3 hours. How far does it travel?\nAnswer:", "A train travels at 60 miles per hour for 3 hours. How far does it travel?", "180 miles.", 32),
    # ---- pattern / list continuation
    ("days", "pattern", "The days of the week are Monday, Tuesday,", "List the days of the week.", "Wednesday, Thursday, Friday, Saturday, Sunday (in order, no extras, no repeats).", 32),
    ("months", "pattern", "The twelve months of the year are January, February,", "List the twelve months of the year in order.", "March through December in order, no repeats.", 48),
    ("evens", "pattern", "Counting by twos: 2, 4, 6, 8, 10,", "Continue the sequence: 2, 4, 6, 8, 10, ...", "12, 14, 16, 18, ... (even numbers in order).", 32),
    ("opposites", "pattern", "Q: What is the opposite of hot?\nA: cold\nQ: What is the opposite of big?\nA:", "What is the opposite of big?", "small (or little); ideally keeps the Q/A format.", 24),
    # ---- definitions
    ("prime_def", "definition", "A prime number is a number that", "What is a prime number?", "A whole number greater than 1 whose only divisors are 1 and itself; examples like 2, 3, 5, 7.", 48),
    ("noun_def", "definition", "In grammar, a noun is a word that", "What is a noun?", "Names a person, place, thing or idea; examples.", 48),
    # ---- narrative / register
    ("story", "narrative", "Once upon a time, in a small village by the sea, there lived", "Write the opening paragraph of a short story about a small village by the sea.",
     "A coherent story opening: a character, a setting, some development; no repetition loops; consistent tense and names.", 96),
    ("email", "narrative", "Dear Ms. Johnson,\n\nI am writing to", "Write a short, polite email to a teacher asking for a two-day extension on an assignment.",
     "Polite, coherent request in an appropriate register; a reason and a closing are a plus.", 80),
    # ---- Q/A with a reason
    ("why_sky", "qa", "Question: Why is the sky blue?\nAnswer:", "Why is the sky blue?",
     "Sunlight scatters off air molecules and shorter (blue) wavelengths scatter more (Rayleigh scattering).", 64),
    ("why_seasons", "qa", "Question: Why does the Earth have seasons?\nAnswer:", "Why does the Earth have seasons?",
     "The tilt of Earth's axis (about 23.5 degrees) changes how directly sunlight hits each hemisphere through the year; NOT distance from the sun.", 64),
]

SUITE = [{"id": i, "category": c, "completion": comp, "chat": chat, "expect": exp, "max_new_tokens": n} for i, c, comp, chat, exp, n in _P]
CATEGORIES = sorted({p["category"] for p in SUITE})
BY_ID = {p["id"]: p for p in SUITE}
assert len(BY_ID) == len(SUITE), "duplicate prompt id"

JUDGE_INSTRUCTIONS = """You are scoring outputs from a SMALL language model (a few hundred million parameters) that is being
trained from scratch. The purpose is to track improvement over training, so be consistent and use the whole scale.

Each item gives you the prompt, how the model was used, the model's output, and what a good answer contains
("expect"). Score three rubrics, each an integer from 1 to 5:

correctness  5 = factually / logically / syntactically correct where it matters; 4 = right with a minor slip;
             3 = partly right or right with a real error; 2 = mostly wrong; 1 = wrong, empty, or nonsense.
             For stories and prose, judge plausibility and internal consistency rather than a fixed answer.
coherence    5 = fluent, grammatical, on topic, no repetition; 4 = minor awkwardness; 3 = readable but drifts,
             or some repetition; 2 = largely incoherent or a repetition loop after a sensible start;
             1 = word salad or an immediate loop.
task         5 = does exactly what the prompt implies (answers the question, completes the function, continues
             the list) in the natural format; 3 = related but incomplete, wrong format, or answers a different
             question; 1 = ignores the task entirely.

Rules:
- mode "completion" means the model was given the prompt as the start of a document and CONTINUED it. Judge the
  continuation. For short-answer prompts, correctness is decided by the first sentence; text that follows only
  affects coherence and task (a correct answer followed by a repetition loop is correctness 5, coherence 1 or 2).
- mode "chat" means the model answered as an assistant. Judge the answer as an answer.
- Code: mentally run it. A function that returns the wrong value is correctness 1 or 2 even if it looks tidy.
- Do not reward length. Do not penalise a short correct answer.
- You do not know which training stage an item comes from. Score each item on its own.

Return ONLY a JSON array, one object per item, in the same order, exactly:
[{"item_id": "...", "correctness": 1-5, "coherence": 1-5, "task": 1-5, "note": "<= 15 words"}]
"""

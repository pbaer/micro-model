"""Factual-recall probe: 194 short, unambiguous questions (capitals, science basics, history dates, authors, units,
language) scored by whether the greedy continuation contains an accepted answer. Two prompt styles:

    completion  "The capital of France is"            -> first 12 generated tokens must contain "Paris"
    chat        <|user|>What is the capital of France?<|end|><|assistant|>...   (SFT/RL checkpoints)

It measures knowledge stored in the weights, which the loss and multiple-choice suites only reach indirectly.
Answers are matched case-insensitively as whole words; aliases separated by "|". Baseline: the 149M base (5B tokens)
scores 43% in completion form, its instruct checkpoint 46% in chat form (capitals ~65%, units ~15%).

    python -m slm.eval.facts --checkpoint runs/<run>/checkpoints/final.pt [--chat] [--out ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

from slm.data.chat import format_chat
from slm.data.tokenizer import SlmTokenizer
from slm.eval.reasoning import load_model
from slm.utils.sdpa import sdpa_context

# (completion prompt, chat question, accepted answers)
CAPITALS = [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Japan", "Tokyo"), ("China", "Beijing"), ("Russia", "Moscow"),
            ("Canada", "Ottawa"), ("Australia", "Canberra"), ("Egypt", "Cairo"), ("Brazil", "Brasilia|Brasília"), ("India", "New Delhi|Delhi"), ("Mexico", "Mexico City"),
            ("the United Kingdom", "London"), ("the United States", "Washington"), ("Turkey", "Ankara"), ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Sweden", "Stockholm"),
            ("Norway", "Oslo"), ("Poland", "Warsaw"), ("Austria", "Vienna"), ("Ireland", "Dublin"), ("the Netherlands", "Amsterdam"), ("Belgium", "Brussels"),
            ("Argentina", "Buenos Aires"), ("Peru", "Lima"), ("Kenya", "Nairobi"), ("Nigeria", "Abuja"), ("South Korea", "Seoul"), ("Thailand", "Bangkok"),
            ("Vietnam", "Hanoi"), ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Saudi Arabia", "Riyadh"), ("Cuba", "Havana"), ("Chile", "Santiago"), ("Finland", "Helsinki"),
            ("Denmark", "Copenhagen"), ("Hungary", "Budapest"), ("Czechia", "Prague"), ("Switzerland", "Bern"), ("Indonesia", "Jakarta"), ("Pakistan", "Islamabad")]
SCIENCE = [("The chemical symbol for gold is", "What is the chemical symbol for gold?", "Au"), ("The chemical symbol for iron is", "What is the chemical symbol for iron?", "Fe"),
           ("The chemical symbol for sodium is", "What is the chemical symbol for sodium?", "Na"), ("The chemical symbol for silver is", "What is the chemical symbol for silver?", "Ag"),
           ("The chemical symbol for potassium is", "What is the chemical symbol for potassium?", "K"), ("The chemical symbol for lead is", "What is the chemical symbol for lead?", "Pb"),
           ("Water is made of hydrogen and", "Water is made of hydrogen and which other element?", "oxygen"), ("The gas that plants absorb from the air is", "Which gas do plants absorb from the air?", "carbon dioxide|CO2"),
           ("The largest planet in the solar system is", "What is the largest planet in the solar system?", "Jupiter"), ("The planet closest to the Sun is", "Which planet is closest to the Sun?", "Mercury"),
           ("The red planet is", "Which planet is known as the red planet?", "Mars"), ("The planet with the most prominent rings is", "Which planet is famous for its rings?", "Saturn"),
           ("At sea level, water boils at a temperature of", "At what temperature does water boil at sea level, in Celsius?", "100"), ("Water freezes at a temperature of", "At what temperature does water freeze, in Celsius?", "0|zero"),
           ("The speed of light is approximately", "What is the speed of light in km per second, approximately?", "300,000|300000|299,792|299792|3 x 10"), ("The hardest natural substance is", "What is the hardest natural substance?", "diamond"),
           ("The powerhouse of the cell is the", "Which organelle is called the powerhouse of the cell?", "mitochondri"), ("The molecule that carries genetic information is", "Which molecule carries genetic information in cells?", "DNA|deoxyribonucleic"),
           ("The number of bones in the adult human body is", "How many bones are in the adult human body?", "206"), ("The largest organ of the human body is the", "What is the largest organ of the human body?", "skin"),
           ("Sound travels faster in water than in", "Does sound travel faster in water or in air?", "air|water"), ("The chemical formula for table salt is", "What is the chemical formula for table salt?", "NaCl"),
           ("The atomic number of carbon is", "What is the atomic number of carbon?", "6|six"), ("The atomic number of hydrogen is", "What is the atomic number of hydrogen?", "1|one"),
           ("The process by which plants make food using sunlight is called", "What is the process by which plants make food using sunlight?", "photosynthesis"),
           ("The force that pulls objects toward the Earth is", "What force pulls objects toward the Earth?", "gravity"), ("The center of an atom is called the", "What is the center of an atom called?", "nucleus"),
           ("Light travels in a", "Does light travel in a straight line or a curve?", "straight"), ("The most abundant gas in Earth's atmosphere is", "What is the most abundant gas in Earth's atmosphere?", "nitrogen"),
           ("The star at the center of our solar system is the", "What is the star at the center of our solar system?", "Sun"), ("Earth's only natural satellite is the", "What is Earth's only natural satellite?", "Moon"),
           ("The unit of electrical resistance is the", "What is the unit of electrical resistance?", "ohm"), ("The unit of force in the SI system is the", "What is the SI unit of force?", "newton"),
           ("The unit of electric current is the", "What is the unit of electric current?", "ampere|amp"), ("The scientist who proposed the theory of general relativity was", "Who proposed the theory of general relativity?", "Einstein"),
           ("The scientist who formulated the laws of motion and universal gravitation was", "Who formulated the laws of motion and universal gravitation?", "Newton"),
           ("The theory of evolution by natural selection was proposed by", "Who proposed the theory of evolution by natural selection?", "Darwin"),
           ("Penicillin was discovered by", "Who discovered penicillin?", "Fleming"), ("The element with the chemical symbol O is", "Which element has the chemical symbol O?", "oxygen"),
           ("The number of planets in the solar system is", "How many planets are in the solar system?", "8|eight")]
HISTORY = [("World War II ended in the year", "In what year did World War II end?", "1945"), ("World War I began in the year", "In what year did World War I begin?", "1914"),
           ("The American Declaration of Independence was signed in", "In what year was the American Declaration of Independence signed?", "1776"),
           ("The first human landed on the Moon in", "In what year did the first human land on the Moon?", "1969"), ("The Berlin Wall fell in", "In what year did the Berlin Wall fall?", "1989"),
           ("Christopher Columbus reached the Americas in", "In what year did Columbus reach the Americas?", "1492"), ("The French Revolution began in", "In what year did the French Revolution begin?", "1789"),
           ("The first president of the United States was", "Who was the first president of the United States?", "Washington"), ("The Titanic sank in the year", "In what year did the Titanic sink?", "1912"),
           ("The Roman Empire's first emperor was", "Who was the first Roman emperor?", "Augustus|Octavian"), ("The ancient Egyptian writing system is called", "What is the ancient Egyptian writing system called?", "hieroglyph"),
           ("The Great Wall is located in", "In which country is the Great Wall?", "China"), ("The pyramids of Giza are in", "In which country are the pyramids of Giza?", "Egypt"),
           ("The Magna Carta was signed in the year", "In what year was the Magna Carta signed?", "1215"), ("Napoleon was defeated at the Battle of", "At which battle was Napoleon finally defeated?", "Waterloo"),
           ("The printing press was invented by", "Who invented the printing press?", "Gutenberg"), ("The telephone was invented by", "Who is credited with inventing the telephone?", "Bell"),
           ("The light bulb is most associated with the inventor", "Which inventor is most associated with the light bulb?", "Edison"), ("The first man in space was", "Who was the first man in space?", "Gagarin"),
           ("The United Nations was founded in", "In what year was the United Nations founded?", "1945"), ("The Soviet Union dissolved in", "In what year did the Soviet Union dissolve?", "1991"),
           ("The Wright brothers made the first powered flight in", "In what year did the Wright brothers make the first powered flight?", "1903"),
           ("Julius Caesar was assassinated in the year", "In what year was Julius Caesar assassinated?", "44"), ("The Renaissance began in the country of", "In which country did the Renaissance begin?", "Italy"),
           ("The leader of the Indian independence movement known for nonviolence was", "Who led India's nonviolent independence movement?", "Gandhi"),
           ("The Black Death reached Europe in the", "In which century did the Black Death reach Europe?", "14th|fourteenth|1347|1348|1340"), ("The capital of the Roman Empire was", "What was the capital of the Roman Empire?", "Rome"),
           ("The first modern Olympic Games were held in", "In what year were the first modern Olympic Games held?", "1896"), ("The Eiffel Tower was completed in", "In what year was the Eiffel Tower completed?", "1889"),
           ("The Statue of Liberty was a gift from", "Which country gave the Statue of Liberty to the United States?", "France")]
CULTURE = [("Romeo and Juliet was written by", "Who wrote Romeo and Juliet?", "Shakespeare"), ("Pride and Prejudice was written by", "Who wrote Pride and Prejudice?", "Austen"),
           ("The Odyssey is attributed to the poet", "Who is the Odyssey attributed to?", "Homer"), ("1984 was written by", "Who wrote the novel 1984?", "Orwell"),
           ("Moby-Dick was written by", "Who wrote Moby-Dick?", "Melville"), ("The Mona Lisa was painted by", "Who painted the Mona Lisa?", "da Vinci|Leonardo"),
           ("The Starry Night was painted by", "Who painted The Starry Night?", "van Gogh|Gogh"), ("The composer of the Ninth Symphony with the Ode to Joy was", "Who composed the Ninth Symphony with the Ode to Joy?", "Beethoven"),
           ("Sherlock Holmes was created by", "Who created Sherlock Holmes?", "Doyle"), ("Harry Potter was written by", "Who wrote the Harry Potter books?", "Rowling"),
           ("The Lord of the Rings was written by", "Who wrote The Lord of the Rings?", "Tolkien"), ("Frankenstein was written by", "Who wrote Frankenstein?", "Shelley"),
           ("The play Hamlet was written by", "Who wrote Hamlet?", "Shakespeare"), ("The Divine Comedy was written by", "Who wrote the Divine Comedy?", "Dante"),
           ("Don Quixote was written by", "Who wrote Don Quixote?", "Cervantes"), ("The Adventures of Tom Sawyer was written by", "Who wrote The Adventures of Tom Sawyer?", "Twain"),
           ("The Iliad is attributed to", "Who is the Iliad attributed to?", "Homer"), ("The Sistine Chapel ceiling was painted by", "Who painted the Sistine Chapel ceiling?", "Michelangelo"),
           ("The Four Seasons was composed by", "Who composed The Four Seasons?", "Vivaldi"), ("The Great Gatsby was written by", "Who wrote The Great Gatsby?", "Fitzgerald")]
GEO = [("The longest river in the world is the", "What is the longest river in the world?", "Nile|Amazon"), ("The largest ocean is the", "What is the largest ocean?", "Pacific"),
       ("The highest mountain in the world is", "What is the highest mountain in the world?", "Everest"), ("The largest desert in the world is the", "What is the largest hot desert in the world?", "Sahara|Antarctic"),
       ("The largest country by area is", "What is the largest country by area?", "Russia"), ("The most populous country in the world is", "What is the most populous country?", "India|China"),
       ("The smallest country in the world is", "What is the smallest country in the world?", "Vatican"), ("The largest continent is", "What is the largest continent?", "Asia"),
       ("The Amazon rainforest is mostly in", "In which country is most of the Amazon rainforest?", "Brazil"), ("Mount Fuji is in", "In which country is Mount Fuji?", "Japan"),
       ("The Alps are a mountain range in", "On which continent are the Alps?", "Europe"), ("The Sahara desert is in", "On which continent is the Sahara?", "Africa"),
       ("The Great Barrier Reef is off the coast of", "Off which country's coast is the Great Barrier Reef?", "Australia"), ("The Andes mountains are in", "On which continent are the Andes?", "South America"),
       ("The Danube flows into the", "Into which sea does the Danube flow?", "Black Sea"), ("The largest lake in Africa is Lake", "What is the largest lake in Africa?", "Victoria"),
       ("The Thames flows through the city of", "Which city does the Thames flow through?", "London"), ("Sicily is an island belonging to", "Which country does Sicily belong to?", "Italy"),
       ("The Panama Canal connects the Atlantic and the", "The Panama Canal connects the Atlantic with which ocean?", "Pacific"), ("Greenland is a territory of", "Which country does Greenland belong to?", "Denmark")]
UNITS = [("There are 60 seconds in a", "How many seconds are in a minute?", "minute|60"), ("The number of minutes in an hour is", "How many minutes are in an hour?", "60|sixty"),
         ("The number of hours in a day is", "How many hours are in a day?", "24|twenty-four"), ("The number of days in a leap year is", "How many days are in a leap year?", "366"),
         ("The number of days in a week is", "How many days are in a week?", "7|seven"), ("The number of months in a year is", "How many months are in a year?", "12|twelve"),
         ("One kilometer is equal to", "How many meters are in a kilometer?", "1000|1,000|thousand"), ("One kilogram is equal to", "How many grams are in a kilogram?", "1000|1,000|thousand"),
         ("The number of centimeters in a meter is", "How many centimeters are in a meter?", "100|hundred"), ("A dozen is equal to", "How many items are in a dozen?", "12|twelve"),
         ("The number of sides of a hexagon is", "How many sides does a hexagon have?", "6|six"), ("The number of sides of a triangle is", "How many sides does a triangle have?", "3|three"),
         ("The number of degrees in a right angle is", "How many degrees are in a right angle?", "90|ninety"), ("The number of degrees in a circle is", "How many degrees are in a full circle?", "360"),
         ("The square root of 144 is", "What is the square root of 144?", "12|twelve"), ("The number of continents is", "How many continents are there?", "7|seven"),
         ("The number of letters in the English alphabet is", "How many letters are in the English alphabet?", "26|twenty-six"), ("The Roman numeral X stands for", "What number does the Roman numeral X represent?", "10|ten"),
         ("The Roman numeral C stands for", "What number does the Roman numeral C represent?", "100|hundred"), ("A century is a period of", "How many years are in a century?", "100|hundred")]
LANGUAGE = [("The opposite of hot is", "What is the opposite of hot?", "cold"), ("The opposite of up is", "What is the opposite of up?", "down"), ("The plural of child is", "What is the plural of child?", "children"),
            ("The plural of mouse is", "What is the plural of mouse?", "mice"), ("The past tense of go is", "What is the past tense of go?", "went"), ("The past tense of eat is", "What is the past tense of eat?", "ate"),
            ("A baby dog is called a", "What is a baby dog called?", "puppy|pup"), ("A baby cat is called a", "What is a baby cat called?", "kitten"), ("A group of lions is called a", "What is a group of lions called?", "pride"),
            ("The first letter of the alphabet is", "What is the first letter of the English alphabet?", "A"), ("The color of the sky on a clear day is", "What color is the sky on a clear day?", "blue"),
            ("The color of grass is", "What color is grass?", "green"), ("The opposite of big is", "What is the opposite of big?", "small|little"), ("The language spoken in Brazil is", "What language is spoken in Brazil?", "Portuguese"),
            ("The language spoken in Mexico is", "What language is spoken in Mexico?", "Spanish"), ("The language spoken in Japan is", "What language is spoken in Japan?", "Japanese"),
            ("A word that means the same as happy is", "Give a word that means the same as happy.", "glad|joyful|cheerful|content|pleased|joyous"), ("A synonym for large is", "Give a synonym for large.", "big|huge|great|enormous"),
            ("The number that comes after nine is", "What number comes after nine?", "10|ten"), ("The day that comes after Monday is", "What day comes after Monday?", "Tuesday")]


def items() -> list[dict]:
    out = []
    for c, a in CAPITALS:
        out.append({"cat": "capitals", "completion": f"The capital of {c} is", "question": f"What is the capital of {c}?", "answers": a})
    for cat, lst in (("science", SCIENCE), ("history", HISTORY), ("culture", CULTURE), ("geography", GEO), ("units", UNITS), ("language", LANGUAGE)):
        for comp, q, a in lst:
            out.append({"cat": cat, "completion": comp, "question": q, "answers": a})
    return out


def _hit(text: str, answers: str) -> bool:
    t = text.lower()
    for a in answers.split("|"):
        a = a.strip().lower()
        if re.search(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", t):
            return True
    return False


@torch.no_grad()
def run_facts(model, tok: SlmTokenizer, chat: bool = False, think_required: bool = False, max_new: int = 12, batch: int = 32) -> dict:
    from slm.rl.rollout import sample_completions

    its = items()
    if chat:
        enc = [format_chat(tok, [{"role": "user", "content": it["question"]}], add_generation_prompt=True, think_required=think_required).ids for it in its]
        max_new = max(max_new, 96 if think_required else 32)
    else:
        enc = [[tok.bos_id, *tok.encode(it["completion"])] for it in its]
    by_len: dict[int, list[int]] = {}
    for i, ids in enumerate(enc):
        by_len.setdefault(len(ids), []).append(i)
    outs = [""] * len(its)
    with sdpa_context("decode"):
        for group in by_len.values():
            for b in range(0, len(group), batch):
                idxs = group[b : b + batch]
                comps = sample_completions(model, tok, [enc[i] for i in idxs], max_new, 0.0)
                for i, c in zip(idxs, comps):
                    outs[i] = tok.decode(c, skip_special=True)
    per_cat: dict[str, list[int]] = {}
    rows = []
    for it, o in zip(its, outs):
        if chat and "<|/think|>" in o:
            o = o.split("<|/think|>")[-1]
        ok = _hit(o, it["answers"])
        per_cat.setdefault(it["cat"], []).append(int(ok))
        rows.append({**it, "output": o.strip()[:80], "correct": ok})
    total = sum(r["correct"] for r in rows) / len(rows)
    return {"accuracy": total, "n": len(rows), "per_category": {k: sum(v) / len(v) for k, v in per_cat.items()}, "rows": rows, "mode": "chat" if chat else "completion"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=r"C:\slm-data\tokenizer\v1")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--think", action="store_true", help="chat mode with a forced think span (reasoning checkpoints)")
    ap.add_argument("--device", default="cuda", help="cpu lets the probe run while the GPU trains")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tok = SlmTokenizer.load(a.tokenizer)
    model = load_model(a.checkpoint, a.device)
    t0 = time.time()
    res = run_facts(model, tok, chat=a.chat or a.think, think_required=a.think)
    res["checkpoint"] = a.checkpoint
    print(f"facts ({res['mode']}): {res['accuracy'] * 100:.1f}% of {res['n']}  " + "  ".join(f"{k} {v * 100:.0f}%" for k, v in res["per_category"].items()) + f"  [{time.time() - t0:.0f}s]")
    for r in res["rows"][:6]:
        print(f"  {r['completion'] if res['mode'] == 'completion' else r['question']!r} -> {r['output']!r} {'OK' if r['correct'] else 'x'}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()

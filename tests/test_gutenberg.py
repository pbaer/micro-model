"""PG-19 narrative-prose filter (slm/data/gutenberg.py) on synthetic books."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from slm.data import gutenberg as G
from slm.data.gutenberg import GutenbergConfig, book_stats, eligible, normalize, segment_bounds, strip_boilerplate, verdict

CFG = GutenbergConfig()


def wrap(text: str, width: int = 70) -> str:
    """Hard-wrap each paragraph like a Project Gutenberg file."""
    out = []
    for para in text.split("\n\n"):
        line, lines = "", []
        for w in para.split():
            if line and len(line) + 1 + len(w) > width:
                lines.append(line)
                line = w
            else:
                line = f"{line} {w}".strip()
        lines.append(line)
        out.append("\n".join(lines))
    return "\n\n".join(out)


NARRATION = ("The evening had come down over the valley before the travellers reached the inn, and the lamps in "
             "the windows made long yellow bars across the road. Nobody in the house expected them, and the "
             "landlord, who was counting his money at the table by the fire, looked up with a start when the door "
             "opened and the cold air came in with them.")
DIALOGUE = ['"We have walked since morning," said the elder of the two, "and we should be glad of a fire."',
            '"You shall have one," said the landlord. "Sit down, sit down, and I will call my wife."',
            '"Is there any news from the town?" asked the younger one, pulling off his gloves.']


def novel(n: int = 40, dialogue: bool = True) -> str:
    paras = []
    for i in range(n):
        paras.append(NARRATION)
        if dialogue:
            paras.append(DIALOGUE[i % 3])
    return wrap("\n\n".join(paras))


def test_strip_full_gutenberg_markers_and_crlf():
    body = novel(5)
    raw = ("The Project Gutenberg EBook of Nothing, by Nobody\r\n\r\nThis eBook is for the use of anyone anywhere.\r\n\r\n"
           "*** START OF THIS PROJECT GUTENBERG EBOOK NOTHING ***\r\n\r\n" + body.replace("\n", "\r\n") +
           "\r\n\r\n*** END OF THIS PROJECT GUTENBERG EBOOK NOTHING ***\r\n\r\nSection 1. General Terms of Use\r\n")
    s = strip_boilerplate(raw)
    assert s == body
    assert "Project Gutenberg" not in s and "\r" not in s


def test_strip_pg19_residue_credit_and_end_line():
    body = novel(10)
    raw = "\n\n\n\nProduced by Some Volunteer and the Online Distributed\nProofreading Team\n\n\n\n\nTHE INN\n\nCHAPTER I\n\n" + body + \
          "\n\n\n\n\nEnd of the Project Gutenberg EBook of The Inn, by Nobody\n\n*** \n"
    s = strip_boilerplate(raw)
    assert s.startswith("THE INN") and s.endswith(body[-200:])
    assert "Produced by" not in s and "End of the Project Gutenberg" not in s and not s.rstrip().endswith("***")
    # a mention in the first half of the book is text, not the closing line
    early = "End of Project Gutenberg's hopes, he thought.\n\n" + body
    assert strip_boilerplate(early).startswith("End of Project Gutenberg's hopes")


def test_date_rule_is_applied_before_download():
    meta = {"1": {"publication_date": 1849}, "2": {"publication_date": 1850}, "3": {"publication_date": 1911}}
    assert eligible(["1", "2", "3"], meta, 1850) == ["2", "3"]


def test_dialogue_density_separates_fiction_from_essays():
    fiction, essay = book_stats(novel(), CFG), book_stats(novel(dialogue=False), CFG)
    assert fiction["dialogue"] > 0.2 and essay["dialogue"] == 0.0
    assert verdict(fiction, CFG) is None
    assert verdict(essay, CFG) == "dialogue"
    # British single-quote dialogue counts; an apostrophe does not
    brit = book_stats(wrap("\n\n".join([NARRATION, "'We are late,' said he. 'Come in.'"] * 30)), CFG)
    assert brit["dialogue"] > 0.1
    assert book_stats(wrap("\n\n".join(["It was the landlord's house and nobody's fault."] * 60)), CFG)["dialogue"] == 0.0


def test_verse_is_dropped_but_an_embedded_poem_is_not():
    stanza = "The wind is in the willow tree,\nThe rain is on the hill,\nAnd all the birds have gone to sea,\nAnd all the world is still."
    poems = "\n\n".join([stanza, '"Hark," she said, "the bells."'] * 150)
    st = book_stats(poems, CFG)
    assert st["verse_share"] > 0.5 and verdict(st, CFG) == "verse"
    prose_with_poem = novel(40) + "\n\n" + stanza + "\n\n" + novel(40)
    st2 = book_stats(prose_with_poem, CFG)
    assert 0 < st2["verse_share"] < CFG.max_verse_share and verdict(st2, CFG) is None


def test_archaic_rate():
    old = "\n\n".join([NARRATION, '"Thou art welcome, and thy brother hath a bed," said he. "Doth it please thee?"'] * 30)
    st = book_stats(wrap(old), CFG)
    assert st["archaic_per_1k"] > 20 and verdict(st, CFG) == "archaic"
    assert book_stats(novel(), CFG)["archaic_per_1k"] == 0


def test_caps_and_table_lines_catch_plays():
    play = "\n\n".join(["HAMLET.\nTo be, or not to be, that is the question of the day and the night.",
                        "OPHELIA.\nGood my lord, how does your honour for this many a day?"] * 60)
    st = book_stats(play + "\n\n" + novel(5), CFG)
    assert st["caps_share"] > CFG.max_caps_share and verdict(st, CFG) == "caps"
    table = "\n".join(["Wheat      12      14      19", "Barley     10      11      12", "Chapter I ........ 5"] * 50)
    assert book_stats(table + "\n\n" + novel(10), CFG)["caps_share"] > CFG.max_caps_share


def test_non_english_and_short_books():
    german = wrap(" ".join(["Der Wanderer kam spät in das Dorf und fand die Tür der Herberge verschlossen."] * 400))
    assert verdict(book_stats(german, CFG), CFG) == "english"
    assert verdict(book_stats(novel(2), CFG), CFG) == "min_words"


def test_normalize_unwraps_prose_and_keeps_verse():
    stanza = "The wind is in the willow tree,\nThe rain is on the hill,\nAnd all the birds have gone to sea."
    text = wrap(NARRATION) + "\n\n" + stanza + "\n\n[Illustration: The inn at night]\n\nIt was _very_ late."
    n = normalize(text)
    paras = n.split("\n\n")
    assert paras[0] == NARRATION  # hard wraps joined
    assert paras[1] == stanza  # verse layout kept
    assert paras[2] == "It was very late."  # illustration tag and italics underscores gone
    assert "\n" not in normalize(wrap(NARRATION), unwrap=True) and "\n" in normalize(wrap(NARRATION), unwrap=False)


def test_segment_bounds_cut_at_paragraphs_and_respect_max():
    n, mx = 100_000, 32768
    cuts = np.arange(0, n, 997)
    b = segment_bounds(n, cuts, mx)
    assert b[0][0] == 0 and b[-1][1] == n and all(x[1] == y[0] for x, y in zip(b, b[1:]))
    assert all(0 < e - s <= mx for s, e in b)
    assert all(s in set(cuts.tolist()) for s, _ in b[1:])  # every internal cut is a paragraph start
    assert segment_bounds(100, cuts, mx) == [(0, 100)]
    hard = segment_bounds(n, np.array([], dtype=np.int64), mx)  # no paragraph starts at all: hard cuts, still <= max
    assert all(0 < e - s <= mx for s, e in hard) and hard[-1][1] == n


def _book(bid, split, dialogue, tokens, sha=None, **kw):
    st = {"book_id": bid, "fi": 0 if split == "train" else 1, "rg": 0, "row": int(bid), "words": 10_000, "stopword_share": 0.5,
          "caps_share": 0.0, "verse_share": 0.0, "archaic_per_1k": 0.0, "dialogue": dialogue, "sha1": sha or bid, "n_ids": tokens}
    st.update(kw)
    return st


def test_select_ranks_by_dialogue_until_target_and_val_uses_the_cutoff():
    cfg = GutenbergConfig(target_tokens=250, segment_tokens=1000)
    books = [_book("1", "train", 0.50, 100), _book("2", "train", 0.40, 100), _book("3", "train", 0.30, 100),
             _book("4", "train", 0.20, 100), _book("5", "train", 0.05, 100), _book("6", "train", 0.45, 100, archaic_per_1k=9.0),
             _book("7", "val", 0.35, 100), _book("8", "val", 0.10, 100), _book("9", "train", 0.60, 100, sha="7")]
    sel = G.select(books, {0: "train", 1: "validation"}, cfg, val_min_tokens=0)
    by = {b["book_id"]: b for b in books}
    assert [b for b in "12345679" if by[b]["selected"]] == ["1", "2", "3", "7"]  # 102+102+102 >= 250 after book 3
    assert sel["dialogue_cutoff"] == 0.30
    assert by["4"]["verdict"] == "not_selected" and by["5"]["verdict"] == "dialogue" and by["6"]["verdict"] == "archaic"
    assert by["9"]["verdict"] == "duplicate"  # duplicates a val book: val wins, so no train/val leak
    assert by["8"]["verdict"] == "not_selected"  # below the train cutoff
    G.select(books, {0: "train", 1: "validation"}, cfg, val_min_tokens=150)
    assert by["8"]["selected"]  # val floor: relax the cutoff until val has enough tokens


def test_prepare_end_to_end(tmp_path, monkeypatch):
    """Tiny raw dir -> shards + manifest with the filter block; the portal's normalizer reads it."""
    from slm.data.tokenizer import SlmTokenizer, train_bpe

    raw = tmp_path / "raw"
    monkeypatch.setattr(G, "raw_dir", lambda: raw)
    tok_dir = tmp_path / "tok"
    SlmTokenizer(train_bpe([novel(20), novel(5, dialogue=False)], vocab_size=600)).save(tok_dir)
    tok = SlmTokenizer.load(tok_dir)
    books = {"101": (1901, novel(60)), "102": (1840, novel(60)), "103": (1880, novel(60, dialogue=False)),
             "104": (1890, "\n\n".join(["HAMLET.\nWords, words, words, my lord, and nothing but the words."] * 400)),
             "105": (1910, novel(50)), "201": (1905, novel(30)), "301": (1899, novel(31))}
    splits = {"train": ["101", "102", "103", "104", "105"], "validation": ["201"], "test": ["301"]}
    (raw / "data").mkdir(parents=True)
    (raw / "metadata.csv").write_text("".join(f"{i},Book {i} by Someone,{y},http://www.gutenberg.org/ebooks/{i}\n" for i, (y, _) in books.items()), encoding="utf-8")
    for s, ids in splits.items():
        (raw / "data" / f"{s}_files.txt").write_text("".join(f"{s}/{i}.txt\n" for i in ids), encoding="utf-8")
        keep = [i for i in ids if books[i][0] >= 1850]
        (raw / s).mkdir()
        pq.write_table(pa.table({"book_id": keep, "short_book_title": [f"Book {i}" for i in keep], "publication_date": [books[i][0] for i in keep],
                                 "url": ["u"] * len(keep), "text": ["Produced by X\n\n" + books[i][1] + "\n\nEnd of the Project Gutenberg EBook of X\n"
                                                                     for i in keep]}), raw / s / f"{s}-00000.parquet", row_group_size=2)
    cfg = GutenbergConfig(min_words=500, target_tokens=1, segment_tokens=512)
    out = tmp_path / "tokenized"
    m = G.prepare(cfg, tok, out, str(tok_dir), workers=2)
    d = out / "gutenberg-pg19"
    assert json.loads((d / "manifest.json").read_text(encoding="utf-8")) == json.loads(json.dumps(m))
    f = m["filter"]
    rules = {r["rule"]: r for r in f["rules"]}
    assert rules["date"]["removed"] == 1 and rules["dialogue"]["removed"] == 1 and rules["caps"]["removed"] == 1
    assert f["books_in"] == {"train": 5, "validation": 1, "test": 1} and m["books"]["train"] == 1 and m["books"]["val"] == 2
    assert m["train_tokens"] > 0 and m["val_tokens"] > 0 and m["train_docs"] > 1  # the kept book is segmented
    ids = np.fromfile(d / "train" / "shard_00000.bin", dtype=np.uint16)
    starts = np.load(d / "train" / "shard_00000.idx.npy")
    assert len(ids) == m["train_tokens"] and all(ids[s] == tok.bos_id for s in starts)
    assert max(np.diff(np.append(starts, len(ids)))) <= cfg.segment_tokens + 2
    text = tok.decode(ids.tolist(), skip_special=True)
    assert "Produced by" not in text and "Project Gutenberg" not in text and "said the landlord" in text
    assert len((d / "books.jsonl").read_text(encoding="utf-8").splitlines()) == 6

    from slm.portal.services.datasets import _prov_line, normalize_manifest

    p = normalize_manifest("gutenberg-pg19", m)
    assert p["made_by"] == "prepare" and p["train_tokens"] == m["train_tokens"]
    assert "dialogue" in _prov_line("gutenberg-pg19", p)


def test_generic_prepare_refuses_the_custom_source():
    from slm.data.sources import SOURCES

    assert SOURCES["gutenberg-pg19"].custom_prepare == "slm.data.gutenberg"
    import sys

    from slm.data import prepare as P

    monkey_argv = ["prepare", "gutenberg-pg19", "--tokenizer", "C:/nowhere"]
    old = sys.argv
    sys.argv = monkey_argv
    try:
        with pytest.raises(SystemExit, match="slm.data.gutenberg"):
            P.main()
    finally:
        sys.argv = old


# ----------------------------------------------------------------------------------- gutenberg-canon
def _cbook(bid, title, n_ids=1000, **kw):
    b = _book(bid, "train", 0.05, n_ids, **kw)
    b.update(title=title, year=1900, pg19_split="train")
    return b


def test_canon_title_matching_folds_accents_punctuation_and_by():
    assert G.split_title("Ethan Frome By Edith Wharton") == ("Ethan Frome", "Edith Wharton")
    assert G.split_title("Plays by Anton Chekhov Second Series") == ("Plays", "Anton Chekhov Second Series")
    assert G.split_title("The Complete Memoires of Jacques Casanova") == ("The Complete Memoires of Jacques Casanova", "")
    assert G.fold("L'Assommoir") == "l assommoir" and G.fold("Thérèse Raquin") == "therese raquin"
    assert G.title_match("l'assommoir", "L'Assommoir") and G.title_match("therese raquin", "Therese Raquin")
    assert G.title_match("=soil", "Soil") and not G.title_match("=soil", "The Soil of France")
    assert G.title_match("salom|salome", "Salome") and not G.title_match("bovary", "Bovaryism")  # whole words
    assert G.volume_info("Sentimental Education Volume II") == ("sentimental education", 2, False)
    assert G.volume_info("In a Glass Darkly v. 1/3") == ("in a glass darkly", 1, False)
    assert G.volume_info("Fromont and Risler Vols. 1-4") == ("fromont and risler", None, True)
    assert G.volume_info("Original Short Stories Complete") == ("original short stories", None, True)
    assert G.volume_info("Home Life in Russia Volumes 1 and 2") == ("home life in russia", None, True)


def test_canon_dedupes_editions_and_never_repeats_gutenberg_pg19():
    cfg, cc = GutenbergConfig(), G.CanonConfig(val_share=0.0)
    books = [_cbook("1", "Madame Bovary by Gustave Flaubert", 100), _cbook("2", "Madame Bovary by Gustave Flaubert", 150),
             _cbook("3", "Sentimental Education Vol 1 by Gustave Flaubert"), _cbook("4", "Sentimental Education Volume II by Gustave Flaubert"),
             _cbook("5", "Original Short Stories Complete by Guy de Maupassant", 9000),
             _cbook("6", "Original Short Stories Vol. 11 of 13 by Guy de Maupassant", 700),
             _cbook("7", "Bel Ami by Guy de Maupassant"), _cbook("8", "Bel Ami by Guy de Maupassant", 2000),
             _cbook("9", "Ghosts A Domestic Tragedy by Henrik Ibsen", 300), _cbook("10", "Ghosts by Henrik Ibsen", 400),
             _cbook("11", "Madame Bovary: A Study by Somebody Else"),
             _cbook("12", "The Man That Corrupted Hadleyburg by Mark Twain"),
             _cbook("13", "The Man That Corrupted Hadleyburg and Other Stories by Mark Twain", 5000),
             _cbook("14", "Sentimental Education Vol 1 by Gustave Flaubert", sha="3")]  # same text as book 3, other id
    entries = (G.CanonEntry("t", "flaubert", ("madame bovary", "sentimental education")),
               G.CanonEntry("t", "maupassant", ("original short stories", "bel ami")),
               G.CanonEntry("t", "ibsen", ("=ghosts|ghosts a domestic tragedy",)),
               G.CanonEntry("t", "mark twain", ("corrupted hadleyburg",)),
               G.CanonEntry("t", "nobody", ("the lost novel",)))
    sel = G.canon_select(books, cfg, cc, pg19_ids={"7", "12"}, entries=entries)
    got = sorted(b["book_id"] for b in sel["books"])
    # longest Bovary; both volumes of the Education; the complete stories only; one Ghosts; no Bel Ami (another
    # edition is in gutenberg-pg19); no Hadleyburg collection (titled after a work gutenberg-pg19 holds); nothing by
    # the other author called "Madame Bovary"
    assert got == ["10", "2", "3", "4", "5"]
    st = {r["title"]: r["status"] for r in sel["report"]}
    assert st["bel ami"] == "already in gutenberg-pg19" and st["corrupted hadleyburg"] == "already in gutenberg-pg19"
    assert st["the lost novel"] == "not in PG-19"
    assert all(b["relaxed"] == ["dialogue"] for b in sel["books"])  # dialogue 0.05 < 0.10: the rule relaxed for them


def test_canon_relaxes_archaic_and_dialogue_only_in_canon_mode():
    cfg, cc = GutenbergConfig(), G.CanonConfig()
    st = {"words": 50_000, "stopword_share": 0.5, "caps_share": 0.01, "verse_share": 0.0, "archaic_per_1k": 2.5, "dialogue": 0.02}
    assert G.failing_rules(st, cfg) == ["archaic", "dialogue"]
    assert G.failing_rules(st, cfg, cc) == []
    assert G.failing_rules({**st, "archaic_per_1k": 3.5}, cfg, cc) == ["archaic"]  # relaxed, not removed
    assert G.failing_rules({**st, "verse_share": 0.5, "caps_share": 0.2}, cfg, cc) == ["caps", "verse"]  # hygiene unchanged
    assert verdict(st, cfg) == "archaic"  # gutenberg-pg19's verdict is untouched


def test_canon_play_format_passes_caps_only_as_a_play():
    cfg, cc = GutenbergConfig(), G.CanonConfig()
    speech = ["HEDDA.", "I really did not look closely at it, Miss Tesman, and I do not think I shall.",
              "MISS TESMAN.", "Let me tell you it's the first time I have worn it, the very first time, George.",
              "TESMAN.", "And a very nice bonnet it is too, quite a beauty, as anyone in the town would say."]
    play = "\n\n".join(speech * 60)
    st = {**book_stats(play, cfg), **G.play_stats(play, cfg, cc.min_speaker_share)}
    assert st["play"] and st["caps_share"] > cfg.max_caps_share and st["caps_share_play"] <= cfg.max_caps_share
    assert "caps" in G.failing_rules(st, cfg) and "caps" not in G.failing_rules(st, cfg, cc)
    headings = "\n\n".join(f"CHAPTER {'I' * (i % 7 + 1)}{'V' * (i // 7)}.\n\n{NARRATION}" for i in range(40))  # distinct headings
    assert not G.play_stats(headings, cfg, cc.min_speaker_share)["play"]


def test_canon_per_author_cap_and_val_split():
    cfg = GutenbergConfig()
    books = [_cbook(str(i), f"Novel Number {w} by Emile Zola", 1000 + i) for i, w in enumerate("one two three four five six seven eight".split(), 1)]
    books[0]["pg19_split"] = "validation"  # PG-19's own held-out book stays held out
    fill = (G.CanonEntry("t", r"\bzola\b"),)
    sel = G.canon_select(books, cfg, G.CanonConfig(per_author_cap=3, val_share=0.0), set(), fill)
    assert [b["book_id"] for b in sel["books"]] == ["1", "2", "3"]  # no titles: PG-19 id order until the cap
    titled = (G.CanonEntry("t", r"\bzola\b", ("number eight", "number two", "number five"), cap=2),)
    sel = G.canon_select(books, cfg, G.CanonConfig(val_share=0.0), set(), titled)
    assert [b["book_id"] for b in sel["books"]] == ["8", "2"]  # list order; the entry's own cap wins
    assert sel["report"][-1]["status"] == "over the entry's cap (2)"
    sel = G.canon_select(books, cfg, G.CanonConfig(per_author_cap=8, val_share=0.25, seed=0), set(), fill)
    val = [b["book_id"] for b in sel["books"] if b["split"] == "val"]
    assert "1" in val and len(val) == 2  # 25% of 8 books, whole books


def test_canon_prepare_end_to_end(tmp_path, monkeypatch):
    """gutenberg-pg19 first, then the canon beside it: the canon never repeats a pg19 book, the manifest lists
    every book with the relaxed rule, and the portal names it a curated supplement."""
    from slm.data.tokenizer import SlmTokenizer, train_bpe

    raw = tmp_path / "raw"
    monkeypatch.setattr(G, "raw_dir", lambda: raw)
    tok_dir = tmp_path / "tok"
    SlmTokenizer(train_bpe([novel(20), novel(5, dialogue=False)], vocab_size=600)).save(tok_dir)
    tok = SlmTokenizer.load(tok_dir)
    books = {"101": ("Chatty Book by Anybody", 1901, novel(60)), "102": ("Quiet Book by Emile Zola", 1880, novel(60, dialogue=False)),
             "103": ("Another Quiet Book by Emile Zola", 1885, novel(40, dialogue=False)),
             "104": ("A Play by Henrik Ibsen", 1890, "\n\n".join(["NORA.\n\nWords, words, words, my lord, and nothing but the words today.",
                                                               "HELMER.\n\nThen say them again, and say them slowly, so that I can hear."] * 200)),
             "201": ("Held Out by Emile Zola", 1905, novel(30, dialogue=False))}
    splits = {"train": ["101", "102", "103", "104"], "validation": ["201"], "test": []}
    (raw / "data").mkdir(parents=True)
    (raw / "metadata.csv").write_text("".join(f"{i},{t},{y},u\n" for i, (t, y, _) in books.items()) + "999,Nana by Emile Zola,1840,u\n", encoding="utf-8")
    for s, ids in splits.items():
        (raw / "data" / f"{s}_files.txt").write_text("".join(f"{s}/{i}.txt\n" for i in ids), encoding="utf-8")
        (raw / s).mkdir()
        if ids:
            pq.write_table(pa.table({"book_id": ids, "short_book_title": [books[i][0] for i in ids], "publication_date": [books[i][1] for i in ids],
                                     "url": ["u"] * len(ids), "text": [books[i][2] for i in ids]}), raw / s / f"{s}-00000.parquet", row_group_size=2)
    cfg = GutenbergConfig(min_words=500, target_tokens=1, segment_tokens=512)
    out = tmp_path / "tokenized"
    G.prepare(cfg, tok, out, str(tok_dir), workers=1, val_min_tokens=0)
    entries = (G.CanonEntry("t", r"\bzola\b", ("=quiet book", "another quiet book", "held out", "=nana")), G.CanonEntry("t", "ibsen", ("a play",)),
               G.CanonEntry("t", "anybody", ("chatty book",)))
    m = G.prepare_canon(cfg, G.CanonConfig(val_share=0.0), tok, out, str(tok_dir), workers=1, entries=entries)
    d = out / "gutenberg-canon"
    assert json.loads((d / "manifest.json").read_text(encoding="utf-8")) == json.loads(json.dumps(m))
    c = m["canon"]
    got = {b["book_id"]: b for b in c["books"]}
    assert sorted(got) == ["102", "103", "104", "201"] and got["201"]["split"] == "val"  # PG-19 val stays val
    assert got["102"]["relaxed"] == ["dialogue"] and "caps (play format)" in got["104"]["relaxed"]
    st = {r["title"]: r["status"] for r in c["report"]}
    assert st["chatty book"] == "already in gutenberg-pg19"
    assert st["=nana"].startswith("not downloaded: PG-19 dates it before 1850")
    assert m["books"]["train"] == 3 and m["books"]["val"] == 1 and m["train_tokens"] > 0 and m["val_tokens"] > 0
    ids = np.fromfile(d / "train" / "shard_00000.bin", dtype=np.uint16)
    starts = np.load(d / "train" / "shard_00000.idx.npy")
    assert len(ids) == m["train_tokens"] and all(ids[s] == tok.bos_id for s in starts)
    assert max(np.diff(np.append(starts, len(ids)))) <= cfg.segment_tokens + 2

    from slm.portal.services.datasets import _prov_line, normalize_manifest

    p = normalize_manifest("gutenberg-canon", m)
    assert p["from"] == ["gutenberg-pg19"] and p["train_tokens"] == m["train_tokens"]
    assert _prov_line("gutenberg-canon", p) == "curated supplement: 4 books, adult literary canon past the dialogue rule"

"""PG-19 (Project Gutenberg books published before 1919) as a narrative-prose pretraining source.

The Hub repo `deepmind/pg19` holds no text: it is a loading script plus the split lists
(`data/{train,validation,test}_files.txt`); the books and `metadata.csv` live in the release's public
GCS bucket. Acquisition is therefore two passes, like python-edu's Software Heritage pass:

    python -m slm.data.download gutenberg-pg19 --all            # the three split lists from the Hub
    python -m slm.data.gutenberg download                       # metadata + books (publication_date >= min_year) -> parquet
    python -m slm.data.gutenberg prepare --tokenizer C:/slm-data/tokenizer/v1 --dry-run   # selection report only
    python -m slm.data.gutenberg prepare --tokenizer C:/slm-data/tokenizer/v1             # shards + manifest
    python -m slm.data.gutenberg canon --tokenizer C:/slm-data/tokenizer/v1 [--dry-run]   # gutenberg-canon supplement

Every threshold is a `GutenbergConfig` field and can be overridden as `key=value` on the command line
(`canon.<field>=value` for `CanonConfig`). See docs/design.md §4 for the rules and why each exists.

`canon` builds `gutenberg-canon`, a small curated supplement: the books on the `CANON` list (adult literary
canon: naturalism, the Russians, the Victorian sensation novel, the decadents, the problem play) that the
dialogue-density ranking left out. They pass the same hygiene rules except that the dialogue rule is not applied,
the archaic threshold is relaxed (translations and Hardy run higher) and a play in speaker-line format is not
counted as ALL-CAPS noise. Books already in `gutenberg-pg19` are never repeated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from slm.config import apply_overrides, from_dict
from slm.data.sources import SOURCES, TOKENIZED_DIR

SOURCE_NAME = "gutenberg-pg19"
ASSET_ROOT = "https://storage.googleapis.com/deepmind-gutenberg/"
SPLITS = ("train", "validation", "test")
VAL_SPLITS = ("validation", "test")  # PG-19's own held-out books become our val split


@dataclass
class GutenbergConfig:
    min_year: int = 1850  # publication_date (PG-19 metadata); applied before download
    min_words: int = 2000  # fragments, pamphlets
    min_stopword_share: float = 0.30  # English backstop: share of words in a small English function-word list
    max_caps_share: float = 0.06  # share of non-empty lines that are ALL-CAPS or table-like
    verse_line_chars: int = 55  # a line shorter than this is "short"
    verse_run: int = 4  # ... and a run of this many consecutive short lines is verse
    max_verse_share: float = 0.15  # share of non-empty lines inside verse runs
    max_archaic_per_1k: float = 1.5  # thee/thou/thy/thine/hath/doth/dost/hast/shalt per 1K words
    min_dialogue: float = 0.10  # share of non-empty lines with a quotation mark
    target_tokens: float = 5e8  # candidates sorted by dialogue density, taken until this many train tokens
    segment_tokens: int = 32768  # books are cut at paragraph boundaries into documents of at most this many tokens
    min_doc_tokens: int = 16
    unwrap: bool = True  # join PG's hard-wrapped (~70 char) lines inside prose paragraphs
    seed: int = 0  # write order of the selected books (never density order)


@dataclass
class CanonConfig:
    """gutenberg-canon: which rules are relaxed for the curated list, and how the set is split."""
    max_archaic_per_1k: float = 3.0  # instead of 1.5: translations (Flaubert, Zola), Hardy and Adam Bede run higher
    per_author_cap: int = 6  # at most this many books per list entry unless the entry sets its own cap
    min_speaker_share: float = 0.04  # a play: >= this share of lines are speaker names (see play_stats)
    val_share: float = 0.02  # whole books held out (PG-19's own validation/test books first, then seeded draws)
    seed: int = 0  # val draws and write order


# ----------------------------------------------------------------------------------- download
def raw_dir() -> Path:
    return SOURCES[SOURCE_NAME].local_dir


def split_ids(split: str, root: Path | None = None) -> list[str]:
    """Book ids of one PG-19 split, from the Hub's `data/<split>_files.txt` (fetched by slm.data.download)."""
    p = (root or raw_dir()) / "data" / f"{split}_files.txt"
    if not p.exists():
        from huggingface_hub import hf_hub_download

        p = Path(hf_hub_download(SOURCES[SOURCE_NAME].repo, f"data/{split}_files.txt", repo_type="dataset", local_dir=str(root or raw_dir())))
    ids = [Path(line.strip()).stem for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(ids, key=lambda x: (len(x), x))  # numeric order without assuming every id is an int


def read_metadata(root: Path | None = None) -> dict[str, dict]:
    p = (root or raw_dir()) / "metadata.csv"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ASSET_ROOT + "metadata.csv", p)
    out = {}
    with open(p, encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                out[row[0]] = {"short_book_title": row[1], "publication_date": int(row[2]), "url": row[3]}
    return out


def eligible(ids: list[str], meta: dict[str, dict], min_year: int) -> list[str]:
    """The date rule, applied before download: books whose PG-19 publication_date is >= min_year."""
    return [i for i in ids if meta[i]["publication_date"] >= min_year]


def _fetch(url: str, tries: int = 5) -> str:
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 -- transient network errors; the last one is raised
            if k == tries - 1:
                raise
            time.sleep(2 ** k)
    raise RuntimeError("unreachable")


def download(min_year: int, workers: int = 24, chunk_books: int = 1000, rg_books: int = 50) -> None:
    """Fetch every book of every split with publication_date >= min_year into
    raw/gutenberg-pg19/<split>/<split>-NNNNN.parquet (chunk_books books per file, rg_books per row group).
    Chunk membership is fixed by the sorted id list, so a rerun skips finished chunks."""
    meta = read_metadata()
    root = raw_dir()
    summary = {"asset_root": ASSET_ROOT, "min_year": min_year, "splits": {}}
    for split in SPLITS:
        ids = split_ids(split)
        keep = eligible(ids, meta, min_year)
        summary["splits"][split] = {"books": len(ids), "downloaded": len(keep), "skipped_by_date": len(ids) - len(keep)}
        out = root / split
        out.mkdir(parents=True, exist_ok=True)
        chunks = [keep[i : i + chunk_books] for i in range(0, len(keep), chunk_books)]
        t0, n_bytes = time.time(), 0
        for k, chunk in enumerate(chunks):
            p = out / f"{split}-{k:05d}.parquet"
            if p.exists():
                continue
            with ThreadPoolExecutor(workers) as ex:
                texts = list(ex.map(lambda i: _fetch(f"{ASSET_ROOT}{split}/{i}.txt"), chunk))
            n_bytes += sum(len(t) for t in texts)
            tbl = pa.table({
                "book_id": chunk, "short_book_title": [meta[i]["short_book_title"] for i in chunk],
                "publication_date": pa.array([meta[i]["publication_date"] for i in chunk], pa.int32()),
                "url": [meta[i]["url"] for i in chunk], "text": texts,
            })
            tmp = p.with_suffix(".tmp")
            pq.write_table(tbl, tmp, row_group_size=rg_books, compression="zstd")
            tmp.replace(p)
            el = time.time() - t0
            print(f"[{split}] chunk {k + 1}/{len(chunks)}: {len(chunk)} books, {n_bytes / 1e9:.2f} GB this session, {n_bytes / 1e6 / max(el, 1e-9):.1f} MB/s", flush=True)
    (root / "download.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


# ----------------------------------------------------------------------------------- cleaning
_START = re.compile(r"^\*{3}\s*START OF (?:THE |THIS )?PROJECT GUTENBERG.*$", re.M | re.I)
_END = re.compile(r"^\*{3}\s*END OF (?:THE |THIS )?PROJECT GUTENBERG.*$", re.M | re.I)
# PG-19 removed the licence but kept the closing line ("End of the Project Gutenberg EBook of X, by Y", or
# "End of Project Gutenberg's X") and whatever followed it (usually a stray "***").
_END_LINE = re.compile(r"^[ \t*]*End of (?:the |this )?Project Gutenberg.*$", re.M | re.I)
_CREDIT = re.compile(r"\s*(?:Produced by|E-?text prepared by|This e-?text was|Transcribed (?:from|by)|Prepared by|"
                     r"Scanned by|Distributed Proofread|Proofreading Team|Online Distributed|\[?Transcriber'?s? note)", re.I)
_ILLUSTRATION = re.compile(r"\[Illustration(?:[:.][^\]]{0,500})?\]", re.I)
_ITALIC = re.compile(r"(?<![\w_])_([^_\n]+(?:\n[^_\n]+){0,3})_(?![\w_])")


def strip_boilerplate(text: str) -> str:
    """Remove Project Gutenberg framing: START/END markers when present (full PG files), PG-19's surviving
    "End of the Project Gutenberg ..." line and everything after it, and production-credit / transcriber
    paragraphs in the first 3,000 characters. The book's own text is left as released."""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    if m := _START.search(t):
        t = t[m.end():]
    if m := _END.search(t):
        t = t[: m.start()]
    ends = [m for m in _END_LINE.finditer(t) if m.start() > len(t) // 2]
    if ends:
        t = t[: ends[0].start()]
    head, rest = t[:3000], t[3000:]
    paras = re.split(r"(\n[ \t]*\n)", head)
    head = "".join(p for p in paras if not _CREDIT.match(p))
    t = (head + rest).strip("\n")
    return re.sub(r"(?:\n[ \t*]*)+$", "", t)  # trailing lines of asterisks


def normalize(text: str, unwrap: bool = True, verse_line_chars: int = 55) -> str:
    """Text as the model will see it: illustration tags and _italic_ underscores removed, and (unwrap) the
    hard line wraps inside prose paragraphs joined, so fiction does not teach a newline every ~70 characters.
    A paragraph whose lines are mostly short (verse, letters, lists, tables) keeps its line breaks."""
    t = _ILLUSTRATION.sub("", text)
    t = _ITALIC.sub(r"\1", t)
    if not unwrap:
        return re.sub(r"\n{3,}", "\n\n", t).strip()
    out = []
    for para in re.split(r"\n[ \t]*\n", t):
        lines = [ln.rstrip() for ln in para.split("\n") if ln.strip()]
        if not lines:
            continue
        short = sum(len(ln.strip()) < verse_line_chars for ln in lines[:-1])
        if len(lines) > 1 and short > (len(lines) - 1) / 2:
            out.append("\n".join(lines))  # verse-like: keep the layout
        else:
            out.append(" ".join(ln.strip() for ln in lines))
    return "\n\n".join(out)


# ----------------------------------------------------------------------------------- statistics
_QUOTE = re.compile(r"[\"“”]|(?:^|\s)'(?=[A-Z])")  # double quotes, or a single quote opening a capitalised word
_ARCHAIC = re.compile(r"\b(?:thee|thou|thy|thine|hath|doth|dost|hast|shalt)\b", re.I)
_WORD = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")
_TABLE = re.compile(r"\S(?: {3,}|\t)\S.*\S(?: {3,}|\t)\S|\|.*\||\.{5,}|(?:\. ){4,}")
STOPWORDS = frozenset(
    "the of and to a in that is was he it for with as his on be at by i had not are but from or have an they which "
    "you were her all she there would their we him been has when who will more no if out so said what up its about "
    "into than them can only other then do any my now over such our me even most made after also did many before "
    "must through back where much your way well down should because each just those how too little very make still "
    "see own here both between being under never same another know while last might us".split())


def book_stats(text: str, cfg: GutenbergConfig) -> dict:
    """Cheap per-book signals, measured on the stripped text *before* unwrapping (line structure matters)."""
    lines = text.split("\n")
    nonempty = [ln for ln in lines if ln.strip()]
    n = max(1, len(nonempty))
    dialogue = sum(1 for ln in nonempty if _QUOTE.search(ln)) / n
    caps = 0
    for ln in nonempty:
        s = ln.strip()
        n_letters = sum(c.isalpha() for c in s)
        if (n_letters >= 4 and not any(c.islower() for c in s)) or _TABLE.search(s) or (len(s) >= 8 and n_letters < 0.4 * len(s)):
            caps += 1
    # verse: runs of >= verse_run consecutive non-empty short lines (prose wraps at ~70 chars; a paragraph's
    # short last line is followed by a blank line, so it rarely forms a run)
    in_verse, run = 0, 0
    for ln in lines + [""]:
        s = ln.strip()
        if s and len(s) < cfg.verse_line_chars:
            run += 1
            continue
        if run >= cfg.verse_run:
            in_verse += run
        run = 0
    words = _WORD.findall(text)
    n_words = len(words)
    sample = [w.lower() for w in words[:50000]]
    return {
        "chars": len(text), "words": n_words, "lines": len(nonempty),
        "dialogue": round(dialogue, 4), "caps_share": round(caps / n, 4), "verse_share": round(in_verse / n, 4),
        "archaic_per_1k": round(len(_ARCHAIC.findall(text)) * 1000 / max(1, n_words), 3),
        "stopword_share": round(sum(w in STOPWORDS for w in sample) / max(1, len(sample)), 4),
        "sha1": hashlib.sha1(re.sub(r"\s+", " ", text).encode("utf-8", errors="ignore")).hexdigest(),
    }


RULES = ("min_words", "english", "duplicate", "caps", "verse", "archaic", "dialogue")


def failing_rules(st: dict, cfg: GutenbergConfig, canon: CanonConfig | None = None) -> list[str]:
    """Every rule a book fails, in RULES order (the first one is its verdict). `duplicate` is set by the caller.

    With `canon` (the gutenberg-canon supplement) the dialogue rule is not applied, the archaic threshold is
    `canon.max_archaic_per_1k`, and a play in speaker-line format (`st["play"]`, see `play_stats`) is judged on its
    caps share without the speaker-name lines. Every other rule is unchanged."""
    caps = st["caps_share_play"] if canon is not None and st.get("play") else st["caps_share"]
    archaic = canon.max_archaic_per_1k if canon is not None else cfg.max_archaic_per_1k
    checks = {"min_words": st["words"] < cfg.min_words, "english": st["stopword_share"] < cfg.min_stopword_share,
              "duplicate": bool(st.get("duplicate")), "caps": caps > cfg.max_caps_share,
              "verse": st["verse_share"] > cfg.max_verse_share, "archaic": st["archaic_per_1k"] > archaic,
              "dialogue": canon is None and st["dialogue"] < cfg.min_dialogue}
    return [r for r in RULES if checks[r]]


def verdict(st: dict, cfg: GutenbergConfig) -> str | None:
    """First rule the book fails, or None if it is a candidate."""
    f = failing_rules(st, cfg)
    return f[0] if f else None


# ----------------------------------------------------------------------------------- segmentation
def segment_bounds(n_tokens: int, cut_tokens: np.ndarray, max_tokens: int) -> list[tuple[int, int]]:
    """Split [0, n_tokens) into ceil(n / (0.9 max)) near-equal pieces, each cut moved to the nearest paragraph
    start (`cut_tokens`, sorted token indices) within +-5% of max, else cut hard. Every piece is <= max_tokens."""
    if n_tokens <= max_tokens:
        return [(0, n_tokens)]
    k = -(-n_tokens // int(0.9 * max_tokens))
    slack = max_tokens // 20
    bounds, prev = [], 0
    for j in range(1, k):
        t = round(j * n_tokens / k)
        lo, hi = max(prev + 1, t - slack), min(n_tokens - 1, t + slack, prev + max_tokens)
        i = int(np.searchsorted(cut_tokens, t))
        best = None
        for cand in (cut_tokens[i - 1] if i > 0 else None, cut_tokens[i] if i < len(cut_tokens) else None):
            if cand is not None and lo <= cand <= hi and (best is None or abs(int(cand) - t) < abs(best - t)):
                best = int(cand)
        c = best if best is not None else min(max(t, lo), hi)
        bounds.append((prev, c))
        prev = c
    bounds.append((prev, n_tokens))
    return bounds


def tokenize_book(tok, text: str, max_tokens: int) -> list[np.ndarray]:
    """One tokenization of the whole book, cut into <= max_tokens pieces at paragraph starts."""
    enc = tok.tok.encode(text, add_special_tokens=False)
    ids = np.asarray(enc.ids, dtype=np.uint16)
    if len(ids) <= max_tokens:
        return [ids]
    starts = np.asarray([o[0] for o in enc.offsets], dtype=np.int64)
    para = np.asarray([m.end() for m in re.finditer(r"\n[ \t]*\n\s*", text)], dtype=np.int64)
    cut = np.unique(np.clip(np.searchsorted(starts, para, side="right") - 1, 1, len(ids) - 1))
    return [ids[a:b] for a, b in segment_bounds(len(ids), cut, max_tokens)]


def n_doc_tokens(n_ids: int, max_tokens: int) -> int:
    """Tokens a book occupies on disk: its ids plus <|bos|>/<|eos|> around each segment."""
    k = 1 if n_ids <= max_tokens else -(-n_ids // int(0.9 * max_tokens))
    return n_ids + 2 * k


# ----------------------------------------------------------------------------------- stats pass (cached)
_TOK = None


def _stats_task(args: tuple) -> list[dict]:
    """Worker: stats + exact token count for every book of one row group (texts never cross processes)."""
    global _TOK
    path, fi, rg, cfg_d, tok_dir = args
    cfg = GutenbergConfig(**cfg_d)
    if _TOK is None:
        from slm.data.tokenizer import SlmTokenizer

        _TOK = SlmTokenizer.load(tok_dir)
    rows = pq.ParquetFile(path).read_row_group(rg, columns=["book_id", "short_book_title", "publication_date", "text"]).to_pylist()
    out = []
    for i, r in enumerate(rows):
        t = strip_boilerplate(r["text"])
        st = book_stats(t, cfg)
        n_ids = len(_TOK.tok.encode(normalize(t, cfg.unwrap, cfg.verse_line_chars), add_special_tokens=False).ids)
        st.update(book_id=r["book_id"], title=r["short_book_title"], year=int(r["publication_date"]), fi=fi, rg=rg, row=i,
                  raw_chars=len(r["text"]), n_ids=n_ids)
        out.append(st)
    return out


def book_table(files: list[Path], cfg: GutenbergConfig, tok, tok_dir: str, workers: int) -> list[dict]:
    """Per-book stats for every downloaded book, cached in raw/gutenberg-pg19/book_stats.json. The cache is keyed
    by the settings the stats depend on (line thresholds, unwrap, tokenizer) and by the file list."""
    key = {"verse_line_chars": cfg.verse_line_chars, "verse_run": cfg.verse_run, "unwrap": cfg.unwrap,
           "tokenizer_sha256": tok.sha256, "files": [f"{p.parent.name}/{p.name}:{p.stat().st_size}" for p in files], "v": 1}
    cache = raw_dir() / "book_stats.json"
    if cache.exists():
        c = json.loads(cache.read_text(encoding="utf-8"))
        if c.get("key") == key:
            print(f"[{SOURCE_NAME}] stats: cached ({len(c['books'])} books)", flush=True)
            return c["books"]
    from multiprocessing import Pool

    tasks = [(str(p), fi, rg, asdict(cfg), tok_dir) for fi, p in enumerate(files) for rg in range(pq.ParquetFile(p).num_row_groups)]
    books, t0 = [], time.time()
    with Pool(workers) as pool:
        for k, res in enumerate(pool.imap_unordered(_stats_task, tasks)):
            books += res
            if (k + 1) % 50 == 0 or k + 1 == len(tasks):
                print(f"[{SOURCE_NAME}] stats: {k + 1}/{len(tasks)} row groups, {len(books)} books, {time.time() - t0:.0f}s", flush=True)
    books.sort(key=lambda b: (b["fi"], b["rg"], b["row"]))
    cache.write_text(json.dumps({"key": key, "books": books}), encoding="utf-8")
    return books


# ----------------------------------------------------------------------------------- selection
RULE_TEXT = {
    "date": "publication_date >= min_year (applied before download)",
    "min_words": "at least min_words words after stripping",
    "english": "share of words in a small English function-word list >= min_stopword_share",
    "duplicate": "not an exact duplicate (whitespace-normalised) of an earlier book; val books win over train",
    "caps": "share of ALL-CAPS / table-like / mostly-non-letter lines <= max_caps_share (plays, indexes, tables)",
    "verse": "share of lines inside runs of >= verse_run lines shorter than verse_line_chars <= max_verse_share",
    "archaic": "thee/thou/thy/thine/hath/doth/dost/hast/shalt per 1K words <= max_archaic_per_1k",
    "dialogue": "share of lines with a quotation mark >= min_dialogue",
    "not_selected": "passed every rule but fell below the dialogue-density cutoff once target_tokens was reached",
}


def select(books: list[dict], split_of_file: dict[int, str], cfg: GutenbergConfig, val_min_tokens: float) -> dict:
    """Apply the rules, rank train candidates by dialogue density, take them until target_tokens; val books
    (PG-19 validation + test) pass the same rules and the same density cutoff. Mutates `books` (verdict fields)."""
    order = sorted(books, key=lambda b: (split_of_file[b["fi"]] == "train", b["fi"], b["rg"], b["row"]))  # val books first: they win duplicates
    seen: set[str] = set()
    for b in order:
        b["split"] = "val" if split_of_file[b["fi"]] in VAL_SPLITS else "train"
        b["duplicate"] = b["sha1"] in seen
        seen.add(b["sha1"])
        b["fails"] = failing_rules(b, cfg)
        b["verdict"] = b["fails"][0] if b["fails"] else None
        b["tokens"] = n_doc_tokens(b["n_ids"], cfg.segment_tokens)
        b["selected"] = False
    rank = lambda b: (-b["dialogue"], len(b["book_id"]), b["book_id"])  # noqa: E731
    cands = sorted((b for b in books if b["split"] == "train" and b["verdict"] is None), key=rank)
    acc, cutoff = 0, None
    for b in cands:
        if acc >= cfg.target_tokens:
            b["verdict"] = "not_selected"
            continue
        b["selected"] = True
        acc += b["tokens"]
        cutoff = b["dialogue"]
    vc = sorted((b for b in books if b["split"] == "val" and b["verdict"] is None), key=rank)
    vacc = 0
    for b in vc:
        if (cutoff is not None and b["dialogue"] >= cutoff) or vacc < val_min_tokens:
            b["selected"] = True
            vacc += b["tokens"]
        else:
            b["verdict"] = "not_selected"
    return {"dialogue_cutoff": cutoff, "train_candidates": len(cands), "val_candidates": len(vc)}


def filter_summary(books: list[dict], cfg: GutenbergConfig, sel: dict, meta: dict) -> dict:
    """The manifest's `filter` block: thresholds, the funnel, and per rule how many books it removed (first
    failing rule, which sums to the total) and how many fail it at all, with example titles."""
    ids = {s: split_ids(s) for s in SPLITS}
    books_in = {s: len(v) for s, v in ids.items()}
    by_date = {s: len(v) - len(eligible(v, meta, cfg.min_year)) for s, v in ids.items()}
    old = sorted(set(ids["train"]) - set(eligible(ids["train"], meta, cfg.min_year)), key=lambda i: (len(i), i))
    tr = [b for b in books if b["split"] == "train"]
    thresholds = {"date": cfg.min_year, "min_words": cfg.min_words, "english": cfg.min_stopword_share, "duplicate": None,
                  "caps": cfg.max_caps_share, "verse": cfg.max_verse_share, "archaic": cfg.max_archaic_per_1k,
                  "dialogue": cfg.min_dialogue, "not_selected": sel["dialogue_cutoff"]}
    rules = [{"rule": "date", "threshold": cfg.min_year, "text": RULE_TEXT["date"], "removed": by_date["train"],
              "removed_val": by_date["validation"] + by_date["test"], "fails": by_date["train"],
              "examples": [f"{meta[i]['short_book_title']} ({meta[i]['publication_date']})" for i in old[:: max(1, len(old) // 6)][:6]]}]
    for r in (*RULES, "not_selected"):
        first = [b for b in tr if b["verdict"] == r]
        rules.append({"rule": r, "threshold": thresholds[r], "text": RULE_TEXT[r], "removed": len(first),
                      "removed_val": sum(1 for b in books if b["split"] == "val" and b["verdict"] == r),
                      "fails": len(first) if r == "not_selected" else sum(1 for b in tr if r in b["fails"]),
                      "removed_tokens": sum(b["tokens"] for b in first),
                      "examples": [f"{b['title']} ({b['year']})" for b in sorted(first, key=lambda b: b["book_id"])[:: max(1, len(first) // 6)][:6]]})
    kept = sorted((b for b in tr if b["selected"]), key=lambda b: -b["dialogue"])
    return {
        "config": asdict(cfg), "rules": rules, "books_in": books_in, "downloaded": {"train": len(tr), "val": len(books) - len(tr)},
        "candidates": sel["train_candidates"], "selected": {"train": len(kept), "val": sum(1 for b in books if b["split"] == "val" and b["selected"])},
        "dialogue_cutoff": sel["dialogue_cutoff"],
        "kept_examples": [f"{b['title']} ({b['year']})" for b in kept[:: max(1, len(kept) // 12)][:12]],
        "val_rule": "PG-19's validation + test books through the same rules and dialogue cutoff (never trained on)",
    }


# ----------------------------------------------------------------------------------- tokenize + write
def _tok_task(args: tuple) -> list[tuple[str, list[np.ndarray]]]:
    global _TOK
    path, rg, rows, cfg_d, tok_dir = args
    cfg = GutenbergConfig(**cfg_d)
    if _TOK is None:
        from slm.data.tokenizer import SlmTokenizer

        _TOK = SlmTokenizer.load(tok_dir)
    tbl = pq.ParquetFile(path).read_row_group(rg, columns=["book_id", "text"]).to_pylist()
    return [(tbl[i]["book_id"], tokenize_book(_TOK, normalize(strip_boilerplate(tbl[i]["text"]), cfg.unwrap, cfg.verse_line_chars), cfg.segment_tokens)) for i in rows]


def prepare(cfg: GutenbergConfig, tok, out_root: Path, tok_dir: str, dry_run: bool = False, workers: int = 16,
            val_min_tokens: float = 2e6) -> dict:
    from slm.data.prepare import SHARD_TOKENS, ShardWriter

    t0 = time.time()
    root = raw_dir()
    files = sorted(p for s in SPLITS for p in (root / s).glob(f"{s}-*.parquet"))
    assert files, f"no parquet under {root}; run `python -m slm.data.gutenberg download` first"
    meta = read_metadata()
    for s in SPLITS:  # a partial download would silently shrink the candidate pool
        want = len(eligible(split_ids(s), meta, cfg.min_year))
        have = sum(pq.ParquetFile(p).metadata.num_rows for p in files if p.parent.name == s)
        assert have == want, f"{s}: {have} books on disk, {want} expected for min_year {cfg.min_year}; rerun download"
    split_of_file = {fi: p.parent.name for fi, p in enumerate(files)}
    books = book_table(files, cfg, tok, tok_dir, workers)
    sel = select(books, split_of_file, cfg, val_min_tokens)
    summary = filter_summary(books, cfg, sel, meta)
    chosen = [b for b in books if b["selected"]]
    tr = [b for b in chosen if b["split"] == "train"]
    va = [b for b in chosen if b["split"] == "val"]
    est = {"train_tokens": sum(b["tokens"] for b in tr), "val_tokens": sum(b["tokens"] for b in va)}
    print(f"[{SOURCE_NAME}] selected {len(tr)} train books ({est['train_tokens'] / 1e6:.1f}M tokens) and {len(va)} val books "
          f"({est['val_tokens'] / 1e6:.2f}M); dialogue cutoff {sel['dialogue_cutoff']}", flush=True)
    for r in summary["rules"]:
        print(f"  {r['rule']:13s} removed {r['removed']:6d} train / {r['removed_val']:3d} val   (fails: {r['fails']})", flush=True)
    out = out_root / SOURCE_NAME
    if dry_run:
        rep = root / "selection_dryrun.json"
        rep.write_text(json.dumps({"summary": summary, "estimate": est, "books": books}, indent=0), encoding="utf-8")
        print(f"[{SOURCE_NAME}] dry run: report in {rep}")
        return {"filter": summary, **est}

    # read in file order (each row group once), write in a seeded shuffle: the stream must not run from the
    # most to the least dialogue-dense book, and a book's segments stay together
    from multiprocessing import Pool

    by_rg: dict[tuple[int, int], list[int]] = {}
    for b in chosen:
        by_rg.setdefault((b["fi"], b["rg"]), []).append(b["row"])
    tasks = [(str(files[fi]), rg, rows, asdict(cfg), tok_dir) for (fi, rg), rows in sorted(by_rg.items())]
    segs: dict[str, list[np.ndarray]] = {}
    with Pool(workers) as pool:
        for k, res in enumerate(pool.imap_unordered(_tok_task, tasks)):
            segs.update(res)
            if (k + 1) % 50 == 0 or k + 1 == len(tasks):
                print(f"[{SOURCE_NAME}] tokenize: {k + 1}/{len(tasks)} row groups, {len(segs)} books, {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(cfg.seed)
    train_order = [tr[i] for i in rng.permutation(len(tr))]
    val_order = sorted(va, key=lambda b: (len(b["book_id"]), b["book_id"]))
    writers = {"train": ShardWriter(out / "train", SHARD_TOKENS), "val": ShardWriter(out / "val", SHARD_TOKENS // 10)}
    n_short = 0
    for split, order in (("train", train_order), ("val", val_order)):
        w = writers[split]
        for b in order:
            b["segments"] = 0
            for s in segs.pop(b["book_id"]):
                if len(s) < cfg.min_doc_tokens:
                    n_short += 1
                    continue
                w.add(np.concatenate([[tok.bos_id], s, [tok.eos_id]]).astype(np.uint16), (b["fi"], b["rg"], b["row"]))
                b["segments"] += 1
        w.flush()
    trw, vaw = writers["train"], writers["val"]
    manifest = {
        "source": SOURCE_NAME, "name": SOURCE_NAME, "kind": "prose", "tokenizer_sha256": tok.sha256, "min_doc_tokens": cfg.min_doc_tokens,
        "train_tokens": trw.total_tokens, "train_docs": trw.total_docs, "train_shards": trw.shard_idx,
        "val_tokens": vaw.total_tokens, "val_docs": vaw.total_docs, "val_shards": vaw.shard_idx,
        "docs_seen": len(books), "docs_dropped": len(books) - len(chosen), "segments_too_short": n_short,
        "books": {"train": len(tr), "val": len(va), "mean_tokens_per_book": trw.total_tokens / max(1, len(tr)),
                  "segment_tokens": cfg.segment_tokens, "doc_unit": "book segment (<= segment_tokens, cut at paragraph starts)"},
        "filter": summary, "files": [str(p) for p in files], "sidecar": "src", "seconds": time.time() - t0,
    }
    keep = ("book_id", "title", "year", "split", "words", "dialogue", "caps_share", "verse_share", "archaic_per_1k",
            "stopword_share", "n_ids", "tokens", "verdict", "fails", "selected", "segments", "fi", "rg", "row")
    with open(out / "books.jsonl", "w", encoding="utf-8") as f:
        for b in books:
            f.write(json.dumps({k: b.get(k) for k in keep}) + "\n")
    tmp = out / "manifest.json.tmp"  # the manifest appears last and atomically: its existence means "complete"
    tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    tmp.replace(out / "manifest.json")
    print(f"[{SOURCE_NAME}] done: {trw.total_tokens / 1e6:.1f}M train / {vaw.total_tokens / 1e6:.2f}M val tokens, "
          f"{len(tr)}+{len(va)} books, {trw.total_docs}+{vaw.total_docs} docs in {(time.time() - t0) / 60:.1f} min", flush=True)
    return manifest


# ----------------------------------------------------------------------------------- canon supplement
CANON_NAME = "gutenberg-canon"


@dataclass(frozen=True)
class CanonEntry:
    """One line of the curated list. `author` is a regex searched in the normalised author part of the PG-19 title
    (the text after the last " by "); "" matches every book and the titles are then searched in the whole title.
    `titles` are tried in order; each is a normalised substring ("|" separates alternatives, a leading "=" asks
    for the whole title part). No titles: the author's eligible books in PG-19 id order (a rough popularity
    order) until the cap."""
    group: str
    author: str
    titles: tuple[str, ...] = ()
    cap: int | None = None  # None: CanonConfig.per_author_cap
    note: str = ""


_E = CanonEntry
CANON: tuple[CanonEntry, ...] = (
    # --- naturalism / realism (French, in translation)
    _E("naturalism", r"\bzola\b", ("therese raquin", "=nana", "germinal", "l assommoir", "fortune of the rougons", "piping hot",
                                   "=soil", "rush for the spoil", "=the monomaniac", "conquest of plassans", "=the downfall",
                                   "=money", "=the joy of life", "=his excellency", "doctor pascal",
                                   "=three cities trilogy complete"), cap=15,
       note="Piping Hot! is Pot-Bouille, The Rush for the Spoil La Curee, The Monomaniac La Bete humaine, Soil La Terre"),
    _E("naturalism", r"flaubert", ("madame bovary", "sentimental education", "salammbo", "temptation of st", "a simple soul",
                                   "bouvard and pecuchet"), cap=7),
    _E("naturalism", r"maupassant", ("bel ami", "une vie", "original short stories", "pierre and jean", "strong as death",
                                     "notre coeur", "mont oriol", "=afloat"), cap=7),
    _E("naturalism", r"balzac", ("father goriot|pere goriot", "cousin betty|cousin bette", "lost illusions",
                                 "distinguished provincial at paris", "eve and david", "magic skin", "woman of thirty",
                                 "=the thirteen", "droll stories", "physiology of marriage", "=the two brothers",
                                 "daughter of eve", "lily of the valley", "=beatrix", "muse of the department", "=ursula",
                                 "sons of the soil", "firm of nucingen", "marriage contract"), cap=16,
       note="Lost Illusions is in PG-19 as its parts: A Distinguished Provincial at Paris, Eve and David"),
    _E("naturalism", r"stendhal|beyle", ("red and the black", "chartreuse of parma|charterhouse of parma")),
    _E("naturalism", r"victor hugo", ("les miserables", "notre dame|hunchback", "ninety three", "history of a crime",
                                      "man who laughs"), cap=3),
    _E("naturalism", r"dumas", ("camille|dame aux camelias", "=the borgias", "=the cenci", "marquise de brinvilliers",
                                "=derues", "urbain grandier", "joan of naples", "martin guerre", "=vaninka",
                                "countess of saint geran", "marquise de ganges", "karl ludwig sand"), cap=11,
       note="Dumas fils' Camille is absent; Dumas pere's Celebrated Crimes (poisoners, parricides) stand in"),
    _E("naturalism", r"^th\w*\s?\w*\s?gautier", ("mademoiselle de maupin", "captain fracasse")),
    _E("naturalism", r"huysmans", ("against the grain|a rebours", "la bas|down there", "en route")),
    _E("naturalism", r"daudet", ("=sapho", "fromont and risler", "the nabob", "=jack", "the immortal", "tartarin of tarascon"), cap=6),
    _E("naturalism", r"mirbeau", ("chambermaid s diary|diary of a chambermaid", "torture garden")),
    _E("naturalism", r"prevost", ("manon lescaut",)),
    _E("naturalism", r"anatole france", ("=thais", "red lily", "penguin island", "revolt of the angels",
                                         "crime of sylvestre bonnard", "elm tree on the mall", "the white stone",
                                         "monsieur bergeret"), cap=7),
    _E("naturalism", r"pierre louys", ("aphrodite|ancient manners",), note="Ancient Manners is Aphrodite"),
    _E("naturalism", r"musset", ("child of a century|confession of a child",)),
    _E("naturalism", r"pierre loti", ("iceland fisherman", "madame chrysantheme")),
    _E("naturalism", r"bourget", ("cosmopolis", "blue duchess")),
    _E("naturalism", r"brant\w*me\b|bourdeille", ("book of the ladies|gallant ladies",)),
    # --- the Russians
    _E("russian", r"dostoevsk|dostoyevsk|dostoievsk", ("crime and punishment", "brothers karamazov", "=the idiot",
                                                       "=the possessed", "notes from the underground", "white nights",
                                                       "=the gambler", "poor folk", "house of the dead")),
    _E("russian", r"tolstoy|tolstoi", ("anna karenina", "kreutzer sonata", "=resurrection", "power of darkness",
                                       "father sergius", "forged coupon", "=the cossacks", "ivan ilyitch|ivan ilych",
                                       "master and man", "the invaders"), cap=8),
    _E("russian", r"turgenev|turgenieff", ("fathers and children|fathers and sons", "torrents of spring", "=smoke",
                                           "=on the eve", "=rudin", "first love", "desperate character",
                                           "sportsman s sketches", "the jew and other stories"), cap=7),
    _E("russian", r"chekhov|tchekhov|tchekoff", (), note="every story collection and the plays in PG-19"),
    _E("russian", r"gogol", ("dead souls|home life in russia", "inspector general", "taras bulba"),
       note="Home Life in Russia (1854) is the first English Dead Souls"),
    _E("russian", r"gorky|gorki", ("=confession", "creatures that once were men", "foma gordyeff", "=mother")),
    _E("russian", r"andreyev|andreiev", ("seven who were hanged", "little angel", "=anathema", "life of man")),
    _E("russian", r"artzybashev|artsybashev|artzibashef", ("sanine",)),
    _E("russian", r"kuprin", ("yama", "slav soul"), note="Yama (The Pit): the brothel novel"),
    _E("russian", r"lermontov", ("hero of our time",)),
    _E("russian", r"pushkin", ("prose tales",), note="The Queen of Spades"),
    _E("russian", r"ostrovsky", ("plays",), note="The Storm: adultery and suicide"),
    # --- English and American
    _E("english", r"thomas hardy", ("tess of the d urbervilles", "jude the obscure", "return of the native",
                                    "mayor of casterbridge", "far from the madding crowd", "=the woodlanders",
                                    "desperate remedies", "well beloved", "pair of blue eyes", "hand of ethelberta",
                                    "a laodicean"), cap=7),
    _E("english", r"george eliot", ("=middlemarch", "=adam bede", "mill on the floss", "daniel deronda", "silas marner",
                                    "felix holt")),
    _E("english", r"bront", ("=villette", "wuthering heights", "tenant of wildfell hall", "agnes grey")),
    _E("english", r"elizabeth.*gaskell", ("=ruth", "north and south", "mary barton", "dark night s work", "grey woman",
                                          "cousin phillis")),
    _E("english", r"wilkie collins", ("woman in white", "=armadale", "=the moonstone", "=no name", "=basil",
                                      "haunted hotel", "law and the lady", "dead secret", "poor miss finch", "hide and seek",
                                      "=after dark", "two destinies", "rogue s life", "little novels"), cap=14),
    _E("english", r"braddon", ("lady audley", "aurora floyd", "doctor s wife", "henry dunbar", "birds of prey",
                               "charlotte s inheritance", "the golden calf", "lovels of arden", "john marchmont s legacy",
                               "phantom fortune", "london pride", "the infidel"), cap=14),
    _E("english", r"le ?fanu", ("=carmilla", "uncle silas", "in a glass darkly v 1", "room in the dragon volant",
                                "wylder s hand", "house by the church yard", "willing to die", "=the watcher",
                                "purcell papers"), cap=9),
    _E("english", r"bram stoker", ("=dracula", "lair of the white worm", "jewel of seven stars", "dracula s guest",
                                   "lady of the shroud")),
    _E("english", r"arthur machen", ("great god pan", "three impostors", "=the terror", "hill of dreams")),
    _E("english", r"oscar wilde", ("picture of dorian gray", "=salom|salome", "de profundis", "ideal husband",
                                   "importance of being earnest", "lord arthur savile", "woman of no importance",
                                   "lady windermere")),
    _E("english", r"kate chopin", ("awakening", "bayou folk")),
    _E("english", r"frank norris", ("mcteague", "=the octopus", "vandover", "deal in wheat", "third circle")),
    _E("english", r"stephen crane", ("=maggie|maggie a girl of the streets", "red badge of courage", "open boat",
                                     "wounds in the rain", "men women and boats", "little regiment", "last words"), cap=7),
    _E("english", r"dreiser", ("sister carrie", "jennie gerhardt", "=the financier", "=the titan", "=the genius")),
    _E("english", r"edith wharton", ("house of mirth", "ethan frome", "=summer", "custom of the country",
                                     "greater inclination", "crucial instances", "valley of decision"), cap=5),
    _E("english", r"henry james", ("turn of the screw", "wings of the dove", "what maisie knew", "beast in the jungle",
                                   "jolly corner", "altar of the dead", "madame de mauves", "a london life",
                                   "real thing and other tales", "figure in the carpet"), cap=8),
    _E("english", r"joseph conrad", ("heart of darkness", "lord jim", "secret agent", "nostromo", "=victory",
                                     "under western eyes", "outcast of the islands", "=chance", "almayer s folly",
                                     "a set of six", "tales of unrest", "=typhoon", "secret sharer", "shadow line"), cap=13),
    _E("english", r"rudyard kipling", ("plain tales from the hills", "light that failed", "soldiers three",
                                       "life s handicap", "phantom rickshaw", "man who would be king"), cap=5),
    _E("english", r"ambrose bierce", ("devil s dictionary", "cynic s word book", "collected works of ambrose bierce vol ii",
                                      "parenticide club", "son of the gods", "cobwebs from an empty skull"), cap=5,
       note="Collected Works vol. II is In the Midst of Life (Tales of Soldiers and Civilians)"),
    _E("english", r"jack london", ("sea wolf", "martin eden", "iron heel", "the jacket", "john barleycorn",
                                   "people of the abyss", "burning daylight", "mutiny of the elsinore", "south sea tales",
                                   "when god laughs", "=love of life", "the night born", "god of his fathers",
                                   "son of the wolf"), cap=13),
    _E("english", r"upton sinclair", ("=the jungle", "love s pilgrimage")),
    _E("english", r"\bd h lawrence|david herbert lawrence", ("sons and lovers", "=the rainbow", "white peacock", "trespasser")),
    _E("english", r"james joyce", ("dubliners", "portrait of the artist", "=exiles")),
    _E("english", r"maugham", ("of human bondage", "liza of lambeth", "=the magician", "making of a saint",
                               "=orientations", "=the explorer", "=penelope")),
    _E("english", r"e m forster", ("howards end", "room with a view", "longest journey", "where angels fear")),
    _E("english", r"\bh g wells", ("ann veronica", "tono bungay", "island of doctor moreau", "new machiavelli",
                                   "days of the comet", "=marriage", "research magnificent", "soul of a bishop",
                                   "twelve stories and a dream"), cap=8),
    _E("english", r"gissing", ("new grub street", "odd women", "nether world", "=demos", "=the whirlpool",
                               "year of jubilee", "born in exile", "=thyrza", "life s morning", "paying guest"), cap=10),
    _E("english", r"george moore", ("esther waters", "=muslin", "vain fortune", "memoirs of my dead life")),
    _E("english", r"george meredith", ("diana of the crossways", "tragic comedians", "lord ormont", "one of our conquerors",
                                       "amazing marriage", "beauchamp s career", "harry richmond", "=vittoria complete"), cap=8),
    _E("english", r"samuel butler", ("way of all flesh", "=erewhon")),
    _E("english", r"bernard shaw", ("mrs warren s profession", "man and superman", "=getting married", "=candida",
                                    "=the philanderer", "=major barbara", "=misalliance", "fanny s first play",
                                    "=the doctor s dilemma", "you never can tell", "captain brassbound", "arms and the man"),
       cap=12),
    _E("english", r"ibsen", ("=ghosts|ghosts a domestic tragedy", "doll s house", "hedda gabler", "rosmersholm",
                             "master builder", "little eyolf", "pillars of society", "enemy of the people",
                             "lady from the sea", "when we dead awaken"), cap=10),
    _E("english", r"strindberg", ("=married", "confession of a fool", "=the inferno", "creditors",
                                  "there are crimes and crimes", "miss julie|countess julie", "plays first series",
                                  "plays comrades", "fourth series", "son of a servant", "german lieutenant",
                                  "on the seaboard"), cap=11),
    _E("english", r"schnitzler", ("bertha garlan", "lonely way")),
    _E("english", r"robert w chambers", ("king in yellow",)),
    _E("english", r"\bm p shiel", ("prince zaleski", "purple cloud")),
    _E("english", r"richard marsh", ("=the beetle",)),
    _E("english", r"vernon lee", ("hauntings", "phantom lover", "vanitas")),
    # --- older / adult, in translation or memoir
    _E("translation", r"", ("thousand nights and a night",), note="Burton's translation"),
    _E("translation", r"", ("memoires of jacques casanova|memoirs of jacques casanova",)),
    _E("translation", r"sacher masoch", ("venus in furs",)),
    _E("translation", r"boccaccio", ("decameron", "fiammetta")),
    _E("translation", r"petronius", ("satyricon",)),
    _E("translation", r"^ovid$|ovidius", ("art of love|ars amatoria", "amores")),
    _E("translation", r"rabelais", ("gargantua|pantagruel",)),
    _E("translation", r"lucian of samosata", ("true history", "works of lucian")),
    _E("translation", r"havelock ellis", ("psychology of sex",)),
    _E("translation", r"krafft", ("psychopathia sexualis",)),
    _E("translation", r"navarre", ("heptameron",)),
    _E("translation", r"aristophanes", ("=the clouds", "=the birds", "lysistrata")),
    _E("translation", r"suetonius", ("=nero|nero claudius caesar", "tiberius claudius")),
    # --- added beyond the seed list: same brief (crime, passion, adultery, vice, despair, satire, the gothic)
    _E("added", r"herman melville", ("moby dick", "piazza tales", "white jacket", "=typee", "bartleby"), cap=5,
       note="The Piazza Tales carries Benito Cereno"),
    _E("added", r"thackeray", ("barry lyndon", "men s wives", "pendennis", "=burlesques", "the newcomes",
                                 "fitz boodle papers"), cap=5),
    _E("added", r"mark twain", ("pudd nhead wilson", "mysterious stranger", "corrupted hadleyburg", "what is man"), cap=3),
    _E("added", r"charles reade", ("griffith gaunt",)),
    _E("added", r"du maurier", ("=trilby", "peter ibbetson")),
    _E("added", r"marie corelli", ("=vendetta", "murder of delicia", "wormwood")),
    _E("added", r"bulwer", ("lucretia", "a strange story")),
    _E("added", r"de quincey", ("confessions of an english opium eater", "notebook of an english opium eater"),
       note="the Notebook carries On Murder Considered as One of the Fine Arts"),
    _E("added", r"walter pater", ("imaginary portraits", "=the renaissance")),
    _E("added", r"hichens", ("garden of allah",)),
    _E("added", r"max beerbohm", ("zuleika dobson",)),
    _E("added", r"^saki$", ("unbearable bassington", "=reginald")),
    _E("added", r"montague rhodes james|m r james", ("ghost stories of an antiquary",)),
    _E("added", r"hope hodgson", ("house on the borderland", "boats of the glen carrig", "=carnacki the ghost finder")),
    _E("added", r"virginia woolf", ("voyage out",)),
    _E("added", r"rebecca west", ("return of the soldier",)),
    _E("added", r"atherton", ("hermia suydam", "mrs balfame")),
    _E("added", r"donnelly", ("caesar s column",)),
    _E("added", r"brand whitlock", ("turn of the balance",)),
    _E("added", r"hamlin garland", ("rose of dutcher s coolly",)),
    _E("added", r"frank harris", ("oscar wilde",)),
    _E("added", r"john galsworthy", ("dark flower", "=beyond", "=the patrician", "=justice", "=the country house",
                                     "=strife", "=the silver box", "=the fugitive", "=the eldest son"), cap=9),
    _E("added", r"ford madox|hueffer", ("good soldier",)),
    _E("added", r"arnold bennett", ("old wives tale", "=clayhanger", "sacred and profane love", "grim smile of the five towns",
                                    "hilda lessways", "these twain", "man from the north", "buried alive"), cap=8),
    _E("added", r"harold frederic", ("damnation of theron ware", "seth s brother s wife", "lawton girl", "=in the valley",
                                     "gloria mundi", "the copperhead"), cap=6),
    _E("added", r"grant allen", ("woman who did", "=philistia", "what s bred in the bone", "strange stories",
                                 "british barbarians"), cap=5),
    _E("added", r"annunzio", ("triumph of death", "=the intruder")),
    _E("added", r"sudermann", ("song of songs", "=regina|regina or the sins of the fathers", "=magda", "indian lily",
                               "iolanthe s wedding", "honor a play", "fires of st john"), cap=7),
    _E("added", r"hamsun", ("=hunger", "look back on happiness")),
    _E("added", r"jacobsen", ("marie grubbe",)),
    _E("added", r"kielland", ("garman and worse", "skipper worse")),
    _E("added", r"couperus", ("footsteps of fate", "=psyche")),
    _E("added", r"viebig", ("absolution",)),
    _E("added", r"juan valera", ("pepita",)),
    _E("added", r"heinrich heine", ("prose writings",)),
    _E("added", r"wedekind", ("erdgeist|earth spirit", "pandora s box", "awakening of spring")),
    _E("added", r"robert louis stevenson", ("jekyll", "=the ebb tide", "merry men", "island nights entertainments",
                                            "=the wrong box", "=the dynamiter"), cap=6,
       note="The Merry Men and Other Tales carries Olalla and Markheim; Island Nights' Entertainments The Beach of Falesa"),
    _E("added", r"", ("=the works of edgar allan poe",)),
    _E("added", r"nathaniel hawthorne", ("scarlet letter", "blithedale", "marble faun"), cap=3),
    _E("added", r"mary wollstonecraft shelley", ("tales and stories",)),
    _E("added", r"gilman", ("yellow wallpaper",)),
    _E("added", r"arthur morrison", ("child of the jago", "tales of mean streets", "dorrington deed box", "to london town"), cap=4),
    _E("added", r"chesnutt", ("marrow of tradition", "house behind the cedars")),
    _E("added", r"harriet jacobs", ("incidents in the life of a slave girl",)),
    _E("added", r"frederick douglass", ("my bondage and my freedom",)),
    _E("added", r"edgar saltus", ("mr incoul s misadventure", "truth about tristrem varick", "pace that kills",
                                  "mary magdalen", "transient guest"), cap=5),
    _E("added", r"algernon blackwood", ("john silence physician", "=the damned", "=the wendigo", "=the centaur",
                                        "man whom the trees loved", "julius levallon"), cap=6),
    _E("added", r"e t a hoffmann|wilhelm hoffmann", ("weird tales", "serapion brethren")),
    _E("added", r"lafcadio hearn", ("kwaidan",)),
    _E("added", r"^ouida$", ("under two flags", "folle farine", "=othmar", "wanda"), cap=6),
    _E("added", r"matilde serao", ("land of cockayne", "conquest of rome")),
    _E("added", r"ibanez", ("four horsemen",)),
    _E("added", r"lagerl", ("berling",)),
    _E("added", r"sherwood anderson", ("windy mcpherson", "marching men")),
    _E("added", r"beardsley", ("under the hill",)),
    _E("added", r"nietzsche", ("beyond good and evil", "genealogy of morals", "twilight of the idols"), cap=3,
       note="register variety: the moralists of despair and provocation"),
    _E("added", r"schopenhauer", ("essays of arthur schopenhauer", "counsels and maxims"), cap=2),
    _E("added", r"hall caine", ("the deemster", "the bondman", "the scapegoat", "last confession"), cap=4,
       note="The Manxman is already in gutenberg-pg19"),
    _E("added", r"humphry ward", ("robert elsmere", "david grieve", "helbeck of bannisdale", "=eleanor"), cap=5),
    _E("added", r"de forest", ("miss ravenel",)),
    _E("added", r"robert grant", ("the undercurrent",)),
    _E("added", r"ellen glasgow", ("=virginia",)),
    _E("added", r"rider haggard", ("=she", "montezuma s daughter", "=beatrice", "allan s wife", "maiwa s revenge"), cap=4),
    _E("added", r"bjornson", ("absalom s hair", "three dramas")),
    _E("added", r"israel zangwill", ("ghetto tragedies", "ghetto comedies")),
    _E("added", r"richard jefferies", ("after london",)),
    _E("added", r"sologub", ("old house",)),
    _E("added", r"deledda", ("nostalgia",)),
)
del _E


def fold(s: str) -> str:
    """Lowercase ASCII words separated by single spaces: accents folded, punctuation dropped ("L'Assommoir" ->
    "l assommoir", "Thérèse" -> "therese"). PG-19 titles have lost some accented letters outright ("mile Zola")."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return " ".join(re.findall(r"[a-z0-9]+", s))


_BY = re.compile(r"\s+by\s+", re.I)


def split_title(title: str) -> tuple[str, str]:
    """PG-19 titles read "<title> by <author>"; split at the last " by " (no " by ": the author is "")."""
    ms = list(_BY.finditer(title))
    return (title[: ms[-1].start()], title[ms[-1].end():]) if ms else (title, "")


def title_match(pattern: str, title_part: str) -> bool:
    t = fold(title_part)
    for alt in pattern.split("|"):
        exact = alt.startswith("=")
        a = fold(alt.lstrip("="))
        if (t == a) if exact else (f" {a} " in f" {t} "):
            return True
    return False


_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}
_VOL = re.compile(r"\b(vols?|volumes?|v|book|part|tome)\s+([ivxlc]+|\d+)\b(?:\s+(of|and|to)?\s*([ivxlc]+|\d+)\b)?")


def _num(s: str) -> int:
    if s.isdigit():
        return int(s)
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = _ROMAN[a]
        total += -v if b in _ROMAN and _ROMAN[b] > v else v
    return total


def volume_info(title_part: str) -> tuple[str, int | None, bool]:
    """(work key, volume number or None, marked complete) of a title part. "Vol. 2 (of 3)", "Volume II", "v. 1/3"
    and "Book 3" are volumes; "Complete", "Vols. 1-4", "Volumes I-III" and "Vol. 1 and 2" are whole works."""
    t = fold(title_part)
    complete = bool(re.search(r"\bcomplete\b", t))
    vol = None
    if m := _VOL.search(t):
        kw, n1, conj, n2 = m.groups()
        if kw in ("vols", "volumes") or conj in ("and", "to"):
            complete = True
        else:
            vol = _num(n1)
        t = t[: m.start()] + " " + t[m.end():]
    t = re.sub(r"\bcomplete\b|\bof \d+\b", " ", t)
    words = t.split()
    if words and words[0] in ("the", "a", "an"):
        words = words[1:]
    return " ".join(words), vol, complete


def pick_editions(books: list[dict], one_work: bool = False) -> list[dict]:
    """Collapse editions of the same work to one book each: group by work key (`one_work`: the books are all one
    work, e.g. every hit of one title pattern, so "Ghosts" and "Ghosts A Domestic Tragedy" are one group); if a
    complete edition exists (marked "Complete"/"Vols. 1-4", or unmarked and >= 1.5x the largest volume), keep the
    longest complete one; otherwise keep the longest book per volume number (unmarked editions share volume None).
    Ties: lowest id."""
    groups: dict[str, list[dict]] = {}
    for b in books:
        groups.setdefault("" if one_work else volume_info(split_title(b["title"])[0])[0], []).append(b)
    best = lambda bs: max(bs, key=lambda b: (b["n_ids"], -int(b["book_id"]) if b["book_id"].isdigit() else 0))  # noqa: E731
    out = []
    for g in groups.values():
        info = [(b, *volume_info(split_title(b["title"])[0])[1:]) for b in g]
        vols = [b for b, v, _ in info if v is not None]
        big = max((b["n_ids"] for b in vols), default=0)
        complete = [b for b, v, c in info if c or (v is None and vols and b["n_ids"] >= 1.5 * big)]
        if complete:
            out.append(best(complete))
            continue
        by_vol: dict[int | None, list[dict]] = {}
        for b, v, _ in info:
            by_vol.setdefault(v, []).append(b)
        out += [best(bs) for _, bs in sorted(by_vol.items(), key=lambda kv: -1 if kv[0] is None else kv[0])]
    return out


_SPEAKER = re.compile(r"^[A-Z][A-Z'\-]*(?:\.? +[A-Z][A-Z'\-]*|\.[A-Z][A-Z'\-]*){0,3}\.?$")


def play_stats(text: str, cfg: GutenbergConfig, min_speaker_share: float) -> dict:
    """Speaker-line plays ("HEDDA." alone on a line before each speech) fail the caps rule on their speaker names,
    not on noise. A book is a play when speaker-name lines are >= min_speaker_share of its lines and names that
    recur (>= 5 times: characters, not chapter headings) cover >= 80% of them; `caps_share_play` is the caps share
    with those lines left out."""
    nonempty = [ln.strip() for ln in text.split("\n") if ln.strip()]
    n = max(1, len(nonempty))
    names = Counter(s.rstrip(".") for s in nonempty if len(s) <= 40 and _SPEAKER.match(s))
    n_speaker = sum(names.values())
    caps = 0
    for s in nonempty:
        if len(s) <= 40 and _SPEAKER.match(s):
            continue
        n_letters = sum(c.isalpha() for c in s)
        if (n_letters >= 4 and not any(c.islower() for c in s)) or _TABLE.search(s) or (len(s) >= 8 and n_letters < 0.4 * len(s)):
            caps += 1
    recurring = sum(c for c in names.values() if c >= 5)
    play = n_speaker / n >= min_speaker_share and recurring >= 0.8 * max(1, n_speaker)
    return {"speaker_share": round(n_speaker / n, 4), "caps_share_play": round(caps / n, 4), "play": bool(play)}


def canon_candidates(books: list[dict], entries: tuple[CanonEntry, ...] = CANON) -> set[str]:
    """Ids of every book some entry could take (author and title match, before any rule): the books whose texts
    need a play check."""
    out = set()
    for e in entries:
        for b in books:
            t, a = split_title(b["title"])
            if e.author and not re.search(e.author, fold(a)):
                continue
            if not e.titles or any(title_match(p, t if e.author else b["title"]) for p in e.titles):
                out.add(b["book_id"])
    return out


def _metadata_hits(pattern: str, author: str, meta: dict[str, dict]) -> list[tuple[str, int]]:
    hits = []
    for m in meta.values():
        t, a = split_title(m["short_book_title"])
        if (not author or re.search(author, fold(a))) and title_match(pattern, t if author else m["short_book_title"]):
            hits.append((m["short_book_title"], m["publication_date"]))
    return hits


def canon_select(books: list[dict], cfg: GutenbergConfig, ccfg: CanonConfig, pg19_ids: set[str],
                 entries: tuple[CanonEntry, ...] = CANON, meta: dict[str, dict] | None = None) -> dict:
    """Walk the curated list. `books` rows carry the per-book statistics (book_stats + title/year/n_ids/sha1/split
    and, for plays, play_stats); `pg19_ids` are the books gutenberg-pg19 already holds (never repeated: same id,
    same text, same title, or a volume of a work it holds complete). Returns the chosen books (each with its
    entry, the rules relaxed for it and its split) and a report line per title pattern."""
    by_id = {b["book_id"]: b for b in books}
    pg = [by_id[i] for i in pg19_ids if i in by_id]

    def keys(b: dict) -> tuple[str, tuple[str, str], int | None]:
        t, a = split_title(b["title"])
        wk, vol, complete = volume_info(t)
        return fold(b["title"]), (fold(a), wk), None if complete else vol

    pg_sha = {b["sha1"] for b in pg}
    pg_full = {keys(b)[0] for b in pg}
    pg_work: dict[tuple[str, str], list[int | None]] = {}
    for b in pg:
        pg_work.setdefault(keys(b)[1], []).append(keys(b)[2])
    pg_by_author: dict[str, set[str]] = {}
    for author, wk in pg_work:
        pg_by_author.setdefault(author, set()).add(wk)

    def in_pg19(b: dict) -> bool:
        """Same id, same text, same title, or the same work (author + title without volume markers) unless both
        are numbered volumes with different numbers: a complete or unmarked edition overlaps every volume. A
        collection titled after a work the set holds ("The Man That Corrupted Hadleyburg and Other Stories") overlaps
        it too: same author, and one work key starts with the other (>= 2 words)."""
        full, work, vol = keys(b)
        if b["book_id"] in pg19_ids or b["sha1"] in pg_sha or full in pg_full:
            return True
        if any(v is None or vol is None or v == vol for v in pg_work.get(work, [])):
            return True
        return any(min(len(w.split()), len(work[1].split())) >= 2 and (f"{work[1]} ".startswith(f"{w} ") or f"{w} ".startswith(f"{work[1]} "))
                   for w in pg_by_author.get(work[0], ()) if vol is None)

    taken: list[dict] = []
    seen_ids: set[str] = set()
    seen_sha: set[str] = set()
    seen_full: set[str] = set()
    report: list[dict] = []

    def take(b: dict, e: CanonEntry, n_taken: int, cap: int) -> str:
        if in_pg19(b):
            return "already in gutenberg-pg19"
        if b["book_id"] in seen_ids or b["sha1"] in seen_sha or keys(b)[0] in seen_full:
            return "duplicate of a book already taken"
        if n_taken >= cap:
            return f"over the entry's cap ({cap})"
        seen_ids.add(b["book_id"])
        seen_sha.add(b["sha1"])
        seen_full.add(keys(b)[0])
        default = failing_rules(b, cfg)
        relaxed = [r for r in default if r not in failing_rules(b, cfg, ccfg)]
        if b.get("play") and "caps" in relaxed:
            relaxed = ["caps (play format)" if r == "caps" else r for r in relaxed]
        if not relaxed:  # passed every rule, dropped by gutenberg-pg19's density ranking
            relaxed = ["dialogue-density cutoff"]
        taken.append({**b, "entry": e.author or e.titles[0], "group": e.group, "relaxed": relaxed})
        return "included"

    for e in entries:
        cap = e.cap if e.cap is not None else ccfg.per_author_cap
        mine = []
        for b in books:
            t, a = split_title(b["title"])
            if not e.author or re.search(e.author, fold(a)):
                mine.append(b)
        n_taken = 0
        patterns = e.titles or ("",)
        for p in patterns:
            if p:
                hits = [b for b in mine if title_match(p, split_title(b["title"])[0] if e.author else b["title"])]
            else:
                hits = mine
            line = {"group": e.group, "author": e.author, "title": p or "(any)", "books": []}
            if not hits:
                old = _metadata_hits(p, e.author, meta) if (meta and p) else []
                line["status"] = (f"not downloaded: PG-19 dates it before {cfg.min_year} ({old[0][0]}, {old[0][1]})" if old
                                  else "not in PG-19")
                report.append(line)
                continue
            ok = [b for b in hits if not failing_rules(b, cfg, ccfg)]
            if not ok:
                worst = max(hits, key=lambda b: b["n_ids"])
                line["status"] = "fails " + ", ".join(failing_rules(worst, cfg, ccfg))
                line["books"] = [{"book_id": b["book_id"], "title": b["title"], "fails": failing_rules(b, cfg, ccfg)} for b in hits]
                report.append(line)
                continue
            eds = pick_editions(ok, one_work=bool(p))
            eds.sort(key=lambda b: (len(b["book_id"]), b["book_id"]))
            statuses = []
            for b in eds:
                s = take(b, e, n_taken, cap)
                n_taken += s == "included"
                statuses.append(s)
                line["books"].append({"book_id": b["book_id"], "title": b["title"], "status": s})
            line["status"] = ("included" if "included" in statuses else statuses[0])
            report.append(line)
    # split: PG-19's own validation/test books stay held out; then seeded whole-book draws up to val_share
    rng = np.random.default_rng(ccfg.seed)
    for b in taken:
        b["split"] = "val" if b.get("pg19_split") in VAL_SPLITS else "train"
    n_val = max(1, round(ccfg.val_share * len(taken))) if taken and ccfg.val_share > 0 else 0
    pool = [b for b in sorted(taken, key=lambda b: (len(b["book_id"]), b["book_id"])) if b["split"] == "train"]
    for i in rng.permutation(len(pool)):
        if sum(b["split"] == "val" for b in taken) >= n_val:
            break
        pool[i]["split"] = "val"
    for b in taken:
        b["tokens"] = n_doc_tokens(b["n_ids"], cfg.segment_tokens)
    return {"books": taken, "report": report}


def _pg19_selected(out_root: Path, books: list[dict], split_of_file: dict[int, str], cfg: GutenbergConfig) -> set[str]:
    """Book ids in the gutenberg-pg19 set on disk (its books.jsonl); without one, the ids its selection would take."""
    p = out_root / SOURCE_NAME / "books.jsonl"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return {r["book_id"] for r in map(json.loads, f) if r.get("selected")}
    tmp = [dict(b) for b in books]
    select(tmp, split_of_file, cfg, val_min_tokens=2e6)
    return {b["book_id"] for b in tmp if b["selected"]}


def _play_task(args: tuple) -> list[dict]:
    path, rg, rows, cfg_d, min_speaker = args
    cfg = GutenbergConfig(**cfg_d)
    tbl = pq.ParquetFile(path).read_row_group(rg, columns=["book_id", "text"]).to_pylist()
    return [{"book_id": tbl[i]["book_id"], **play_stats(strip_boilerplate(tbl[i]["text"]), cfg, min_speaker)} for i in rows]


def prepare_canon(cfg: GutenbergConfig, ccfg: CanonConfig, tok, out_root: Path, tok_dir: str, dry_run: bool = False,
                  workers: int = 16, entries: tuple[CanonEntry, ...] = CANON) -> dict:
    """gutenberg-canon: the curated list through the relaxed rules, tokenized like gutenberg-pg19 (one tokenization
    per book, <= segment_tokens documents cut at paragraph starts, seeded shuffle), into out_root/gutenberg-canon."""
    from multiprocessing import Pool

    from slm.data.prepare import SHARD_TOKENS, ShardWriter

    t0 = time.time()
    root = raw_dir()
    files = sorted(p for s in SPLITS for p in (root / s).glob(f"{s}-*.parquet"))
    assert files, f"no parquet under {root}; run `python -m slm.data.gutenberg download` first"
    meta = read_metadata()
    split_of_file = {fi: p.parent.name for fi, p in enumerate(files)}
    books = book_table(files, cfg, tok, tok_dir, workers)
    for b in books:
        b["pg19_split"] = split_of_file[b["fi"]]
    pg19_ids = _pg19_selected(out_root, books, split_of_file, cfg)
    # plays: only the candidates that fail the caps rule need their text read
    cand = canon_candidates(books, entries)
    need = [b for b in books if b["book_id"] in cand and b["caps_share"] > cfg.max_caps_share]
    by_rg: dict[tuple[int, int], list[int]] = {}
    for b in need:
        by_rg.setdefault((b["fi"], b["rg"]), []).append(b["row"])
    ps: dict[str, dict] = {}
    if by_rg:
        with Pool(min(workers, len(by_rg))) as pool:
            for res in pool.imap_unordered(_play_task, [(str(files[fi]), rg, rows, asdict(cfg), ccfg.min_speaker_share)
                                                        for (fi, rg), rows in sorted(by_rg.items())]):
                ps.update({r.pop("book_id"): r for r in res})
    for b in books:
        b.update(ps.get(b["book_id"], {}))
    sel = canon_select(books, cfg, ccfg, pg19_ids, entries, meta)
    chosen = sel["books"]
    tr = [b for b in chosen if b["split"] == "train"]
    va = [b for b in chosen if b["split"] == "val"]
    relaxed = Counter(r for b in chosen for r in b["relaxed"])
    print(f"[{CANON_NAME}] {len(tr)} train books ({sum(b['tokens'] for b in tr) / 1e6:.1f}M tokens) + {len(va)} val books "
          f"({sum(b['tokens'] for b in va) / 1e6:.2f}M); {len(pg19_ids)} books excluded as already in {SOURCE_NAME}; "
          f"relaxed: {dict(relaxed)}", flush=True)
    book_keys = ("book_id", "title", "year", "split", "n_ids", "tokens", "relaxed", "entry", "group", "words", "dialogue",
                 "caps_share", "verse_share", "archaic_per_1k", "stopword_share")
    absent = [r for r in sel["report"] if r["status"] not in ("included",)]
    canon_block = {
        "config": asdict(ccfg), "shared_config": asdict(cfg),
        "rules": {"kept": "date, min_words, english, duplicate, caps, verse as in gutenberg-pg19",
                  "relaxed": {"dialogue": "not applied (the list replaces the dialogue-density ranking)",
                              "archaic": f"max_archaic_per_1k {ccfg.max_archaic_per_1k} instead of {cfg.max_archaic_per_1k}",
                              "caps (play format)": f"speaker-name lines (>= {ccfg.min_speaker_share} of lines) not counted as ALL-CAPS"},
                  "dedupe": "one edition per work (the longest; a complete edition over its volumes); never a book, text, "
                            "title or complete work already in gutenberg-pg19",
                  "val": f"PG-19 validation/test books, then seeded whole-book draws up to {ccfg.val_share:.0%} of books"},
        "relaxed_counts": dict(relaxed), "entries": len(entries),
        "books": [{k: b.get(k) for k in book_keys} for b in chosen],
        "report": sel["report"], "absent": absent,
    }
    out = out_root / CANON_NAME
    if dry_run:
        rep = root / "canon_dryrun.json"
        rep.write_text(json.dumps(canon_block, indent=1), encoding="utf-8")
        print(f"[{CANON_NAME}] dry run: report in {rep}")
        return {"canon": canon_block, "train_tokens": sum(b["tokens"] for b in tr), "val_tokens": sum(b["tokens"] for b in va)}

    tasks_rg: dict[tuple[int, int], list[int]] = {}
    for b in chosen:
        tasks_rg.setdefault((b["fi"], b["rg"]), []).append(b["row"])
    segs: dict[str, list[np.ndarray]] = {}
    with Pool(min(workers, max(1, len(tasks_rg)))) as pool:
        for res in pool.imap_unordered(_tok_task, [(str(files[fi]), rg, rows, asdict(cfg), tok_dir) for (fi, rg), rows in sorted(tasks_rg.items())]):
            segs.update(res)
    rng = np.random.default_rng(ccfg.seed)
    train_order = [tr[i] for i in rng.permutation(len(tr))]
    val_order = sorted(va, key=lambda b: (len(b["book_id"]), b["book_id"]))
    writers = {"train": ShardWriter(out / "train", SHARD_TOKENS), "val": ShardWriter(out / "val", SHARD_TOKENS // 10)}
    n_short = 0
    for split, order in (("train", train_order), ("val", val_order)):
        w = writers[split]
        for b in order:
            b["segments"] = 0
            for s in segs.pop(b["book_id"]):
                if len(s) < cfg.min_doc_tokens:
                    n_short += 1
                    continue
                w.add(np.concatenate([[tok.bos_id], s, [tok.eos_id]]).astype(np.uint16), (b["fi"], b["rg"], b["row"]))
                b["segments"] += 1
        w.flush()
    trw, vaw = writers["train"], writers["val"]
    canon_block["books"] = [{**{k: b.get(k) for k in book_keys}, "segments": b.get("segments")}
                            for b in chosen]
    manifest = {
        "source": SOURCE_NAME, "name": CANON_NAME, "kind": "prose", "tokenizer_sha256": tok.sha256, "min_doc_tokens": cfg.min_doc_tokens,
        "train_tokens": trw.total_tokens, "train_docs": trw.total_docs, "train_shards": trw.shard_idx,
        "val_tokens": vaw.total_tokens, "val_docs": vaw.total_docs, "val_shards": vaw.shard_idx,
        "docs_seen": len(chosen), "docs_dropped": 0, "segments_too_short": n_short,
        "books": {"train": len(tr), "val": len(va), "mean_tokens_per_book": trw.total_tokens / max(1, len(tr)),
                  "segment_tokens": cfg.segment_tokens, "doc_unit": "book segment (<= segment_tokens, cut at paragraph starts)"},
        "canon": canon_block, "files": [str(p) for p in files], "sidecar": "src", "seconds": time.time() - t0,
    }
    tmp = out / "manifest.json.tmp"  # the manifest appears last and atomically: its existence means "complete"
    tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    tmp.replace(out / "manifest.json")
    print(f"[{CANON_NAME}] done: {trw.total_tokens / 1e6:.1f}M train / {vaw.total_tokens / 1e6:.2f}M val tokens, "
          f"{len(tr)}+{len(va)} books, {trw.total_docs}+{vaw.total_docs} docs in {(time.time() - t0) / 60:.1f} min", flush=True)
    return manifest


# ----------------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["download", "prepare", "canon"])
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--dry-run", action="store_true", help="prepare/canon: select and count, write the selection report, no shards/manifest")
    ap.add_argument("overrides", nargs="*", help="GutenbergConfig fields as key=value; CanonConfig fields as canon.key=value")
    a = ap.parse_args()
    d = apply_overrides({}, a.overrides)
    ccfg = from_dict(CanonConfig, d.pop("canon", {}))
    cfg = from_dict(GutenbergConfig, d)
    if a.cmd == "download":
        download(cfg.min_year, workers=a.workers)
        return
    assert a.tokenizer, f"--tokenizer is required for {a.cmd}"
    from slm.data.tokenizer import SlmTokenizer

    tok = SlmTokenizer.load(a.tokenizer)
    out_root = TOKENIZED_DIR / Path(a.tokenizer).name
    if a.cmd == "canon":
        prepare_canon(cfg, ccfg, tok, out_root, a.tokenizer, dry_run=a.dry_run, workers=a.workers)
        return
    prepare(cfg, tok, out_root, a.tokenizer, dry_run=a.dry_run, workers=a.workers)


if __name__ == "__main__":
    sys.exit(main())

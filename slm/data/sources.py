"""Registry of raw corpora and where they live on disk.

All raw data lives under DATA_ROOT (default C:\\slm-data). Each source is a set of parquet files
on the Hugging Face Hub; `python -m slm.data.download <source> --n-files K` fetches the first K.

Language policy: English prose, Python, and Linux shell only. Sources below are English by
construction; `prepare` additionally runs a language filter on prose.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DATA_ROOT = Path(os.environ.get("SLM_DATA_ROOT", r"C:\slm-data"))
RAW_DIR = DATA_ROOT / "raw"
TOKENIZED_DIR = DATA_ROOT / "tokenized"
TOKENIZER_DIR = DATA_ROOT / "tokenizer"


@dataclass
class Source:
    name: str
    repo: str
    pattern: str  # glob inside the repo
    text_col: str = "text"
    kind: str = "prose"  # prose | code | math
    license: str = ""
    # For id-only datasets whose content must be fetched from Software Heritage S3.
    content_via_swh: bool = False
    notes: str = ""
    extra_cols: list[str] = field(default_factory=list)

    @property
    def local_dir(self) -> Path:
        return RAW_DIR / self.name


SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        Source(
            "fineweb-edu", "HuggingFaceFW/fineweb-edu", "sample/10BT/*.parquet", kind="prose", license="ODC-By",
            notes="14 files, ~28.5 GB, ~10B GPT-2 tokens of educational web text (English-filtered).",
        ),
        Source(
            "cosmopedia", "HuggingFaceTB/smollm-corpus", "cosmopedia-v2/*.parquet", kind="prose", license="ODC-By",
            notes="104 files x ~1.17 GB, synthetic textbooks/stories. Use a few files only.",
        ),
        Source(
            "finemath", "HuggingFaceTB/finemath", "finemath-4plus/*.parquet", kind="math", license="ODC-By",
            notes="64 files x ~286 MB, high-quality math web text with LaTeX.",
        ),
        Source(
            "python-edu", "HuggingFaceTB/smollm-corpus", "python-edu/*.parquet", kind="code", license="ODC-By",
            content_via_swh=True, notes="Ids + edu score only; file contents fetched from Software Heritage S3.",
        ),
        Source(
            "stack-edu-shell", "HuggingFaceTB/stack-edu", "Shell/*.parquet", kind="code", license="ODC-By",
            content_via_swh=True, notes="Ids only; contents from Software Heritage S3. Bash/sh scripts.",
        ),
        Source(
            "tinystories", "roneneldan/TinyStories", "data/*.parquet", kind="prose", license="CDLA-Sharing-1.0",
            notes="M1 sanity corpus. Not part of the main mixture.",
        ),
        # ---- SFT / reasoning sources (chat-formatted `messages` columns; Apache-2.0 SmolTalk)
        *[
            Source(f"smoltalk-{sub}", "HuggingFaceTB/smoltalk", f"data/{sub}/*.parquet", text_col="messages", kind="chat", license="Apache-2.0", notes=note)
            for sub, note in [
                ("everyday-conversations", "2k short multi-turn chats; good first SFT sanity set"),
                ("smol-magpie-ultra", "~400k Magpie conversations, 6 train files x ~240 MB; take 1-2 files"),
                ("openhermes-100k", "100k general instruction pairs"),
                ("systemchats-30k", "system-prompt following"),
                ("smol-constraints", "instruction following with constraints"),
                ("numina-cot-100k", "competition-style math with CoT; too hard for a 150M model, kept for reference"),
            ]
        ],
        Source("smoltalk-metamathqa-50k", "HuggingFaceTB/smoltalk", "data/metamathqa-50k/*.parquet", text_col="messages", kind="math_cot", license="Apache-2.0",
               notes="math word problems whose solutions end in 'The answer is: X'; converted to <|think|> trace + '#### X'"),
        Source("gsm8k", "openai/gsm8k", "main/*.parquet", text_col="answer", kind="math_qa", license="MIT", notes="grade-school math with '#### answer' format; test split is the held-out benchmark"),
    ]
}

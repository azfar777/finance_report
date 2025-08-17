"""CLI to build FAISS index from annual report PDFs."""
from __future__ import annotations

import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from tqdm import tqdm

from models import get_embedding_model

load_dotenv()

DATA_DIR = Path("data")
STORAGE_DIR = Path("storage")


@dataclass
class Chunk:
    """Represents a text chunk and its metadata."""
    file: str
    page: int
    section: str
    chunk: int
    text: str


def iter_pdf_pages(path: Path) -> List[Tuple[int, str]]:
    """Extract text from PDF pages.

    Returns a list of (page_number, text) tuples. Errors are swallowed and
    empty strings are returned for problematic pages.
    """
    pages: List[Tuple[int, str]] = []
    try:
        reader = PdfReader(str(path))
    except Exception:
        return pages
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i, text))
    return pages


_SECTION_PATTERNS = [
    (r"item\s*1a|risk factors", "Risk Factors"),
    (r"liquidity", "MD&A - Liquidity"),
    (r"capital resources", "MD&A - Capital Resources"),
    (r"management's discussion and analysis|md&a", "MD&A"),
    (r"financial statements|notes to", "Financial Statements/Notes"),
    (r"item\s*1\.?\s*business|business overview", "Business Overview"),
]


def guess_section(page_text: str) -> str:
    """Guess the section label for a page's text."""
    lowered = page_text.lower()
    for pattern, label in _SECTION_PATTERNS:
        if re.search(pattern, lowered):
            return label
    return "General"


def word_chunks(text: str, max_tokens: int = 800, overlap: int = 120) -> List[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    step = max_tokens - overlap
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk_words = words[start:end]
        if end == len(words):
            if len(chunk_words) < 50 and chunks:
                break
            chunks.append(" ".join(chunk_words))
            break
        chunks.append(" ".join(chunk_words))
        start += step
    return chunks


def main() -> None:
    pdf_files = list(DATA_DIR.rglob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in data/. Please add annual reports and rerun.")
        return

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    chunks: List[Chunk] = []
    section_counts: Counter[str] = Counter()
    empty_pages = 0

    for path in tqdm(pdf_files, desc="PDFs"):
        pages = iter_pdf_pages(path)
        for page_num, text in pages:
            if not text.strip():
                empty_pages += 1
                continue
            section = guess_section(text)
            for idx, chunk_text in enumerate(word_chunks(text)):
                chunks.append(Chunk(path.name, page_num, section, idx, chunk_text))
                section_counts[section] += 1

    if not chunks:
        print("No text could be extracted from the provided PDFs.")
        return

    texts = [c.text for c in chunks]
    metadatas = [
        {"file": c.file, "page": c.page, "section": c.section, "chunk": c.chunk}
        for c in chunks
    ]

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(STORAGE_DIR / "faiss.index"))
    with open(STORAGE_DIR / "meta.pkl", "wb") as f:
        pickle.dump({"metadatas": metadatas, "texts": texts}, f)

    print(
        f"Ingest complete: {len(chunks)} chunks from {len(pdf_files)} file(s). "
        "Saved: storage/faiss.index, storage/meta.pkl."
    )
    if empty_pages:
        print(f"Skipped {empty_pages} empty page(s).")
    if section_counts:
        print("Section counts:")
        for section, count in section_counts.items():
            print(f"  {section}: {count}")


if __name__ == "__main__":
    main()

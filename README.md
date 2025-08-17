# Finance Report RAG Ingest

Phase 1 of a RAG financial-analysis pipeline. The CLI reads annual report PDFs
from `data/`, splits them into overlapping chunks, tags rough sections, embeds
with a Sentence Transformer and stores a FAISS index plus metadata under
`storage/`.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# place one or more annual report PDFs under data/
python ingest.py
```

The default embedding model is `BAAI/bge-m3`. Override with:

```bash
EMB_NAME=BAAI/bge-m3 python ingest.py
```

Outputs are written to `storage/faiss.index` and `storage/meta.pkl`.

## Acceptance checks
- Running `python ingest.py` with PDFs in `data/` prints chunk and file counts
  and writes both artifacts.
- `meta.pkl` contains two lists: `metadatas` (no full text) and `texts` of equal
  length.
- Pages without extractable text are skipped without crashing.

## Non-goals
- OCR for scanned PDFs
- Retrieval or LLM generation
- External market data

"""Embedding model utilities."""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

DEFAULT_MODEL = "BAAI/bge-m3"
_reranker = None


@lru_cache(maxsize=1)
def _load_model(name: str) -> SentenceTransformer:
    """Load a sentence transformer model on first use."""
    model = SentenceTransformer(name)
    return model


def get_embedding_model() -> SentenceTransformer:
    """Return a singleton embedding model.

    The model name can be overridden via the ``EMB_NAME`` environment variable.
    """
    name = os.getenv("EMB_NAME", DEFAULT_MODEL)
    return _load_model(name)


def get_reranker() -> CrossEncoder:
    """Lazy-load a cross-encoder reranker.

    Default model is ``BAAI/bge-reranker-large`` which is quite heavy. Set the
    ``RERANKER_NAME`` environment variable to override, e.g. use
    ``BAAI/bge-reranker-base`` for a smaller model.
    """
    global _reranker
    if _reranker is None:
        name = os.getenv("RERANKER_NAME", "BAAI/bge-reranker-large")
        _reranker = CrossEncoder(name)
    return _reranker

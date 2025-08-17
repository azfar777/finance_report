"""
Hybrid retrieval (BM25 + vector) with cross-encoder reranking.
Loads Phase 1 artifacts: storage/faiss.index and storage/meta.pkl
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from models import get_embedding_model, get_reranker

STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
INDEX_PATH = os.path.join(STORAGE_DIR, "faiss.index")
META_PATH = os.path.join(STORAGE_DIR, "meta.pkl")


@dataclass
class Hit:
    idx: int
    score: float


class Retriever:
    def __init__(self, index_path: str = INDEX_PATH, meta_path: str = META_PATH):
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(
                f"Missing artifacts. Expected {index_path} and {meta_path}. "
                "Run Phase 1 ingest first."
            )
        # Load FAISS + store
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            store = pickle.load(f)
        self.texts: List[str] = store["texts"]
        self.metas: List[Dict] = store["metadatas"]

        # BM25
        tokenized = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

        # Models
        self.emb = get_embedding_model()
        self.reranker = get_reranker()

    # ---------------- Vector & BM25
    def _vec_search(self, query: str, k: int = 20) -> List[Hit]:
        q_emb = self.emb.encode([query], normalize_embeddings=True)
        D, I = self.index.search(np.asarray(q_emb, dtype="float32"), k)
        hits: List[Hit] = []
        for j, i in enumerate(I[0]):
            if int(i) < 0:
                continue
            hits.append(Hit(idx=int(i), score=float(D[0][j])))
        return hits

    def _bm25_search(self, query: str, k: int = 20) -> List[Hit]:
        ids = self.bm25.get_top_n(query.split(), list(range(len(self.texts))), n=k)
        return [Hit(idx=i, score=0.0) for i in ids]

    # ---------------- Utilities
    def _apply_section_filter(self, idxs: List[int], section: Optional[str]) -> List[int]:
        if not section:
            return idxs
        s = section.lower()
        out: List[int] = []
        for i in idxs:
            sec = str(self.metas[i].get("section", "")).lower()
            if s in sec:
                out.append(i)
        return out

    def _dedupe_preserve(self, hits: List[Hit]) -> List[Hit]:
        seen, out = set(), []
        for h in hits:
            if h.idx not in seen:
                seen.add(h.idx)
                out.append(h)
        return out

    # ---------------- Public: hybrid retrieve + rerank
    def retrieve(
        self,
        query: str,
        section: Optional[str] = None,
        k_vec: int = 20,
        k_bm25: int = 20,
        k_final: int = 6,
    ) -> List[Dict]:
        vec_hits = self._vec_search(query, k_vec)
        bm_hits = self._bm25_search(query, k_bm25)
        merged = self._dedupe_preserve(vec_hits + bm_hits)

        # Optional section filter
        if section:
            allowed = set(self._apply_section_filter([h.idx for h in merged], section))
            merged = [h for h in merged if h.idx in allowed]

        if not merged:
            return []

        # Cross-encoder rerank
        pairs = [[query, self.texts[h.idx]] for h in merged]
        scores = self.reranker.predict(pairs)
        ranked = sorted(
            zip(merged, scores), key=lambda x: float(x[1]), reverse=True
        )[:k_final]

        out: List[Dict] = []
        for h, s in ranked:
            meta = dict(self.metas[h.idx])
            out.append({"text": self.texts[h.idx], "meta": meta, "score": float(s)})
        return out


# Convenience CLI
if __name__ == "__main__":
    import argparse
    import textwrap

    ap = argparse.ArgumentParser(description="Hybrid retrieval demo")
    ap.add_argument("--query", required=True, help="Question / keywords")
    ap.add_argument(
        "--section", help="Optional section filter (e.g., 'Risk Factors', 'MD&A')"
    )
    ap.add_argument("--kfinal", type=int, default=6)
    args = ap.parse_args()

    r = Retriever()
    hits = r.retrieve(args.query, section=args.section, k_final=args.kfinal)
    if not hits:
        print("No results.")
    else:
        for i, h in enumerate(hits, 1):
            m = h["meta"]
            print(
                f"[{i}] score={h['score']:.3f} — {m.get('file')} p.{m.get('page')} — {m.get('section')}"
            )
            snippet = textwrap.shorten(
                h["text"].replace("\n", " "), width=300, placeholder="…"
            )
            print("    ", snippet)

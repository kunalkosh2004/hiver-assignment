"""Hybrid retriever: normalized BM25 + dense scores blended by ``alpha``.

Both score vectors are min-max normalized across the whole corpus before the
linear combination, so the two very different scales are comparable without
retrieving only the union. alpha=0.5 keeps BM25 and dense balanced; the
evaluation tunes alpha on held-out relevance pairs (never golden labels).
"""
from __future__ import annotations

import numpy as np

from .base import RetrievalHit
from .bm25 import BM25Retriever
from .dense import DenseRetriever, _rank


class HybridRetriever:
    name = "hybrid"

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, alpha: float = 0.5):
        self.bm25 = bm25
        self.dense = dense
        self.alpha = float(alpha)
        self.store = bm25.store

    @staticmethod
    def _minmax(x: np.ndarray) -> np.ndarray:
        lo, hi = float(np.min(x)), float(np.max(x))
        if not np.isfinite(hi - lo) or hi - lo <= 1e-12:
            return np.zeros_like(x, dtype=float)
        return (x - lo) / (hi - lo)

    def score_all(self, query: str, context=None) -> np.ndarray:
        a = self.alpha
        return a * self._minmax(self.dense.score_all(query, context)) + \
            (1 - a) * self._minmax(self.bm25.score_all(query, context))

    def search(self, query: str, *, context=None, top_k: int = 5,
               timestamp_cutoff: str | None = None,
               exclude_conversation_ids: set[int] | None = None) -> list[RetrievalHit]:
        scores = self.score_all(query, context)
        return _rank(self.store, scores, top_k, timestamp_cutoff, exclude_conversation_ids)
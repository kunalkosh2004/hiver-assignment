"""TF-IDF cosine retriever (baseline).

Fits sklearn's TfidfVectorizer on the doc store and scores queries with a
sparse dot product. Persisted as ``vec.pkl`` + ``matrix.npz`` + ``meta.json``
under the artifact dir; loading reattaches to the (parquet-backed) case store.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import RetrievalHit
from .corpus import CaseStore


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, store: CaseStore, vec: TfidfVectorizer, X: sparse.csr_matrix):
        self.store = store
        self.vec = vec
        self.X = X

    @classmethod
    def fit(cls, store: CaseStore, *, max_features: int = 200_000, **kwargs) -> "TfidfRetriever":
        vec = TfidfVectorizer(lowercase=True, sublinear_tf=True, max_features=max_features)
        X = vec.fit_transform(store.texts())
        return cls(store, vec, X)

    def _text(self, query: str, context) -> str:
        return self.store.query_text(query, context)

    def score_all(self, query: str, context=None) -> np.ndarray:
        qv = self.vec.transform([self._text(query, context)])
        res = qv @ self.X.T
        dense = res.toarray() if sparse.issparse(res) else np.asarray(res)
        return dense.ravel()

    def search(self, query: str, *, context=None, top_k: int = 5,
               timestamp_cutoff: str | None = None,
               exclude_conversation_ids: set[int] | None = None) -> list[RetrievalHit]:
        scores = self.score_all(query, context)
        mask = self.store.filter_mask(timestamp_cutoff, exclude_conversation_ids)
        scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[: top_k]
        return [self.store.hit(int(i), float(scores[i])) for i in order if np.isfinite(float(scores[i]))]

    def save(self, base: Path) -> None:
        base.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(self.vec, base / "vec.pkl")
        sparse.save_npz(base / "matrix.npz", self.X.tocsr())
        (base / "meta.json").write_text(json.dumps({
            "model": self.name,
            "include_context": self.store.include_context,
            "n_docs": self.store.n(),
            "n_terms": self.X.shape[1],
        }))

    @classmethod
    def load(cls, base: Path, store: CaseStore) -> "TfidfRetriever":
        import joblib

        return cls(store, joblib.load(base / "vec.pkl"), sparse.load_npz(base / "matrix.npz"))
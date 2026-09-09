"""Self-contained Okapi BM25 retriever (no external bm25 package).

Scores with scipy sparse TF counts:
    score(d, q) = sum_t idf[t] * tf(d,t)*(k1+1) / (tf(d,t) + k1*(1 - b + b*len(d)/avgdl))

The vocabulary is built once and persisted, so reloads are byte-identical.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

from .base import RetrievalHit
from .corpus import CaseStore
from .text import tokenize


class BM25Retriever:
    name = "bm25"

    def __init__(self, store: CaseStore, tf: sparse.csr_matrix, idf: np.ndarray,
                 dl: np.ndarray, avgdl: float, k1: float = 1.5, b: float = 0.75,
                 vocab: list[str] | None = None):
        self.store = store
        self.tf = tf.tocsr()
        self.idf = np.asarray(idf, dtype=float)
        self.dl = np.asarray(dl, dtype=float)
        self.avgdl = float(avgdl)
        self.k1 = k1
        self.b = b
        self.vocab = vocab
        self._vocab_map = {t: i for i, t in enumerate(vocab)} if vocab else None

    @classmethod
    def fit(cls, store: CaseStore, *, k1: float = 1.5, b: float = 0.75,
            min_df: int = 2, max_terms: int = 500_000) -> "BM25Retriever":
        n_docs = store.n()
        texts = store.texts()
        tok_docs = [tokenize(t) for t in texts]

        counts = Counter()
        for doc in tok_docs:
            counts.update({t for t in doc})
        vocab_list = sorted(
            (t for t, c in counts.items() if c >= min_df),
            key=lambda t: (-counts[t], t),
        )[:max_terms]
        vocab_map = {t: i for i, t in enumerate(vocab_list)}

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        dl = np.zeros(n_docs, dtype=float)
        for i, doc in enumerate(tok_docs):
            dl[i] = len(doc)
            if not doc:
                continue
            for t, c in Counter(doc).items():
                j = vocab_map.get(t)
                if j is None:
                    continue
                rows.append(i)
                cols.append(j)
                data.append(float(c))
        tf = sparse.csr_matrix((data, (rows, cols)), shape=(n_docs, len(vocab_list)))
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        avgdl = float(dl.mean())
        return cls(store, tf, idf, dl, avgdl, k1=k1, b=b, vocab=vocab_list)

    def score_all(self, query: str, context=None) -> np.ndarray:
        scores = np.zeros(self.store.n(), dtype=float)
        toks = set(tokenize(self._query_text(query, context)))
        k1, b, avgdl = self.k1, self.b, self.avgdl
        for t in toks:
            j = self._vocab_map.get(t)
            if j is None:
                continue
            tf_col = self.tf.getcol(j)
            nz = tf_col.indices
            if len(nz) == 0:
                continue
            vals = tf_col.data
            dl_nz = self.dl[nz]
            denom = vals + k1 * (1 - b + b * dl_nz / avgdl)
            scores[nz] += self.idf[j] * (vals * (k1 + 1)) / denom
        return scores

    def _query_text(self, query: str, context) -> str:
        return self.store.query_text(query, context)

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
        sparse.save_npz(base / "tf.npz", self.tf)
        np.save(base / "idf.npy", self.idf)
        np.save(base / "dl.npy", self.dl)
        np.save(base / "vocab.npy", np.array(self.vocab, dtype="object"), allow_pickle=True)
        (base / "meta.json").write_text(json.dumps({
            "model": self.name,
            "include_context": self.store.include_context,
            "n_docs": self.store.n(),
            "n_terms": len(self.vocab),
            "avgdl": self.avgdl,
            "k1": self.k1,
            "b": self.b,
        }))

    @classmethod
    def load(cls, base: Path, store: CaseStore) -> "BM25Retriever":
        tf = sparse.load_npz(base / "tf.npz")
        idf = np.load(base / "idf.npy")
        dl = np.load(base / "dl.npy")
        vocab = (np.load(base / "vocab.npy", allow_pickle=True)).tolist()
        meta = json.loads((base / "meta.json").read_text())
        return cls(store, tf, idf, dl, meta["avgdl"], k1=meta["k1"], b=meta["b"], vocab=vocab)
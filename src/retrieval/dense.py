"""Dense retriever over the historical corpus using a local sentence encoder.

Uses sentence-transformers ``all-MiniLM-L6-v2`` (384-dim), which is already
cached offline. Embeddings are L2-normalized and searched with an exact cosine
dot product over a numpy matrix — a deterministic local index, no FAISS.

Artifacts:
    data/retrieval/embeddings/<variant>/embeddings.npy   (N x 384 float32)
    data/retrieval/embeddings/<variant>/manifest.json    (capability manifest)
    data/retrieval/index/dense/<variant>/meta.json       (index meta)

The manifest double-checks model / dimension / corpus version so a stale cache
cannot be silently reused after a corpus rebuild.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .base import RetrievalHit
from .corpus import CaseStore

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _embed_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def corpus_version(store: CaseStore) -> str:
    cases = store.cases
    if not cases:
        return "empty"
    return f"cases:{len(cases)}:{cases[0]['case_id']}:{cases[-1]['case_id']}"


class DenseRetriever:
    name = "dense"

    def __init__(self, store: CaseStore, embeddings: np.ndarray, model_name: str = MODEL_NAME):
        self.store = store
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.model_name = model_name

    # -- embeddings ------------------------------------------------------- #
    @classmethod
    def _cache_paths(cls, cache_dir: Path, variant: str):
        d = Path(cache_dir) / variant
        return d / "embeddings.npy", d / "manifest.json", d

    @classmethod
    def fit(cls, store: CaseStore, *, cache_dir: Path | str | None = None,
            batch_size: int = 256) -> "DenseRetriever":
        cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "embeddings"
        npy_path, manifest_path, d = cls._cache_paths(cache_dir, "ctx" if store.include_context else "msg")
        version = corpus_version(store)
        n = store.n()
        if npy_path.exists() and manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            if (m.get("corpus_version") == version and m.get("embedding_model") == MODEL_NAME
                    and m.get("embedding_dimension") == EMBED_DIM
                    and int(m.get("n_docs", -1)) == n):
                print(f"[dense] reuse cached embeddings {npy_path}", flush=True)
                return cls(store, np.load(npy_path))
        d.mkdir(parents=True, exist_ok=True)
        progress_path = d / "progress.json"
        start = 0
        if progress_path.exists():
            p = json.loads(progress_path.read_text())
            if p.get("corpus_version") == version and npy_path.exists():
                start = int(p.get("rows_done", 0))
        mode = "w+" if not npy_path.exists() else "r+"
        emb = np.lib.format.open_memmap(npy_path, mode=mode, dtype=np.float32, shape=(n, EMBED_DIM))
        model = _get_model()
        texts = store.texts()
        t0 = time.time()
        dev = _embed_device()
        for i in range(start, n, batch_size):
            batch = texts[i: i + batch_size]
            vecs = model.encode(
                batch, normalize_embeddings=True, convert_to_numpy=True,
                device=dev, show_progress_bar=False,
            )
            emb[i: i + batch_size] = np.asarray(vecs, dtype=np.float32)
            emb.flush()
            progress_path.write_text(json.dumps({
                "corpus_version": version, "rows_done": i + batch_size, "n_docs": n,
            }))
            if (i // batch_size) % 8 == 0:
                print(f"[dense] embedded {min(i+batch_size, n)}/{n} docs", flush=True)
        emb.flush()
        del emb
        manifest_path.write_text(json.dumps({
            "corpus_version": version,
            "embedding_model": MODEL_NAME,
            "embedding_dimension": EMBED_DIM,
            "preprocessing_version": "retrieval.text.v1",
            "n_docs": n,
            "variant": "ctx" if store.include_context else "msg",
            "embed_seconds": round(time.time() - t0, 1),
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2))
        progress_path.unlink(missing_ok=True)
        print(f"[dense] embedded {n} docs in {time.time()-t0:.0f}s", flush=True)
        return cls(store, np.load(npy_path))

    def _query_vec(self, query: str, context) -> np.ndarray:
        vec = _get_model().encode(
            [self.store.query_text(query, context)], normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vec[0], dtype=np.float32)

    # -- scoring ---------------------------------------------------------- #
    def score_all(self, query: str, context=None) -> np.ndarray:
        q = self._query_vec(query, context)
        return (self.embeddings @ q).astype(np.float32)

    def search(self, query: str, *, context=None, top_k: int = 5,
               timestamp_cutoff: str | None = None,
               exclude_conversation_ids: set[int] | None = None) -> list[RetrievalHit]:
        scores = self.score_all(query, context)
        return _rank(self.store, scores, top_k, timestamp_cutoff, exclude_conversation_ids)

    # -- persistence ------------------------------------------------------ #
    def save(self, base: Path) -> None:
        base.mkdir(parents=True, exist_ok=True)
        (base / "meta.json").write_text(json.dumps({
            "model": self.name,
            "variant": "ctx" if self.store.include_context else "msg",
            "include_context": self.store.include_context,
            "n_docs": self.store.n(),
            "embedding_model": self.model_name,
            "embedding_dimension": int(self.embeddings.shape[1]),
            "cache_key": corpus_version(self.store),
        }))

    @classmethod
    def load(cls, base: Path, store: CaseStore, *, cache_dir: Path | str | None = None) -> "DenseRetriever":
        cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "embeddings"
        npy_path, _, __ = cls._cache_paths(cache_dir, "ctx" if store.include_context else "msg")
        emb = np.load(npy_path, mmap_mode="r")
        return cls(store, np.array(emb))


def _rank(store: CaseStore, scores: np.ndarray, top_k: int,
          timestamp_cutoff: str | None, exclude_conversation_ids: set[int] | None) -> list[RetrievalHit]:
    mask = store.filter_mask(timestamp_cutoff, exclude_conversation_ids)
    scores = np.where(mask, scores, -np.inf)
    order = np.argsort(-scores)[: top_k]
    return [store.hit(int(i), float(scores[i])) for i in order if np.isfinite(float(scores[i]))]
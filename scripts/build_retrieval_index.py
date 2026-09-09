"""Build local retrieval indexes over the historical AmazonHelp corpus.

Writes under data/retrieval/index/<model>/<variant>/ and a top-level
manifest.json. Everything is deterministic and rebuildable:

    python scripts/build_retrieval_index.py                     # all models, both variants
    python scripts/build_retrieval_index.py --models bm25       # lexical only, both variants
    python scripts/build_retrieval_index.py --variant msg       # message-only only

Models: tfidf | bm25 | dense | hybrid. Variants: msg | ctx.
The golden benchmark + reserved holdout are already excluded from the corpus;
we additionally assert no golden conversation id ever reaches an index.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.retrieval.corpus import (  # noqa: E402
    CaseStore,
    assert_no_golden_overlap,
    load_corpus,
    save_slim_corpus,
)

INDEX_DIR = config.DATA_DIR / "retrieval" / "index"


def _holds_out() -> set[int]:
    with (config.DATA_DIR / "golden" / "conversation_holdout.json").open() as fh:
        return set(json.load(fh).get("holdout_conversation_ids", [])) or set()


def build_model(store: CaseStore, model: str, variant: str) -> dict:
    out = INDEX_DIR / model / variant
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model": model, "variant": variant, "include_context": store.include_context}
    t0 = time.time()
    if model == "tfidf":
        from src.retrieval.tfidf import TfidfRetriever

        r = TfidfRetriever.fit(store)
        r.save(out)
    elif model == "bm25":
        from src.retrieval.bm25 import BM25Retriever

        r = BM25Retriever.fit(store)
        r.save(out)
    elif model == "dense":
        from src.retrieval.dense import DenseRetriever

        r = DenseRetriever.fit(store, cache_dir=config.DATA_DIR / "retrieval" / "embeddings")
        r.save(out)
    elif model == "hybrid":
        from src.retrieval.dense import DenseRetriever
        from src.retrieval.hybrid import HybridRetriever

        bm = _load_existing("bm25", variant, store)
        dn = DenseRetriever.load(
            INDEX_DIR / "dense" / variant,
            store,
            cache_dir=config.DATA_DIR / "retrieval" / "embeddings",
        )
        r = HybridRetriever(bm, dn, alpha=0.5)
        (out / "meta.json").write_text(json.dumps({
            "model": "hybrid", "variant": variant,
            "include_context": store.include_context,
            "alpha": 0.5, "components": ["bm25", "dense"],
        }))
        return {"model": "hybrid", "variant": variant, "n_docs": store.n(), "build_s": time.time() - t0}
    else:
        raise ValueError(model)

    return {"model": model, "variant": variant, "n_docs": store.n(), "build_s": time.time() - t0}


def _load_existing(model: str, variant: str, store: CaseStore):
    from src.retrieval.bm25 import BM25Retriever

    if model == "bm25":
        return BM25Retriever.load(INDEX_DIR / "bm25" / variant, store)
    raise ValueError(model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tfidf,bm25,dense,hybrid")
    ap.add_argument("--variant", default="both", choices=("msg", "ctx", "both"))
    args = ap.parse_args()

    cases = load_corpus()
    print(f"[index] {len(cases)} cases", flush=True)
    save_slim_corpus(cases, str(config.DATA_DIR / "retrieval" / "corpus.slim.parquet"))

    indexed_ids = {int(c["conversation_id"]) for c in cases}
    assert_no_golden_overlap(indexed_ids)

    req_models = [m for m in args.models.split(",") if m]
    variants = ["msg", "ctx"] if args.variant == "both" else [args.variant]
    built = []
    for variant in variants:
        store = CaseStore(cases, include_context=(variant == "ctx"))
        for model in req_models:
            info = build_model(store, model, variant)
            built.append(info)
            print(f"[index] built {model}/{variant} in {info['build_s']:.1f}s", flush=True)

    underlying_n = len(cases)
    manifest = {
        "corpus": {
            "n_cases": underlying_n,
            "usable_responses": sum(bool(c["has_usable_response"]) for c in cases),
            "conversations": len({c["conversation_id"] for c in cases}),
        },
        "preprocessing_version": "retrieval.text.v1",
        "dashboard": built,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (INDEX_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[index] done; manifest at {INDEX_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
"""Corpus-size scaling experiment for retrieval (message-only inputs).

    python scripts/evaluate_retrieval_scaling.py [--models bm25,dense] \
                    [--sizes 10000,50000,100000,full]

Subsets are nested: one seeded shuffle (seed 42) of the full corpus order, so
10k < 50k < 100k < full. Dense subsets slice the full embedding matrix; lexical
retrievers are refit on the subset. R@5/MRR come from the human relevance
benchmark; per-intent / confidence stats can optionally use the weak-intent
proxy (explicitly labeled, never ground truth).

Outputs:
    reports/retrieval_scaling.{csv,json}   size x model x R@5/MRR/latency/index bytes
    figures/retrieval_scaling_curve.png
    figures/retrieval_latency.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.corpus import CaseStore, load_corpus  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402
from src.retrieval.tfidf import TfidfRetriever  # noqa: E402

REL_DIR = config.DATA_DIR / "retrieval" / "relevance"
EMB_DIR = config.DATA_DIR / "retrieval" / "embeddings"
REPORTS = config.DATA_DIR.parent / "reports"
SEED = 42


class _DenseSubset(DenseRetriever):
    """Dense retriever over a slice of the full (message) embedding matrix."""
    pass


def load_benchmark():
    queries = [json.loads(l) for l in (REL_DIR / "queries.jsonl").open()]
    gold = {}
    for line in (REL_DIR / "relevance_annotations.jsonl").open():
        r = json.loads(line)
        gold.setdefault(r["query_id"], {})[r["candidate_case_id"]] = int(r["level"])
    return queries, gold


def metrics(retriever, queries, gold) -> dict:
    r5s, mrrs, lats = [], [], []
    for q in queries:
        t0 = time.perf_counter()
        hits = retriever.search(
            q["customer_message"], context=None, top_k=5,
            timestamp_cutoff=q.get("created_at"),
        )
        lats.append((time.perf_counter() - t0) * 1000)
        rel = gold.get(q["query_id"], {})
        relevant = {cid for cid, lv in rel.items() if lv >= 1}
        tot = max(1, len(relevant))
        r5 = sum(1 for h in hits if h.case_id in relevant) / tot
        mr = 0.0
        for i, h in enumerate(hits, start=1):
            if h.case_id in relevant:
                mr = 1.0 / i
                break
        r5s.append(r5)
        mrrs.append(mr)
    return {
        "R@5": float(np.mean(r5s)), "MRR": float(np.mean(mrrs)),
        "lat_ms_avg": float(np.mean(lats)), "lat_ms_p95": float(np.quantile(lats, 0.95)),
    }


def subset_retriever(model: str, cases, idx, emb_full=None):
    sub = [cases[i] for i in idx]
    store = CaseStore(sub, include_context=False)
    t0 = time.time()
    if model == "bm25":
        r = BM25Retriever.fit(store)
        size_bytes = r.tf.data.nbytes + r.idf.nbytes + r.dl.nbytes
    elif model == "tfidf":
        r = TfidfRetriever.fit(store)
        size_bytes = r.X.data.nbytes + r.X.indices.nbytes + r.X.indptr.nbytes
    elif model == "dense":
        r = _DenseSubset(store, emb_full[idx])
        size_bytes = int(idx.size) * 384 * 4
    else:
        raise ValueError(model)
    return r, time.time() - t0, size_bytes, store.n()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="bm25,dense")
    ap.add_argument("--sizes", default="10000,50000,100000,full")
    args = ap.parse_args()

    models = args.models.split(",")
    sizes = [int(s) if s != "full" else None for s in args.sizes.split(",")]

    queries, gold = load_benchmark()
    print(f"[scale] {len(queries)} queries, {sum(len(v) for v in gold.values())} judged pairs", flush=True)
    cases = load_corpus()
    n = len(cases)
    order = list(range(n))
    rng = np.random.RandomState(SEED)
    rng.shuffle(order)

    emb_full = None
    if "dense" in models:
        variant = "msg"
        npy = EMB_DIR / variant / "embeddings.npy"
        emb_full = np.load(npy)
        print(f"[scale] full embeddings {emb_full.shape}", flush=True)

    rows = []
    for size in sizes:
        k = n if size is None else size
        idx = np.array(order[: k], dtype=int)
        print(f"[scale] subset {len(idx)}", flush=True)
        for model in models:
            r, build_s, bytes_used, n_docs = subset_retriever(model, cases, idx, emb_full)
            m = metrics(r, queries, gold)
            rows.append({
                "corpus_size": k, "model": model, "build_seconds": round(build_s, 2),
                "index_bytes": int(bytes_used), "n_docs": n_docs,
                **m,
            })
            print(f"  {model:5s} n={k:>6d} R@5={m['R@5']:.3f} MRR={m['MRR']:.3f} "
                  f"lat={m['lat_ms_avg']:.0f}ms build={build_s:.1f}s", flush=True)

    (REPORTS).mkdir(exist_ok=True)
    with (REPORTS / "retrieval_scaling.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (REPORTS / "retrieval_scaling.json").write_text(json.dumps({
        "seed": SEED, "nested_subsets": True, "input_variant": "message-only",
        "metric_source": "human relevance benchmark",
        "weak_intent_proxy_note": "not used for headline R@5/MRR; proxy only for intent breakdowns",
        "rows": rows,
    }, indent=2))
    _plot(rows)


def _plot(rows):
    by = {}
    for r in rows:
        by.setdefault(r["model"], {"sizes": [], "R5": [], "lat": []})
        by[r["model"]]["sizes"].append(r["corpus_size"])
        by[r["model"]]["R5"].append(r["R@5"])
        by[r["model"]]["lat"].append(r["lat_ms_avg"])
    fig, ax1 = plt.subplots(figsize=(8, 5))
    for model, d in by.items():
        ax1.plot(d["sizes"], d["R5"], marker="o", label=f"{model} R@5")
        ax1.set_xscale("symlog")
    ax1.set_xlabel("corpus size")
    ax1.set_ylabel("R@5")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "retrieval_scaling_curve.png", dpi=130)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    for model, d in by.items():
        ax2.plot(d["sizes"], d["lat"], marker="s", label=f"{model} lat avg (ms)")
    ax2.set_xscale("symlog")
    ax2.set_xlabel("corpus size")
    ax2.set_ylabel("query latency (ms)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(REPORTS / "figures" / "retrieval_latency.png", dpi=130)


if __name__ == "__main__":
    main()
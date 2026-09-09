"""Evaluate retrieval against the human-labeled relevance benchmark.

    python scripts/evaluate_retrieval.py

Runs every {model x input-variant} over the golden-derived query set, applying
the temporal filter (retrieved interaction must precede the query) and computes
pool-based recall (unjudged top-K hits count as misses — standard pooling
assumption, disclosed in reports). Reports:

reports/retrieval_results.{csv,json}   R@1/3/5/10, MRR, useful@5, canned@5,
                                       no-relevant-top5, latency
reports/retrieval_raw_hits.jsonl       top-10 hit dump for error analysis
figures/retrieval_recall_at_k.png      recall curve by retriever
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.corpus import CaseStore, load_corpus  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402
from src.retrieval.tfidf import TfidfRetriever  # noqa: E402

REL_DIR = config.DATA_DIR / "retrieval" / "relevance"
INDEX_DIR = config.DATA_DIR / "retrieval" / "index"
EMB_DIR = config.DATA_DIR / "retrieval" / "embeddings"
REPORTS = config.DATA_DIR.parent / "reports"

K_LIST = [1, 3, 5, 10]
MODELS = ["tfidf", "bm25", "dense", "hybrid"]

LEVEL_RELEVANT = 1
LEVEL_USEFUL = 2


def load_annotations() -> dict[str, dict[str, int]]:
    out = {}
    for line in (REL_DIR / "relevance_annotations.jsonl").open():
        r = json.loads(line)
        out.setdefault(r["query_id"], {})[r["candidate_case_id"]] = int(r["level"])
    return out


def load_queries() -> list[dict]:
    return [json.loads(l) for l in (REL_DIR / "queries.jsonl").open()]


def build_retrievers(cases: list[dict]) -> dict[tuple[str, str]]:
    out = {}
    for variant in ("msg", "ctx"):
        store = CaseStore(cases, include_context=(variant == "ctx"))
        for model, cls, kw in (
            ("tfidf", TfidfRetriever, {}),
            ("bm25", BM25Retriever, {}),
        ):
            out[(model, variant)] = cls.load(INDEX_DIR / model / variant, store, **kw)
        out[("dense", variant)] = DenseRetriever.load(INDEX_DIR / "dense" / variant, store, cache_dir=EMB_DIR)
        out[("hybrid", variant)] = HybridRetriever(
            out[("bm25", variant)], out[("dense", variant)], alpha=0.5
        )
    return out


def evaluate_query(retriever, q: dict, gold: dict[str, int]) -> dict:
    t0 = time.perf_counter()
    hits = retriever.search(
        q["customer_message"], context=q.get("conversation_context") or "", top_k=10,
        timestamp_cutoff=q.get("created_at"),
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    ranked = [h.case_id for h in hits]
    judged = [gold.get(cid) for cid in ranked]
    relevant_at = [i + 1 for i, lv in enumerate(judged) if lv is not None and lv >= LEVEL_RELEVANT]
    useful_at = [i + 1 for i, lv in enumerate(judged) if lv is not None and lv >= LEVEL_USEFUL]

    total_relevant = sum(1 for lv in gold.values() if lv >= LEVEL_RELEVANT)
    total_useful = sum(1 for lv in gold.values() if lv >= LEVEL_USEFUL)

    m = {"latency_ms": latency_ms, "ranked": ranked,
         "relevant_at": relevant_at, "useful_at": useful_at,
         "total_relevant": total_relevant, "total_useful": total_useful,
         "canned_ranked": [bool(h.metadata.get("is_canned_template")) for h in hits]}
    m["relevant_ids"] = {cid for cid in ranked if gold.get(cid) is not None and gold.get(cid) >= LEVEL_RELEVANT}
    return m


def recall(m: dict, k: int, mode: str = "relevant") -> float:
    key = "relevant_at" if mode == "relevant" else "useful_at"
    total = m[f"total_{mode}"]
    if total == 0:
        return 0.0
    return float(sum(x <= k for x in m[key])) / total


def mrr(m: dict) -> float:
    if not m["relevant_at"]:
        return 0.0
    return 1.0 / m["relevant_at"][0]


def sum_metric(name, results) -> dict:
    vals = [r[name] for r in results]
    return {"avg": sum(vals) / len(vals), "p95": _p95(vals), "n": len(vals)}


def _p95(xs):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    return float(xs[int(0.95 * (len(xs) - 1))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, nargs="*", default=[0.3, 0.5, 0.7],
                    help="hybrid alphas to report")
    args = ap.parse_args()

    queries = load_queries()
    gold = load_annotations()
    print(f"[eval] {len(queries)} queries, {sum(len(v) for v in gold.values())} judged pairs", flush=True)

    cases = load_corpus()
    retrievers = build_retrievers(cases)
    store = CaseStore(cases, include_context=False)

    results = []
    raw_rows = []
    for (model, variant), retriever in sorted(retrievers.items()):
        agg = []
        for q in queries:
            m = evaluate_query(retriever, q, gold.get(q["query_id"], {}))
            agg.append({"query_id": q["query_id"], "model": model, "variant": variant, **m})
            for r in raw_rows_fmt(model, variant, q, m):
                raw_rows.append(r)
        # alpha variants for hybrid
        results.append(aggregate_row(model, variant, agg))
        if model == "hybrid":
            for a in args.alpha:
                if abs(a - 0.5) < 1e-9:
                    continue
                rh = HybridRetriever(retrievers[("bm25", variant)], retrievers[("dense", variant)], alpha=a)
                agg2 = [evaluate_query(rh, q, gold.get(q["query_id"], {})) | {"query_id": q["query_id"]}
                        for q in queries]
                results.append(aggregate_row(f"hybrid_a{int(a*10):02d}", variant, agg2))

    _write_outputs(results, raw_rows, args)

    table = ["{:<14} {:<4} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>6} {:>6}".format(
        "model", "in", "R@1", "R@3", "R@5", "R@10", "MRR", "use5", "nocanR@5", "t_ms")]
    for r in results:
        table.append("{:<14} {:<4} {:>5.3f} {:>5.3f} {:>5.3f} {:>5.3f} {:>5.3f} {:>5.3f} {:>6.3f} {:>6.1f}".format(
            r["model"], r["variant"], r["R@1"], r["R@3"], r["R@5"], r["R@10"], r["MRR"],
            r["useful_R@5"], r["no_canned_R@5"], r["latency_ms_avg"]))
    print("\n".join(table))


def raw_rows_fmt(model, variant, q, m):
    out = {"query_id": q["query_id"], "model": model, "variant": variant,
           "primary_intent": q["primary_intent"], "language": q["language"],
           "difficulty": q["difficulty"]}
    for i, cid in enumerate(m["ranked"]):
        out[f"hit{i+1}"] = cid
    return [out]


def aggregate_row(model, variant, agg: list[dict]) -> dict:
    n = len(agg)
    r = {"model": model, "variant": variant, "n_queries": n}
    for k in K_LIST:
        r[f"R@{k}"] = sum(recall(a, k) for a in agg) / n
        r[f"useful_R@{k}"] = sum(recall(a, k, "useful") for a in agg) / n
    r["MRR"] = sum(mrr(a) for a in agg) / n
    r["no_relevant_top5"] = sum(1 for a in agg if not a["relevant_at"] or a["relevant_at"][0] > 5) / n
    r["no_useful_top5"] = sum(1 for a in agg if not a["useful_at"] or a["useful_at"][0] > 5) / n
    r["canned_rate_at5"] = sum(1 for a in agg if any(a["canned_ranked"][:5])) / n
    r["latency_ms_avg"] = sum(a["latency_ms"] for a in agg) / n
    r["latency_ms_p95"] = _p95([a["latency_ms"] for a in agg])
    # no-canned variant: drop canned hits, refill to top K
    def nc_recall(a, k):
        nc = [cid for cid, can in zip(a["ranked"], a["canned_ranked"]) if not can][:k]
        hits = set(nc) & a["relevant_ids"]
        return len(hits) / a["total_relevant"] if a["total_relevant"] else 0.0

    r["no_canned_R@5"] = sum(nc_recall(a, 5) for a in agg) / n
    r["no_canned_R@10"] = sum(nc_recall(a, 10) for a in agg) / n
    return r


def _write_outputs(results, raw_rows, args) -> None:
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    import csv

    csv_path = REPORTS / "retrieval_results.csv"
    with csv_path.open("w", newline="") as fh:
        cols = ["model", "variant", "n_queries", *[f"R@{k}" for k in K_LIST],
                *[f"useful_R@{k}" for k in K_LIST],
                "MRR", "no_relevant_top5", "no_useful_top5", "canned_rate_at5",
                "no_canned_R@5", "no_canned_R@10", "latency_ms_avg", "latency_ms_p95"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({c: r.get(c) for c in cols})

    (REPORTS / "retrieval_results.json").write_text(json.dumps({
        "k_list": K_LIST,
        "alpha_values": args.alpha,
        "pooling_notes": ("R@K is pool-based: candidates in the judged pool are the only "
                          "positives/unjudged top-K hits count as misses. Hybrid alphas are "
                          "reported for every value; alpha is NOT tuned on golden labels."),
        "results": results,
        "raw_hits": raw_rows,
    }, indent=2))

    _plot_recall_at_k(results)


def _plot_recall_at_k(results) -> None:
    by = {}
    for r in results:
        by[(r["model"], r["variant"])] = [r[f"R@{k}"] for k in K_LIST]
    fig, ax = plt.subplots(figsize=(8, 5))
    for (model, variant), ys in sorted(by.items()):
        ax.plot(K_LIST, ys, marker="o", label=f"{model}/{variant}")
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K (pool-based)")
    ax.set_xticks(K_LIST)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "retrieval_recall_at_k.png", dpi=130)


if __name__ == "__main__":
    main()
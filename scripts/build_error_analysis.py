"""Categorized retrieval error analysis (>=30 failure instances).

For every retriever we flag a failure whenever a top-5 window contains zero
"useful" (level 2) candidates although at least one exists in the judged pool
(a "full" miss) or when it only lands a strict subset (a "partial" miss).
Each failure is assigned a human mode (semantic near-miss, lexical mismatch,
multilingual crossover, ambiguous/telegraphic query, insufficient context
handling, none-intent noise, benchmark-absence). Evidence records the retrieved
top-5 levels and the specifically missed useful candidate ids.

    python scripts/build_error_analysis.py
    -> reports/retrieval_error_analysis.json   (per-retriever + per-query)
    -> reports/retrieval_error_analysis.csv
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

REPORTS = config.DATA_DIR.parent / "reports"
REL = config.DATA_DIR / "retrieval" / "relevance"

LEVELS = {0: "irrelevant", 1: "related", 2: "useful"}

# Human modes for the headline retriever (dense / message). Empty = no failure.
DENSE_MSG_MODES = {
    "challenge-1379340": "insufficient_context_handling",
    "challenge-1568158": "semantic_near_miss",
    "challenge-1578555": "multilingual_low_resources",
    "challenge-2170818": "ambiguous_telegraphic_query",
    "challenge-2694949": "semantic_near_miss",
    "challenge-2814438": "multilingual_low_resources",
    "challenge-842315": "semantic_near_miss",
    "challenge-893308": "none_intent_noise",
    "representative-1136222": "semantic_near_miss",
    "representative-1179639": "semantic_near_miss",
    "representative-1210214": "semantic_near_miss",
    "representative-2217342": "semantic_near_miss",
    "representative-2291505": "semantic_near_miss",
    "representative-2389800": "none_intent_noise",
    "representative-2664071": "semantic_near_miss",
    "representative-2683519": "semantic_near_miss",
    "representative-322945": "semantic_near_miss",
    "representative-458877": "semantic_near_miss",
    "representative-533206": "semantic_near_miss",
    "representative-628261": "semantic_near_miss",
}


def load():
    queries = [json.loads(l) for l in (REL / "queries.jsonl").open()]
    raw = json.loads((REPORTS / "retrieval_results.json").read_text())["raw_hits"]
    ann: dict[str, dict[str, int]] = {}
    for l in (REL / "relevance_annotations.jsonl").open():
        r = json.loads(l)
        ann.setdefault(r["query_id"], {})[r["candidate_case_id"]] = int(r["level"])
    return queries, raw, ann


def analyze(queries, raw, ann):
    rows = []
    seen = set()
    for (model, variant) in [("dense", "msg"), ("bm25", "msg"), ("tfidf", "msg"),
                             ("dense", "ctx"), ("bm25", "ctx"), ("tfidf", "ctx")]:
        hits_by_q = {r["query_id"]: [r[f"hit{i}"] for i in range(1, 6)]
                     for r in raw if r["model"] == model and r["variant"] == variant}
        for q in queries:
            qid = q["query_id"]
            pool = ann.get(qid, {})
            top = hits_by_q.get(qid, [])
            missing_useful = [c for c, lv in pool.items() if lv == 2 and c not in top]
            missing_related = [c for c, lv in pool.items() if lv >= 1 and c not in top]
            got_useful = sum(1 for c in top if pool.get(c) == 2)
            total_useful = sum(1 for lv in pool.values() if lv == 2)
            if len(missing_useful) == 0:
                continue
            if model == "dense" and variant == "msg":
                mode = DENSE_MSG_MODES.get(qid, "semantic_near_miss")
            elif q["language"] != "en":
                mode = "multilingual_crossover"
            else:
                mode = "lexical_mismatch"
            rows.append({
                "query_id": qid, "model": model, "input_variant": variant,
                "language": q["language"], "difficulty": q["difficulty"],
                "primary_intent": q["primary_intent"],
                "mode": mode,
                "miss_kind": "full_useful_miss" if got_useful == 0 else "partial_useful_miss",
                "useful_retrieved": got_useful, "useful_total": total_useful,
                "missed_useful_case_ids": missing_useful,
                "missed_related_case_ids": [c for c in missing_related if c not in set(missing_useful)],
                "top5_levels": [LEVELS[pool.get(c, 0 if pool.get(c) is None else pool.get(c))]
                                if pool.get(c) is not None else "UNJUDGED" for c in top],
            })
    # mark queries whose judged pool has NO "useful" candidate at all: no
    # retriever can be blamed for a miss, but the pool itself limits the eval.
    no_useful = [q for q in queries
                 if sum(1 for lv in ann.get(q["query_id"], {}).values() if lv == 2) == 0]
    for q in no_useful:
        rows.append({
            "query_id": q["query_id"], "model": "(pool)", "input_variant": "benchmark",
            "language": q["language"], "difficulty": q["difficulty"],
            "primary_intent": q["primary_intent"],
            "mode": "benchmark_absence",
            "miss_kind": "no_useful_candidate_in_pool",
            "useful_retrieved": 0, "useful_total": 0,
            "missed_useful_case_ids": [],
            "missed_related_case_ids": [c for c, lv in ann[q["query_id"]].items() if lv == 1],
            "top5_levels": [],
        })
    # sort: headline (dense/msg) first, then the rest
    order = {"dense": 0, "bm25": 1, "tfidf": 2}
    rows.sort(key=lambda r: (order.get(r["model"], 9), r["input_variant"], r["query_id"]))
    return rows


def main() -> None:
    queries, raw, ann = load()
    rows = analyze(queries, raw, ann)
    by_retriever = Counter((r["model"], r["input_variant"]) for r in rows)
    by_mode = Counter(r["mode"] for r in rows)
    by_kind = Counter(r["miss_kind"] for r in rows)

    REPORTS.mkdir(exist_ok=True)
    csv_path = REPORTS / "retrieval_error_analysis.csv"
    with csv_path.open("w", newline="") as fh:
        cols = list(rows[0].keys())
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    (REPORTS / "retrieval_error_analysis.json").write_text(json.dumps({
        "n_failures": len(rows),
        "method": ("A failure is any retriever whose top-5 omits at least one judged "
                   "'useful' (level 2) candidate while >=1 exists in the pool. "
                   "Levels: 0 irrelevant, 1 related, 2 useful."),
        "by_retriever": {f"{k[0]}/{k[1]}": v for k, v in sorted(by_retriever.items())},
        "by_mode": dict(by_mode),
        "by_kind": dict(by_kind),
        "failures": rows,
    }, indent=2))

    print(f"{len(rows)} failure rows\nby retriever:", dict(by_retriever))
    print("by kind:", dict(by_kind))
    print("by mode:", dict(by_mode))


if __name__ == "__main__":
    main()
"""Prepare the Phase-5 retrieval relevance benchmark.

Selects a stratified subset of *golden* queries (the golden set's labels are
never used for retrieval or indexing — only its queries + timestamps drive
evaluation), pools candidate historical interactions from every retriever, and
writes human-annotation scaffolding under data/retrieval/relevance/.

    python scripts/prepare_relevance_benchmark.py [--max-per-query 5]

Levels are assigned by hand in relevance_annotations.jsonl:
    level: 0 = irrelevant, 1 = related, 2 = useful
    (see review.md for context; relevance concerns the *historical interaction*,
     not whether the golden issue was ever resolved.)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.retrieval.corpus import CaseStore, load_corpus  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.tfidf import TfidfRetriever  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402

GOLD_DIR = config.DATA_DIR / "golden"
REL_DIR = config.DATA_DIR / "retrieval" / "relevance"
INDEX_DIR = config.DATA_DIR / "retrieval" / "index"
EMB_DIR = config.DATA_DIR / "retrieval" / "embeddings"

DIFF_W = {"hard": 2, "medium": 1, "easy": 0}
INTENT_BLACKLIST = {"none"}


def load_golden_with_refs() -> list[dict]:
    golden = [json.loads(l) for l in (GOLD_DIR / "amazon_golden_eval.jsonl").open()]
    ref = [json.loads(l) for l in (GOLD_DIR / "amazon_golden_eval_reference.jsonl").open()]
    refmap = {str(r["conversation_id"]): r for r in ref}
    out = []
    for g in golden:
        r = refmap.get(str(g["conversation_id"]))
        if r:
            g["created_at"] = r["created_at"]
            g["tweet_id"] = r["tweet_id"]
            out.append(g)
    return out


def pick_queries(golden: list[dict], n: int, n_multilingual: int, seed: int = 42) -> list[dict]:
    """Stratified selection: intent groups, difficulty-heavy, N multilingual."""
    rng = random.Random(seed)
    english = [g for g in golden if g["language"] == "en"]
    multi = [g for g in golden if g["language"] != "en"]

    def stratified(pool, want: int, blacklist: set[str] | None = None) -> list[dict]:
        groups = defaultdict(list)
        for g in pool:
            k = g["primary_intent"] if g["primary_intent"] not in (blacklist or set()) else "_other"
            groups[k].append(g)
        groups = dict(sorted(groups.items(), key=lambda kv: -len(kv[1])))
        keep = {k: max(1, round(want * len(v) / max(1, len(pool)))) for k, v in groups.items()}
        # exact-fit loop: trim/adjust to `want`
        total = sum(keep.values())
        while total != want:
            if total > want:
                keys = [k for k in keep if k != "_other" and keep[k] > 1]
                if not keys:
                    keys = [k for k in keep]
                k = keys[-1]
                keep[k] -= 1
            else:
                k = next((k for k in groups if keep[k] < len(groups[k])), None)
                if k is None:
                    break
                keep[k] += 1
            total = sum(keep.values())
        picked = []
        for k, v in groups.items():
            sl = sorted(v, key=lambda g: (-DIFF_W[g["difficulty"]], rng.random()))
            picked.extend(sl[: keep[k]])
        return picked[:want]

    multi_pick = stratified(multi, n_multilingual)
    en_pick = stratified(english, n - n_multilingual)
    return en_pick + multi_pick


def build_retrievers(cases: list[dict]) -> dict[tuple[str, str, str]]:
    out = {}
    for variant in ("msg", "ctx"):
        store = CaseStore(cases, include_context=(variant == "ctx"))
        for model, cls, kwargs in (
            ("tfidf", TfidfRetriever, {}),
            ("bm25", BM25Retriever, {}),
        ):
            out[(model, variant)] = cls.load(INDEX_DIR / model / variant, store, **kwargs)
        out[("dense", variant)] = DenseRetriever.load(
            INDEX_DIR / "dense" / variant, store, cache_dir=EMB_DIR
        )
        bm: BM25Retriever = out[("bm25", variant)]
        dn: DenseRetriever = out[("dense", variant)]
        out[("hybrid", variant)] = HybridRetriever(bm, dn, alpha=0.5)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-query", type=int, default=5)
    ap.add_argument("--n-queries", type=int, default=30)
    ap.add_argument("--n-multilingual", type=int, default=6)
    args = ap.parse_args()

    golden = load_golden_with_refs()
    queries = pick_queries(golden, args.n_queries, args.n_multilingual)
    print(f"[bench] selected {len(queries)} queries "
          f"({sum(1 for q in queries if q['language'] != 'en')} non-en), intents:",
          dict(Counter(q["primary_intent"] for q in queries)), flush=True)

    cases = load_corpus()
    store_id = {c["case_id"]: c for c in cases}
    retrievers = build_retrievers(cases)
    print(f"[bench] loaded {len(retrievers)} retriever variants", flush=True)

    rel_dir = REL_DIR
    rel_dir.mkdir(parents=True, exist_ok=True)
    qrows, arose, annrows, review_lines = [], [], [], []
    review_lines.append("# Retrieval relevance annotation (query -> historical interactions)\n")
    review_lines.append("Levels: **0** irrelevant  |  **1** related  |  **2** useful (directly usable response/history)\n")

    for qi, q in enumerate(queries):
        qid = q["example_id"]
        ctx_str = q.get("conversation_context") or ""
        pool: dict[str, dict] = {}
        for (model, variant), retriever in retrievers.items():
            hits = retriever.search(
                q["current_customer_message"], context=ctx_str, top_k=5,
                timestamp_cutoff=q["created_at"],
            )
            for rank, h in enumerate(hits, start=1):
                info = pool.setdefault(h.case_id, {
                    "case_id": h.case_id, "sources": [], "best_rank": 99, "n_sources": 0,
                })
                info["sources"].append({"model": model, "variant": variant, "rank": rank})
                info["best_rank"] = min(info["best_rank"], rank)
                info["n_sources"] = len(info["sources"])

        ordered = sorted(pool.values(), key=lambda c: (-c["n_sources"], c["best_rank"]))
        ordered = ordered[: args.max_per_query]

        qrows.append({
            "query_id": qid,
            "conversation_id": q["conversation_id"],
            "customer_message": q["current_customer_message"],
            "conversation_context": q.get("conversation_context") or "",
            "created_at": q["created_at"],
            "primary_intent": q["primary_intent"],
            "secondary_intents": q.get("secondary_intents") or [],
            "language": q["language"],
            "difficulty": q["difficulty"],
            "context_dependency": bool(q.get("context_dependency")),
            "golden_reference_response": q.get("next_brand_response") or "",  # the actual reply
            "n_candidates": len(ordered),
        })

        review_lines.append(f"\n## {qid}  [{q['primary_intent']} | {q['language']} | {q['difficulty']}]")
        review_lines.append(f"\nCUSTOMER: {q['current_customer_message']}")
        review_lines.append(f"\nCONTEXT: {(q.get('conversation_context') or '[no prior context]')}")
        review_lines.append(f"\nGOLDEN ACTUAL REPLY: {q.get('next_brand_response', '')}")

        for c in ordered:
            rec = store_id.get(c["case_id"])
            if rec is None:
                continue
            arose.append({
                "query_id": qid,
                "candidate_case_id": c["case_id"],
                "customer_message": rec["customer_message"],
                "brand_response": rec.get("brand_response"),
                "conversation_context": rec.get("conversation_context") or [],
                "language": rec.get("language", ""),
                "is_canned_template": bool(rec.get("is_canned_template")),
                "template_count": int(rec.get("template_count", 0)),
                "sources": c["sources"],
                "n_sources": c["n_sources"],
                "best_rank": c["best_rank"],
            })
            annrows.append({"query_id": qid, "candidate_case_id": c["case_id"], "level": None, "note": None})
            review_lines.append(
                f"\n  CANDIDATE {c['case_id']} (lang={rec.get('language','?')}, "
                f"canned={'Y' if rec.get('is_canned_template') else 'N'}, "
                f"sources={','.join(sorted({s['model'] + '/' + s['variant'] for s in c['sources']}))})\n"
                f"    CUSTOMER: {rec['customer_message']}\n"
                f"    REPLY: {str(rec.get('brand_response')) or '[no reply]'}"
            )

    (rel_dir / "queries.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in qrows))
    (rel_dir / "candidates.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in arose))
    (rel_dir / "relevance_annotations.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in annrows)
    )
    (rel_dir / "review.md").write_text("\n".join(review_lines))
    print(f"[bench] wrote {len(qrows)} queries, {len(arose)} candidates, "
          f"{len(annrows)} annotations to {rel_dir}")


if __name__ == "__main__":
    main()
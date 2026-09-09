"""Phase 5A — build the clean historical AmazonHelp support corpus.

Run:
    python scripts/build_retrieval_corpus.py

Outputs:
    data/retrieval/corpus.jsonl          clean historical cases (git-ignored)
    data/retrieval/corpus_stats.json     forensic statistics table
    data/retrieval/corpus_stats.md       human-readable statistics

The golden benchmark (inside the reserved holdout) is excluded and the
`assert_no_golden_overlap` guard runs as a hard gate. No brand reply is ever
claimed to mean "resolved".
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.retrieval.corpus import (  # noqa: E402
    assert_no_golden_overlap,
    build_cases,
    forensics,
    load_holdout_ids,
)

RETRIEVAL_DATA = ROOT / "data" / "retrieval"
CORPUS_PARQUET = ROOT / "data" / "intermediate" / "amazon_customer_corpus.parquet"


def main():
    RETRIEVAL_DATA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    corpus = pd.read_parquet(CORPUS_PARQUET)
    holdout = load_holdout_ids()

    from src import amazon
    twcs = amazon.load_with_conversations()

    cases = build_cases(corpus, holdout, twcs=twcs)
    assert_no_golden_overlap({int(c["conversation_id"]) for c in cases})

    with (RETRIEVAL_DATA / "corpus.jsonl").open("w") as f:
        for c in sorted(cases, key=lambda c: int(c["case_id"])):
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    stats = forensics(cases)
    stats["build_seconds"] = round(time.time() - t0, 1)
    stats["excluded_holdout_conversations"] = len(holdout)
    (RETRIEVAL_DATA / "corpus_stats.json").write_text(json.dumps(stats, indent=2))

    lines = ["# Historical AmazonHelp Support Corpus — Statistics", ""]
    key_rows = [
        ("Total AmazonHelp customer messages (after build)", stats["total_cases"]),
        ("Cases with usable customer→brand response", stats["cases_with_usable_response"]),
        ("Percentage usable", f"{stats['pct_with_usable_response']}%"),
        ("Distinct conversations", stats["distinct_conversations"]),
        ("Average conversation length (turns)", stats["avg_conv_length"]),
        ("Median conversation length (turns)", stats["median_conv_length"]),
        ("First-message (no prior context) cases", f"{stats['pct_first_message']}%"),
        ("Average response length (chars)", stats["avg_response_chars"]),
        ("Median response length (chars)", stats["median_response_chars"]),
        ("Unique customer messages", stats["unique_customer_messages"]),
        ("Unique brand responses", stats["unique_responses"]),
        ("Unique exact (message, response) pairs", stats["unique_exact_pairs"]),
        ("Canned-template responses (>=20 copies)", f"{stats['canned_template_rate']}%"),
        ("Low-content messages (near-empty)", f"{stats['low_content_rate']}%"),
        ("Build time (s)", stats["build_seconds"]),
        ("Excluded holdout conversations (golden+reserve)", stats["excluded_holdout_conversations"]),
        ("Responses with a support link (URL + reach/here context)", f"{stats['feature_rates'].get('contains_support_link')}%"),
    ]
    lines += [f"| {k} | {v} |" for k, v in key_rows]
    lines.append("")
    lines.append("## Response-pattern feature rates (among cases with a response)")
    for k, v in stats["feature_rates"].items():
        lines.append(f"- {k}: {v}%")
    lines.append("")
    lines.append("## Language distribution")
    for k, v in stats["language_dist"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Top exact-duplicate customer messages")
    for k, v in stats["top_dup_messages"].items():
        lines.append(f"- `{k}`: {v}")
    (RETRIEVAL_DATA / "corpus_stats.md").write_text("\n".join(lines) + "\n")

    print(f"\nWrote data/retrieval/corpus.jsonl ({len(cases)} cases) in {stats['build_seconds']}s")
    print(f"Golden-overlap assertion: PASSED")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
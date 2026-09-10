"""Agent failure analysis over the golden benchmark (Phase 9).

Classifies each of the 200 golden rows produced by the final agent into a
categorical failure taxonomy and reports totals + per-category examples:

    intent_primary_miss          predicted intent != golden primary intent
      |- close_secondary         predicted intent is one of the secondary intents
      |- predicted_none          predicted 'none' although a primary intent exists
      `- off_target              predicted intent in neither primary nor secondary
    escalation_conservative      escalated although confident + evidence present
    escalation_false_auto        auto-handled although no usable evidence surfaced
    drafting_ungrounded          auto_handle produced the generic template only
    retrieval_lang_mismatch      top-1 evidence language != query language

Each row gets a (possibly empty) list of failure codes; rows with no code are
successes. At least 30 failure rows are expected (intent misses alone dominate).

Input:  per-row dump from evaluate_agent.py --dump-rows
Output: reports/agent_error_analysis.{json,csv}

Usage:
    python scripts/analyze_agent_errors.py reports/agent_rows.json
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.baselines import load_golden_set  # noqa: E402

KEY = lambda x: x["example_id"]  # noqa: E731
REPORTS = config.PROJECT_ROOT / "reports"


def _reuse_tokens(draft: str, reply: str, min_tokens: int = 2) -> bool:
    """Draft reuses >= min_tokens distinct reply words (len>=5, non-url)."""
    import re  # noqa: PLC0415

    def words(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-zA-Z]{5,}", (text or "").lower())
                if not w.startswith(("http", "tco"))}
    return bool(len(words(draft) & words(reply)) >= min_tokens)


def classify(row: dict, golden: dict) -> list[str]:
    codes: list[str] = []
    true = row["true_intent"]
    pred = row["intent"]

    if pred != true:
        secondaries = golden.get(row["example_id"], {}).get("secondary_intents") or []
        if pred in secondaries:
            codes.append("intent_close_secondary")
        elif pred == "none":
            codes.append("intent_predicted_none")
        else:
            codes.append("intent_off_target")

    sims = [h["score"] for h in row["evidence"]]
    sim_mean = sum(sims[:3]) / max(1, len(sims[:3]))
    usable = any(isinstance(h.get("brand_response"), str) and h["brand_response"] for h in row["evidence"])

    if row["action"] == "escalate" and row["intent_prob"] >= 0.30 and usable and sim_mean >= 0.12:
        codes.append("escalation_conservative")
    if row["action"] == "auto_handle" and (not usable or sim_mean < 0.12):
        codes.append("escalation_false_auto")

    if row["action"] == "auto_handle":
        replies = [h.get("brand_response") or "" for h in row["evidence"]
                   if isinstance(h.get("brand_response"), str)]
        if not any(_reuse_tokens(row["draft"], r) for r in replies):
            codes.append("drafting_ungrounded")

    if row.get("language") and row["evidence"] and row["evidence"][0].get("language"):
        if row["language"] != row["evidence"][0]["language"] and row["language"] != "en":
            codes.append("retrieval_lang_mismatch")
    return codes


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rows = json.loads(open(sys.argv[1]).read())
    golden = {r["example_id"]: r for _, r in load_golden_set().reset_index(drop=True).iterrows()}

    failures: list[dict] = []
    per_code: Counter[str] = Counter()
    for row in rows:
        codes = classify(row, golden)
        per_code.update(codes)
        failures.append({
            "example_id": row["example_id"],
            "codes": codes,
            "true_intent": row["true_intent"],
            "intent": row["intent"],
            "action": row["action"],
            "confidence": row["action_confidence"],
            "language": row["language"],
        })

    cooc = Counter(tuple(sorted(f["codes"])) for f in failures if f["codes"])
    row_by_id = {r["example_id"]: r for r in rows}
    examples = {}
    for code, _count in per_code.most_common():
        cand = [f for f in failures if code in f["codes"]]
        examples[code] = [
            {
                "example_id": str(r["example_id"]),
                "language": r["language"],
                "true_intent": r["true_intent"],
                "predicted_intent": r["intent"],
                "action": r["action"],
                "query": (r["query"] or "")[:220],
                "draft": (r["draft"] or "")[:260],
            }
            for r in (row_by_id[f["example_id"]] for f in cand[:2])
        ]
    summary = {
        "n_golden": len(rows),
        "n_failure_rows": sum(1 for f in failures if f["codes"]),
        "n_success_rows": sum(1 for f in failures if not f["codes"]),
        "category_counts": dict(sorted(per_code.items(), key=lambda kv: -kv[1])),
        "co_occurrence": {str(k): v for k, v in sorted(cooc.items(), key=lambda kv: -kv[1])},
        "examples": examples,
    }
    (REPORTS / "agent_error_analysis.json").write_text(json.dumps(summary, indent=2))
    with (REPORTS / "agent_error_analysis.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "example_id", "codes", "true_intent", "intent", "action",
            "confidence", "language"])
        w.writeheader()
        for f in sorted(failures, key=lambda x: (-len(x["codes"]), str(x["example_id"]))):
            w.writerow({**f, "codes": ";".join(f["codes"])})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
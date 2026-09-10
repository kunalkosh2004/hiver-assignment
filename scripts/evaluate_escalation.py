"""Escalation decision evaluation against the hand-labelled benchmark.

Compares the agent's auto_handle/escalate decision (from the per-row dump,
``reports/agent_rows.json``) against the human-expected decision in
``data/golden/escalation_gold.jsonl``.

Reports (and writes ``reports/escalation_gold_results.{json,csv}``):
  * 2x2 confusion auto_handle/escalate predicted vs expected
  * accuracy, and precision/recall/F1 for the *escalate* decision
  * ``false_auto`` (expected escalate, handled automatically) = the dangerous
    direction; ``false_escalate`` (expected auto, escalated) = conservative cost
  * per-intent breakdown (n, expected/predicted escalate rates)

The benchmark is evaluation-only: it never feeds the threshold fit.

Usage:
    python scripts/evaluate_escalation.py reports/agent_rows.json
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

GOLD = config.DATA_DIR / "golden" / "escalation_gold.jsonl"


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rows = json.loads(open(sys.argv[1]).read())
    gold = {json.loads(l)["example_id"]: json.loads(l) for l in GOLD.open()}

    paired = []
    for r in rows:
        g = gold.get(r["example_id"])
        if g is None:
            print(f"no escalation label for {r['example_id']}")
            return 1
        paired.append({
            "example_id": r["example_id"],
            "split": g["split"],
            "language": g["language"],
            "primary_intent": g["primary_intent"],
            "expected": g["expected_decision"],
            "expected_reason": g["reason"],
            "predicted": r["action"],
            "confidence": r["action_confidence"],
            "reasons": r["action_reasons"],
        })

    def _confusion(pairs):
        m = {"auto_handle": {"auto_handle": 0, "escalate": 0},
             "escalate": {"auto_handle": 0, "escalate": 0}}
        for p in pairs:
            m[p["expected"]][p["predicted"]] += 1
        return m

    m = _confusion(paired)
    aa, ae, ea, ee = m["auto_handle"]["auto_handle"], m["auto_handle"]["escalate"], m["escalate"]["auto_handle"], m["escalate"]["escalate"]
    n = len(paired)
    acc = (aa + ee) / n
    prec_e = ee / (ee + ae) if (ee + ae) else 0.0
    rec_e = ee / (ee + ea) if (ee + ea) else 0.0
    prec_a = aa / (aa + ea) if (aa + ea) else 0.0
    rec_a = aa / (aa + ae) if (aa + ae) else 0.0

    per_intent = []
    by_i = defaultdict(list)
    for p in paired:
        by_i[p["primary_intent"]].append(p)
    for intent, ps in sorted(by_i.items()):
        nexp = sum(1 for p in ps if p["expected"] == "escalate")
        npred = sum(1 for p in ps if p["predicted"] == "escalate")
        per_intent.append({
            "intent": intent,
            "n": len(ps),
            "expected_escalate": nexp,
            "predicted_escalate": npred,
            "expected_escalate_rate": round(nexp / len(ps), 3),
            "predicted_escalate_rate": round(npred / len(ps), 3),
        })

    summary = {
        "n": n,
        "confusion": m,
        "accuracy": round(acc, 4),
        "escalate_precision": round(prec_e, 4),
        "escalate_recall": round(rec_e, 4),
        "escalate_f1": round(_f1(prec_e, rec_e), 4),
        "auto_precision": round(prec_a, 4),
        "auto_recall": round(rec_a, 4),
        "false_auto_dangerous": ea,
        "false_escalate_cost": ae,
        "expected_decision_rate": dict(Counter(p["expected"] for p in paired)),
        "predicted_decision_rate": dict(Counter(p["predicted"] for p in paired)),
        "per_intent": per_intent,
        "note": ("false_auto = expected escalate but auto-handled (dangerous); "
                 "false_escalate = expected auto but escalated (conservative cost). "
                 "Benchmark is hand-labelled single-annotator; evaluation-only."),
    }

    reports = config.PROJECT_ROOT / "reports"
    (reports / "escalation_gold_results.json").write_text(json.dumps(summary, indent=2))
    with (reports / "escalation_gold_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_intent[0].keys()))
        w.writeheader()
        w.writerows(per_intent)

    print(f"escalation benchmark: {n} rows | conf: auto={aa}/{ae} esc={ea}/{ee}")
    print(f"  accuracy={acc:.3f}  escalate P={prec_e:.3f} R={rec_e:.3f} F1={_f1(prec_e, rec_e):.3f}")
    print(f"  false_auto(dangerous)={ea}  false_escalate(cost)={ae}")
    for row in per_intent:
        print(f"  {row['intent']:<32} n={row['n']:<3} exp↑{row['expected_escalate_rate']:.2f} pred↑{row['predicted_escalate_rate']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Phase C: consolidated benchmark (all methods, one table) + risk-policy
comparison + per-intent confusion matrix for escalation decisions.

Consumes the existing reports (no training, no API):
  * reports/agent_rows.json             — final agent's per-golden-row output
  * reports/baseline_results.json       — intent baselines on golden_all
  * reports/escalation_gold_results.json— hand-labelled decision benchmark
  * data/golden/escalation_gold.jsonl   — expected decisions

Produces:
  * one benchmark table where EVERY method is scored on the same golden_all
    200 rows (intent acc/F1) and, for escalation policies, on the same
    200-row human-labelled decision benchmark (acc / escalate-F1 / false_auto)
  * a policy comparison: always-auto, always-escalate, risk-intent heuristic
    vs the agent's dev-calibrated confidence floor
  * a per-intent 2x2 confusion matrix (expected x predicted escalate) for the
    final agent
  -> reports/phase_c_benchmark.{json,csv,md}

Usage:
    python scripts/consolidate_benchmark.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

RISK_INTENTS = {
    "account_access", "charge_issue", "delivered_but_not_received",
    "delivered_wrong_location", "refund_request", "cancellation",
    "product_return_and_replacement", "service_complaint_escalation",
    "other_unclear",
}


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def decision_metrics(pairs: list[tuple[str, str]]):
    """pairs = list of (expected, predicted); both in {auto_handle, escalate}."""
    m = {(e, p): 0 for e in ("auto_handle", "escalate") for p in ("auto_handle", "escalate")}
    for e, p in pairs:
        m[(e, p)] += 1
    aa, ae, ea, ee = m[("auto_handle", "auto_handle")], m[("auto_handle", "escalate")], \
        m[("escalate", "auto_handle")], m[("escalate", "escalate")]
    n = len(pairs)
    return {
        "n": n,
        "accuracy": round((aa + ee) / n, 4),
        "escalate_precision": round(ee / (ee + ae), 4) if (ee + ae) else 0.0,
        "escalate_recall": round(ee / (ee + ea), 4) if (ee + ea) else 0.0,
        "escalate_f1": round(f1(ee / (ee + ae) if (ee + ae) else 0.0,
                                ee / (ee + ea) if (ee + ea) else 0.0), 4),
        "false_auto_dangerous": ea,
        "false_escalate_cost": ae,
        "escalate_rate": round((ee + ae) / n, 4),
    }


def main() -> int:
    rows = json.loads((REPORTS / "agent_rows.json").read_text())
    gold = {json.loads(l)["example_id"]: json.loads(l)
            for l in (ROOT / "data" / "golden" / "escalation_gold.jsonl").open()}

    paired = []
    for r in rows:
        g = gold[r["example_id"]]
        paired.append({"row": r, "expected": g["expected_decision"],
                       "predicted": r["action"], "intent": r["intent"],
                       "true_intent": r["true_intent"]})

    # ---- 1. risk-policy comparison on the same decision benchmark ----------
    policies = {
        "always_auto": [("auto_handle" if True else p["predicted"]) for p in paired],
        "always_escalate": ["escalate"] * len(paired),
        "risk_intent_policy": ["escalate" if p["intent"] in RISK_INTENTS else "auto_handle"
                               for p in paired],
        "risk_true_intent_policy": ["escalate" if p["true_intent"] in RISK_INTENTS else "auto_handle"
                                    for p in paired],
        "agent_floor_policy": [p["predicted"] for p in paired],
    }
    policy_table = {name: decision_metrics(list(zip([q["expected"] for q in paired], pred)))
                    for name, pred in policies.items()}

    # ---- 2. final agent per-intent confusion matrix (expected x predicted) --
    per_intent = {}
    for p in paired:
        key = p["intent"]
        per_intent.setdefault(key, Counter()).update(
            {(p["expected"], p["predicted"]): 1})
    per_intent_rows = []
    for intent in sorted(per_intent, key=lambda k: -sum(per_intent[k].values())):
        c = per_intent[intent]
        per_intent_rows.append({
            "predicted_intent": intent,
            "n": sum(c.values()),
            "expected_escalate": c[("escalate", "auto_handle")] + c[("escalate", "escalate")],
            "predicted_escalate": c[("auto_handle", "escalate")] + c[("escalate", "escalate")],
            "auto_as_auto": c[("auto_handle", "auto_handle")],
            "auto_as_escalate": c[("auto_handle", "escalate")],
            "escalate_as_auto": c[("escalate", "auto_handle")],
            "escalate_as_escalate": c[("escalate", "escalate")],
        })

    # ---- 3. consolidated method table (golden_all 200, intent) -------------
    baseline = json.loads((REPORTS / "baseline_results.json").read_text())
    agent_report = json.loads((REPORTS / "agent_results.json").read_text())
    BL = {name: baseline["results"][name]["overall"]
          for name in baseline["results"] if name.endswith("__golden_all")}

    intent_rows = [
        ("majority_label", "majority", BL["message_only__majority__golden_all"], 0, None),
        ("logistic_msg_only", "logistic", BL["message_only__logistic__golden_all"], 0, None),
        ("svm_msg_only", "svm", BL["message_only__svm__golden_all"], 0, None),
        ("svm_msg+ctx", "svm (msg+context)", BL["message_context__svm__golden_all"], 0, None),
        ("final_agent", "agent (weak-201k intent)", None,
         agent_report["intent_learning_curve"][2]["accuracy"],
         agent_report["intent_learning_curve"][2]["macro_f1"]),
    ]

    method_table = []
    for key, label, bl, agent_acc, agent_f1 in intent_rows:
        acc = bl["accuracy"] if bl else agent_acc
        mf1 = bl["macro_f1"] if bl else agent_f1
        method_table.append({
            "method": label,
            "intent_accuracy": round(acc, 4),
            "intent_macro_f1": round(mf1, 4),
            "escalation_rate": (policy_table["agent_floor_policy"]["escalate_rate"]
                                if key == "final_agent" else None),
            "decision_accuracy": (policy_table["agent_floor_policy"]["accuracy"]
                                  if key == "final_agent" else None),
            "escalate_f1": (policy_table["agent_floor_policy"]["escalate_f1"]
                            if key == "final_agent" else None),
            "false_auto": (policy_table["agent_floor_policy"]["false_auto_dangerous"]
                           if key == "final_agent" else None),
            "grounded_reuse_pct": (agent_report["drafting"]["resolution_reuse_rate_auto"] * 100
                                   if key == "final_agent" else None),
            "empty_drafts": (agent_report["drafting"]["empty_drafts"]
                             if key == "final_agent" else None),
            "avg_draft_chars": (agent_report["drafting"]["avg_draft_chars"]
                                if key == "final_agent" else None),
            "source": "baseline_results.json" if bl else "agent_results.json",
        })

    # ---- persist -----------------------------------------------------------
    out = {
        "n_rows": len(paired),
        "note": ("All rows are the SAME 200 golden rows. Intent columns from "
                 "baseline_results.json (golden_all) and agent_results.json; decision "
                 "columns from the human-labelled escalation benchmark via the SAME "
                 "reports/agent_rows.json per-row dump. RISK_INTENTS policy is a "
                 "human-risk heuristic (escalate iff predicted intent is risky); "
                 "it is an offline policy probe, not the shipped agent."),
        "policy_table": policy_table,
        "per_intent_confusion": per_intent_rows,
        "method_table": method_table,
    }
    (REPORTS / "phase_c_benchmark.json").write_text(json.dumps(out, indent=2))

    with (REPORTS / "phase_c_benchmark.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(method_table[0].keys()))
        w.writeheader()
        w.writerows(method_table)

    lines = [
        "# Consolidated benchmark (Phase C) — same 200 golden rows every method",
        "",
        "| method | intent acc | macro-F1 | esc-rate | decision acc | esc-F1 | false_auto | grounded% | empty | chars |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in method_table:
        if m["decision_accuracy"] is None:
            lines.append(f"| {m['method']} | {m['intent_accuracy']:.3f} | {m['intent_macro_f1']:.3f} | – | – | – | – | – | – | – |")
        else:
            lines.append(f"| {m['method']} | {m['intent_accuracy']:.3f} | {m['intent_macro_f1']:.3f} "
                         f"| {m['escalation_rate']:.3f} | {m['decision_accuracy']:.3f} | {m['escalate_f1']:.3f} "
                         f"| {m['false_auto']} | {m['grounded_reuse_pct']:.1f} | {m['empty_drafts']} | {m['avg_draft_chars']:.0f} |")
    lines += ["", "Policy probe on the decision benchmark (same 200 rows):", "",
              "| policy | accuracy | escalate-F1 | false_auto | false_escalate | esc-rate |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, m in policy_table.items():
        lines.append(f"| {name} | {m['accuracy']:.3f} | {m['escalate_f1']:.3f} | "
                     f"{m['false_auto_dangerous']} | {m['false_escalate_cost']} | {m['escalate_rate']:.3f} |")
    (REPORTS / "phase_c_benchmark.md").write_text("\n".join(lines) + "\n")

    print("Methods (golden_all):")
    for m in method_table:
        print(f"  {m['method']:<26} acc={m['intent_accuracy']:.3f} mF1={m['intent_macro_f1']:.3f}"
              + (f"  dec_acc={m['decision_accuracy']:.3f} escF1={m['escalate_f1']:.3f} "
                 f"false_auto={m['false_auto']} grounded={m['grounded_reuse_pct']:.1f}%" if m["decision_accuracy"] is not None else ""))
    print("Policies:")
    for name, m in policy_table.items():
        print(f"  {name:<22} acc={m['accuracy']:.3f} escF1={m['escalate_f1']:.3f} "
              f"false_auto={m['false_auto_dangerous']:>3} false_esc={m['false_escalate_cost']:>3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
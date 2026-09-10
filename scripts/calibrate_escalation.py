"""Calibrate escalation thresholds on the *dev* set (never on golden).

Probes the agent pipeline over the 200 labelled dev rows (no golden data) and
picks COMB_LOW (a floor on the combined confidence 0.6*intent_prob +
0.4*mean-similarity) so that the escalate share on dev lands at the requested
fraction (default 25%). The chosen value is written to
data/retrieval/models/escalation_thresholds.json, which the agent loads at
runtime; the policy is frozen there, never retuned on golden.

The calibration uses the dev set's intent labels only to describe the resulting
slices (share escalated per true intent); it never optimizes a golden metric.

Usage:
    python scripts/calibrate_escalation.py [--target-rate 0.25]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.agent.agent import Agent  # noqa: E402
from src.agent.escalation import COMB_LOW, W_INTENT, W_SIM  # noqa: E402
from src.baselines import load_dev_set  # noqa: E402

OUT = config.DATA_DIR / "retrieval" / "models" / "escalation_thresholds.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-rate", type=float, default=0.25)
    args = ap.parse_args()

    agent = Agent.load()
    dev = load_dev_set().reset_index(drop=True)

    recs = []
    for _, r in dev.iterrows():
        intent, p, margin = agent.intent.predict(r["current_customer_message"])
        hits = agent.retriever.search(r["current_customer_message"], top_k=5)
        sims = [h.score for h in hits]
        sim = sum(sims[:3]) / max(1, len(sims[:3]))
        usable = any(isinstance(h.brand_response, str) and h.brand_response for h in hits[:3])
        conf = W_INTENT * p + W_SIM * sim
        recs.append({"intent": intent, "p": p, "margin": margin, "sim": sim,
                     "usable": usable, "conf": conf, "true": r["primary_intent"]})

    n = len(recs)
    if args.target_rate >= 1.0:
        comb_low = float("inf")
    else:
        ordered = sorted(c["conf"] for c in recs)
        k = min(int(round(args.target_rate * (n - 1))), n - 1)
        comb_low = float(ordered[k])

    sim_split = defaultdict(int)
    for c in recs:
        esc = (c["intent"] in ("none", "other_unclear")
               or not c["usable"] or c["conf"] < comb_low)
        sim_split[c["true"]] += int(esc)

    result = {
        "source": "dev-200 (evaluation-visible only; golden untouched)",
        "target_escalate_rate": args.target_rate,
        "n_dev": n,
        "thresholds": {"comb_low": round(comb_low, 4)},
        "defaults": {"comb_low": COMB_LOW, "w": {"intent": W_INTENT, "sim": W_SIM}},
        "escalate_per_true_intent_dev": dict(sorted(sim_split.items())),
        "notes": [
            "weak-model probabilities saturate near 1.0 on dev; a single combined-"
            "confidence floor is used instead of per-feature gates",
            "calibrated on dev ONLY; frozen thereafter (never retuned on golden)",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
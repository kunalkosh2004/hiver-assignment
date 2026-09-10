"""Assemble the hand-labelled escalation benchmark (assignment §11).

Reads the frozen single-annotator labels (``data/golden/labels/escalation_gold.tsv``,
one `example_id<TAB>expected_decision<TAB>reason` per row) and joins them onto the
golden set to produce ``data/golden/escalation_gold.jsonl``:

    {example_id, split, language, primary_intent, expected_decision, reason}

Every golden row must be labelled exactly once (validated). The benchmark is a
*decision* benchmark: it answers "should this message be auto-handled or
escalated to a human?" for 200 golden examples. It is used only for evaluation
and never for training/threshold fitting.

Usage:
    python scripts/build_escalation_benchmark.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.baselines import load_golden_set  # noqa: E402

LABELS_TSV = config.DATA_DIR / "golden" / "labels" / "escalation_gold.tsv"
OUT = config.DATA_DIR / "golden" / "escalation_gold.jsonl"


def main() -> int:
    labels: dict[str, tuple[str, str]] = {}
    for line in LABELS_TSV.read_text().splitlines():
        if not line.strip():
            continue
        example_id, decision, reason = line.split("\t", 2)
        assert decision in ("auto_handle", "escalate"), example_id
        labels[example_id] = (decision, reason.strip())

    golden = load_golden_set().reset_index(drop=True)
    missing = sorted(set(golden["example_id"]) - set(labels))
    extra = sorted(set(labels) - set(golden["example_id"]))
    if missing or extra:
        print(f"label coverage error: missing={missing} extra={extra}")
        return 1

    rows = []
    for _, r in golden.iterrows():
        decision, reason = labels[r["example_id"]]
        rows.append({
            "example_id": r["example_id"],
            "split": r["split"],
            "language": r["language"],
            "primary_intent": r["primary_intent"],
            "expected_decision": decision,
            "reason": reason,
        })
    rows.sort(key=lambda x: x["example_id"])
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))

    from collections import Counter  # noqa: PLC0415
    print(f"wrote {len(rows)} rows -> {OUT}")
    print("expected_decision:", dict(Counter(r["expected_decision"] for r in rows)))
    print("per split      :", dict(Counter((r['split'], r['expected_decision']) for r in rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
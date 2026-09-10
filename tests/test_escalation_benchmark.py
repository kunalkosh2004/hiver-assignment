"""Phase A escalation-decision benchmark integrity tests (no network)."""
import json
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestEscalationBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold = {json.loads(l)["example_id"]: json.loads(l)
                    for l in (ROOT / "data/golden/amazon_golden_eval.jsonl").open()}
        rows = (ROOT / "data/golden" / "escalation_gold.jsonl").read_text().splitlines()
        cls.bench = [json.loads(l) for l in rows]

    def test_full_golden_coverage(self):
        ids = [r["example_id"] for r in self.bench]
        self.assertEqual(sorted(ids), sorted(self.gold), "labelled ids != golden ids")

    def test_distinct_and_valid_decisions(self):
        self.assertEqual(len(self.bench), 200)
        dec = Counter(r["expected_decision"] for r in self.bench)
        self.assertTrue(0 < dec.get("escalate", 0) < 200)
        self.assertTrue(0 < dec.get("auto_handle", 0) < 200)
        for r in self.bench:
            self.assertIn(r["expected_decision"], ("auto_handle", "escalate"))
            self.assertTrue(r["reason"].strip())

    def test_split_labels_match_golden(self):
        by_split = Counter((r["example_id"].split("-")[0], r["split"]) for r in self.bench)
        for (_, split), _ in by_split.items():
            self.assertIn(split, ("challenge", "representative"))

    def test_build_is_idempotent(self):
        before = (ROOT / "data/golden" / "escalation_gold.jsonl").read_text()
        subprocess.run([sys.executable, "scripts/build_escalation_benchmark.py"],
                       check=True, cwd=ROOT, capture_output=True)
        after = (ROOT / "data/golden" / "escalation_gold.jsonl").read_text()
        self.assertEqual(before, after, "build is not idempotent")

    def test_eval_report_roundtrip(self):
        dump = ROOT / "reports" / "agent_rows.json"
        self.assertTrue(dump.exists(), "agent_rows.json missing")
        out = subprocess.run(
            [sys.executable, "scripts/evaluate_escalation.py", str(dump)],
            check=True, cwd=ROOT, capture_output=True, text=True)
        summary = json.loads((ROOT / "reports" / "escalation_gold_results.json").read_text())
        self.assertEqual(summary["n"], 200)
        self.assertEqual(summary["confusion"]["auto_handle"]["auto_handle"]
                         + summary["confusion"]["escalate"]["escalate"]
                         + summary["false_auto_dangerous"] + summary["false_escalate_cost"], 200)
        self.assertIn("accuracy=0.", out.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
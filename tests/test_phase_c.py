"""Phase C consolidated-benchmark integrity tests (no network)."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestConsolidatedBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dump = json.loads((ROOT / "reports" / "agent_rows.json").read_text())
        cls.gold = {json.loads(l)["example_id"]: json.loads(l)
                    for l in (ROOT / "data" / "golden" / "escalation_gold.jsonl").open()}

    def _run(self):
        return subprocess.run([sys.executable, "scripts/consolidate_benchmark.py"],
                              check=True, cwd=ROOT, capture_output=True, text=True)

    def test_every_row_paired_to_gold(self):
        for r in self.dump:
            self.assertIn(r["example_id"], self.gold)

    def test_policies_computed(self):
        self._run()
        d = json.loads((ROOT / "reports" / "phase_c_benchmark.json").read_text())
        for name in ("always_auto", "always_escalate", "risk_intent_policy",
                     "risk_true_intent_policy", "agent_floor_policy"):
            self.assertIn(name, d["policy_table"])
        pt = d["policy_table"]
        self.assertEqual(pt["always_auto"]["n"], 200)
        self.assertEqual(pt["agent_floor_policy"]["false_auto_dangerous"]
                         + pt["agent_floor_policy"]["false_escalate_cost"], 200
                         - (sum(1 for r in self.dump
                                if r["action"] == self.gold[r["example_id"]]["expected_decision"])))

    def test_risk_true_intent_ceiling_above_floor(self):
        d = json.loads((ROOT / "reports" / "phase_c_benchmark.json").read_text())
        pt = d["policy_table"]
        self.assertGreater(pt["risk_true_intent_policy"]["accuracy"],
                           pt["agent_floor_policy"]["accuracy"])
        self.assertLess(pt["risk_true_intent_policy"]["false_auto_dangerous"],
                        pt["agent_floor_policy"]["false_auto_dangerous"])

    def test_method_table_rows(self):
        d = json.loads((ROOT / "reports" / "phase_c_benchmark.json").read_text())
        m = {row["method"] for row in d["method_table"]}
        self.assertIn("agent (weak-201k intent)", m)
        agent = next(r for r in d["method_table"] if r["method"] == "agent (weak-201k intent)")
        self.assertEqual(agent["empty_drafts"], 0)
        self.assertGreaterEqual(agent["grounded_reuse_pct"], 90.0)

    def test_idempotent(self):
        before = (ROOT / "reports" / "phase_c_benchmark.json").read_text()
        self._run()
        self.assertEqual(before, (ROOT / "reports" / "phase_c_benchmark.json").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
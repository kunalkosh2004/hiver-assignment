"""Phase D reproduction-timing integrity tests (no network)."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestReproductionTiming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = json.loads((ROOT / "reports" / "reproduction_timing.json").read_text())

    def test_under_15_minutes(self):
        self.assertLess(self.t["total_minutes"], 15.0)

    def test_steps_sum_matches_total(self):
        self.assertAlmostEqual(sum(self.t["steps_seconds"].values()),
                               self.t["total_seconds"], places=1)
        self.assertAlmostEqual(self.t["total_seconds"] / 60, self.t["total_minutes"], places=2)

    def test_all_reproduction_steps_present(self):
        for step in ("train_agent_intent", "calibrate_escalation", "evaluate_agent",
                     "analyze_agent_errors", "build_escalation_benchmark",
                     "evaluate_escalation", "consolidate_benchmark", "unit_tests"):
            self.assertIn(step, self.t["steps_seconds"])

    def test_dominant_steps_documented(self):
        s = self.t["steps_seconds"]
        self.assertGreater(s["evaluate_agent"], s["analyze_agent_errors"])
        self.assertGreater(s["train_agent_intent"], 60)


if __name__ == "__main__":
    unittest.main(verbosity=2)
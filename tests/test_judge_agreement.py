"""Offline validation of the judge-agreement metric maths (no network).

Locks the Phase-B protocol so that the LLM-run ``scripts/evaluate_judge_agreement.py``
only has to substitute live judgment rows — the metrics themselves are fixed and
tested: self-agreement, judge-vs-deterministic Cohen kappa, the rate-limit path,
and the decision rule from ``reports/judge_rubric.md``.
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_judge_agreement import (  # noqa: E402  (sys.path above)
    _cohen_kappa,
    _deterministic_grounded,
    _summarize,
)


class TestJudgeMetrics(unittest.TestCase):
    def test_cohen_kappa_perfect(self):
        obs = [(True, True), (True, True), (False, False)]
        self.assertEqual(_cohen_kappa(obs), 1.0)

    def test_cohen_kappa_expected_lower(self):
        obs = [(True, True), (True, False), (False, True), (False, False)]
        k = _cohen_kappa(obs)
        self.assertIsNotNone(k)
        self.assertLess(k, 1.0)

    def test_kappa_empty(self):
        self.assertIsNone(_cohen_kappa([]))

    def test_deterministic_grounded_token_overlap(self):
        self.assertTrue(_deterministic_grounded(
            "Your order will arrive today.",
            "Checking your order ships today, expect it by 8pm."))
        self.assertFalse(_deterministic_grounded(
            "The cat is on the mat.",
            "Return window is fourteen days."))

    def test_summarize_empty_is_rate_limit(self):
        s = _summarize([])
        self.assertEqual(s, {"note": "no rows judged (rate limit)"})

    def test_summarize_metrics(self):
        rows = [
            {"human_helpful": True, "human_helpful_repeat": True, "draft_helpful": False,
             "draft_grounded": True, "draft_hallucination_free": True,
             "det_grounded": True},
            {"human_helpful": True, "human_helpful_repeat": False, "draft_helpful": False,
             "draft_grounded": False, "draft_hallucination_free": False,
             "det_grounded": False},
            {"human_helpful": False, "human_helpful_repeat": False, "draft_helpful": False,
             "draft_grounded": None, "draft_hallucination_free": True,
             "det_grounded": False},
        ]
        s = _summarize(rows)
        self.assertEqual(s["n_rows_judged"], 3)
        self.assertEqual(s["judge_human_helpful_rate"], round(2 / 3, 4))
        self.assertEqual(s["judge_repeat_self_agreement"], round(2 / 3, 4))
        self.assertEqual(s["human_draft_helpful_gap"], round(2 / 3, 4))
        # grounded oracle pairs: rows 1 (T,T) and 2 (F,F) => perfect
        self.assertEqual(s["judge_vs_deterministic_grounded_kappa"], 1.0)
        self.assertEqual(s["judge_vs_deterministic_pairs"], 2)
        self.assertEqual(s["det_grounded_rate"], 0.5)

    def test_rubric_decision_gate(self):
        s = _summarize([
            {"human_helpful": True, "human_helpful_repeat": True, "draft_helpful": True,
             "draft_grounded": True, "draft_hallucination_free": True, "det_grounded": True},
            {"human_helpful": True, "human_helpful_repeat": True, "draft_helpful": False,
             "draft_grounded": False, "draft_hallucination_free": True, "det_grounded": False},
        ])
        # gate: self_agreement>=0.8, human_helpful>=0.7, kappa>=0.4
        self.assertGreaterEqual(s["judge_repeat_self_agreement"], 0.8)
        self.assertGreaterEqual(s["judge_human_helpful_rate"], 0.7)
        kappa = s["judge_vs_deterministic_grounded_kappa"]
        self.assertIsNotNone(kappa)
        self.assertGreaterEqual(kappa, 0.4)

    def test_script_help_flag(self):
        out = subprocess.run([sys.executable, "scripts/evaluate_judge_agreement.py", "--help"],
                             check=True, cwd=ROOT, capture_output=True, text=True)
        self.assertIn("--sample", out.stdout)

    def test_rubric_doc_committed(self):
        doc = (ROOT / "reports" / "judge_rubric.md").read_text()
        self.assertIn("judge_repeat_self_agreement", doc)
        self.assertIn("judge_vs_deterministic_grounded_kappa", doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
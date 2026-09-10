"""Unit tests for the retrieval-augmented agent (drafting, escalation, pipeline)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.agent import Agent  # noqa: E402
from src.agent.draft import draft_auto, draft_escalation, sanitize  # noqa: E402
from src.agent.escalation import decide  # noqa: E402
from src.agent.intent import IntentClassifier  # noqa: E402
from src.retrieval.base import RetrievalHit  # noqa: E402


def hit(score=0.4, reply="Sorry for the trouble! Please contact us via the link above.", msg="m"):
    return RetrievalHit("c1", score=score, customer_message=msg,
                        brand_response=reply, language="en")


class TestSanitize(unittest.TestCase):
    def test_removes_handles_urls_and_order_ids(self):
        out = sanitize("@USER please check order 1234567890123 at https://t.co/xyz ok?")
        self.assertNotIn("@USER", out)
        self.assertNotIn("1234567890123", out)
        self.assertNotIn("http", out)

    def test_removes_phone_numbers(self):
        out = sanitize("call 1-800-555-0199 now")
        self.assertNotIn("1-800", out)

    def test_removes_scrub_markers(self):
        out = sanitize("please visit us here: ^RB thanks")
        self.assertNotIn("^RB", out)


class TestEscalationPolicy(unittest.TestCase):
    def test_low_intent_prob_escalates(self):
        d = decide(intent="delivery_delay", intent_prob=0.20, margin=0.15, hits=[hit()])
        self.assertEqual(d.action, "escalate")
        self.assertTrue(d.reasons)

    def test_none_intent_escalates(self):
        d = decide(intent="none", intent_prob=0.9, margin=0.8, hits=[hit()])
        self.assertEqual(d.action, "escalate")

    def test_below_combined_floor_escalates(self):
        d = decide(intent="delivery_delay", intent_prob=0.3, margin=0.10, hits=[hit()],
                   thresholds={"comb_low": 0.7})
        self.assertEqual(d.action, "escalate")

    def test_no_evidence_escalates_even_confident(self):
        d = decide(intent="refund_request", intent_prob=0.95, margin=0.9, hits=[])
        self.assertEqual(d.action, "escalate")

    def test_strong_evidence_autohandles(self):
        d = decide(intent="delivery_delay", intent_prob=0.85, margin=0.6,
                   hits=[hit(0.45), hit(0.40), hit(0.39)])
        self.assertEqual(d.action, "auto_handle")
        self.assertTrue(0.0 <= d.confidence <= 1.0)

    def test_high_risk_intent_escalates_despite_confidence(self):
        d = decide(intent="charge_issue", intent_prob=0.99, margin=0.9,
                   hits=[hit(0.6), hit(0.55), hit(0.5)])
        self.assertEqual(d.action, "escalate")
        self.assertTrue(any("high-risk" in r for r in d.reasons))

    def test_high_risk_intent_set_from_config(self):
        from src.agent.escalation import RISK_INTENTS  # noqa: PLC0415
        self.assertIn("charge_issue", RISK_INTENTS)
        self.assertIn("account_access", RISK_INTENTS)
        self.assertNotIn("delivery_delay", RISK_INTENTS)


class TestDrafter(unittest.TestCase):
    def test_draft_is_grounded_and_short(self):
        text = draft_auto("my order is late", "delivery_delay",
                          "Sorry for this! To check the delivery, please send us your order number and email.")
        self.assertIn("order number", text)
        self.assertLess(len(text), 600)

    def test_draft_strips_full_order_numbers(self):
        text = draft_auto("q", "i", "Refund of 1234567890123 processed on your card ****1234.")
        self.assertNotIn("1234567890123", text)

    def test_escalation_draft_is_safe(self):
        text = draft_escalation("charge_issue")
        self.assertIn("specialists", text)


class TestAgentPipeline(unittest.TestCase):
    def test_respond_shape_with_fake_retriever(self):
        class Fake(int):
            name = "fake"

            def search(self, query, *, context=None, top_k=5, timestamp_cutoff=None, **kw):
                return [hit(0.5), hit(0.3)]

        class Dummy01:
            classes_ = ["delivery_delay"]

            def predict_proba(self, xs):
                return [[0.9]] if True else [[0.0]]

        model = IntentClassifier(Dummy01(), ["delivery_delay"], "fake")
        agent = Agent(model, Fake(0))
        out = agent.respond("why is my order late", created_at="2017-01-01")
        self.assertEqual(out["intent"], "delivery_delay")
        self.assertIn(out["action"], {"auto_handle", "escalate"})
        self.assertEqual(out["action_reasons"], ["confident intent + evidence present"])
        self.assertIn("draft", out)
        self.assertEqual(len(out["evidence"]), 2)


if __name__ == "__main__":
    unittest.main()
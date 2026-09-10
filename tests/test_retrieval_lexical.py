"""Unit tests for the lexical retrieval baselines (TF-IDF + BM25).

No golden data, no network. Verifies scoring determinism, reload round-trips,
and the temporal filter. Run:
    python -m unittest tests.test_retrieval_lexical
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.corpus import CaseStore  # noqa: E402
from src.retrieval.tfidf import TfidfRetriever  # noqa: E402

SAMPLE = [
    {"case_id": "1", "conversation_id": 10, "position_in_conversation": 0,
     "conv_length": 2, "customer_timestamp": "2021-01-01", "response_timestamp": "2021-01-01",
     "language": "en", "customer_message": "where is my order it says delivered",
     "conversation_context": [], "brand_response": "sorry to hear that", "has_usable_response": True,
     "is_canned_template": False, "template_count": 1, "template_key": "t1", "metadata": {}},
    {"case_id": "2", "conversation_id": 11, "position_in_conversation": 0,
     "conv_length": 2, "customer_timestamp": "2021-02-01", "response_timestamp": "2021-02-01",
     "language": "es", "customer_message": "donde esta mi pedido", "conversation_context": [],
     "brand_response": "gracias", "has_usable_response": True, "is_canned_template": False,
     "template_count": 1, "template_key": "t2", "metadata": {}},
    {"case_id": "3", "conversation_id": 12, "position_in_conversation": 0,
     "conv_length": 2, "customer_timestamp": "2022-01-01", "response_timestamp": "2022-01-01",
     "language": "en", "customer_message": "how do I refund my echo dot", "conversation_context": [],
     "brand_response": "here you go", "has_usable_response": True, "is_canned_template": True,
     "template_count": 50, "template_key": "t3", "metadata": {}},
]


class LexicalRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.cases = SAMPLE
        self.store_msg = CaseStore(self.cases, include_context=False)

    def test_tfidf_ranks_relevant_first(self):
        r = TfidfRetriever.fit(self.store_msg)
        hits = r.search("where is my order", top_k=2)
        self.assertEqual(hits[0].case_id, "1")
        self.assertGreater(hits[0].score, hits[1].score)

    def test_tfidf_reload_roundtrip(self):
        r = TfidfRetriever.fit(self.store_msg)
        with tempfile.TemporaryDirectory() as d:
            r.save(Path(d))
            r2 = TfidfRetriever.load(Path(d), self.store_msg)
        a = r.score_all("refund echo dot")
        b = r2.score_all("refund echo dot")
        self.assertTrue((a == b).all())

    def test_bm25_ranks_relevant_first(self):
        r = BM25Retriever.fit(self.store_msg, min_df=1)
        hits = r.search("where is my order", top_k=2)
        self.assertEqual(hits[0].case_id, "1")

    def test_bm25_temporal_filter(self):
        r = BM25Retriever.fit(self.store_msg)
        hits = r.search("my order", top_k=3, timestamp_cutoff="2021-06-01")
        self.assertEqual({h.case_id for h in hits}, {"1", "2"})

    def test_exclude_conversation(self):
        r = BM25Retriever.fit(self.store_msg)
        hits = r.search("order", top_k=5, exclude_conversation_ids={10})
        self.assertNotIn("1", {h.case_id for h in hits})

    def test_bm25_reload_roundtrip(self):
        r = BM25Retriever.fit(self.store_msg)
        with tempfile.TemporaryDirectory() as d:
            r.save(Path(d))
            r2 = BM25Retriever.load(Path(d), self.store_msg)
        self.assertEqual(r.score_all("order").tolist(), r2.score_all("order").tolist())


if __name__ == "__main__":
    unittest.main()
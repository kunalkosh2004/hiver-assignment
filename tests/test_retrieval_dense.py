"""Unit tests for the dense + hybrid retrievers (local sentence encoder).

Runs on a 3-doc toy corpus (embedding the tiny set is fast); no network
(cache hit for all-MiniLM-L6-v2). Run:
    python -m unittest tests.test_retrieval_dense
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.corpus import CaseStore  # noqa: E402
from src.retrieval.dense import DenseRetriever, corpus_version  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402

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
     "brand_response": "here you go", "has_usable_response": True, "is_canned_template": False,
     "template_count": 1, "template_key": "t3", "metadata": {}},
]


class DenseRetrrieverTest(unittest.TestCase):
    def setUp(self):
        self.store = CaseStore(SAMPLE, include_context=False)

    def test_fit_ranks_nearest(self):
        r = DenseRetriever.fit(self.store, cache_dir=tempfile.mkdtemp())
        hits = r.search("where is my order", top_k=2)
        self.assertEqual(hits[0].case_id, "1")
        self.assertTrue(hits[0].score > hits[1].score)

    def test_manifest_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / "emb"
            r = DenseRetriever.fit(self.store, cache_dir=cache)
            self.assertEqual(corpus_version(self.store), "cases:3:1:3")
            manifest = json.loads((cache / "msg" / "manifest.json").read_text())
            self.assertEqual(manifest["embedding_model"], "all-MiniLM-L6-v2")
            r2 = DenseRetriever.fit(self.store, cache_dir=cache)  # reuse path
            self.assertEqual((r.score_all("order") == r2.score_all("order")).all(), True)

    def test_hybrid_in_bounds(self):
        bm = BM25Retriever.fit(self.store)
        dn = DenseRetriever.fit(self.store, cache_dir=tempfile.mkdtemp())
        h = HybridRetriever(bm, dn, alpha=0.4)
        s = h.score_all("order")
        self.assertEqual(s.shape, (3,))
        self.assertTrue((((s >= 0) | (s < 1e-9)).all()))
        hits = h.search("order", top_k=2)
        self.assertEqual(len(hits), 2)


if __name__ == "__main__":
    unittest.main()
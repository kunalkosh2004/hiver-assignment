"""Retrieval-augmented support agent (Phase 6 core).

Pipeline per customer message:
1. intent classification (offline TF-IDF + LogReg, message-only),
2. dense message-only retrieval over the historical corpus (temporal filter),
3. fixed confidence-based escalation decision,
4. deterministic grounded drafting.

Golden benchmark is not used anywhere inside this module or its models.
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import config
from .escalation import decide, build_draft
from .intent import IntentClassifier


class Agent:
    def __init__(self, intent_model: IntentClassifier, retriever,
                 thresholds: dict | None = None) -> None:
        self.intent = intent_model
        self.retriever = retriever
        self.thresholds = thresholds

    @classmethod
    def load(cls, intent_dir: Path | None = None,
             dense_index: Path | None = None,
             thresholds_path: Path | None = None) -> "Agent":
        from ..retrieval.corpus import CaseStore, load_corpus
        from ..retrieval.dense import DenseRetriever
        from .escalation import load_thresholds

        intent_dir = intent_dir or config.DATA_DIR / "retrieval" / "models" / "intent_weak"
        intent = IntentClassifier.load(intent_dir)

        index = dense_index or config.DATA_DIR / "retrieval" / "index" / "dense" / "msg"
        emb = config.DATA_DIR / "retrieval" / "embeddings"
        cases = load_corpus()
        store = CaseStore(cases, include_context=False)
        retriever = DenseRetriever.load(index, store, cache_dir=emb)
        thresholds = load_thresholds(
            thresholds_path or config.DATA_DIR / "retrieval" / "models" / "escalation_thresholds.json")
        return cls(intent, retriever, thresholds)

    def respond(self, customer_message: str, *,
                conversation_context: str | None = None,
                created_at: str | None = None,
                top_k: int = 5) -> dict:
        intent, intent_prob, margin = self.intent.predict(customer_message, conversation_context)

        hits = self.retriever.search(
            customer_message,
            context=conversation_context,
            top_k=top_k,
            timestamp_cutoff=created_at,
        )

        decision = decide(
            intent=intent, intent_prob=intent_prob, margin=margin, hits=hits,
            thresholds=self.thresholds,
        )
        draft = build_draft(decision, customer_message, hits)

        return {
            "intent": intent,
            "intent_prob": round(intent_prob, 4),
            "action": decision.action,
            "action_confidence": round(decision.confidence, 4),
            "action_reasons": decision.reasons,
            "draft": draft,
            "evidence": [h.to_record() for h in hits],
        }

    def evidence_stats(self, hits: list) -> dict:
        usable = sum(1 for h in hits if h.brand_response)
        same_lang = sum(1 for h in hits if h.language)
        return {"n_hits": len(hits), "with_reply": usable, "lang_matched": same_lang}
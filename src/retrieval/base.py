"""Retriever protocol + result records.

The retrieval unit is a **historical support interaction**: a customer message,
the prior in-conversation context that was available before it, and the brand
response that *followed* it. A retrieved record never implies the issue was
resolved — it is evidence of what the customer asked and how Amazon responded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class RetrievalHit:
    case_id: str
    score: float
    customer_message: str
    brand_response: str | None
    conversation_context: list[dict] = field(default_factory=list)
    language: str = ""
    customer_timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "case_id": self.case_id,
            "score": round(float(self.score), 6),
            "customer_message": self.customer_message,
            "brand_response": self.brand_response,
            "conversation_context": self.conversation_context,
            "language": self.language,
            "customer_timestamp": self.customer_timestamp,
            "metadata": self.metadata,
        }


class Retriever(Protocol):
    name: str

    def search(
        self,
        query: str,
        *,
        context: list[dict] | str | None = None,
        include_context: bool = False,
        top_k: int = 10,
        timestamp_cutoff: str | None = None,
    ) -> list[RetrievalHit]:
        """Return the top-k most useful historical supports for `query`.

        `include_context` selects which index representation to query
        (message-only vs message+context). `timestamp_cutoff`, when provided,
        restricts candidates to interactions whose customer timestamp <= cutoff
        (chronological / anti-future-leakage retrieval).
        """
        ...
"""Shared text serialization + tokenization for retrieval.

Two input representations mirror the Phase-4 baselines:

- **message**: the current customer message only.
- **message_context**: up to 6 prior turns (role-tagged) + current message.

This is the *only* place retrieval input strings are built, so Experiment A vs
Experiment B cannot silently drift between the corpus index and query side.
"""
from __future__ import annotations

import re

MAX_CONTEXT_TURNS = 6


def serialize_message(text: str) -> str:
    return str(text)


def serialize_context_and_message(text: str, context: list[dict] | str | None) -> str:
    parts: list[str] = []

    if isinstance(context, str) and context.strip():
        parts.append(context.strip())
    elif context:
        for turn in context[:MAX_CONTEXT_TURNS]:
            role = str(turn.get("role", "customer")).strip().lower()
            t = str(turn.get("text", "")).strip()
            if not t:
                continue
            label = "Previous customer" if role == "customer" else "Previous support"
            parts.append(f"{label}: {t}")

    parts.append(f"Current customer: {str(text)}")
    return "\n".join(parts)


def document_text(customer_message: str, context: list[dict] | str | None, include_context: bool) -> str:
    if include_context:
        return serialize_context_and_message(customer_message, context)
    return serialize_message(customer_message)


_TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Deterministic lowercase tokenization used by BM25.

    Kept independent of TF-IDF's vectorizer (which uses sklearn defaults), but
    with the same spirit: lowercase, drop pure-digits and 1-char tokens.
    """
    toks = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    return [t for t in toks if not t.isdigit() and len(t) > 1]
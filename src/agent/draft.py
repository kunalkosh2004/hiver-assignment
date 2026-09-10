"""Deterministic groundable response drafting for the agent.

The draft is a *transformation* of the best retrieved historical resolution
(grounded, never fabricated): handles/URLs are sanitized, obviously-private
order numbers are stripped, and only three content pieces are ever emitted —
an opener, a grounded re-statement of the retrieved brand reply, and a
next-step cue. No order IDs, phone numbers, dates or amounts are invented.
"""
from __future__ import annotations

import re

_HANDLE_RE = re.compile(r"@[\w\d]+")
_URL_RE = re.compile(r"https?://\S+|(?:t\.co/\S+)")
_ORDER_ID_RE = re.compile(r"\b\d{13,17}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
_SCRUB_TAG_RE = re.compile(r"\^[A-Za-z0-9]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

APOLOGY_CUES = ("sorry", "apolog", "regret", "unfortunate")
CONTACT_CUES = ("contact", "message us", "reach out", "get in touch", "reply to this", "chat")
LINK_CUES = ("link", "email", "http", "applicable policy", "details and timeline")
ASK_CUES = ("order id", "order number", "provide", "please share", "email address", "preference")
THANKS_CUES = ("thank", "welcome", "helpful")

_OPENERS = {
    "thank": "Hello! Thank you for taking the time to write to us.",
    "sorry": "Hello! We are sorry to hear about this experience.",
    "default": "Hello! This is Amazon Customer Service.",
}
_NEXT_STEP = (
    "To help, please share a little more detail (for example the order number "
    "or the exact message you see) and we'll take it from there."
)
_ESCALATION = (
    "Thank you for raising this. I have included our specialists on this thread "
    "so they can look into it for you; they will follow up with next steps shortly."
)


def sanitize(text: str) -> str:
    out = _URL_RE.sub("", text)
    out = _PHONE_RE.sub("", out)
    out = _ORDER_ID_RE.sub("", out)
    out = _HANDLE_RE.sub("", out)
    out = _SCRUB_TAG_RE.sub("", out)
    return re.sub(r"\s+", " ", out).strip(" .,;-")


def _cue(line: str) -> str:
    low = line.lower()
    if any(c in low for c in APOLOGY_CUES):
        return "sorry"
    if any(c in low for c in CONTACT_CUES) or any(c in low for c in LINK_CUES):
        return "contact"
    if any(c in low for c in ASK_CUES):
        return "ask"
    if any(c in low for c in THANKS_CUES):
        return "thank"
    return "default"


def _grounded_body(retrieved_reply: str, max_chars: int = 220) -> str:
    text = sanitize(retrieved_reply)
    if not text:
        return ""
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s]
    body = " ".join(sentences)
    while len(body) > max_chars and " " in body:
        body = body.rsplit(" ", 1)[0]
    if sentences and body != text and not body.endswith((".", "!", "?")):
        body += "."
    return body


def draft_auto(query: str, intent: str, retrieved_reply: str | None) -> str:
    cue = _cue(retrieved_reply) if retrieved_reply else "default"
    opener_key = "sorry" if cue == "sorry" else ("thank" if cue == "thank" else "default")
    opened = _OPENERS[opener_key]
    body = _grounded_body(retrieved_reply) if retrieved_reply else _NEXT_STEP
    parts = [opened]
    if body:
        parts.append(body)
    if retrieved_reply:
        parts.append(_NEXT_STEP)
    return " ".join(parts)


def draft_escalation(intent: str) -> str:
    return f"{_OPENERS.get(_cue('sorry'), _OPENERS['default'])} {_ESCALATION}"
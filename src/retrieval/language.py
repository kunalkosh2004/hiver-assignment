"""Deterministic, offline, fast language detection for short support tweets.

Deliberately avoids langdetect on the ~200k-row corpus (too slow). Non-Latin
scripts are detected by Unicode range (fast and unambiguous). Latin-script text
is scored with compact per-language lexicons (function words + support-domain
terms); the highest-scoring language wins, defaulting to ``en`` when there is no
signal. Coarse by construction: ``en`` vs a ``multilingual`` bucket is what the
retrieval evaluation needs.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-záàâäãéèêëíìîïóòôöõúùûüñç’']+", re.UNICODE)

_LEXICON: dict[str, set[str]] = {
    "en": {
        "the", "and", "for", "your", "my", "you", "was", "will", "this", "that",
        "have", "has", "with", "from", "please", "thanks", "thank", "order",
        "delivery", "package", "not", "they", "but", "are", "can", "get", "got",
        "still", "waiting", "received", "get something", "us", "me", "we", "is",
    },
    "es": {
        "el", "la", "los", "las", "de", "que", "por", "para", "con", "una",
        "gracias", "pedido", "entrega", "mi", "me", "esta", "estoy", "como",
        "cuando", "donde", "no", "se", "les", "muy", "quiero", "dice", "enviar",
    },
    "fr": {
        "le", "la", "les", "des", "pour", "avec", "mon", "ma", "mes", "je",
        "vous", "merci", "commande", "colis", "livraison", "une", "pas", "pas de",
        "bonjour", "votre", "suis", "est", "ont", "être", "avoir", "dans",
    },
    "de": {
        "der", "die", "das", "und", "für", "mit", "ich", "sie", "mein",
        "meine", "bestellung", "vielen", "danke", "nicht", "eine", "ein",
        "auf", "bei", "war", "wird", "lieferung", "paket", "keine", "noch",
    },
    "it": {
        "il", "la", "lo", "gli", "per", "con", "mio", "mia", "ordine",
        "grazie", "non", "sono", "sto", "una", "un", "ho", "ha", "è", "stato",
        "consegna", "mi", "se", "dove", "quando", "pacchetto",
    },
    "pt": {
        "o", "a", "os", "as", "de", "do", "da", "para", "com", "meu",
        "minha", "pedido", "obrigado", "obrigada", "não", "está", "estou",
        "entrega", "eu", "me", "uma", "um", "na", "no", "foi", "aqui",
    },
    "nl": {
        "de", "het", "een", "en", "voor", "met", "mijn", "ik", "bestelling",
        "dank", "niet", "op", "aan", "ze", "we", "pakket", "levering", "nog",
        "al", "geen", "wel", "dan",
    },
}


def _script_hint(text: str) -> str | None:
    counts = {"ja": 0, "zh": 0, "ko": 0, "ar": 0, "hi": 0}
    for ch in text:
        o = ord(ch)
        if 0x3040 <= o <= 0x30FF or ch == "\u30fc":
            counts["ja"] += 1
        elif 0x4E00 <= o <= 0x9FFF:
            counts["zh"] += 1
        elif 0xAC00 <= o <= 0xD7AF:
            counts["ko"] += 1
        elif 0x0600 <= o <= 0x06FF:
            counts["ar"] += 1
        elif 0x0900 <= o <= 0x097F:
            counts["hi"] += 1
    hits = {k: v for k, v in counts.items() if v > 0}
    if not hits:
        return None
    if hits.get("ja") and hits.get("zh"):
        return "ja" if counts["ja"] >= counts["zh"] else "zh"
    return max(hits, key=hits.get)


def detect_language(text: str) -> str:
    if not text or not text.strip():
        return "unknown"
    hint = _script_hint(text)
    if hint:
        return hint
    toks = _TOKEN_RE.findall(text.lower())
    if not toks:
        return "unknown"
    scores = {lang: sum(1 for t in toks if t in words) for lang, words in _LEXICON.items()}
    best_lang, best = max(scores.items(), key=lambda kv: kv[1])
    if best >= 2:
        return best_lang
    if best == 1 and best_lang in ("en", "es", "fr", "de", "it", "pt", "nl"):
        return best_lang
    return "en" if any(ord(ch) < 128 for ch in text) else "other"
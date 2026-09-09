"""Helpers for building the AmazonHelp customer-message corpus.

The unit of analysis for intent discovery is a **conversation-aware customer
message**: each customer tweet in an AmazonHelp conversation, annotated with the
conversation context that precedes it (so context-dependent messages like "still
waiting" remain interpretable) and the brand response that follows it (so we can
study historical resolution patterns later).

All context assembly is vectorized (groupby shifts + merge lookups) so building
the ~200k-row corpus is fast.
"""
from __future__ import annotations

import re

import pandas as pd

from . import config, data_io

BRAND = "AmazonHelp"
CONTEXT_WINDOW = 6  # how many prior turns to carry as conversation context


def load_with_conversations() -> pd.DataFrame:
    """Return the full corpus with a ``conversation_id`` column attached."""
    df = data_io.load_twcs()
    conv_cache = config.DATA_DIR / "cache" / "conversation_ids.parquet"
    if conv_cache.exists():
        conv = pd.read_parquet(conv_cache).set_index("tweet_id")["conversation_id"]
        df["conversation_id"] = df["tweet_id"].map(conv)
    else:
        from . import threads  # lazy import keeps startup light

        df, _ = threads.assign_conversation_ids(df)
        df[["tweet_id", "conversation_id"]].to_parquet(conv_cache)
    return df


def amazon_conversations(df: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    """Return (tweets-in-amazon-conversations, set of amazon conversation_ids)."""
    am = df[df["author_id"] == BRAND]
    am_convs = set(am["conversation_id"].dropna())
    sub = df[df["conversation_id"].isin(am_convs)]
    return sub, am_convs


def clean_text(s: str) -> str:
    """Light normalization for analysis: drop URLs, blanket @mentions.

    Kept intentionally light: we do NOT strip stopwords or punctuation here —
    those decisions belong to each specific analysis (embedding vs TF-IDF).
    """
    s = re.sub(r"https?://\S+", "URL", s)
    s = re.sub(r"@\w+", "@USER", s)
    return re.sub(r"\s+", " ", s).strip()


def build_customer_corpus() -> pd.DataFrame:
    """Build the conversation-aware AmazonHelp customer-message corpus.

    Returns one row per customer message in an AmazonHelp conversation, carrying
    conversation context and the following brand response. This is the primary
    analysis unit for Phase 2.
    """
    df = load_with_conversations()
    sub, _ = amazon_conversations(df)

    # Chronological sort within each conversation (deterministic).
    sub = sub.sort_values(["conversation_id", "created_at", "tweet_id"]).reset_index(drop=True)
    sub["pos_in_conv"] = sub.groupby("conversation_id").cumcount()
    sub["conv_length"] = sub.groupby("conversation_id")["tweet_id"].transform("size")

    role = pd.Series("CUSTOMER", index=sub.index)
    role[~sub["inbound"]] = "BRAND"
    sub["_role"] = role
    sub["_rendered_clean"] = sub["_role"] + ": " + sub["text"].astype(str).map(clean_text)

    sub["conversation_context"] = _context_series(sub, sub["_rendered_clean"])
    sub["previous_brand_message"] = _lookup_turn(sub, "text", "BRAND", before=True)
    sub["next_brand_response"] = _lookup_turn(sub, "text", "BRAND", before=False)

    cust = sub[sub["inbound"]].copy()
    cust["customer_text"] = cust["text"]
    cust["customer_text_clean"] = cust["text"].astype(str).map(clean_text)

    keep = [
        "conversation_id", "tweet_id", "created_at", "author_id",
        "customer_text", "customer_text_clean", "conversation_context",
        "previous_brand_message", "next_brand_response",
        "pos_in_conv", "conv_length",
    ]
    return cust[keep].reset_index(drop=True)


def _context_series(sub: pd.DataFrame, rendered: pd.Series) -> pd.Series:
    """Join up to CONTEXT_WINDOW preceding turns into one context string per row."""
    g = sub.groupby("conversation_id")
    parts = []
    for k in range(1, CONTEXT_WINDOW + 1):
        parts.append(g["_rendered_clean"].shift(k))
    frame = pd.concat(parts, axis=1)
    # Efficient row-wise join across the K shifted columns.
    arr = frame.to_numpy()
    out = []
    for row in arr:
        vals = [str(v) for v in row if pd.notna(v)]
        out.append(" | ".join(vals))
    return pd.Series(out, index=sub.index)


def _lookup_turn(sub: pd.DataFrame, col: str, role: str, before: bool) -> pd.Series:
    """For each row, the text of the nearest BRAND turn before/after it (in-conversation)."""
    b = sub[sub["_role"] == role][["conversation_id", "pos_in_conv", col]].rename(
        columns={"pos_in_conv": "bpos", col: "btext"}
    )
    m = sub[["conversation_id", "pos_in_conv"]].merge(b, on="conversation_id")
    if before:
        m = m[m["bpos"] < m["pos_in_conv"]]
        m = m.sort_values(["conversation_id", "pos_in_conv", "bpos"]).groupby(
            ["conversation_id", "pos_in_conv"]
        ).tail(1)
    else:
        m = m[m["bpos"] > m["pos_in_conv"]]
        m = m.sort_values(["conversation_id", "pos_in_conv", "bpos"]).groupby(
            ["conversation_id", "pos_in_conv"]
        ).head(1)
    if m.empty:
        return pd.Series([None] * len(sub), index=sub.index, dtype="object")
    table = m.set_index(pd.MultiIndex.from_arrays(
        [m["conversation_id"], m["pos_in_conv"]]
    ))["btext"].to_dict()
    idx = pd.MultiIndex.from_arrays([sub["conversation_id"], sub["pos_in_conv"]])
    return idx.map(table)

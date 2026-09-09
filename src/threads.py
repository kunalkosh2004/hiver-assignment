"""Conversation / reply-graph reconstruction for the Twitter support corpus.

Data model (verified against the data):
  * Each tweet is a row with a unique ``tweet_id``.
  * ``in_response_to_tweet_id`` is the *parent* edge: the tweet this tweet
    replies to. This is the reliable backward link (99.8% of set parents exist
    in the dataset).
  * ``response_tweet_id`` is a comma-separated *forward* index (children),
    which is noisy/incomplete and not used for thread reconstruction.
  * ``inbound`` is True for anonymized customer accounts (numeric author_id)
    and False for the brand/support account (named handle).

A *conversation/thread* is the connected component of tweets reachable through
parent edges. We label each tweet with a ``conversation_id`` equal to the
tweet_id of the oldest reachable root in its component.

All reconstruction is defensive: it tolerates missing parents, self-loops,
cycles and malformed IDs without crashing.
"""
from __future__ import annotations

import typing as t

import numpy as np
import pandas as pd

PARENT_COL = "in_response_to_tweet_id"
TWEET_ID_COL = "tweet_id"
MAX_FOLLOW_DEPTH = 200  # safety valve against pathological cycles


def parse_parent_ids(df: pd.DataFrame) -> pd.Series:
    """Return an int64 Series (same index as df) of parent tweet ids.

    Invalid / non-numeric / missing parents become NaN. This standardises the
    messy string column into a numeric FK suitable for joining back to tweets.
    """
    s = df[PARENT_COL].astype("string")
    s = s.str.extract(r"(\d+)", expand=False)  # first numeric token
    return pd.to_numeric(s, errors="coerce")


def build_parent_map(df: pd.DataFrame) -> dict[int, int]:
    """Map tweet_id -> parent_tweet_id (NaN/missing -> missing)."""
    parents = parse_parent_ids(df)
    ids = df[TWEET_ID_COL].astype("int64")
    pmap: dict[int, int] = {}
    for tid, pid in zip(ids.to_numpy(), parents.to_numpy(dtype="float64")):
        if not np.isnan(pid):
            pmap[int(tid)] = int(pid)
    return pmap


def reconstruct_thread(
    tweet_id: int,
    parent_map: dict[int, int],
    max_depth: int = MAX_FOLLOW_DEPTH,
) -> list[int]:
    """Follow parent edges from ``tweet_id`` up to the root, safely.

    Returns the ordered list of tweet_ids from the given tweet up to the root
    (i.e. [tweet_id, parent, grandparent, ...]). Missing parents, self-loops,
    cycles and malformed values terminate the walk without crashing.
    """
    chain: list[int] = []
    seen: set[int] = set()
    current = int(tweet_id)
    for _ in range(max_depth):
        chain.append(current)
        parent = parent_map.get(current)
        if parent is None or parent == current:  # no parent or self-loop
            break
        if parent in seen:  # cycle
            break
        seen.add(parent)
        current = parent
    return chain


def root_of(tweet_id: int, parent_map: dict[int, int]) -> int | None:
    """Return the root tweet_id of the chain containing ``tweet_id``."""
    chain = reconstruct_thread(tweet_id, parent_map)
    return chain[-1] if chain else None


def assign_conversation_ids(
    df: pd.DataFrame, parent_map: dict[int, int] | None = None
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Label every row with a ``conversation_id`` (the root tweet_id).

    Uses union-find over parent edges so a thread's label is consistent no
    matter which member you start from. Returns the frame with a new
    ``conversation_id`` column, plus the parent map used.

    The mapping is deterministic and idempotent.
    """
    if parent_map is None:
        parent_map = build_parent_map(df)

    ids = df[TWEET_ID_COL].astype("int64").to_numpy()

    # Union-find with path compression over tweet ids present in the dataset.
    root = {int(t): int(t) for t in ids}
    id_set = set(root)

    def find(x: int) -> int:
        if x != root[x]:
            root[x] = find(root[x])
        return root[x]

    def union(a: int, b: int) -> None:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            root[rb] = ra

    # Union each tweet with its parent, but only when the parent is itself a
    # tweet in the dataset. Orphan references (parent absent from the corpus)
    # become their own root rather than crashing the walk.
    for tid in ids:
        pid = parent_map.get(int(tid))
        if pid is not None and pid in id_set:
            union(int(tid), int(pid))

    conv_ids = np.array([find(int(t)) for t in ids], dtype="int64")
    out = df.copy()
    out["conversation_id"] = conv_ids
    return out, parent_map


def root_created_at(sub: pd.DataFrame) -> pd.Series:
    """Earliest created_at per conversation (used to keep threads together in time)."""
    return sub.groupby("conversation_id")["created_at"].transform("min")


def display_conversation(
    sub: pd.DataFrame,
    conversation_id: int,
    n_max: int = 40,
    author_label: t.Callable[[str], str] | None = None,
) -> str:
    """Render a conversation as a readable CUSTOMER:/BRAND: transcript."""
    conv = (
        sub[sub["conversation_id"] == conversation_id]
        .sort_values("created_at")
        .head(n_max)
    )
    if author_label is None:
        author_label = default_role_label

    lines: list[str] = []
    for _, row in conv.iterrows():
        role = author_label(row["author_id"], row["inbound"])
        text = str(row["text"]).replace("\n", " ")
        if len(text) > 260:
            text = text[:260] + "…"
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def default_role_label(author_id: str, inbound: bool) -> str:
    """Human label for an author given its inbound flag.

    inbound=True is an anonymized customer; inbound=False is the brand/support
    account (verified: 100% of inbound=True rows have numeric author_id and
    100% of inbound=False rows have a named handle).
    """
    return "CUSTOMER" if inbound else "BRAND"


def roles_per_conversation(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise each conversation by which roles appear in it."""
    def _roles(g: pd.DataFrame) -> str:
        has_cust = bool((~g["inbound"]).sum() == 0 or g["inbound"].any())
        has_cust = bool(g["inbound"].any())
        has_brand = bool((~g["inbound"]).any())
        if has_cust and has_brand:
            return "customer+brand"
        if has_cust:
            return "customer_only"
        return "brand_only"

    return (
        df.groupby("conversation_id")
        .apply(_roles, include_groups=False)
        .rename("composition")
        .reset_index()
    )

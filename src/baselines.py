"""Phase-4 baseline intent classifiers (deterministic, reproducible).

Data model
----------
The developer set (`data/golden/amazon_dev_set.jsonl`) is the supervised
TRAIN/VAL/INTERNAL-TEST source. It is conversation-disjoint from the 200-example
golden benchmark (`data/golden/amazon_golden_eval.jsonl`), which is used ONLY for
final evaluation. `next_brand_response` never enters a model.

Split safety
------------
All splits happen at `conversation_id` level (never individual tweets) and assert
disjointness across train/validation/golden. Exact duplicate customer messages
crossing split boundaries are counted and reported (leakage can also come from
repeated/canned text, not only shared conversation ids).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

GOLDEN_DIR = config.DATA_DIR / "golden"
DEV_SET = GOLDEN_DIR / "amazon_dev_set.jsonl"
GOLDEN_SET = GOLDEN_DIR / "amazon_golden_eval.jsonl"
HOLDOUT = GOLDEN_DIR / "conversation_holdout.json"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_dev_set() -> pd.DataFrame:
    """Load the labelled developer set as a DataFrame."""
    recs = [json.loads(l) for l in DEV_SET.open()]
    return pd.DataFrame(recs)


def load_golden_set() -> pd.DataFrame:
    """Load the golden benchmark as a DataFrame (evaluation only)."""
    recs = [json.loads(l) for l in GOLDEN_SET.open()]
    return pd.DataFrame(recs)


def load_holdout() -> set:
    return set(json.loads(HOLDOUT.read_text())["conversation_ids"])


# --------------------------------------------------------------------------- #
# Input serialization (message-only vs message + context)
# --------------------------------------------------------------------------- #
def serialize_message(row) -> str:
    """Experiment A: current_customer_message ONLY."""
    return str(row["current_customer_message"])


def serialize_message_context(row, max_context_turns: int | None = 6) -> str:
    """Experiment B: conversation context + current message.

    Deterministic serialization; NEVER includes next_brand_response. Context is
    rendered with role labels; only turns that occurred before the current
    message are included (the model sees nothing from the future).
    """
    parts: list[str] = []
    ctx = row.get("conversation_context") or []
    for turn in ctx[:max_context_turns]:
        role = str(turn.get("role", "customer")).strip().lower()
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        label = "Previous customer" if role == "customer" else "Previous support"
        parts.append(f"{label}:\n{text}")
    parts.append(f"Current customer:\n{str(row['current_customer_message'])}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Conversation-level splitting
# --------------------------------------------------------------------------- #
def split_by_conversations(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict:
    """Split labelled records into train/validation/internal-test at the
    conversation level (conversation-disjoint), deterministic with `seed`.

    Returns dict of DataFrames: train, validation, test.
    """
    convs = df["conversation_id"].unique()
    rng = np.random.default_rng(seed)
    convs = convs.copy()
    rng.shuffle(convs)

    n_train = int(round(len(convs) * train_frac))
    n_val = int(round(len(convs) * val_frac))

    train_c, val_c, test_c = convs[:n_train], convs[n_train:n_train + n_val], convs[n_train + n_val:]
    group = df.set_index("conversation_id")
    return {
        "train": group.loc[train_c].reset_index(),
        "validation": group.loc[val_c].reset_index(),
        "test": group.loc[test_c].reset_index(),
    }


def assert_no_conversation_overlap(*frames: pd.DataFrame) -> None:
    """Assert all pairs of conversation-id sets are disjoint."""
    sets = [set(f["conversation_id"]) for f in frames if len(f)]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            assert not overlap, (
                f"conversation-level leakage between splits: {len(overlap)} shared "
                f"conversations (e.g. {sorted(overlap)[:5]})."
            )


def assert_no_golden_leakage(*frames: pd.DataFrame) -> None:
    """Assert training/val/test conversations never appear in the golden benchmark."""
    golden_ids = set(load_golden_set()["conversation_id"])
    for f in frames:
        overlap = set(f["conversation_id"]) & golden_ids
        assert not overlap, f"{len(overlap)} conversation(s) leak into the golden benchmark."


def count_cross_split_duplicate_messages(*frames: pd.DataFrame) -> int:
    """Count exact duplicate customer messages that cross any split boundary.

    Repeated/canned text is a leakage risk even when conversation ids differ,
    so we report how many distinct texts appear in >1 split.
    """
    if len(frames) < 2:
        return 0
    per_split = [set(f["current_customer_message"].astype(str)) for f in frames]
    crossing = set()
    for i in range(len(per_split)):
        for j in range(i + 1, len(per_split)):
            crossing |= per_split[i] & per_split[j]
    return len(crossing)


# --------------------------------------------------------------------------- #
# Label handling
# --------------------------------------------------------------------------- #
def supported_labels() -> list[str]:
    """Ordered label vocabulary (frozen 12 intents + none + other_unclear).

    Both special labels are present in the developer set, so the supervised
    baseline is trained/evaluated on the full label set (no silent dropping).
    ``next_brand_response`` is never used.
    """
    import yaml as _yaml
    tax_path = config.PROJECT_ROOT / "config" / "amazon_intents.yaml"
    tax = _yaml.safe_load(tax_path.read_text())
    order = [i["id"] for i in tax["intents"]]
    return order + ["none", "other_unclear"]


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def weighted_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def macro_precision(y_true, y_pred) -> float:
    from sklearn.metrics import precision_score
    return float(precision_score(y_true, y_pred, average="macro", zero_division=0))


def macro_recall(y_true, y_pred) -> float:
    from sklearn.metrics import recall_score
    return float(recall_score(y_true, y_pred, average="macro", zero_division=0))

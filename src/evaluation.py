"""Golden-set evaluation helpers (Phase 3).

Provides `assert_no_conversation_overlap` used to guarantee conversation-level
leakage safety between the golden eval set and any downstream train split.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence


def conversation_ids(records: Sequence[dict]) -> set:
    """Extract the set of conversation ids from a list of record dicts."""
    ids = set()
    for rec in records:
        cid = rec.get("conversation_id")
        if cid is not None:
            ids.add(cid)
    return ids


def assert_no_conversation_overlap(
    golden: Sequence[dict],
    train: Optional[Iterable[dict]] = None,
    holdout: Optional[Sequence[int]] = None,
    allow_overlap: bool = False,
) -> bool:
    """Assert that no conversation appears in both the golden set and the
    (future) training set, and (optionally) that none of the golden examples
    fall into the reserved conversation holdout.

    The golden set was built by splitting conversations FIRST (see Phase 2/3):
    the conversations used here were reserved at selection time, so overlap
    with any later train split is prevented. This check is a belt-and-braces
    guard that must pass before a classifier is trained on the golden-convenient
    train split.

    Returns True when all checks pass. Raises AssertionError otherwise.
    """
    golden_ids = conversation_ids(golden)
    assert golden_ids, "golden set is empty"

    if train is not None:
        train_ids = conversation_ids(list(train))
        overlap = golden_ids & train_ids
        assert not overlap, (
            f"conversation-level leakage detected: {len(overlap)} conversations "
            f"appear in both the golden eval set and the train set (e.g. {sorted(overlap)[:5]})."
        )
        assert not allow_overlap or not overlap

    if holdout is not None:
        holdset = set(holdout)
        leaked = golden_ids & holdset
        assert not leaked, (
            f"{len(leaked)} golden conversations are from the reserved holdout: {sorted(leaked)[:5]}"
        )

    return True


def label_overlap_metrics(
    records: Sequence[dict],
    categories: Sequence[str] = ("primary_intent", "difficulty", "label_confidence",
                                 "message_type", "context_dependency"),
) -> dict:
    """Return a small distribution summary for the given record fields."""
    summary = {}
    for cat in categories:
        counts: dict[str, int] = {}
        for rec in records:
            val = rec.get(cat)
            if val is None:
                val = "MISSING"
            counts[val] = counts.get(val, 0) + 1
        summary[cat] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    return summary

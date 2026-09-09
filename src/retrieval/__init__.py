"""Phase-5 historical support retrieval layer.

Type-B (unlabelled) historical interactions power the retrieval index; Type A
(human labels) is used only for evaluation; Type C (LLM outputs) never enters
the index and is never treated as ground truth. See
`scripts/build_retrieval_index.py` and `scripts/evaluate_retrieval.py`.
"""
from .base import RetrievalHit, Retriever
from .corpus import assert_no_golden_overlap, build_cases, forensics, load_holdout_ids
from .language import detect_language

__all__ = [
    "RetrievalHit",
    "Retriever",
    "assert_no_golden_overlap",
    "build_cases",
    "forensics",
    "load_holdout_ids",
    "detect_language",
]
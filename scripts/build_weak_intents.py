"""Weak-intent proxy labels for the historical corpus (analysis ONLY).

Because ~200k unlabelled interactions cannot be human-labeled at budget, the
retrieval scaling curve and intent breakdowns use a *proxy*: a TF-IDF +
logistic-regression intent classifier trained on the 200 dev labels
(message-only), applied to every corpus case. Outputs are EXPLICITLY marked as
proxy in every report that uses them and are never treated as ground truth.

    python scripts/build_weak_intents.py
    -> data/retrieval/weak_intents.jsonl  (case_id, weak_intent, weak_conf)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src import config  # noqa: E402
from src.retrieval.corpus import load_corpus  # noqa: E402

REL = config.DATA_DIR / "retrieval"


def main() -> None:
    dev = [json.loads(l) for l in (config.DATA_DIR / "golden" / "amazon_dev_set.jsonl").open()]
    dev = [d for d in dev if d["primary_intent"] != "none"]
    clf = make_pipeline(
        TfidfVectorizer(sublinear_tf=True),
        LogisticRegression(max_iter=1200, C=4.0),
    )
    clf.fit(
        [d["current_customer_message"] for d in dev],
        [d["primary_intent"] for d in dev],
    )
    classes = list(clf.classes_)

    cases = load_corpus()
    msgs = [c["customer_message"] for c in cases]
    probs = clf.predict_proba(msgs)
    best = probs.argmax(axis=1)
    conf = probs[np.arange(len(msgs)), best]

    with (REL / "weak_intents.jsonl").open("w") as fh:
        for c, j, p in zip(cases, best, conf):
            fh.write(json.dumps({
                "case_id": c["case_id"],
                "weak_intent": str(classes[j]),
                "weak_conf": round(float(p), 3),
            }) + "\n")
    print("[weak] proxied", len(cases), "cases with", len(classes), "classes:", classes)


if __name__ == "__main__":
    main()
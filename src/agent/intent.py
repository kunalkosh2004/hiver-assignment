"""Intent classification component for the retrieval-augmented agent.

Two training regimes, both deterministic and offline:

- ``fit_weak``        train on the ~201k weak-intent proxy labels produced by
                      ``scripts/build_weak_intents.py`` (labels explicitly NOT
                      ground truth; the classifier itself is the agent's model).
- ``fit_dev``         train on the 140-row Phase-4 conversation-safe dev split
                      (human labels) — used for the classifier scaling curve.

The golden benchmark is never shown to these models; it is reserved for the
final harness (``scripts/evaluate_agent.py``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from ..baselines import split_by_conversations, supported_labels, load_dev_set
from .. import config


class IntentClassifier:
    """TF-IDF + logistic-regression intent model with save/load + predict."""

    name = "intent"

    def __init__(self, pipeline, classes: list[str], source: str):
        self.pipeline = pipeline
        self.classes = [str(c) for c in classes]
        self.source = source
        self._idx = {c: i for i, c in enumerate(self.classes)}

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit_weak(
        cls, msgs: list[str], weak_intents: list[str],
        *, C: float = 4.0, max_features: int = 200_000, seed: int = 42,
    ) -> "IntentClassifier":
        pipe = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, max_features=max_features),
            LogisticRegression(max_iter=1200, C=C, random_state=seed),
        )
        pipe.fit(msgs, weak_intents)
        return cls(pipe, list(pipe.classes_), source="weak-201k")

    @classmethod
    def fit_dev(
        cls, *, seed: int = 42, input_variant: str = "message",
    ) -> tuple["IntentClassifier", dict]:
        from ..baselines import serialize_message, serialize_message_context

        dev = load_dev_set()
        split = split_by_conversations(dev, train_frac=0.70, val_frac=0.15, seed=seed)
        frame = split["train"]
        xs = [serialize_message(r) if input_variant == "message"
              else serialize_message_context(r) for _, r in frame.iterrows()]
        pipe = make_pipeline(
            TfidfVectorizer(sublinear_tf=True),
            LogisticRegression(max_iter=1200, C=4.0, random_state=seed),
        )
        pipe.fit(xs, list(frame["primary_intent"]))
        info = {
            "source": f"dev-{len(frame)}",
            "n_train": int(len(frame)),
            "validated_on": "dev/validation",
        }
        return cls(pipe, list(pipe.classes_), source=f"dev-{len(frame)}"), info

    # ---------------------------------------------------------------- predict
    def predict(self, message: str, context: str | list | None = None) -> tuple[str, float, float]:
        """Return (best intent, best probability, top-two margin)."""
        text = message if not context else f"{message}\n{context if isinstance(context, str) else ''}"
        probs = self.pipeline.predict_proba([text])[0]
        order = np.argsort(-np.asarray(probs))
        best = int(order[0])
        margin = float(probs[best] - probs[int(order[1])]) if len(order) > 1 else float(probs[best])
        return self.classes[best], float(probs[best]), margin

    def predict_proba_all(self, message: str) -> dict[str, float]:
        probs = self.pipeline.predict_proba([message])[0]
        return {self.classes[i]: float(p) for i, p in enumerate(probs)}

    # ------------------------------------------------------------------ io
    def save(self, base: Path) -> None:
        import joblib

        base = Path(base)
        base.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, base / "pipeline.pkl")
        (base / "meta.json").write_text(json.dumps({
            "model": self.name, "source": self.source,
            "classes": self.classes, "input_variant": "message",
        }))

    @classmethod
    def load(cls, base: Path) -> "IntentClassifier":
        import joblib

        base = Path(base)
        meta = json.loads((base / "meta.json").read_text())
        return cls(joblib.load(base / "pipeline.pkl"), meta["classes"], meta["source"])


def load_weak_labels() -> tuple[list[str], list[str]]:
    """Customer messages + weak intent labels, aligned to corpus order."""
    from ..retrieval.corpus import load_corpus

    weak = {}
    for l in (config.DATA_DIR / "retrieval" / "weak_intents.jsonl").open():
        r = json.loads(l)
        weak[r["case_id"]] = r["weak_intent"]
    cases = load_corpus()
    msgs, labels = [], []
    for c in cases:
        w = weak.get(str(c["case_id"]))
        if w is None or not c.get("customer_message"):
            continue
        msgs.append(c["customer_message"])
        labels.append(w)
    return msgs, labels
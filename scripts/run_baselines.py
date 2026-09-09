"""Phase-4 baseline runner — deterministic, reproducible and fully offline.

Usage:
    python scripts/run_baselines.py

Steps:
  1. load the labelled developer set + golden benchmark
  2. conversation-disjoint split (train/validation/internal-test), seed 42
  3. assert no leakage (cross-split + golden + holdout + duplicate texts)
  4. train Majority / TF-IDF+LogisticRegression / TF-IDF+LinearSVC
     under message-only and message+context representations
  5. evaluate on the internal test set and the golden benchmark (all/rep/chal)
  6. save metrics, predictions, confusion matrices, error analysis to reports/

The golden benchmark is used ONLY for evaluation; it is never part of training.
The LLM providers are NOT invoked here — this run is deterministic and offline.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import baselines as B  # noqa: E402

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15

TFIDF_PARAMS = dict(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True)
LOGISTIC_PARAMS = dict(max_iter=2000, random_state=SEED, class_weight="balanced")
SVC_PARAMS = dict(max_iter=2000, random_state=SEED, class_weight="balanced")


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def overall_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": B.macro_f1(y_true, y_pred),
        "weighted_f1": B.weighted_f1(y_true, y_pred),
        "macro_precision": B.macro_precision(y_true, y_pred),
        "macro_recall": B.macro_recall(y_true, y_pred),
    }


def per_intent_metrics(y_true, y_pred, labels):
    from sklearn.metrics import classification_report
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True,
                                zero_division=0)
    out = {}
    for lab in labels:
        r = rep.get(lab, {})
        out[lab] = {
            "support": int(rep[lab].get("support", 0)) if lab in rep else 0,
            "precision": float(r.get("precision", 0.0)),
            "recall": float(r.get("recall", 0.0)),
            "f1": float(r.get("f1-score", 0.0)),
        }
    return out


def confusion_matrix_df(y_true, y_pred, labels):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred, labels=labels)


def serialize_inputs(df, representation):
    fn = B.serialize_message if representation == "message_only" else B.serialize_message_context
    return [fn(r) for r in df.to_dict("records")]


def make_models(train, train_y, labels):
    """Return dict of fitted models -> (predict_fn, has_proba)."""
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    fitted = {}
    # Majority
    m = DummyClassifier(strategy="most_frequent")
    m.fit(train, train_y)
    fitted["majority"] = (m, False)
    # LogisticRegression
    m = LogisticRegression(**LOGISTIC_PARAMS)
    m.fit(train, train_y)
    fitted["logistic"] = (m, True)
    # LinearSVC
    m = LinearSVC(**SVC_PARAMS)
    m.fit(train, train_y)
    fitted["svm"] = (m, False)
    return fitted


def predict_row(df, y_pred, model, representation, subset, proba=None, proba_labels=None):
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        rec = {
            "example_id": f"{subset}-{r['conversation_id']}",
            "conversation_id": int(r["conversation_id"]),
            "subset": subset,
            "model": model,
            "input": representation,
            "language": r.get("language", ""),
            "difficulty": r.get("difficulty", ""),
            "context_dependency": r.get("context_dependency", ""),
            "message_type": r.get("message_type", ""),
            "gold_intent": r["primary_intent"],
            "predicted_intent": str(y_pred[i]),
            "correct": int(str(y_pred[i]) == r["primary_intent"]),
            "current_message": str(r["current_customer_message"]),
        }
        if proba is not None:
            proba_labels = list(proba_labels)
            rec["confidence"] = float(np.max(proba[i]))
            rec["predicted_probability"] = float(proba[i][proba_labels.index(str(y_pred[i]))])
        rows.append(rec)
    return rows


def main():
    dev = B.load_dev_set()
    golden = B.load_golden_set()
    labels = B.supported_labels()

    # ---- 1) conversation-disjoint split (reused everywhere) ----------------
    splits = B.split_by_conversations(dev, TRAIN_FRAC, VAL_FRAC, seed=SEED)
    train, val, test = splits["train"], splits["validation"], splits["test"]

    # ---- 2) leakage assertions ---------------------------------------------
    B.assert_no_conversation_overlap(train, val, test, golden)
    B.assert_no_golden_leakage(train, val, test)
    holdout = B.load_holdout()
    golden_ids = set(golden["conversation_id"])
    assert golden_ids <= holdout, "golden conversations must be inside the reserved holdout"

    cross_dup = B.count_cross_split_duplicate_messages(train, val, test)
    dup_gold_train = len(set(golden["current_customer_message"].astype(str))
                         & set(train["current_customer_message"].astype(str)))
    dup_gold_val = len(set(golden["current_customer_message"].astype(str))
                       & set(val["current_customer_message"].astype(str)))
    dup_gold_test = len(set(golden["current_customer_message"].astype(str))
                        & set(test["current_customer_message"].astype(str)))

    split_report = {
        "seed": SEED,
        "sizes": {k: int(len(v)) for k, v in splits.items()},
        "cross_split_dup_messages": int(cross_dup),
        "dup_message_golden_vs_train": int(dup_gold_train),
        "dup_message_golden_vs_val": int(dup_gold_val),
        "dup_message_golden_vs_test": int(dup_gold_test),
        "golden_subsets": dict(Counter(golden["split"])),
        "train_label_dist": dict(Counter(train["primary_intent"])),
        "val_label_dist": dict(Counter(val["primary_intent"])),
        "test_label_dist": dict(Counter(test["primary_intent"])),
    }

    # ---- 3) representations ------------------------------------------------
    reps = {
        "message_only": {
            "text": serialize_inputs(train, "message_only"), "label": "Message"},
        "message_context": {
            "text": serialize_inputs(train, "message_context"), "label": "Message + Context"},
    }

    results = {}      # (repr, model, subset) -> (overall, per_intent, confusion_matrix)
    predictions = []  # rows

    subsets = [
        ("test", test),
        ("val", val),
        ("golden_all", golden),
        ("golden_rep", golden[golden["split"] == "representative"]),
        ("golden_chal", golden[golden["split"] == "challenge"]),
    ]

    for repr_name, rinfo in reps.items():
        # vectorize training once per representation, reuse for all classical models
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(**TFIDF_PARAMS)
        X_train = vec.fit_transform(rinfo["text"])
        # vectorize test/golden once per representation (fit transform only)
        X_cache = {}
        for sub_name, sub_df in subsets:
            X_cache[sub_name] = vec.transform(serialize_inputs(sub_df, repr_name))

        train_y = train["primary_intent"].tolist()

        for model_name in ("majority", "logistic", "svm"):
            if model_name == "majority":
                fit_text = None  # majority doesn't use features
                from sklearn.dummy import DummyClassifier
                model = DummyClassifier(strategy="most_frequent")
                model.fit(train, train_y)
            elif model_name == "logistic":
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(**LOGISTIC_PARAMS)
                model.fit(X_train, train_y)
            else:
                from sklearn.svm import LinearSVC
                model = LinearSVC(**SVC_PARAMS)
                model.fit(X_train, train_y)

            has_proba = model_name == "logistic"
            proba_labels = list(model.classes_) if has_proba else None

            for sub_name, sub_df in subsets:
                Xt = X_cache[sub_name]
                y_true = sub_df["primary_intent"].tolist()
                if model_name == "majority":
                    y_pred = model.predict([[0]] * len(sub_df))  # features ignored
                    proba = None
                else:
                    y_pred = model.predict(Xt)
                    proba = model.predict_proba(Xt) if has_proba else None

                overall = overall_metrics(y_true, y_pred)
                per_intent = per_intent_metrics(y_true, y_pred, labels)
                cm = confusion_matrix_df(y_true, y_pred, labels)
                results[(repr_name, model_name, sub_name)] = (overall, per_intent, cm)
                predictions.extend(
                    predict_row(sub_df, y_pred, model_name, repr_name, sub_name,
                                proba=proba, proba_labels=proba_labels))

    # ---- 4) persistence ----------------------------------------------------
    pd.DataFrame([
        {"model": m, "input": r, "subset": s, **o}
        for (r, m, s), (o, _, _) in results.items()
    ]).to_csv(REPORTS / "baseline_results.csv", index=False)

    results_json = {
        "split": split_report,
        "results": {
            f"{rgn}__{m}__{s}": {"overall": o, "per_intent": pi}
            for (rgn, m, s), (o, pi, _) in results.items()
        },
    }
    (REPORTS / "baseline_results.json").write_text(json.dumps(results_json, indent=2))

    pd.DataFrame(predictions).to_csv(REPORTS / "baseline_predictions.csv", index=False)

    # ---- 5) confusion matrices (classical models, test set) ----------------
    for repr_name in ("message_only", "message_context"):
        for model_name in ("logistic", "svm"):
            cm = results[(repr_name, model_name, "test")][2]
            save_confusion(cm, labels, f"confusion_{model_name}_{repr_name}")

    # ---- 6) headline table -------------------------------------------------
    print("\n=== HEADLINE RESULTS (macro F1 on golden benchmark) ===")
    print(f"{'Model':<10}{'Input':<16}{'Subset':<14}{'Acc':>8}{'MacroF1':>9}{'W-F1':>9}")
    print("-" * 66)
    for (r, m, s), (o, _, _) in results.items():
        print(f"{m:<10}{r:<16}{s:<14}{o['accuracy']:>8.3f}{o['macro_f1']:>9.3f}{o['weighted_f1']:>9.3f}")

    print("\nSplit sizes:", split_report["sizes"])
    print(f"Cross-split duplicate messages (train/val/test): {cross_dup}")
    print(f"Duplicate message text golden-vs-train/val/test: "
          f"{dup_gold_train}/{dup_gold_val}/{dup_gold_test}")
    print("\nWrote reports/baseline_results.csv, baseline_results.json,")
    print("        reports/baseline_predictions.csv, reports/figures/confusion_*.png")

    return results


def save_confusion(cm, labels, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Gold (internal test)")
    thresh = max(1, int(cm.max() * 0.3))
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = int(cm[i, j])
            if v >= thresh:
                ax.text(j, i, str(v), ha="center", va="center", color="white")
            elif v > 0:
                ax.text(j, i, str(v), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"saved reports/figures/{name}.png")


if __name__ == "__main__":
    main()

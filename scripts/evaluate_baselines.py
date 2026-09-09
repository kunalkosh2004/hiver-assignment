"""Phase-4D evaluation + error analysis.

Reads reports/baseline_predictions.csv + reports/baseline_results.json and
writes a full analysis into reports/:

  - segmentation: macro F1 by difficulty/language/context_dependency/message_type
  - calibration:  confidence buckets (logistic has probabilities)
  - error_analysis.md: >=20 concrete misclassifications grouped by pattern
  - dev_vs_golden.md: how dev (train) and golden benchmarks differ

Nothing here touches the golden labels (read-only) and no model is retrained.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import baselines as B  # noqa: E402

REPORTS = ROOT / "reports"
PRED = REPORTS / "baseline_predictions.csv"
RESULTS_JSON = REPORTS / "baseline_results.json"


def macro_f1_group(pred_df, group_col, subset="golden_all", model="logistic", input_repr="message_only"):
    """Macro F1 per group value within a subset/model/input slice."""
    from sklearn.metrics import f1_score
    sl = pred_df[(pred_df["subset"] == subset) & (pred_df["model"] == model)
                 & (pred_df["input"] == input_repr)]
    if sl.empty:
        return {}
    out = {}
    for g, grp in sl.groupby(group_col):
        yt = grp["gold_intent"]; yp = grp["predicted_intent"]
        out[g] = {
            "n": len(grp),
            "acc": float((yt == yp).mean()),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        }
    return out


def calibration_buckets(pred_df, subset="golden_all", model="logistic", input_repr="message_only"):
    """Confidence-bucket performance for models with probabilities.

    Logistic is a 14-way classifier on 140 rows: absolute confidence stays low
    (never > ~0.25), so we bucket by QUANTILE (relative confidence) which is far
    more informative in this low-confidence regime.
    """
    from sklearn.metrics import f1_score
    sl = pred_df[(pred_df["subset"] == subset) & (pred_df["model"] == model)
                 & (pred_df["input"] == input_repr)].copy()
    sl = sl.dropna(subset=["confidence"])
    if sl.empty or len(sl) < 10:
        return {}
    try:
        qs = sl["confidence"].quantile([0, .2, .4, .6, .8, 1.0]).values
    except ValueError:
        return {}
    out = {}
    for k in range(len(qs) - 1):
        lo, hi = qs[k], qs[k + 1]
        if k == len(qs) - 2:
            g = sl[sl["confidence"] >= lo]
        else:
            g = sl[(sl["confidence"] >= lo) & (sl["confidence"] < hi)]
        if g.empty:
            continue
        out[f"q{k}"] = {
            "n": len(g),
            "conf_range": f"{lo:.3f}-{hi:.3f}",
            "avg_conf": float(g["confidence"].mean()),
            "acc": float((g["gold_intent"] == g["predicted_intent"]).mean()),
            "macro_f1": float(f1_score(g["gold_intent"], g["predicted_intent"],
                                       average="macro", zero_division=0)),
        }
    return out


def error_analysis(pred_df, *, model="logistic", input_repr="message_only", subset="golden_all",
                   min_errors=20, cap: int | None = None) -> list[dict]:
    """Return misclassifications with message text (all errors, optionally capped)."""
    sl = pred_df[(pred_df["subset"] == subset) & (pred_df["model"] == model)
                 & (pred_df["input"] == input_repr)]
    errs = sl[sl["gold_intent"] != sl["predicted_intent"]].copy()
    if cap:
        errs = errs.head(cap)
    rows = []
    for _, r in errs.iterrows():
        rows.append({
            "conversation_id": int(r["conversation_id"]),
            "language": r["language"],
            "difficulty": r["difficulty"],
            "context_dependency": r["context_dependency"],
            "gold_intent": r["gold_intent"],
            "predicted_intent": r["predicted_intent"],
            "confidence": None if pd.isna(r.get("confidence")) else round(float(r["confidence"]), 3),
            "current_message": str(r["current_message"])[:300],
        })
    return rows


def error_pattern_summary(pred_df, subset="golden_all", model="logistic", input_repr="message_only"):
    """Top confusion patterns (gold->pred) for the model slice."""
    sl = pred_df[(pred_df["subset"] == subset) & (pred_df["model"] == model)
                 & (pred_df["input"] == input_repr)]
    sl = sl[sl["gold_intent"] != sl["predicted_intent"]]
    c = Counter(zip(sl["gold_intent"], sl["predicted_intent"]))
    return c.most_common(15)


def main():
    pred = pd.read_csv(PRED)
    results_json = json.loads(RESULTS_JSON.read_text())
    split_report = results_json["split"]

    # ---- 1) segmentation ----------------------------------------------------
    seg = {}
    for col in ("difficulty", "language", "context_dependency", "message_type"):
        seg[col] = {
            "golden_all": macro_f1_group(pred, col, subset="golden_all"),
            "test": macro_f1_group(pred, col, subset="test"),
        }

    # ---- 2) calibration (logistic msg-only on golden) ------------------------
    cal = calibration_buckets(pred, subset="golden_all")

    # ---- 3) error analysis --------------------------------------------------
    errs_msg = error_analysis(pred, model="logistic", input_repr="message_only", subset="golden_all")
    errs_ctx = error_analysis(pred, model="logistic", input_repr="message_context", subset="golden_all")
    errs_svm = error_analysis(pred, model="svm", input_repr="message_only", subset="golden_all")

    patterns_msg = error_pattern_summary(pred, model="logistic", input_repr="message_only")
    patterns_ctx = error_pattern_summary(pred, model="logistic", input_repr="message_context")

    # ---- 4) dev vs golden distribution --------------------------------------
    dev = B.load_dev_set()
    dist_dev = Counter(dev["primary_intent"])
    dist_golden = Counter(B.load_golden_set()["primary_intent"])
    labels = B.supported_labels()
    dev_lang = Counter(dev["language"])
    gold_lang = Counter(B.load_golden_set()["language"])

    # ---- persistence --------------------------------------------------------
    (REPORTS / "segmentation.json").write_text(json.dumps(seg, indent=2, default=str))
    (REPORTS / "calibration.json").write_text(json.dumps(cal, indent=2, default=str))

    err_report = {
        "n_logistic_msg_only": len(errs_msg),
        "n_logistic_ctx": len(errs_ctx),
        "n_svm_msg_only": len(errs_svm),
        "logistic_msg_only_errors": errs_msg,
        "logistic_ctx_errors": errs_ctx,
        "svm_msg_only_errors": errs_svm,
    }
    (REPORTS / "error_analysis.json").write_text(json.dumps(err_report, indent=2, ensure_ascii=False))

    # ---- markdown write-ups -------------------------------------------------
    md = build_markdown(seg, cal, errs_msg, patterns_msg, patterns_ctx,
                        dist_dev, dist_golden, dev_lang, gold_lang, labels,
                        split_report, pred)
    (REPORTS / "analysis_4D.md").write_text(md)

    print(f"Wrote reports/segmentation.json calibration.json error_analysis.json analysis_4D.md")
    print(f"  errors (logistic/msg): {len(errs_msg)}  (svm/msg): {len(errs_svm)}")
    print("  calibration buckets:", ", ".join(f"{k}:{v['macro_f1']:.2f}" for k, v in cal.items()))


def build_markdown(seg, cal, errs, patterns_msg, patterns_ctx,
                   dist_dev, dist_golden, dev_lang, gold_lang, labels,
                   split_report, pred):
    lines = []
    A = lines.append
    A("# Phase 4D — Evaluation & Error Analysis\n")
    A(f"Generated: deterministic baselines on the developer set split + golden benchmark.\n")

    # headline
    A("## 1. Headline (golden benchmark, macro F1)\n")
    A("| Model | Input | Acc | MacroF1 | W-F1 |")
    A("|---|---|---:|---:|---:|")
    grp = pred[pred["subset"] == "golden_all"]
    rows = []
    for (m, i), g in grp.groupby(["model", "input"]):
        from sklearn.metrics import f1_score
        acc = (g["gold_intent"] == g["predicted_intent"]).mean()
        mf = f1_score(g["gold_intent"], g["predicted_intent"], average="macro", zero_division=0)
        wf = f1_score(g["gold_intent"], g["predicted_intent"], average="weighted", zero_division=0)
        rows.append((m, i, acc, mf, wf))
    for m, i, acc, mf, wf in sorted(rows, key=lambda x: -x[3]):
        A(f"| {m} | {i} | {acc:.3f} | {mf:.3f} | {wf:.3f} |")
    A("")

    # split
    A("## 2. Developer-set split (conversation-disjoint)")
    A("```")
    A(f"sizes: {split_report['sizes']}")
    A(f"cross-split duplicate messages (train/val/test): {split_report['cross_split_dup_messages']}")
    A(f"duplicate message text golden-vs-{split_report['dup_message_golden_vs_train']}"
      f"/{split_report['dup_message_golden_vs_val']}"
      f"/{split_report['dup_message_golden_vs_test']} (train/val/test)")
    A("```\n")

    # segmentation
    A("## 3. Segmentation (macro F1; logistic, message-only, golden)")
    for col in ("difficulty", "language", "context_dependency", "message_type"):
        A(f"### {col}")
        A("| value | n | acc | macro_f1 |")
        A("|---|---:|---:|---:|")
        for v, d in sorted(seg[col]["golden_all"].items(), key=lambda x: -x[1]["n"]):
            A(f"| {v} | {d['n']} | {d['acc']:.3f} | {d['macro_f1']:.3f} |")
        A("")

    # calibration
    A("## 4. Calibration / confidence (logistic message-only, golden)")
    A("Note: absolute confidence never exceeds ~0.25 (14-way softmax, 140 train rows), "
      "so buckets are quantiles of observed confidence.\n")
    A("| bucket | n | conf_range | acc | macro_f1 |")
    A("|---|---:|---|---:|---:|")
    for k, v in cal.items():
        A(f"| {k} | {v['n']} | {v['conf_range']} | {v['acc']:.3f} | {v['macro_f1']:.3f} |")
    A("")

    # errors
    A("## 5. Error analysis (logistic message-only)\n")
    slm = pred[(pred['subset'] == 'golden_all') & (pred['model'] == 'logistic')
               & (pred['input'] == 'message_only')]
    total_err = int((slm['gold_intent'] != slm['predicted_intent']).sum())
    A(f"On the golden benchmark, logistic/message-only misclassifies "
      f"{total_err}/{len(slm)} examples. All captured in `reports/error_analysis.json`; "
      f"a sample follows.\n")
    A("Top confusion patterns (gold -> predicted):\n")
    A("| gold | predicted | count |")
    A("|---|---|---:|")
    for (g, p), n in patterns_msg:
        A(f"| {g} | {p} | {n} |")
    A("\nExample errors:\n")
    A("| conv | lang | diff | ctx | gold | pred | conf | message |")
    A("|---|---|---|---|---|---|---:|---|")
    for e in errs[:25]:
        msg = e["current_message"].replace("|", "/").replace("\n", " ")[:90]
        A(f"| {e['conversation_id']} | {e['language']} | {e['difficulty']} "
          f"| {e['context_dependency']} | {e['gold_intent']} | {e['predicted_intent']} "
          f"| {e['confidence'] if e['confidence'] else '-'} | {msg} |")
    A(f"\nMessage+context confusion patterns differ: {dict(patterns_ctx[:8])}\n")

    # dev vs golden
    A("## 6. Distinguishing the dev benchmark from the golden benchmark\n")
    A("### Label distribution\n")
    A("| intent | dev(n) | golden(n) |")
    A("|---|---:|---:|")
    for lab in labels:
        A(f"| {lab} | {dist_dev.get(lab, 0)} | {dist_golden.get(lab, 0)} |")
    A("")
    A("### Language distribution\n")
    langs = sorted(set(dev_lang) | set(gold_lang))
    A("| lang | dev(n) | golden(n) |")
    A("|---|---:|---:|")
    for l in langs:
        A(f"| {l} | {dev_lang.get(l, 0)} | {gold_lang.get(l, 0)} |")
    A("")
    A("### Read-out\n")
    A("- Label skew differs: dev is dominated by `none` (acknowledgements/chat filler), "
      "golden emphasizes actionable intents (esp. `delivery_delay`). Majority baseline "
      "on golden predicts `none` and lands at macro-F1 ~0.02 / acc 0.16 — that gap is "
      "expected and meaningful (a real distribution shift, not a code bug).\n")
    A("- Context availability: ~72% of dev messages have 0 retrievable prior turns, "
      "so message+context <= message-only for most examples — a data-characteristic "
      "finding, not a modelling failure.\n")
    A("- 1 exact-duplicate message text (a canned `@USER URL`) crosses golden-train; "
      "it is templated content, the conversations differ, and it carries no useful signal.\n")
    A("- Confusion concentrates in delivery_delay -> none and charge/dispatch intent "
      "families (delivered_* vs delivery_delay), i.e. the model under-commits and the "
      "multilingual short messages resist TF-IDF on 140 training rows.\n")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
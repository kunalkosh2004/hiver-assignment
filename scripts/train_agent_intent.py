"""Train the agent's intent classifiers (weak-201k + dev-140) and save models.

Models trained here are the *agent's runtime* models. The golden benchmark is
never shown to either fit routine; dev/validation is only used for a sanity
report. Training data:

- weak   201,741 messages labelled by the deterministic proxy from Phase 5E
         (explicitly NOT ground truth),
- dev    140 of the 200 in-domain dev rows, conversation-safe split (seed 42).

Usage:
    python scripts/train_agent_intent.py [--out-dir data/retrieval/models]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.agent.intent import IntentClassifier, load_weak_labels  # noqa: E402
from src.baselines import (  # noqa: E402
    macro_f1, serialize_message, split_by_conversations, load_dev_set,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(config.DATA_DIR / "retrieval" / "models"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    report: dict = {}

    # ---------- weak-201k (agent's runtime model) ----------
    print("[train] loading weak labels + corpus …", flush=True)
    t0 = time.time()
    msgs, labels = load_weak_labels()
    print(f"[train] {len(msgs):,} weak-labelled messages in {time.time()-t0:.1f}s", flush=True)

    print("[train] fitting weak-201k intent classifier …", flush=True)
    t0 = time.time()
    weak = IntentClassifier.fit_weak(msgs, labels)
    weak.save(out / "intent_weak")
    report["weak"] = {
        "n_train": len(msgs),
        "source": weak.source,
        "classes_n": len(weak.classes),
        "classes": weak.classes,
        "fit_seconds": round(time.time() - t0, 1),
        "imbalance_note": "delivery_delay dominates weak labels (~86%); proxy bias disclosed",
    }
    report["weak"]["class_counts"] = {
        c: labels.count(c) for c in weak.classes if c in labels
    }
    print(f"[train] weak fit done in {time.time()-t0:.1f}s → {weak.classes}", flush=True)

    # ---------- dev-140 (classifier scaling-curve point) ----------
    print("[train] fitting dev-140 human-label model …", flush=True)
    dev = load_dev_set()
    split = split_by_conversations(dev, train_frac=0.70, val_frac=0.15, seed=42)
    val = split["validation"]
    xs = [serialize_message(r) for _, r in val.iterrows()]
    ys = list(val["primary_intent"])
    t0 = time.time()
    dev_model, info = IntentClassifier.fit_dev(seed=42)
    dev_model.save(out / "intent_dev")
    vpred = dev_model.pipeline.predict(xs)
    report["dev"] = {
        **info,
        "classes_n": len(dev_model.classes),
        "validation_macro_f1": round(macro_f1(ys, vpred), 4),
        "validation_n": len(ys),
        "fit_seconds": round(time.time() - t0, 1),
    }
    print(f"[train] dev fit done in {time.time()-t0:.1f}s; val macro-F1 "
          f"{report['dev']['validation_macro_f1']:.3f}", flush=True)

    # ---------- weak-model cross-check on dev/validation (diagnostic) ----------
    try:
        wpred = weak.pipeline.predict(xs)
        report["weak"]["dev_validation_proxy_macro_f1"] = round(macro_f1(ys, wpred), 4)
    except Exception as exc:  # noqa: BLE001
        report["weak"]["dev_validation_proxy_error"] = str(exc)

    (config.PROJECT_ROOT / "reports" / "agent_intent_models.json").write_text(
        json.dumps(report, indent=2))
    print("[train] wrote reports/agent_intent_models.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
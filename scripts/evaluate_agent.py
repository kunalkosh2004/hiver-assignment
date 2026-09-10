"""Final evaluation harness for the retrieval-augmented agent (Phases 6-8).

Runs the agent over the 200-row golden benchmark (used ONLY here) with the
temporal filter, and reports:

- intent classifier learning curve  (majority -> dev-140 human -> weak-201k;
                                     LLM-labelled data = future work, dashed point),
- escalation policy behavior        (auto/escalate split, by-true-intent,
                                     confidence, ECE-style calibration on dev),
- deterministic draft grounding     (cue-overlap vs golden human reply + a
                                     grounded-when-reused substring check),
- optional LLM judge                (only if a provider key is present; sampled;
                                     otherwise recorded as disabled).

Reports: reports/agent_results.{json,csv}, figures/agent_learning_curve.png,
figures/agent_escalation_calibration.png

Usage:
    python scripts/evaluate_agent.py [--llm-sample 60] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.agent import Agent, IntentClassifier  # noqa: E402
from src.baselines import (  # noqa: E402
    load_dev_set, load_golden_set, load_holdout, macro_f1, serialize_message,
    split_by_conversations,
)

REFERENCE = config.DATA_DIR / "golden" / "amazon_golden_eval_reference.jsonl"

REPORTS = config.PROJECT_ROOT / "reports"
MODELS = config.DATA_DIR / "retrieval" / "models"
FULL = 201_741

CUE_EXTRACTORS = {
    "url": lambda s: "http" in s.lower(),
    "email": lambda s: ("@" in s and "." in s) and not s.lower().startswith("@user"),
    "apology": lambda s: any(c in s.lower() for c in ("sorry", "apolog", "regret")),
    "ask": lambda s: any(c in s.lower() for c in ("order id", "order number", "please share", "provide", "email address")),
    "contact": lambda s: any(c in s.lower() for c in ("contact", "message us", "get in touch", "reply to this", "chat", "reach out")),
}


def cue_set(text: str) -> set[str]:
    return {name for name, fn in CUE_EXTRACTORS.items() if fn(text or "")}


def ece(confs: list[float], correct: list[int], n_bins: int = 5) -> float:
    if not confs:
        return float("nan")
    edges = [i / n_bins for i in range(n_bins + 1)]
    err = total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = [i for i, c in enumerate(confs) if lo <= c < hi or (hi == 1.0 and c == 1.0)]
        if not idx:
            continue
        conf = sum(confs[i] for i in idx) / len(idx)
        acc = sum(correct[i] for i in idx) / len(idx)
        w = len(idx) / len(confs)
        err += w * abs(conf - acc)
        total += w
    return err / total if total else float("nan")


def calibrate_dev() -> dict:
    """Confidence calibration diagnostic using human-labelled dev (sim-term fixed)."""
    dev = load_dev_set()
    split = split_by_conversations(dev, seed=42)
    model, _ = IntentClassifier.fit_dev(seed=42)
    confs, correct = [], []
    for _, r in split["validation"].iterrows():
        intent, p, _ = model.predict(serialize_message(r))
        conf = 0.6 * p + 0.4 * 0.2  # no retrieval available on dev; sim term constant
        confs.append(conf)
        correct.append(int(intent == r["primary_intent"]))
    return {
        "ece": round(ece(confs, correct), 4), "n": len(confs),
        "sim_term": "unavailable on dev (fixed at 0.2)",
    }


def load_golden_replies() -> dict[str, str]:
    """next_brand_response per example_id from the golden *reference* file."""
    if not REFERENCE.exists():
        return {}
    return {
        rec["example_id"]: (rec.get("next_brand_response") or "")
        for rec in (json.loads(l) for l in REFERENCE.open())
        if isinstance(rec.get("next_brand_response"), str)
    }


def reflect_context(r) -> str | None:
    ctx = r.get("conversation_context") or r.get("previous_brand_message")
    if isinstance(ctx, list):
        parts = [x.get("text", "") for x in ctx if isinstance(x, dict)]
        parts = [p for p in parts if isinstance(p, str)]
        return "\n".join(parts) or None
    return ctx if isinstance(ctx, str) else None


def run_golden(agent: Agent) -> list[dict]:
    replies = load_golden_replies()
    rows = []
    for _, r in load_golden_set().reset_index(drop=True).iterrows():
        out = agent.respond(
            r["current_customer_message"],
            conversation_context=reflect_context(r),
            created_at=r.get("created_at"),
            top_k=5,
        )
        out["example_id"] = r["example_id"]
        out["query"] = r["current_customer_message"]
        out["language"] = r.get("language", "")
        out["true_intent"] = r["primary_intent"]
        out["golden_reply"] = replies.get(r["example_id"], "")
        out["cue_overlap"] = sorted(cue_set(out["draft"]) & cue_set(out["golden_reply"]))
        rows.append(out)
    golden_ids = {str(c) for c in load_golden_set()["conversation_id"]}
    hit_ids = {h["case_id"] for r in rows for h in r["evidence"]}
    assert not (hit_ids & golden_ids), "golden conversations leaked into agent evidence"
    return rows


def ground_tokens(draft: str, reply: str) -> set[str]:
    """Reply words (len>=5, non-handle/url/order) that reappear in the draft."""
    import re  # noqa: PLC0415

    def words(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-zA-Z]{5,}", (text or "").lower())
                if not w.startswith(("http", "tco"))}
    return words(draft) & words(reply)


def grounded_reuse(draft: str, reply: str, min_tokens: int = 2) -> bool:
    """True if the draft reuses >= min_tokens distinct reply words."""
    if not isinstance(reply, str) or not reply or not draft:
        return False
    return len(ground_tokens(draft, reply)) >= min_tokens


def pref_best_reply(rows: list[dict]) -> list[str]:
    replies = []
    for r in rows:
        best = next((h["brand_response"] for h in r["evidence"]
                     if isinstance(h.get("brand_response"), str) and h["brand_response"]), "")
        replies.append(best or "")
    return replies


def intent_metrics_from(true: list[str], preds: list[str]) -> dict:
    return {
        "accuracy": round(sum(a == b for a, b in zip(true, preds)) / len(true), 4),
        "macro_f1": round(macro_f1(true, preds), 4),
    }


def plot_learning_curve(curve: list[dict], out: Path) -> None:
    xs = [pt["n_train"] for pt in curve]
    ys = [pt["macro_f1"] for pt in curve]
    labels = [pt["label"] for pt in curve]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, "o-", color="#1f77b4")
    plt.plot([xs[-1], 800_000], [ys[-1], max(ys) + 0.02], "--", color="#999")
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_xticks(xs + [800_000], labels + ["LLM? future"], rotation=20, ha="right")
    ax.set_xlabel("train rows (log)")
    ax.set_ylabel("macro-F1 on golden (200)")
    ax.set_title("Intent classifier learning curve")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def plot_calibration(confs: list[float], correct: list[int], out: Path) -> None:
    plt.figure(figsize=(6, 4))
    bins = [i / 5 for i in range(6)]
    mean_conf, acc = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = [i for i, c in enumerate(confs) if lo <= c < hi or (hi == 1.0 and c == 1.0)]
        if idx:
            mean_conf.append(sum(confs[i] for i in idx) / len(idx))
            acc.append(sum(correct[i] for i in idx) / len(idx))
    plt.plot([0, 1], [0, 1], "--", color="#999", label="perfect")
    plt.plot(mean_conf, acc, "o-", color="#1f77b4", label="dev")
    plt.xlabel("avg confidence"); plt.ylabel("accuracy")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out, dpi=130); plt.close()


def llm_judge_sample(rows: list[dict], sample: int, seed: int) -> dict:
    from src.llm import make_router  # noqa: PLC0415

    router = make_router()
    if not router.enabled():
        return {"status": "disabled", "note": "no LLM provider key set"}
    import random  # noqa: PLC0415

    sample_rows = random.Random(seed).sample(rows, min(sample, len(rows)))
    schema = {
        "type": "object",
        "properties": {
            "helpful": {"type": "boolean"},
            "grounded_in_retrieval": {"type": "boolean"},
            "hallucination_free": {"type": "boolean"},
            "notes": {"type": "string"},
        },
    }
    judged = []
    for r in sample_rows:
        res = router.complete_structured(
            system_prompt="You grade a support agent draft. Output ONLY the schema fields.",
            user_prompt=(
                f"Customer message:\n{r['draft']}\n\n"
                f"DRAFT reply:\n{r['draft']}\n\n"
                f"Retrieved historical resolutions:\n"
                + " ".join(h.get("brand_response") or "" for h in r["evidence"] if h.get("brand_response"))[:1500]
            ),
            response_schema=schema,
        )
        if res.data:
            judged.append(res.data)
    return {
        "status": "ran", "n": len(judged),
        "helpful_rate": round(sum(1 for j in judged if j.get("helpful")) / max(1, len(judged)), 4),
        "grounded_rate": round(sum(1 for j in judged if j.get("grounded_in_retrieval")) / max(1, len(judged)), 4),
        "hallucination_free_rate": round(sum(1 for j in judged if j.get("hallucination_free")) / max(1, len(judged)), 4),
    }


def escalate_sensitivity(rows: list[dict], floors: list[float]) -> dict:
    """Escalation count under different combined-confidence floors (recomputed
    from the per-row dump; no retuning on golden)."""
    out = {}
    for cl in floors:
        n = 0
        for r in rows:
            usable = any(isinstance(h.get("brand_response"), str) and h["brand_response"]
                         for h in r["evidence"][:3])
            sims = [h["score"] for h in r["evidence"]]
            sim = sum(sims[:3]) / max(1, len(sims[:3]))
            n += int(r["intent"] in ("none", "other_unclear") or not usable
                     or (0.6 * r["intent_prob"] + 0.4 * sim) < cl)
        out[str(cl)] = {"escalate": n, "rate": round(n / max(1, len(rows)), 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-sample", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump-rows", help="also write per-row agent output JSON here")
    args = ap.parse_args()

    start = time.time()
    weak = IntentClassifier.load(MODELS / "intent_weak")
    agent = Agent.load(intent_dir=MODELS / "intent_weak")
    dev_model = IntentClassifier.load(MODELS / "intent_dev")

    rows = run_golden(agent)
    true = [r["true_intent"] for r in rows]
    preds = [r["intent"] for r in rows]
    msg_preds = dev_model.pipeline.predict([r["query"] for r in rows])

    train = split_by_conversations(load_dev_set(), seed=42)["train"]
    maj = Counter(train["primary_intent"]).most_common(1)[0][0]
    curve = [
        {"label": "majority", "n_train": 0, "classes_n": len(set(true)),
         **intent_metrics_from(true, [maj] * len(true))},
        {"label": "dev-140 (human)", "n_train": int(len(train)), "classes_n": len(dev_model.classes),
         **intent_metrics_from(true, msg_preds)},
        {"label": "weak-201k", "n_train": FULL, "classes_n": len(weak.classes),
         **intent_metrics_from(true, preds)},
    ]
    # comparability: shared-intent-subset metrics (drop golden rows outside the
    # model's label space, e.g. 'none' rows are unseen by the weak model).
    full_labels = set(true)
    for pt, plabels in zip(curve, (full_labels, set(dev_model.classes), set(weak.classes))):
        idx = [i for i, t in enumerate(true) if t in plabels]
        pt["on_shared_subset"] = {
            "n": len(idx),
            **intent_metrics_from([true[i] for i in idx], [preds[i] for i in idx]
                                  if plabels == set(weak.classes) else
                                  [msg_preds[i] for i in idx] if plabels == set(dev_model.classes) else
                                  [maj] * len(idx)),
        }

    actions = [r["action"] for r in rows]
    best_replies = pref_best_reply(rows)
    ons = [r for r in rows if r["action"] == "auto_handle"]
    grounded_n = sum(1 for i, r in enumerate(ons) if grounded_reuse(r["draft"], best_replies[rows.index(r)]))
    escalation = {
        "auto_handle": int(actions.count("auto_handle")),
        "escalate": int(actions.count("escalate")),
        "escalation_rate": round(actions.count("escalate") / len(actions), 4),
        "by_true_intent": dict(sorted(Counter(r["true_intent"] for r in rows if r["action"] == "escalate").items())),
        "sensitivity_to_comb_low": escalate_sensitivity(rows, [0.45, 0.60, 0.75, 0.8775, 0.95]),
        "active_comb_low": agent.thresholds.get("comb_low", 0.45),
        "note": "dev-calibrated floor transfers poorly to golden (domain shift) — sensitivity shown",
    }
    drafting = {
        "avg_cue_overlap": round(sum(len(r["cue_overlap"]) for r in rows) / len(rows), 3),
        "overlap_cooc": {str(k): v for k, v in Counter(tuple(r["cue_overlap"]) for r in rows).items()},
        "resolution_reuse_rate_auto": round(grounded_n / max(1, len(ons)), 4),
        "empty_drafts": sum(1 for r in rows if not r["draft"]),
        "avg_draft_chars": round(sum(len(r["draft"]) for r in rows) / len(rows), 1),
    }
    calib = calibrate_dev()
    if args.dump_rows:
        Path(args.dump_rows).write_text(json.dumps(sorted(rows, key=lambda x: str(x["example_id"])), indent=2))
    report = {
        "phase": "phase8-final-agent",
        "n_golden": len(rows),
        "intent_learning_curve": curve,
        "escalation": escalation,
        "drafting": drafting,
        "calibration_dev": calib,
        "llm_judge": llm_judge_sample(rows, args.llm_sample, args.seed),
        "seconds": round(time.time() - start, 1),
    }
    (REPORTS / "agent_results.json").write_text(json.dumps(report, indent=2))
    with (REPORTS / "agent_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "example_id", "true_intent", "intent", "action", "confidence",
            "language", "cue_overlap", "draft"])
        w.writeheader()
        for r in sorted(rows, key=lambda x: str(x["example_id"])):
            w.writerow({
                "example_id": r["example_id"], "true_intent": r["true_intent"],
                "intent": r["intent"], "action": r["action"],
                "confidence": r["action_confidence"], "language": r["language"],
                "cue_overlap": ",".join(r["cue_overlap"]), "draft": r["draft"],
            })
    plot_learning_curve(curve, REPORTS / "figures" / "agent_learning_curve.png")
    plt.figure(figsize=(6, 4))
    bins = [i / 5 for i in range(6)]
    mc, a = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = [i for i, c in enumerate([r["action_confidence"] for r in rows]) if lo <= c < hi or (hi == 1.0 and c == 1.0)]
        if idx:
            mc.append(sum([r["action_confidence"] for r in rows][i] for i in idx) / len(idx))
            a.append(sum(int(preds[i] == true[i]) for i in idx) / len(idx))
    plt.plot([0, 1], [0, 1], "--", color="#999")
    plt.plot(mc, a, "o-", color="#d62728", label="golden")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(REPORTS / "figures" / "agent_escalation_calibration.png", dpi=130); plt.close()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
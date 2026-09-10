"""Confidence-based escalation policy (deterministic, fixed, documented).

The decision rule derives from fixed, documented defaults optionally overridden
by a JSON file of *dev-calibrated* thresholds (written by
scripts/calibrate_escalation.py from the labelled dev set only — the golden
benchmark is never used to set these). Escalation fires on:

1. hard intents   — 'none' / 'other_unclear' (catch-all bucket = no actionable
                    intent);
2. high-risk intent — the intent is a *human-ownable risk category* (security,
                    money, remedy, delivery investigation, complaint — see
                    config/risk_intents.json), irrespective of confidence:
                    Phase A's hand-labelled decision benchmark showed these
                    must reach a human even when the model is confident;
3. missing evidence — no usable historical reply in the top-3, or no hits at
                    all (retrieval failed);
4. low confidence — combined confidence (0.6 * intent_prob + 0.4 * mean(top-3
                    similarity)) below a dev-calibrated floor (now applies only
                    to non-high-risk intents).

The high-risk set comes from a human-labelled 200-row escalation benchmark
(data/golden/escalation_gold.jsonl, Phase A) and is stored in
config/risk_intents.json; the benchmark itself is evaluation-only and is never
merged into training or the threshold fit. The floor (3) is set on dev only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .draft import draft_auto, draft_escalation

P_INTENT_LOW = 0.30
SIM_FLOOR = 0.02
COMB_LOW = 0.45
W_INTENT, W_SIM = 0.6, 0.4

DEFAULT_KEYS = ("p_intent_low", "sim_floor", "comb_low")


def load_thresholds(path: Path | str | None) -> dict:
    """Load dev-calibrated thresholds (no golden involvement)."""
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        data = {**data.get("thresholds", {}), **data}
        return {k: data[k] for k in DEFAULT_KEYS if k in data}
    except Exception:  # noqa: BLE001
        return {}


def _load_risk_intents() -> set[str]:
    """Human-labelled high-risk intent set (Phase A benchmark, traceable)."""
    p = Path(__file__).resolve().parents[2] / "config" / "risk_intents.json"
    try:
        return set(json.loads(p.read_text()))
    except Exception:  # noqa: BLE001
        return set()


RISK_INTENTS = _load_risk_intents()


@dataclass
class EscalationDecision:
    action: str  # 'auto_handle' | 'escalate'
    intent: str
    intent_prob: float
    margin: float
    sim_mean: float
    confidence: float
    reasons: list[str] = field(default_factory=list)

    def to_record(self) -> dict:
        return {
            "action": self.action,
            "intent": self.intent,
            "intent_prob": round(self.intent_prob, 4),
            "margin": round(self.margin, 4),
            "sim_mean": round(self.sim_mean, 4),
            "confidence": round(self.confidence, 4),
            "reasons": self.reasons,
        }


def decide(
    *, intent: str, intent_prob: float, margin: float, hits: list,
    thresholds: dict | None = None,
    w_intent: float = W_INTENT, w_sim: float = W_SIM,
) -> EscalationDecision:
    """Decide auto_handle vs escalate from intent + retrieval evidence."""
    th = thresholds or {}
    comb_low = th.get("comb_low", COMB_LOW)
    sim_floor = th.get("sim_floor", SIM_FLOOR)

    sims = [float(h.score) for h in hits]
    sim_mean = sum(sims[:3]) / max(1, len(sims[:3]))
    usable = any(isinstance(h.brand_response, str) and h.brand_response for h in hits[:3])
    confidence = min(1.0, max(0.0, w_intent * intent_prob + w_sim * sim_mean))

    reasons: list[str] = []
    action = "auto_handle"

    if intent in ("none", "other_unclear"):
        reasons.append(f"intent={intent} (no actionable intent)")
        action = "escalate"
    elif intent in RISK_INTENTS:
        reasons.append(f"intent={intent} (high-risk category)")
        action = "escalate"
    elif not hits:
        reasons.append("no retrieval hits")
        action = "escalate"
    elif not usable:
        reasons.append("no usable reply in top-3 evidence")
        action = "escalate"
    elif sim_mean < sim_floor:
        reasons.append(f"sim_mean={sim_mean:.2f} < {sim_floor}")
        action = "escalate"
    elif confidence < comb_low:
        reasons.append(f"confidence={confidence:.2f} < comb_low={comb_low}")
        action = "escalate"

    if not reasons:
        reasons.append("confident intent + evidence present")

    return EscalationDecision(
        action=action, intent=intent, intent_prob=intent_prob,
        margin=margin, sim_mean=sim_mean, confidence=confidence, reasons=reasons,
    )


def build_draft(decision: EscalationDecision, query: str, hits: list) -> str:
    if decision.action == "escalate":
        return draft_escalation(decision.intent)
    reply = next((h.brand_response for h in hits if isinstance(h.brand_response, str) and h.brand_response), None)
    return draft_auto(query, decision.intent, reply)
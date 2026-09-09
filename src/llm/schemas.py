"""JSON schemas describing the structured outputs the LLM providers must emit.

These are frozen for the LLM-classification job (a classifier has no business
producing free text). `IntentionSchema` mirrors the frozen 12-intent taxonomy
from config/amazon_intents.yaml so downstream consumers can validate against it.
"""
from __future__ import annotations

# Frozen intent id + their canonical human label (from config/amazon_intents.yaml).
# Kept in sync manually; `src.baselines.supported_labels` references the frozen
# taxonomy for the deterministic baselines.
INTENT_IDS = [
    "refund_request",
    "delivery_delay",
    "order_status_query",
    "charge_issue",
    "product_return_and_replacement",
    "delivered_wrong_location",
    "delivered_but_not_received",
    "account_access",
    "account_info_update",
    "device_app_issue",
    "cancellation",
    "service_complaint_escalation",
]

# Sentinel classes beyond the frozen taxonomy (mirrors deterministic labels).
SPECIAL_LABELS = ["none", "other_unclear"]

INTENT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "primary_intent": {"type": "string", "enum": INTENT_IDS + SPECIAL_LABELS},
        "secondary_intents": {
            "type": "array",
            "items": {"type": "string", "enum": INTENT_IDS + SPECIAL_LABELS},
            "maxItems": 3,
        },
        "message_type": {
            "type": "string",
            "enum": [
                "support_request", "complaint", "acknowledgement",
                "follow_up", "clarification", "unknown",
            ],
        },
        "context_dependency": {
            "type": "string",
            "enum": ["self_contained", "context_helpful", "context_required"],
        },
        "label_confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
    },
    "required": [
        "primary_intent",
        "secondary_intents",
        "message_type",
        "context_dependency",
        "label_confidence",
    ],
}

CLASSIFY_SYSTEM_PROMPT = (
    "You are an Amazon Customer Support intent classifier. Classify ONLY the "
    "current customer message using the frozen intent taxonomy provided. Return "
    "strictly valid JSON matching the provided schema. Use primary_intent="
    "\"none\" when the message is not a concrete actionable request (greeting, "
    "thanks, ack) and other_unclear only when genuinely unclassifiable. Never "
    "invent intents."
)
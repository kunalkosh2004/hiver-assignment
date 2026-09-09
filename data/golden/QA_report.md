# Golden Set QA Report (amazon-intents-v1)

- examples: 200 (150 representative + 50 challenge)
- distinct conversations: 200
- languages: {'es': 3, 'en': 180, 'de': 2, 'fr': 5, 'it': 3, 'ja': 5, 'pt': 2}
- primary intents: {'account_access': 6, 'cancellation': 6, 'charge_issue': 11, 'delivered_but_not_received': 11, 'delivered_wrong_location': 7, 'delivery_delay': 43, 'device_app_issue': 14, 'none': 32, 'order_status_query': 13, 'other_unclear': 3, 'product_return_and_replacement': 18, 'refund_request': 8, 'service_complaint_escalation': 27, 'account_info_update': 1}
- message_type: {'complaint': 106, 'support_request': 23, 'follow_up': 36, 'clarification': 5, 'acknowledgement': 26, 'unknown': 4}
- context_dependency: {'context_helpful': 140, 'self_contained': 54, 'context_required': 6}
- difficulty: {'hard': 58, 'easy': 47, 'medium': 95}
- label_confidence: {'medium': 97, 'low': 17, 'high': 86}
- low-confidence examples: 17 (all have notes)
- other_unclear rate: 1.5%
- annotator agreement: single annotator, no IAA available
- leakage checks: no next_brand_response in eval; conversations distinct; all in holdout
- builder validation: counts, taxonomy membership, schema enums, note requirements — all pass

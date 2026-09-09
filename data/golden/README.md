# AmazonHelp Golden Evaluation Set (v1)

Hand-annotated, conversation-safe benchmark for AmazonHelp **support-intent
classification**. Built in Phase 3 of the Hiver SDE-Intern eval pipeline.

**Taxonomy version:** `amazon-intents-v1` (see `config/amazon_intent_version.txt`,
`config/amazon_intent_guidelines.yaml`, taxonomy source `config/amazon_intents.yaml`).

---

## Files

| File | Contents |
|------|----------|
| `amazon_golden_eval.jsonl` | **Eval input**: 200 examples. `current_customer_message` + `conversation_context` (list of `{role,text}`) + gold labels. **No `next_brand_response`** (leak-free, ready for classifier training/eval on the customer side). |
| `amazon_golden_eval_reference.jsonl` | Same 200 examples **plus** `next_brand_response` (brand reply) and metadata (`tweet_id`, timestamps, `previous_brand_message`, holdout flag). For inspection only — never used as model input. |
| `amazon_golden_annotation.csv` | Human-review ledger: every example with its primary/secondary intent, message type, context dependency, difficulty, confidence, and annotation note. |
| `candidates_representative.tsv`, `candidates_challenge.tsv` | Candidate pools from which the 200 were hand selected (review trails). |
| `conversation_holdout.json` | Conversation ids reserved out of training (must never leak). |
| `QA_report.md` | Persisted QA / distribution report. |

## Composition

- **Exactly 200 examples** = 150 representative + 50 challenge.
- **Representative (150):** standard, primarily-English examples tuned to real
  traffic (140 drawn from the English representative pool + 10 standard-English
  challenge examples focused on delivery/status coverage).
- **Challenge (50):** hard examples oversampling **multilingual (20)** and
  **ambiguous / very short / context-dependent / angry (30)** messages.
- **Conversation-safe:** conversations were split *before* selection; every
  golden example maps to a distinct conversation (`distinct conversations = 200`),
  none falls in the reserved holdout, and `src/evaluation.assert_no_conversation_overlap`
  guards any future train split.
- **Turn-accurate:** each example corresponds to the *exact* tweet that was
  hand-reviewed (joined by `tweet_id`, since one conversation has many tweets),
  so the labels describe the `current_customer_message` actually shown.

## Labels

Per example (see `config/amazon_intent_guidelines.yaml` for full definitions):

- `primary_intent` — one of the 12 frozen intents, or `none`
  (conversational layer: thanks/ack), or `other_unclear` (3 used, each with a note).
- `secondary_intents` — genuinely co-occurring intents (never equal to primary).
- `message_type` — support_request / follow_up / clarification / acknowledgement / complaint / unknown.
- `context_dependency` — self_contained / context_helpful / context_required.
- `difficulty` — easy / medium / hard.
- `label_confidence` — high / medium / low (every low + every other_unclear has a note).
- `language`, `annotation_note`.

## Distribution summary

- **Intents (all):** delivery_delay 43, none 32, service_complaint_escalation 27,
  product_return_and_replacement 18, device_app_issue 14, order_status_query 13,
  charge_issue 11, delivered_but_not_received 11, refund_request 8,
  delivered_wrong_location 7, account_access 6, cancellation 6,
  other_unclear 3, account_info_update 1.
- **Languages:** en 180, fr 5, ja 5, es 3, it 3, de 2, pt 2
  (20 of 50 challenge examples are multilingual — vs ~27% in the source corpus).
- **Difficulty:** medium 95, hard 58, easy 47.
- **Confidence:** medium 97, high 86, low 17.
- **Message type:** complaint 106, follow_up 36, acknowledgement 26,
  support_request 23, clarification 5, unknown 4.
- **Context dependency:** context_helpful 140, self_contained 54, context_required 6.

## Annotator agreement

Single-annotator benchmark. **No inter-annotator agreement is available**;
17 low-confidence examples and 3 `other_unclear` examples carry explicit notes
documenting the residual ambiguity. A second-pass review is recommended before
Phase 4.

## `prime_membership` decision

`prime_membership` is intentionally **excluded** as a top-level intent (see
decision log in `config/amazon_intent_guidelines.yaml`): in this corpus "prime"
almost always means (a) promised shipping speed -> `delivery_delay`, (b) Prime
Video / Prime Music / Prime Reading -> `device_app_issue`, or (c) genuine
membership-management messages that co-occur with `charge_issue`, `cancellation`
or `account_access`. Adding it would create a weak, overlapping boundary.

## Quality checks (all pass)

- unique `example_id`s; unique `conversation_id`s; distinct conversations = 200
- no empty messages
- all primary/secondary intents within the frozen 12-intent taxonomy
- no `next_brand_response` leakage into the eval jsonl
- no conversation overlap with the holdout / any train split
- required fields present on every example

# Reply-quality rubric & judge-agreement protocol (Phase B)

This document is the canonical rubric used by `scripts/evaluate_judge_agreement.py`.
It defines (1) the quality dimensions, (2) 1–5 scoring tiers with anchors,
(3) the boolean mapping the judge emits, (4) the agreement metrics used as
*evidence that the LLM judge can be trusted*, and (5) how to read the numbers.

## 1. Dimensions (judged per reply)

| dimension | boolean emitted | question the judge answers |
|---|---|---|
| `helpful` | `helpful` | Does the reply target/resolve the customer's actual concern? |
| `grounded_in_retrieval` | `grounded_in_retrieval` | Is it consistent with the customer's stated facts AND the retrieved historical resolutions it cites? |
| `hallucination_free` | `hallucination_free` | Does it invent order ids, delivery dates, refunds, promises, fees, or contact numbers? |

The judge also returns free-text `notes` (2–4 words) for traceability.

## 2. Numeric tiers (the rubric behind the boolean)

The boolean is derived from the integer tier with `tier >= 4` ⇒ `true`:

| tier | `helpful` | `grounded_in_retrieval` | `hallucination_free` |
|----|----|----|----|
| 1 | ignores the concern / answers a different question | contradicts the customer's own facts | fabricates an order id / refund / promise |
| 2 | tangential acknowledgment, no action or answer | cites a resolution unrelated to the facts | plausible but unverifiable detail asserted as fact |
| 3 | partial: addresses part, omits the ask | correct on the main point, loose on details | generic filler only (no invented specifics, no useful specifics) |
| 4 | directly resolves the concern with a concrete next step | uses customer facts + retrieved resolution correctly | concrete, retrieval-supportable, nothing invented |
| 5 | resolves the concern and anticipates the likely follow-up | integrates multiple retrieved resolutions accurately | fully precise; every claim traceable |

Mapping: `tier >= 4 → helpful=True/grounded=True/hallucination_free=True` (a
2–3 on hallucination (generic only) is treated as *free of invented facts*).

## 3. Protocol (how `evaluate_judge_agreement.py` gathers evidence)

For a fixed sample of golden rows (`--sample 15`, seed 42):

1. **Judge the human reference reply twice** (`golden_reply`), identical prompts.
   → self-consistency / *judge repeatability*.
2. **Judge the agent draft once** (`draft`), with the retrieved resolutions
   shown (`_evid(r["evidence"])`), so `grounded_in_retrieval` is checkable.
3. Sleep ≥11 s between calls to remain inside Gemini free-tier limits; a
   provider failure skips that judgment (counted in `skipped`), never crashes
   the run; OpenAI is the automatic fallback provider.

Prompts are the literal strings in the script: the judge is instructed to
"Reply with JSON only" and the schema constrains the output.

## 4. Agreement metrics (what "trustworthy judge" means)

| metric | meaning | desired reading |
|---|---|---|
| `judge_human_helpful_rate` | how often the judge calls a *human-written* resolution helpful | ≥ 0.7 — humans resolve most support cases; lower ⇒ judge too strict → distrust its helpful numbers |
| `judge_repeat_self_agreement` | % of identical decisions across the two repeated judgments | ≥ 0.8 — the judge is repeatable (noisy judges inflate all gaps) |
| `judge_vs_deterministic_grounded_kappa` | Cohen's κ between the judge's `grounded_in_retrieval` and a reference-derived oracle (draft↔human-reply token overlap ≥ 2) on the same rows | ≥ 0.4 (moderate) — judge correlates with a human-ground-truth proxy |
| `human_draft_helpful_gap` | helpful_rate(human) − helpful_rate(draft) | use *only* along with the three above; a judge that fails its own sanity checks should not be used to rank drafts |

The deterministic oracle (`_deterministic_grounded`, ≥2 shared content tokens)
is noisy by design — it is a *proxy*, so κ is expected to sit below a perfect
correlation; values ≥ 0.4 alongside the two judge-sanity checks are the bar we
call "evidence the judge can be trusted".

## 5. Decision rule for using the judge

Only treat the LLM judge's `helpful` numbers as evidence when ALL hold:
`judge_repeat_self_agreement ≥ 0.8`, `judge_human_helpful_rate ≥ 0.7`, and
`judge_vs_deterministic_grounded_kappa ≥ 0.4`. Otherwise the judge is treated
as unaudited and the deterministic metrics (§8 of the main report) stand alone.
This gate is documented in §6C of `reports/FULL_PROGRESS_REPORT.md`.

## 6. Offline reproducibility

`tests/test_judge_agreement.py` validates the metric maths (self-agreement,
κ, oracle, rate-limit path) against synthetic rows with **no network**; the
full script is green-lighted to run with a funded key via
`python scripts/evaluate_judge_agreement.py --sample 15`. Current committed
`reports/judge_agreement.json` states the live run is **blocked by free-tier
quota** (Gemini 429 `You exceeded your current quota`, OpenAI key exhausted) —
the numbers will be filled in the moment a key allows runs, and `--dump-rows
reports/judge_agreement_rows.json` makes each judged row auditable.
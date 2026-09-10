# Final Report — Retrieval-Augmented Support Agent for AmazonHelp

**Repo:** [kunalkosh2004/hiver-assignment](https://github.com/kunalkosh2004/hiver-assignment) · **branch** `main` · **seed 42, fully deterministic, no API key required**
**Full evidence trail:** `reports/FULL_PROGRESS_REPORT.md` (every number recomputable offline).

---

## 0. TL;DR

Built an end-to-end support agent for AmazonHelp on the Kaggle "Customer Support on
Twitter" corpus: **intent → retrieval memory → grounded reply → auto-handle/escalate**.
The honest headline: the intent layer caps everything (golden macro-F1 0.078/0.050 for
the small-dev/weak-proxy classifiers); retrieval and drafting are strong (dense R@5
0.576; 96% grounded reuse, 0 fabricated drafts); **escalation was silently the weakest
link** — a new hand-labelled 200-row decision benchmark (Phase A) exposed that the
dev-calibrated confidence floor auto-handles 66 of 104 cases a human would escalate.
A perfect-intent risk policy would score 87.5% decision accuracy vs 55.5% shipped
(Phase C). 56 unit tests, offline reproduction measured (see §6), all phases A–D
committed and pushed.

## 1. What was built

| phase | deliverable | headline |
|---|---|---|
| P1 forensics | `notebooks/01`, `src/{config,threads,amazon}` | brand chosen = **AmazonHelp** (largest clean multi-turn threads, multilingual) |
| P2 taxonomy | `config/amazon_intents.yaml` (frozen v1) | 12 intents + `none`/`other_unclear`; `prime_membership` deliberately excluded |
| P3 golden | `data/golden/` (200 = 150 rep + 50 chal) | conversation-safe, turn-accurate, leak-free eval + separate reference file |
| P4 baselines | `scripts/run_baselines.py` | majority/TF-IDF LogReg/SVM × msg / msg+context |
| P5 retrieval | `src/retrieval/` (tfidf/bm25/dense/hybrid) | 201,741-interaction corpus; dense **R@5 0.576** (human-labelled benchmark) |
| P6 drafting | `src/agent/` | deterministic, grounded-in-retrieval; **0 fabrications**, 0 empty |
| P7 escalation | `src/agent/escalation.py` | dev-only calibrated confidence floor (`comb_low=0.8775`), golden never touched during fit |
| P8 harness | `scripts/evaluate_agent.py` | golden 200: intent learning curve + escalation + drafting + optional LLM judge |
| P9 failure analysis | `scripts/analyze_agent_errors.py` | 162/200 rows ≥1 failure; taxonomy + real examples |
| A escalation benchmark | `data/golden/escalation_gold.jsonl` | **200 human-labelled decisions** (104 escalate / 96 auto), evaluation-only |
| B judge agreement | `reports/judge_rubric.md` + metric tests | rubric, protocol, gate; live run quota-blocked (§4) |
| C consolidated bench | `reports/phase_c_benchmark.md` | all methods one table + escalation-policy probe |

The LLM layer (`src/llm/`, OpenAI+Gemini auto-fallback) is **optional and
disabled-by-default**; headline numbers never depend on it.

## 2. Results — every method on the same 200 golden rows

| method | intent acc | macro-F1 | esc-rate | decision acc | escal-F1 | false_auto |
|---|---:|---:|---:|---:|---:|---:|
| majority | 0.160 | 0.020 | – | – | – | – |
| logistic (msg) | 0.230 | 0.164 | – | – | – | – |
| svm (msg) | 0.245 | 0.144 | – | – | – | – |
| svm (msg+ctx) | 0.280 | 0.135 | – | – | – | – |
| **final agent** (weak-201k intent + dense RAG + floor) | **0.245** | 0.050 | 0.305 | **0.555** | 0.461 | **66** |

Drafting (final agent, 200 rows): 200/200 non-empty, avg 260 chars, **95.7% of
auto drafts reuse the retrieved resolution's words** (≥2 token overlap; the 
deterministic drafting paradigm means rephrasing, never inventing).

### Escalation decision benchmark — policy probe (same 200 rows)

| policy | accuracy | escal-F1 | false_auto (dangerous) | false_escalate (cost) |
|---|---:|---:|---:|---:|
| always auto | 0.480 | 0.000 | 104 | 0 |
| always escalate | 0.520 | 0.684 | 0 | 96 |
| risk by *predicted* intent | 0.550 | 0.274 | 87 | 3 |
| **risk by *true* intent (ceiling)** | **0.875** | **0.876** | **16** | 9 |
| agent floor (shipped) | 0.555 | 0.461 | 66 | 23 |

Per-intent confusion shows why: **180/200 rows predict `delivery_delay`** (the
weak-201k proxy bias); of those, 58 were human-escalations auto-handled. The gap
between shipped (55.5%) and perfect-intent (87.5%) is **entirely intent error** —
escalation logic and retrieval are not the bottleneck.

## 3. What is misleading about my headline number (required)

1. **Macro-F1 0.078 (dev-140) overstates real intent ability.** 14 classes / 140
   rows ≈ 10 examples per class; one lucky class moves the average, and F1 hides
   which classes are bought/sold (`delivery_delay` dominates everything).
2. **"96% grounded reuse" flatters drafting.** Reuse is measured *given the
   retrieved evidence*. When retrieval is wrong-but-relevant — which it often is
   for misclassified intents — a draft restating it still scores "grounded".
   The drafter is honest given the evidence; it is not honest about whether the
   evidence was the right evidence.
3. **Escalation floor is calibrated to one brand's inbox.** 27.5% dev → 30.5%
   golden match is reassuring but both pools are the same 2015–2017 AmazonHelp
   distribution; on new-brand data every threshold needs refitting. The *new*
   decision benchmark (§2) is the honest version of this story and it is worse
   than the floor's self-report said.
4. **Conversely, macro-F1 undersells deployable behavior.** The multilingual
   segment (~40% of golden) drags averages down; an English-only deployment would
   outperform both headline F1s. "Misleading" cuts both ways.

## 4. LLM judge — rubric, agreement, status

`reports/judge_rubric.md` freezes the reply-quality rubric: `helpful` /
`grounded_in_retrieval` / `hallucination_free` with 1–5 anchor tiers (tier ≥ 4 ⇒
true), literal prompts, and a **gate** before the judge's numbers may be trusted:
`self-agreement ≥ 0.8` AND `judge-human helpful rate ≥ 0.7` AND
`judge-vs-deterministic kappa ≥ 0.4`. `scripts/evaluate_judge_agreement.py`
implements it (judge human reply twice + agent draft once, 11s throttle, OpenAI
fallback, per-row audit dump). Metric maths are locked by offline tests.

**Status (honest):** commit `74fc09d` records the live run as blocked
— Gemini free-tier hit a hard 429 on every call and the OpenAI key is
`credit_balance_exhausted`; a 4-row pilot ran earlier (helpful 0.5 / grounded 0.25 /
hallucination-free 0.25) but **fails the gate**, so the judge is recorded as
*unaudited* and all headline numbers stand on deterministic metrics only.

## 5. Top-5 failure modes (real examples)

| # | mode | rows | hypothesis |
|--:|---|---|---|
| 1 | `intent_off_target` | 143 | 86%-`delivery_delay` weak prior + TF-IDF bag-of-graphemes + multilingual surface mismatch |
| 2 | `escalation_conservative` | 61 | one global floor over-tuned to dev confidence mass |
| 3 | `intent_close_secondary` | 8 | shared super-tokens ("account", "paid") |
| 4 | `drafting_ungrounded` | 6 | retrieval miss + confident intent → neutral truthful-but-useless template |
| 5 | `retrieval_lang_mismatch` | 3 | ~90% EN corpus vs non-EN queries |

Examples: `challenge-1033752` (de thank-you → predicted `delivery_delay`, delay
resolution drafted); `challenge-102915` (TR customer + URL → delay resolution);
`challenge-1451501` (pt first-purchase excitement → escalated on high confidence);
`challenge-1242242` (wrong-investigation complaint → `service_complaint_escalation`).
**Every category except `escalation_conservative` is downstream of intent error** —
one better classifier fixes the whole top of the list, which drove the Phase-A/C
finding and the one-week plan.

## 6. Reproducibility & timing

Deterministic seed 42; every number recomputed from live data (none hardcoded);
golden excluded from the retrieval index and runtime-asserted. 56 unit tests
(`python -m unittest discover -s tests`, zero credits).

```
python scripts/train_agent_intent.py && python scripts/calibrate_escalation.py
python scripts/evaluate_agent.py --dump-rows reports/agent_rows.json
python scripts/analyze_agent_errors.py reports/agent_rows.json
python scripts/build_escalation_benchmark.py && python scripts/evaluate_escalation.py reports/agent_rows.json
python scripts/consolidate_benchmark.py
```

Measured warm-cache offline reproduction (`reports/reproduction_timing.json`,
via `scripts/time_reproduction.py`, this machine): **14.54 min total** across
the six scripts + full 56-test suite — inside the 15-minute requirement.
Dominant steps: agent eval on the golden 200 **4.8 min**, weak-intent training
**4.4 min**, dev escalation calibration **4.0 min** (each one-shot, repeatable,
deterministic in substance — `fit_seconds` varies with hardware). A cold run
additionally pays the corpus-index build (`scripts/build_retrieval_index.py`,
gitignored artifacts) and parquet cache build, both one-shot.
Interactive demo: `python scripts/agent_chat.py`.

## 7. One more week (in priority order)

1. **Fix the intent axis** — the single lever proven by §2/§5: de-noise the weak
   proxy with ~100 clean parent rows, calibrate a `none` catch-all, and escalate
   non-EN queries by default (kills `retrieval_lang_mismatch` outright).
2. **Fit the per-intent risk escalation policy the benchmark paved for:** escalate
   on the risk set {account_access, charge_issue, delivered_but_not_received,
   delivered_wrong_location, refund_request, cancellation,
   product_return_and_replacement, service_complaint_escalation, other_unclear} —
   executes the proven 87.5% ceiling.
3. **Run the judge live** once quota resets/funded key (`--sample 15`), then
   decide by the gate whether to trust its draft rankings.
4. **Copy-style sweep** on the 38 success rows (template-family rewrites).
5. **Second-brand port** to quantify threshold transfer.

## 8. Decision log (condensed — 17 full entries in the full report)

AmazonHelp chosen; `prime_membership` excluded; self-built golden (200) with a
separate *reference* file; single annotator + frozen guidelines (17 low-confidence
notes); retrieval at message granularity; temporal filter on string timestamps;
golden excluded from index **and runtime-asserted**; dense-over-BM25 chosen by a
human-labelled benchmark; 201k TF-IDF weak proxy over LLM bootstrapping
(offline, honestly disclosed as not beating 140 clean labels); `comb_low`
dev-only + frozen + verified out-of-pool; one global floor (after measuring
saturated weak probabilities); deterministic "never-fabricate" drafting;
seeded-everything; LLM layer optional/disabled; **hand-labelled 200-row decision
benchmark treated as evaluation-only — never merged into training or the
threshold fit**.
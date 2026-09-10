# Hiver SDE-Intern — Full Progress Report (Phases 1–9)

**Generated:** 2026-09-09 · **Repo:** `Hiver-Assignment` · **Branch:** `main`
**Call sign for this workstream:** `QFTHX4`

A single consolidated report of everything done so far: dataset forensics,
intent taxonomy, golden evaluation set, baseline intent classifiers, and the
optional LLM provider layer. The final agent (retrieval, response generation,
escalation) is intentionally **not** built yet (Phase 5+, out of scope of the
current objective).

---

## 0. Snapshot

| | |
|---|---|
| **Phase 1 — Dataset forensics** | ✅ Done — brand chosen: **AmazonHelp** |
| **Phase 2 — Intent discovery** | ✅ Done — frozen 12-intent taxonomy (`amazon-intents-v1`) |
| **Phase 3 — Golden evaluation set** | ✅ Done — 200 hand-labelled examples (150 rep / 50 chal) |
| **Phase 4 — Baseline classifiers** | ✅ Done — majority + TF-IDF LogReg/SVM, msg-only vs msg+context |
| **LLM provider layer** | ✅ Done — resilient OpenAI+Gemini auto-fallback (optional) |
| **Phase 5 — Retrieval (RAG memory)** | ✅ Done — TF-IDF/BM25/dense/hybrid × {msg, ctx} over a 201,741-interaction corpus; human-labelled benchmark + scaling + error analysis |
| **Phases 6–8 — Final agent** | ✅ Done — weak-201k intent model + dense retrieval + dev-calibrated escalation + grounded deterministic drafting; final harness on the golden 200 (LLM judge shipped off when no key) |
| **Phase 9 — Failure analysis** | ✅ Done — per-golden-row taxonomy (162/200 rows ≥1 failure; intent misclassification dominates) |
| **Commits** | 15 (`8f0d25c` → docs/agent) |
| **Reproducibility** | `seed 42`, deterministic, no API key; `pip/requirements` offline |

---

## 1. Phase 1 — Dataset Forensics & Brand Choice

**Deliverable:** `notebooks/01_dataset_forensics.ipynb` (executed, all outputs saved), `scripts/download_data.py`, `src/{config,data_io,threads}.py`.

- Dataset: **Customer Support on Twitter** (Kaggle `thoughtvector/customer-support-on-twitter`), `twcs.csv`.
- Schema: 2,811,774 rows × 7 cols, no duplicate `tweet_id`s.
  - `inbound`: **100%** identified customers are numeric authors; outbound = brand handles (AmazonHelp etc.).
  - `in_response_to_tweet_id` (28%) → reliable parent edge for conversation reconstruction via union-find.
  - `response_tweet_id` (44%) is forward/noisy — treated as such.
- Brand landscape ranked transparently (brand size, response rates, customer conversation structure).
- **Headline result: recommended brand = AmazonHelp** (alternates AppleSupport, Tesco).
- Findings surfaced: Oct–Dec 2017 collection artifact, text-quality/leakage analysis, repeated support patterns used as TF-IDF intent seeds.

## 2. Phase 2 — AmazonHelp Intent Taxonomy

**Deliverable:** `notebooks/02_amazon_intent_discovery.ipynb`, `config/amazon_intents.yaml` (frozen), `config/amazon_intent_guidelines.yaml` (v1 human-label rules), `config/amazon_intent_version.txt` = `amazon-intents-v1`.

From a conversation-aware AmazonHelp customer corpus, iteratively refined to a **frozen 12-intent taxonomy**:

`refund_request, delivery_delay, order_status_query, charge_issue, product_return_and_replacement, delivered_wrong_location, delivered_but_not_received, account_access, account_info_update, device_app_issue, cancellation, service_complaint_escalation`

Plus sentinel labels used downstream: `none` (thanks/ack/chat filler) and `other_unclear` (genuinely unclassifiable). `prime_membership` was deliberately **excluded** (overlapping/weak boundary; "prime" almost always means shipping speed → `delivery_delay`, Prime Video/Music → `device_app_issue`, or membership mgmt co-occurring with `charge_issue`/`cancellation`/`account_access`). Discovery was cluster-reading-driven with the LLM-assisted portions **not** part of the automated run (no API key required).

## 3. Phase 3 — Golden Evaluation Set

**Deliverable:** `data/golden/` (see `data/golden/README.md` and `QA_report.md`).

- **Exactly 200 examples** = **150 representative + 50 challenge**, hand-annotated per frozen guidelines.
  - Challenge half oversamples multilingual (20) and hard/ambiguous/very short/context-dependent/angry (30).
- **Conversation-safe:** every example maps to a distinct conversation; `conversation_holdout.json` reserves those ids; `src/evaluation.assert_no_conversation_overlap` guards future train splits.
- **Turn-accurate:** each example joined to the *exact* reviewed tweet by `tweet_id` (one conversation has many tweets).
- **Leak-free eval jsonl:** `amazon_golden_eval.jsonl` has `current_customer_message` + `conversation_context` + labels, **no `next_brand_response`**. A reference file (`amazon_golden_eval_reference.jsonl`) includes the brand reply for inspection only.
- **Golden composition highlights:** delivery_delay 43, none 32, service_complaint_escalation 27; en 180 + 20 multilingual; difficulty medium 95 / hard 58 / easy 47.
- Single-annotator benchmark: no inter-annotator agreement available; 17 low-confidence + 3 other_unclear carry explicit notes.

## 4. Phase 4 — Baseline Intent Classifiers (+ LLM layer)

### 4A — Developer set & conversation-safe split infrastructure
- Built a **200-example labelled developer set** (`data/golden/amazon_dev_set.jsonl`): 100 screened-but-not-golden candidates (Phase-3 pools) + 100 seed-42 sampled corpus conversations; disjoint from golden and from each other.
- Labelled per frozen guidelines (`scripts/dev_annotations.py` — 200/200 annotations validated against `dev_ids.json`), then assembled with message + context + labels (leak-free, no `next_brand_response`).
- **70/15/15 conversation-level split** at train/val/test = **140/30/30** (seed 42). Leakage asserted: cross-split conversation disjointness, golden holdout exclusion, and duplicate message-text crossings (found: 1 templated `@USER URL` crossing golden∨train, no useful signal).

### 4B/4C — Baselines
`scripts/run_baselines.py`: Majority (DummyClassifier), TF-IDF + LogisticRegression, TF-IDF + LinearSVC, each under **message-only** vs **message+context** serialization. Deterministic, offline.

**Headline results on the golden benchmark (macro F1):**

| Model | Input | Acc | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| logistic | message_only | 0.230 | **0.164** | 0.232 |
| svm | message_only | 0.245 | 0.144 | 0.237 |
| svm | message_context | 0.280 | 0.135 | 0.240 |
| logistic | message_context | 0.245 | 0.121 | 0.218 |
| majority | message_only / context | 0.160 | 0.020 | 0.044 |

### 4D — Evaluation & error analysis (`scripts/evaluate_baselines.py`)
- Per-intent P/R/F1 + confusion matrices for every (model, input, subset) — `reports/`.
- **Error analysis:** all golden misclassifications captured — logistic/msg-only **154**, logistic/msg+ctx **151**, svm/msg-only **151** (≥20 required); top confusion patterns = none→product_return, delivery_delay→service_complaint, delivery_delay→refund_request, etc.
- **Segmentation:** English-only macro F1 **0.173** (n=180) vs non-English ≈ chance (fr/ja/it/de = 0.000).
- **Calibration:** logistic confidence **never exceeds ~0.21**; macro F1 is flat (0.11–0.19) across confidence quantiles → no trustworthy confidence signal for escalation.
- **Dev vs golden distinction write-up:** dev set is `none`-heavy (acknowledgements), golden is actionable-intent-heavy (delivery_delay); ~72% of dev messages have zero prior turns → context adds little here.

### 4E — Documentation & reproduction
- README sections 7–9 (Phase 3, Phase 4, LLM layer), `reports/README.md`, `reports/analysis_4D.md`.
- **Reproduction verified bit-stable** (two consecutive runs produce identical `baseline_results.csv`).

### LLM provider layer (optional, resilient)
`src/llm/`: primary **Gemini** → fallback **OpenAI**, 2 retries with bounded exp backoff + jitter, 30 s timeout, structured JSON output, `max_calls_per_run` guard, no-key ⇒ baselines unaffected. `.env.example`, `scripts/check_llm_providers.py`, **16 mocked unit tests pass** (`python -m unittest tests.test_llm_providers`; no API credits). LLM analysis output goes to `reports/llm_analysis/` marked **not ground truth**, never trained on.

### Honest findings (what the numbers say)
1. Numeric baselines are **weak** (macro F1 ≈ 0.12–0.16) — expected for 14 imbalanced classes, 140 train rows, short/noisy multilingual tweets. This is the reproducible bar an LLM agent must beat.
2. Message-only ≈ message+context here (72% of dev messages have no prior turns) — context adds no signal at this size.
3. English-only works (0.173); multilingual needs more data or transfer learning.
4. Logistic confidence is flat/uncalibrated — unusable for rejection/escalation as-is.
5. Dev→golden distribution shift (`none`-heavy → actionable-heavy) is real and documented, not a code bug.

---

## 5. Phase 5 — Retrieval over the historical corpus

**Goal.** Establish whether the ~203k historical AmazonHelp "interaction" records
(customer message → conversation context → brand response) are a useful **RAG
memory**: does the right prior brand-side treatment surface at the top of the
candidate list? Phase 5 evaluates *retrieval only*; Phases 6–7 will consume the
retrievals for drafting and escalation. Never adapted golden labels; golden was
never indexed.

### 5A — Historical support corpus (`496afd2`)

- Built 201,741 interactions from AmazonHelp reply-graph conversations;
  179,848 (89.1%) have a usable brand response; 82,256 distinct conversations
  (median length 6). Response patterns confirmed: `contains_help_url` 5.7%,
  `contains_instruction` 44.4%, `contains_apology` 27.9%, `claims_resolved`
  only 0.5% → retrieval is about *reusable treatment*, not promised outcomes.
- Fatal-fast issues found: `str(NaN)`-contamination (everything was "usable"),
  and `langdetect` was too slow for 200k → replaced with a fast lexicon+script
  detector (200/200 golden agreement). Canned-template engineering shortlist
  (≥20 copies) = 2% of responses (e.g. `@USER URL` 1733×, `@USER Yes` 159×).
- Don't-delete-but-measure policy applied to duplicates/canned templates.
- Leak protection: golden (200) + holdout (100) conversations asserted disjoint
  from the index (`assert_no_golden_overlap`, build-time gate); temporal filter
  `customer_timestamp <= query created_at` in every search; 0 violations in a
  20k sample. `data/retrieval/` is git-ignored (artifacts regenerable).

### 5B — Lexical retrieval baselines (`9fb2fde`)

Self-contained OKAPI BM25 (scipy sparse, k1=1.5, b=0.75, min_df=2, no external
package) + sklearn TF-IDF, persisted and byte-identical on reload; CaseStore
reworked around a `corpus.slim.parquet` with precomputed `__doc_msg`/`__doc_ctx`
columns so everything reloads in seconds instead of minutes.

### 5C — Dense + hybrid retrieval (`d29e73d`)

- Dense retriever over cached `all-MiniLM-L6-v2` embeddings (201,741×384,
  ~308 MB/variant); hybrid = α·BM25 + (1−α)·dense at α ∈ {0.3,0.5,0.7}.
- Two engineering bug events worth recording: (1) `np.memmap(mode="w+")` writes
  headerless data → embedded files unreadable → switched to
  `np.lib.format.open_memmap` (and salvaged a fully-computed cache by prepending
  the 128-byte npy header); (2) a **silent wrong-results bug in scipy 1.18.1's
  `csr_matrix.getcol()`** (column `.indices` collapse to 0 while values are
  correct) disabled all BM25 scoring until rewritten with a dense query-term
  block (`d102191`, benchmarks re-run afterwards).

### 5D — Benchmark & evaluation (`f245cc2`, `d102191`, `0659cf6`)

- 30 queries chosen with seed 42 (24 en + 6 es/it/pt/de/fr, difficulty-weighted,
  all context-dependent) → 150 candidate pairs collected from all 8 retrievers
  and hand-labelled 0/1/2 (54 / 33 / 63). Recall is pool-based (unjudged top-K =
  miss, disclosed); hybrid alphas reported, not tuned on golden.

| Retriever | Input | R@1 | R@3 | R@5 | R@10 | MRR | useful@5 | no-rel@5 | lat(ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | msg | **0.252** | **0.448** | **0.576** | **0.593** | **0.850** | **0.583** | 0.13 | 268 |
| dense | ctx | 0.184 | 0.302 | 0.358 | 0.366 | 0.683 | 0.403 | 0.30 | 2432 |
| hybrid α=.7 | msg | 0.154 | 0.353 | 0.426 | 0.552 | 0.632 | 0.444 | 0.23 | 368 |
| bm25 | msg | 0.134 | 0.190 | 0.203 | 0.276 | 0.447 | 0.189 | 0.53 | 100 |
| tfidf | msg | 0.063 | 0.194 | 0.241 | 0.280 | 0.388 | 0.214 | 0.40 | 124 |

- **Dense message-only wins**: ≥1 related interaction in top-5 for every query;
  ~58% surface a useful resolution in the top-5. Message-only beats
  message+context at *index* time (context belongs on the query side).

### 5E — Data scaling + error analysis (`0659cf6`)

| Corpus size | dense R@5 | dense MRR | bm25 R@5 |
|---|---:|---:|---:|
| 10,000 | 0.028 | 0.100 | 0.011 |
| 50,000 | 0.192 | 0.525 | 0.088 |
| 100,000 | 0.315 | 0.711 | 0.104 |
| 201,741 | 0.576 | 0.850 | 0.203 |

**Does more data help?** Yes — nested seed-42 subsets show dense R@5 growing
~6× from 10k→full with **no plateau**, index at only 308 MB. This is the honest
argument for LLM weak-label bootstrapping: human labels feed the classifier,
retrieval scales automatically off raw history. A LogReg intent *proxy*
(`build_weak_intents.py`, 13 classes over all 201,741 cases) is produced for
breakdowns only and explicitly marked *not ground truth*.

**Error analysis** (`retrieval_error_analysis.json`): 131 categorized failure
windows across retrievers and inputs — lexical/phrase-gap mismatches (99, all
BM25/TF-IDF), semantic near-misses (14, dense pulls the right theme, wrong
subtype, e.g. anger-spiral over non-delivery), multilingual crossover (12; the
de/es/pt queries also have **no usable candidate inside the 150-pair pool** —
recorded as benchmark-absence), plus rare insufficient-context cases. Live
examples: a Spanish locker/return case, a German joke-reply, a French colis
trucking answer, an Italian Prime-Video compatibility thread. This is exactly
the evidence base Phase 9 (failure analysis of the final agent) will update.

### Phase-5 reproduction (no API key)

```bash
python scripts/build_retrieval_index.py --models tfidf,bm25,dense,hybrid --variant both
python scripts/prepare_relevance_benchmark.py
python scripts/evaluate_retrieval.py
python scripts/build_weak_intents.py
python scripts/evaluate_retrieval_scaling.py
python scripts/build_error_analysis.py
python -m unittest tests.test_retrieval_lexical tests.test_retrieval_dense
```

## 6. Phases 6–9 — Retrieval-Augmented Agent (final)

### 6A — Agent components (intent, retrieval, drafting — `src/agent/`)

- `intent.py` — TF-IDF + LogisticRegression intent models: the agent's runtime
  **weak-201k** (trained on the Phase-5E proxy labels, 12 classes) and a
  **dev-140** variant (14 human classes incl. `none`) for the learning curve.
  Per-call predictions include best probability + top-two *margin*.
- `draft.py` — deterministic grounded drafting: the drafter transforms the best
  retrieved brand reply, stripping handles, URLs, phone numbers, raw order ids
  and `^RB` scrub markers; it never invents order ids/amounts/links.
- `escalation.py` — fixed, documented policy: escalate on catch-all intents
  (`none`/`other_unclear`), missing usable reply in top-3, or combined
  confidence (0.6·intent_prob + 0.4·mean-sim) below a floor.
- `agent.py` — orchestrates intent → dense retrieval (temporal filter) →
  escalation → draft, with golden leakage assertion (no golden conversation is
  ever retrievable).

### 6B — Threshold calibration (dev-only)

`scripts/calibrate_escalation.py` probes the pipeline over the **dev** set
(golden untouched) and picks `comb_low` so ~25% escalate on dev. The weak-model
probabilities saturate near 1.0 on dev, so a *single combined-confidence floor*
(not per-feature gates) is used; the value is frozen in
`data/retrieval/models/escalation_thresholds.json`. Result: `comb_low = 0.8775`,
27.5% escalate on dev.

### 6C — Final evaluation on the golden 200 (`scripts/evaluate_agent.py`)

Intent classifier learning curve (golden accuracy / macro-F1):

| model | train rows | classes | acc | macro-F1 |
| ----: | ---------: | ------: | --: | -------: |
| majority | 0 | 14 | 0.160 | 0.020 |
| dev-140 (human) | 140 | 14 | 0.315 | 0.078 |
| weak-201k (proxy) | 201,741 | 12 | 0.245 | 0.050 |

Honest finding: **more noisy weak data did not beat 140 clean human labels**
(also on label-space-shared subsets: weak 0.29/0.06 vs dev-140 0.32/0.08).
The weak labels are 86% `delivery_delay` (proxy bias) and are auto-correlated
with the same small dev family; the learning curve plateaus where label noise
meets class imbalance. Retrieval memory is not the bottleneck — intent is.

Escalation (Phase 7): **61/200 = 30.5%** on golden with the frozen floor;
sensitivity 1% (0.45) → 7% (0.75) → 31% (0.8775) → 95% (0.95). Spread evenly
across true intents; no escalation ground truth exists, so this is reported as a
distribution + ECE-style calibration diagnostic on dev (ECE 0.12).

Drafting (Phase 6): auto-handled replies reuse retrieved-resolution words in
**≈96%** of rows (token-overlap ≥2, no fabrication); avg 263 chars, 0 empty
drafts. Optional LLM judge disabled (no provider key) and recorded as such.

### 6D — Failure analysis (Phase 9, `scripts/analyze_agent_errors.py`)

Per-golden-row taxonomy over the agent dump:

| category | rows |
| ---- | --: |
| `intent_off_target` | 143 |
| `escalation_conservative` | 61 |
| `intent_close_secondary` | 8 |
| `drafting_ungrounded` | 6 |
| `retrieval_lang_mismatch` | 3 |
| **failure rows (≥1 code)** | **162 / 200** |
| success rows (0 codes) | 38 |

Top combination `intent_off_target` alone (87 rows) outweighs all others. The
biggest lever for the final agent is intent quality — matching the Phases-4/6B
result — while retrieval + grounded drafting already work (dense R@5 0.576 §5C,
draft reuse 96%).

### Phase-6–9 reproduction (no API key)

```bash
python scripts/train_agent_intent.py        # weak-201k + dev-140 -> data/retrieval/models/
python scripts/calibrate_escalation.py      # dev-only floor -> escalation_thresholds.json
python scripts/evaluate_agent.py --dump-rows reports/agent_rows.json
python scripts/analyze_agent_errors.py reports/agent_rows.json
python scripts/build_notebook_p4.py && jupyter nbconvert --execute notebooks/04_retrieval_augmented_agent.ipynb
python -m unittest tests.test_agent          # drafting / escalation / pipeline unit tests
```

## Reproduction (no API key required)

```bash
source .venv/bin/activate
python scripts/run_baselines.py       # trains + evaluates baselines → reports/ (deterministic)
python scripts/evaluate_baselines.py  # segmentation / calibration / error analysis → reports/
python -m unittest tests.test_llm_providers   # mocked LLM-layer tests (optional, no credits)
python scripts/check_llm_providers.py         # optional no-cost LLM health check
```

## Commit log

```
<hash>  docs: document Phases 6–9 agent, eval and failure analysis       (6–9 docs)
<hash>  feat: add agent failure analysis                                  (9)
<hash>  feat: add final agent evaluation harness + escalation calibration (7–8)
<hash>  feat: add retrieval-augmented agent (intent, drafting, escalation) (6)
0659cf6 feat: add retrieval scaling, weak-intent proxy, and error analysis (5E)
d102191 fix: correct bm25 scoring after scipy 1.18 getcol bug
f245cc2 feat: add retrieval evaluation benchmark                         (5D)
d29e73d feat: add dense historical support retrieval                     (5C)
9fb2fde feat: add lexical retrieval baselines                            (5B)
496afd2 feat: build historical AmazonHelp support corpus                 (5A)
6f87a36 docs: Phase-4 README + reports documentation with reproduction   (4E)
1a859a3 feat: add Phase-4D evaluation + error analysis                    (4D)
9f3b665 feat: run majority + TF-IDF logistic/SVM baselines               (4B/4C)
de0206f feat: add resilient OpenAI+Gemini LLM provider with auto-fallback
9bf1de9 feat: add conversation-safe baseline split infra + dev set       (4A)
4dca25a feat: create AmazonHelp golden evaluation set                    (P3)
7136c4f feat: discover AmazonHelp support intent taxonomy                (P2)
8f0d25c feat: add dataset forensics and brand analysis                   (P1)
```

## Next steps (future work, out of scope)

LLM-labelled weak intents (needs a provider key; targets the dominant intent
ceiling), escalation ground-truth labels, LLM-judged drafting-quality sweeps,
hybrid-alpha tuning on a dev pool.
# Hiver SDE-Intern — AI Support Agent

Take-home assignment: build an AI customer-support agent that classifies intents,
drafts replies grounded in a brand's historical resolutions, and decides
auto-handle vs. escalate. The repo currently contains **Phase 1 — Dataset
Forensics**, **Phase 2 — AmazonHelp support-intent discovery**, **Phase 3 —
golden evaluation set**, and **Phase 4 — baseline intent classifiers + LLM
provider layer**. No *final* agent is built yet, by design (see phase plan at
the bottom).

This README is the **single progress & results document**: it summarises what
we did, every headline number from the executed notebooks / baseline runs, and
the discovered intent taxonomy.

---

## 1. Progress so far (completed)

| Step | Status | Deliverable |
| ---- | ------ | ----------- |
| Phase 1 — Dataset forensics | ✅ Done | `notebooks/01_dataset_forensics.ipynb` (executed, all outputs saved) |
| Dataset downloaded to `data/` | ✅ Done | `scripts/download_data.py` |
| Reusable helpers | ✅ Done | `src/{config,data_io,threads}.py` |
| Brand choice (recommendation) | ✅ Done | **AmazonHelp** (rationale in §5) |
| Phase 2 — AmazonHelp intent discovery | ✅ Done | `notebooks/02_amazon_intent_discovery.ipynb` + `config/amazon_intents.yaml` |
| Phase 3 — Golden evaluation set | ✅ Done | `data/golden/` (200 examples, §7) |
| Phase 4A — Baseline split infra | ✅ Done | `data/golden/amazon_dev_set.jsonl` (§8) |
| LLM provider layer | ✅ Done | `src/llm/` (optional, resilient fallback, §11) |
| Phase 4B/4C — Majority + TF-IDF baselines | ✅ Done | `scripts/run_baselines.py`, `reports/baseline_results.*` (§8) |
| Phase 4D — Evaluation & error analysis | ✅ Done | `scripts/evaluate_baselines.py`, `reports/analysis_4D.md` (§8) |
| Phase 5A — Historical support corpus | ✅ Done | ~203k interactions, `src/retrieval/corpus.py` (§9) |
| Phase 5B/5C — Lexical + dense retrieval | ✅ Done | `src/retrieval/{tfidf,bm25,dense,hybrid}.py`, cached embeddings (§9) |
| Phase 5D — Retrieval evaluation | ✅ Done | human-labelled pool (30q/150 pairs), `reports/retrieval_results.*` (§9) |
| Phase 5E — Data scaling + error analysis | ✅ Done | `reports/retrieval_scaling.*`, `retrieval_error_analysis.*` (§9) |
| Phase 6 — Response generation (grounded) | ✅ Done | `src/agent/` draft+intent, deterministic, no fabrication (§10) |
| Phase 7 — Escalation policy | ✅ Done | `src/agent/escalation.py`, dev-calibrated floor (§10) |
| Phase 8 — Final harness + LLM judge | ✅ Done | `scripts/evaluate_agent.py`, `reports/agent_results.*` (§10) |
| Escalation decision benchmark | ✅ Done | hand-labelled 200 (`data/golden/escalation_gold.jsonl`) + `scripts/evaluate_escalation.py` (§10) |
| Phase 9 — Failure analysis & report | ✅ Done | `reports/agent_error_analysis.*`, `notebooks/04_…` (§10) |

---

## 2. Dataset & all notebook outputs

**Customer Support on Twitter** (Kaggle `thoughtvector/customer-support-on-twitter`), file `twcs.csv`.

### Loading & schema
```
Loaded 2,811,774 rows x 7 cols (memory ~599 MB)
Number of columns: 7
Number of rows   : 2811774
Duplicate rows   : 0
Duplicate tweet_id: 0
Unique tweet_id  : 2811774
Unique author_id : 702777
```

| Column | Meaning | Missing |
| ------ | ------- | ------: |
| `tweet_id` | unique anonymized id | 0 |
| `author_id` | customers = numeric ids, brands = named handles | 0 |
| `inbound` | True = customer, False = brand/support | 0 |
| `created_at` | Twitter timestamp | 0 |
| `text` | content, PII masked | 0 |
| `response_tweet_id` | comma-sep ids of tweets responding *to* this one (forward, noisy) | 1,040,629 (44%) |
| `in_response_to_tweet_id` | id of the tweet this is *in response to* (reliable parent) | 794,335 (28%) |

### `inbound` — proven, not assumed
```
                author_id NOT numeric   author_id numeric
inbound=False (outbound)       1,273,931                   0
inbound=True (inbound)                 0           1,537,843
=> inbound=True rows are 100% numeric customers
=> inbound=False rows are 100% named brand handles
```
`1,537,843 customer / 1,273,931 brand` tweets. The flag is a perfect,
deterministic customer/brand separator.

### Reply graph & conversations
- Rows with a set parent: `2,017,439`; without: `794,335`
- Set parents that exist as tweets: `2,013,577` (**99.8%**); orphans: `3,862`
- **Recoverable parents across all rows: ~72%**
- `conversation_id` (union-find over parent edges): **798,197 conversations**
- Messages/conversation: mean `3.52`, median `2`, p90 `6`, max `1,390`

### Temporal — collection artifact
```
Tweets by year:
2008: 2   2010: 7   2011: 11   2012: 56   2013: 86
2014: 199  2015: 427  2016: 1512  2017: 2,809,474
```
Nominal range `2008-05 → 2017-12`, but the corpus is effectively a single
**Oct–Dec 2017** snapshot (2.81 M of 2.81 M tweets in 2017). Pre-2016 dates are
negligible collection noise.

### Text quality (customer sample)
`URL 13.9% · @mention 96.3% · hashtag 9.3% · emoji/non-ascii 27% · all-caps 11.9%`
Length: min 6, max 513, mean 110 chars. Heavy templating confirmed on the
brand side (recurring canned replies e.g. `@118422 Cheers. ^Osebi.,` ×18).

### Leakage risks
Exact-duplicate customer texts (~0.5%) and identical support templates mean the
future eval **must split at conversation level, never tweet level**, with
near-duplicate dedup across units.

---

## 3. Brand landscape (ranked table)

| Rank | Brand | Support tweets | Customer tweets | Conversations | Avg turns | Median |
| ---: | ----- | -------------: | --------------: | ------------: | --------: | -----: |
| 1 | AmazonHelp | 169,840 | 203,598 | 82,556 | 4.5 | 3 |
| 2 | AppleSupport | 106,860 | 131,764 | 80,717 | 3.0 | 2 |
| 3 | Uber_Support | 56,270 | 72,154 | 41,923 | 3.1 | 2 |
| 4 | SpotifyCares | 43,265 | 48,543 | 28,280 | 3.2 | 2 |
| 5 | AmericanAir | 36,764 | 50,054 | 26,386 | 3.3 | 2 |
| 6 | Delta | 42,253 | 45,296 | 26,168 | 3.4 | 2 |
| 7 | comcastcares | 33,031 | 39,579 | 24,063 | 3.0 | 2 |
| 8 | TMobileHelp | 34,317 | 47,158 | 22,820 | 3.6 | 2 |
| 9 | SouthwestAir | 28,977 | 35,370 | 21,636 | 3.0 | 2 |
| 10 | Ask_Spectrum | 25,860 | 33,252 | 18,532 | 3.2 | 2 |
| … | (108 support accounts total) | | | | | |

---

## 4. Which brand is better — the comparison

**Scoring method** (transparent, computed live): `Overall = 0.55·Volume + 0.45·Depth`
where `Volume` = log-normalized conversation count and `Depth` = log-normalized mean
turns. Volume is weighted higher because intent classification and retrieval both
need data above all. `IntentPool` = customer-message count (context for intent
diversity, not double-counted).

| Rank | Brand | Conversations | Avg turns | IntentPool | Volume | Depth | Overall |
| ---: | ----- | ------------: | --------: | ---------: | -----: | ----: | ------: |
| **1** | **AmazonHelp** | 82,556 | 4.5 | 203,598 | 1.00 | 1.00 | **1.000** |
| 2 | Tesco | 16,722 | 4.4 | 34,228 | 0.86 | 0.99 | 0.919 |
| 3 | AppleSupport | 80,717 | 3.0 | 131,764 | 1.00 | 0.81 | 0.915 |
| 4 | VirginTrains | 14,853 | 4.4 | 37,832 | 0.85 | 0.99 | 0.913 |
| 5 | TMobileHelp | 22,820 | 3.6 | 47,158 | 0.89 | 0.90 | 0.895 |
| 6 | Uber_Support | 41,923 | 3.1 | 72,154 | 0.94 | 0.83 | 0.891 |
| 7 | Delta | 26,168 | 3.4 | 45,296 | 0.90 | 0.87 | 0.887 |
| 8 | British_Airways | 16,452 | 3.7 | 31,187 | 0.86 | 0.91 | 0.883 |
| 9 | AmericanAir | 26,386 | 3.3 | 50,054 | 0.90 | 0.86 | 0.882 |
| 10 | SpotifyCares | 28,280 | 3.2 | 48,543 | 0.91 | 0.84 | 0.879 |

> **Note on a rejected metric:** "share of conversations containing a brand reply"
> is ~100% for every brand *by construction* (a brand appears in a conversation only
> by replying), so it is **not** a discriminating feature and was intentionally
> excluded rather than reported misleadingly.

### Verdict — AmazonHelp is the best choice
- **Highest volume *and* among the deepest threads** (82.5 k conversations, mean
  4.5 turns, 203,598 customer messages) — it alone is not the largest by raw
  tweets *or* conversations arbitrarily; it uniquely has both volume and depth.
- Its customer messages show **clear recurring support intents** via TF-IDF:
  `order, delivery, prime, delivered, package, refund, account, ordered, received`
  → ideal for intent classification and RAG-based historical-resolution retrieval.
- Real-world relevance: order status, delivery/delay, refund, account issues are
  exactly the recurring intents a support agent should auto-handle.

**Alternates:** **AppleSupport** (near-equal volume 80.7 k convs, huge intent
diversity across device/software/battery/account, heavy templated learnable
replies; slightly shallower threads) and **Tesco** (deepest multi-turn exchanges,
mean 4.4/median 4, but ~5× less volume than Amazon → weaker intent-data pool).

---

## 5. Phase-1 conclusion & next step

> **Recommended brand: AmazonHelp.** Next (Phase 2): deep-dive AmazonHelp's
> customer messages (deterministic sample, TF-IDF/embeddings) to finalize the
> intent taxonomy, then confirm the conversation-level split for the golden eval
> set. Do **not** build the agent yet.

---

## 5b. Phase-2 result: the AmazonHelp intent taxonomy

`notebooks/02_amazon_intent_discovery.ipynb` builds a **conversation-aware
customer-message corpus** (each customer tweet plus the context before it and the
brand reply after it) and discovers the recurring intents AmazonHelp customers
raise. Everything is recomputed live; sampling is seeded (42). LLM-assisted
reading of clusters is intentionally not part of the automated run (no API key);
clusters are interpreted by hand from the actual output.

### Key Phase-2 numbers (live)
- AmazonHelp conversations: **82,556**; customer messages: **203,598**; brand replies: **203,598**
- **~27.4% of traffic is non-English** (Spanish, Japanese, French, German, …) — a
  first-order routing concern, not an `other` intent bucket
- English subset of the 10,000-sample: **7,289 (72.9%)**
- **12-intent taxonomy** published to `config/amazon_intents.yaml`:

| Intent | What the customer wants |
| ------ | ----------------------- |
| `delivery_delay` | expected delivery is late / not arrived |
| `delivered_but_not_received` | tracking shows delivered, customer got nothing |
| `delivered_wrong_location` | delivered to wrong place / stolen / dumped / missed |
| `order_status_query` | where is my order / when will it ship |
| `refund_request` | initiate / chase a refund |
| `cancellation` | cancel / I did not order this |
| `charge_issue` | wrong / double / unauthorized charge, price discrepancy |
| `product_return_and_replacement` | return / exchange / faulty-damaged item |
| `device_app_issue` | Echo / Alexa / Kindle / Fire / app not working |
| `account_access` | cannot log in / password / hacked / suspended |
| `account_info_update` | change email / address / phone / name |
| `service_complaint_escalation` | dissatisfaction with support, escalation / threat to leave |

### Findings that constrain Phase 3+
- **Context matters:** 16% of English messages are <60 chars and **~56% use
  deictic/referential words** (`it`/`this`/`still`/`again`) — classify
  *message+context*, not isolated tweets; acknowledgements are a messaging layer,
  not a domain intent.
- **Intents overlap** (return↔refund, delay↔status) — a Phase-3 labeler should
  allow co-occurring labels, and the rubric's ~¼ confident-match rate on a random
  sample shows a supervised labeler (not keywords) is needed.
- **Resolution shape:** ~41% of brand replies include a help URL / handoff form —
  a later phase should reproduce *safe handoff*, not expose resolution publicly.

---

## 6. Reproduce Phase 1 (under 15 minutes)

### 1. Create a Python environment (Python 3.10+)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> On Homebrew/macOS you may need `--break-system-packages` style flags only if
> you skip the venv; the venv above avoids that.

### 2. Download the data

```bash
python scripts/download_data.py
```

This uses `kagglehub` to download the Kaggle dataset into a cache and copies
the ~517 MB `twcs.csv` into `data/`. `data/` is git-ignored (too large to
commit). The first notebook run builds a `data/twcs.parquet` cache and a
`data/cache/conversation_ids.parquet` label cache so re-runs are fast.
If you already have the CSV, place it at `data/twcs.csv` and skip this step.

### 3. Run the forensics notebook

```bash
jupyter notebook notebooks/01_dataset_forensics.ipynb
# or, to execute headlessly and capture all outputs:
jupyter nbconvert --to notebook --execute --inplace notebooks/01_dataset_forensics.ipynb
```

The notebook (`notebooks/01_dataset_forensics.ipynb`) answers, with every
number computed live from the data:

1. Dataset discovery & why the full `twcs.csv` is used
2. Schema forensics + data dictionary
3. **`inbound` semantics proven** from the data (100% clean customer/brand split)
4. The reply graph (`in_response_to_tweet_id` parent links, ~72% recoverable)
5. Conversation reconstruction (`reconstruct_thread`) + example transcripts
6. Brand / author analysis (ranked table of 108 support accounts)
7. Temporal analysis, incl. the Oct–Dec 2017 collection artifact
8. Text-quality & leakage analysis
9. Repeated support patterns (TF-IDF intent seeds for AmazonHelp)
10. Transparent brand-scoring table
11. Executive summary + top-3 brand recommendation

**Headline Phase-1 result: recommended brand = AmazonHelp**, with AppleSupport and
Tesco as alternates (details and rationale inside the notebook, section 12 & 16).

### Reproduce Phase 2 (intent discovery)

```bash
python scripts/build_notebook_p2.py
jupyter nbconvert --to notebook --execute --inplace notebooks/02_amazon_intent_discovery.ipynb
```

This regenerates `notebooks/02_amazon_intent_discovery.ipynb`, recomputes every
number, and re-publishes `config/amazon_intents.yaml`. Heavy derived artifacts
(customer corpus, 10k sample, sample embeddings) are cached under
`data/intermediate/` (git-ignored) and rebuilt automatically when absent; the
local sentence-transformer (`all-MiniLM-L6-v2`) downloads once on first run. No
API key is required — the cluster reading is hand-authored markdown from the real
cluster output, and every statistic is computed live.

## 7. Phase 3 — golden evaluation set

See [`data/golden/README.md`](data/golden/README.md) for the full spec. Summary:

- **200 examples** = 150 representative + 50 challenge, hand-labelled from the
  frozen `amazon-intents-v1` taxonomy, mapped to the *exact* reviewed tweet
  (join by `tweet_id`), with `primary_intent`, `secondary_intents`,
  `message_type`, `context_dependency`, `difficulty`, `label_confidence`,
  `language`, and an annotation note.
- Conversation-safe: every golden example maps to a distinct conversation;
  `conversation_holdout.json` reserves those ids; `amazon_golden_eval.jsonl`
  contains **no `next_brand_response`** (leak-free). Use for evaluation ONLY.

## 8. Phase 4 — baseline intent classifiers

**Design.** A 200-example labelled **developer set**
(`data/golden/amazon_dev_set.jsonl`) was built (100 screened-but-not-golden +
100 deterministically sampled corpus conversations, seed 42), shown to be
conversation-disjoint from golden, then split **at the conversation level** into
train / validation / internal-test (140 / 30 / 30, seed 42). Leakage is asserted
(mutual conversation disjointness, golden holdout, and duplicate message-text
crossings). The golden benchmark is used **only** for final evaluation.

**Models & inputs** (`scripts/run_baselines.py`): Majority, TF-IDF +
LogisticRegression, and TF-IDF + Linear SVM, each under *message-only* vs
*message+context* serialization. Deterministic (seed 42), fully offline.

### Headline results — golden benchmark, macro F1

| Model | Input | Acc | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| logistic | message_only | 0.230 | **0.164** | 0.232 |
| svm | message_only | 0.245 | 0.144 | 0.237 |
| svm | message_context | 0.280 | 0.135 | 0.240 |
| logistic | message_context | 0.245 | 0.121 | 0.218 |
| majority | message_only / context | 0.160 | 0.020 | 0.044 |

Detailed per-intent P/R/F1, confusion matrices (`reports/figures/`), full error
analysis (154/151 misclassifications captured), calibration/confidence buckets,
and segmentation by difficulty / language / context dependency / message type
are in `reports/` (see `reports/README.md`) and the write-up in
`reports/analysis_4D.md`.

### Key findings (Phases 4B–4D)

- **Baselines are weak on this benchmark (macro F1 ≈ 0.12–0.16)**, driven by a
  14-way class count with heavy skew, only 140 training rows, and short noisy
  multilingual tweets. This is the honest, reproducible state of the art for a
  numeric baseline here — and the number an LLM agent must beat to justify
  retrieval + generation complexity.
- **Message-only ≈ message+context**: ~72% of dev messages have zero retrievable
  prior turns (most crowd tweets are conversation-first), so adding context
  mostly adds nothing (and slightly hurts TF-IDF sparsity).
- **English-only works better than the pooled result** (macro F1 0.173 on the
  180-English subset) — multilingual short messages are essentially at chance
  with this train size.
- **Calibration is poor**: logistic confidence never exceeds ~0.21 (softmax
  stays flat), and performance is nearly constant across confidence quantiles —
  no trustworthy confidence signal for rejection/escalation yet.
- **Dev vs golden shift**: the developer set is dominated by `none`
  (acknowledgements), golden emphasizes actionable intents (`delivery_delay`).
  See `reports/analysis_4D.md` §6.

### Reproduce Phase 4 (deterministic baselines, no API key)

```bash
python scripts/run_baselines.py       # trains + evaluates, writes reports/
python scripts/evaluate_baselines.py  # segmentation/calibration/error analysis
```

Both regenerate `reports/*.csv|json|md` from the committed dev + golden sets.
They never call an LLM and never modify golden labels.

## 9. Phase 5 — historical-corpus retrieval (RAG memory)

**Goal.** Show whether the ~203k historical AmazonHelp interactions ("customer
message → conversation context → brand response") are useful as a *retrieval
memory* for an agent: does the right prior resolution surface at the top of the
candidate list? Phase 5 evaluates **retrieval only** (Phases 6–7 later consume
the retrieved resolutions for drafting and escalation).

**Design constraints (all enforced in code).**
- Retrieval unit is a **historical interaction** (`case_id`), never a "resolved
  case" — `claims_resolved` fires on only 0.5% of the corpus, so retrieval is
  about reusable treatment, not promised outcomes.
- **Leakage-proof:** the golden 200 + 100-conversation holdout are excluded from
  the index at build time, with a hard `assert_no_golden_overlap()` gate.
- **Temporal filter:** every live query stops the result window at
  `query.created_at` (retrieved interactions must predate the tweet).
- The pool is labelled by hand: 150 candidate pairs over 30 queries
  (24 English + 6 ES/IT/PT/DE/FR), levels *0 = irrelevant, 1 = related,
  2 = useful* (a response you'd actually reuse). Recall is **pool-based**:
  unjudged top-K hits count as misses (standard pooling assumption, disclosed in
  reports). Hybrid alphas are reported, **never tuned on golden labels**.

**Retrievers** (`src/retrieval/`): TF-IDF (sklearn), self-contained OKAPI BM25
(scipy sparse, no external package), dense (`sentence-transformers`
all-MiniLM-L6-v2, 384d, cached embeddings), and hybrid α·bm25 + (1−α)·dense at
α ∈ {0.3, 0.5, 0.7}; inputs = *message-only* vs *message + 6-turn context*.

### Headline results — human benchmark (top-5 of 10)

| Retriever | Input | R@1 | R@3 | R@5 | R@10 | MRR | useful@5 | no-rel@5 | lat(ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | msg | **0.252** | **0.448** | **0.576** | **0.593** | **0.850** | **0.583** | 0.13 | 268 |
| dense | ctx | 0.184 | 0.302 | 0.358 | 0.366 | 0.683 | 0.403 | 0.30 | 2432 |
| hybrid (α=.7) | msg | 0.154 | 0.353 | 0.426 | 0.552 | 0.632 | 0.444 | 0.23 | 368 |
| bm25 | msg | 0.134 | 0.190 | 0.203 | 0.276 | 0.447 | 0.189 | 0.53 | 100 |
| tfidf | msg | 0.063 | 0.194 | 0.241 | 0.280 | 0.388 | 0.214 | 0.40 | 124 |

- **Dense message-only is the clear winner**: every run puts at least one
  *related* interaction in its top-5 for all 30 queries (no-rel@5 = 0.13) and
  ~58% of queries surface a genuinely *useful* resolution in the top-5.
- **Message-only beats message+context at index time** — context belongs to the
  *query side* (what preceded the tweet), not to stuffing each conversation into
  the index. BM25 lexical baselines sit far below dense; hybrid only helps at
  high α (basically dense) — BM25 is not additive here.
- Note: canned-template engineering verdicts — with corrected scoring, no
  retriever lands a canned reply in a top-5 here (canned_rate@5 = 0).

### Does more data help? — nested subsets (seed 42, message-only)

| Corpus size | dense R@5 | dense MRR | bm25 R@5 |
|---|---:|---:|---:|
| 10,000 | 0.028 | 0.100 | 0.011 |
| 50,000 | 0.192 | 0.525 | 0.088 |
| 100,000 | 0.315 | 0.711 | 0.104 |
| 201,741 (full) | 0.576 | 0.850 | 0.203 |

**Yes — and it is still climbing.** Dense R@5 grows ~3.4× from 50k→full with no
plateau (0.19→0.32→0.58), and index size is only 308 MB. This is the honest
motivation for LLM-weak-label bootstrapping: clean human labels capped the intent
classifier at a low ceiling (macro-F1 ~0.08, §8/§10), while retrieval scales
automatically off the raw historical corpus.

### Error analysis — 131 categorized failures

`reports/retrieval_error_analysis.json` classifies every retriever window that
omits a judged-useful candidate: full + partial useful-misses by
*lexical mismatch* (99 — all BM25/TF-IDF, phrase/synonym gaps),
*semantic near-miss* (14 — dense pulls the right theme but the wrong subtype,
e.g. anger/spiral over non-delivery), *multilingual crossover* (12 — non-English
queries leak into wrong-language hits; de/es/pt also lack useful labelled
candidates in the pool), plus rare *insufficient-context* cases. Integrity bug
on the way: scipy 1.18.1's `csr.getcol()` collapses nonzero row indices to 0,
which silently wrecked BM25 scoring until a dense-block rewrite
(`src/retrieval/bm25.py`, commit `d102191`) — benchmarks re-run since.

### Reproduce Phase 5 (deterministic, no API key)

```bash
python scripts/build_retrieval_index.py --models tfidf,bm25,dense,hybrid --variant both
python scripts/prepare_relevance_benchmark.py     # 30 queries / 150 judged pairs (seed 42)
# (read data/retrieval/relevance/review.md, fill relevance_annotations.jsonl levels)
python scripts/evaluate_retrieval.py              # R@K/MRR/useful table -> reports/
python scripts/build_weak_intents.py              # 201,741-case LogReg intent proxy (analysis only)
python scripts/evaluate_retrieval_scaling.py      # 10k/50k/100k/full curve -> reports/
python scripts/build_error_analysis.py            # >=30 categorized failures -> reports/
```

## 10. Phases 6–9 — the retrieval-augmented agent (final)

`src/agent/` closes the agent loop end-to-end; every number below is computed
live in `notebooks/04_retrieval_augmented_agent.ipynb` from committed reports.

**Pipeline (per customer message, deterministic, no API key):** weak-201k intent
classifier → dense message-only retrieval (temporal filter) → confidence-based
escalation → grounded templated drafting. The drafter transforms the best
retrieved brand reply verbatim-limited (handles/URLs/order numbers/phones/
`^RB` scrub-tags stripped, nothing fabricated).

**Intent learning curve on the golden benchmark (200 rows):**

| model | train rows | accuracy | macro-F1 | note |
| ----- | ---------- | -------- | -------- | ---- |
| majority | 0 | 0.160 | 0.020 | `delivery_delay` mode |
| dev-140 (human labels) | 140 | 0.315 | 0.078 | 14 classes incl. `none` |
| weak-201k (proxy labels) | 201,741 | 0.245 | 0.050 | 12 classes, no `none` |

Honest headline: **more noisy weak data did *not* beat 140 clean human labels**
(weak macro-F1 0.05 vs 0.08; on shared-intent subsets 0.29 acc / 0.06 F1 vs
0.32 / 0.08). The weak-201k labels are 86% `delivery_delay` (proxy bias) and
auto-correlated with the same small dev family — the scaling curve plateaus where
label noise meets class imbalance. This is the real argument for clean/LLM
labels, not a coverage gap in retrieval.

**Escalation (Phase 7, dev-calibrated floor):** fires on catch-all intents,
missing usable evidence in top-3, or combined confidence
(0.6·intent_prob + 0.4·mean-sim) below a floor calibrated on the **dev** set
only (golden untouched, frozen after). On golden: **61/200 escalated (30.5%)**,
with sensitivity 1%→7%→31%→95% as the floor moves 0.45→0.75→0.8775→0.95.

**Drafting (Phase 6):** auto-handled replies reuse retrieved-resolution words in
**~96%** of rows (token-overlap, ≥2 tokens); average draft 263 chars, zero empty;
no URLs, phone numbers, raw order IDs or fabricated data. Optional LLM judge was
**disabled** (no provider key) and recorded as such — deterministic rubric metrics
above are the shipped proxy.

**Failure analysis (Phase 9, per-golden-row taxonomy, ≥30):** 162/200 rows carry
at least one failure code — intent misclassification dominates
(`intent_off_target` 143 + `intent_close_secondary` 8), then
`escalation_conservative` 61 (escalated rows that individually looked
answerable), `drafting_ungrounded` 6, `retrieval_lang_mismatch` 3. 38/200 rows
are clean runs. The single largest lever for the agent is a better *intent
model*, not better retrieval: recall is already strong (dense R@5 0.576, §9).

### Reproduce Phases 6–9 (deterministic, no API key)

```bash
python scripts/train_agent_intent.py        # weak-201k + dev-140 models
python scripts/calibrate_escalation.py      # dev-only confidence floor (comb_low)
python scripts/evaluate_agent.py --dump-rows reports/agent_rows.json
python scripts/analyze_agent_errors.py reports/agent_rows.json
python scripts/build_escalation_benchmark.py
python scripts/evaluate_escalation.py reports/agent_rows.json
# notebook: scripts/build_notebook_p4.py && jupyter nbconvert --execute …
```

## 11. LLM provider layer (optional, resilient)

`src/llm/` provides an **OpenAI + Gemini abstraction with automatic fallback**
(primary = Gemini, fallback = OpenAI per spec), bounded exponential backoff with
jitter (2 retries), 30s timeouts, structured JSON output, and a conservative
`max_calls_per_run` guard. It is **not required** for the baselines: with no
`GEMINI_API_KEY` / `OPENAI_API_KEY` (see `.env.example`) the router reports
`enabled=False` and every baseline still runs. LLM analysis output, when used,
is stored separately under `reports/llm_analysis/` and explicitly marked
**not ground truth** and never used to train the deterministic baselines.
Mocked unit tests (no API credits) live in `tests/test_llm_providers.py`
(`python -m unittest tests.test_llm_providers`); `scripts/check_llm_providers.py`
is a no-cost health check.

**Testing with your keys:** keys in the project-root `.env` are loaded
automatically (no `export` needed). The default models are `gemini-3.6-flash`
(Gemini) and `gpt-4o-mini` (OpenAI), overridable with `LLM_MODEL` /
`LLM_FALLBACK_MODEL`. Quick checks:

```
python scripts/check_llm_providers.py        # config only, zero credits
python -m unittest tests.test_llm_providers  # mocked, zero credits
python scripts/evaluate_agent.py --llm-sample 5   # live judge on 5 drafts
python scripts/evaluate_judge_agreement.py --sample 15  # judge-vs-human agreement study (§/judge_rubric.md)
```

Note: Gemini free-tier is capped at ~5 req/min, so the judge paces itself
(12s gap between drafts, skips rows that still hit the limit and reports them
as `skipped`). The OpenAI key is used only as a fallback when Gemini fails.
**Total reproduction time with keys + warm caches ≈ 12–20 min; offline
(deterministic, no keys) ≈ 10–14 min** — the corpus index build and agent eval
dominate; both are one-shot scripts with 15-min README walkthrough in §6.

**Try the agent interactively:**

```
python scripts/agent_chat.py
# My order still hasn't arrived and says "delayed".
# How long do refunds take after a return?
# /inspect   → intent, action, confidence, reasons
# /topk 8    → deeper retrieval;  /reset → new customer;  /quit
```

## Repo layout

```
data/            git-ignored raw csv + parquet caches + large Phase-2 artifacts
data/golden/     Phase-3 golden benchmark (200) + Phase-4 dev set (200) + holdout
config/amazon_intents.yaml       the discovered Phase-2 intent taxonomy
config/amazon_intent_guidelines.yaml  frozen human-label rules (v1)
notebooks/01_dataset_forensics.ipynb   the Phase-1 deliverable
notebooks/02_amazon_intent_discovery.ipynb   the Phase-2 deliverable (intent taxonomy)
notebooks/03_historical_support_corpus.ipynb the Phase-5 retrieval baseline (regenerate: scripts/build_notebook_p3.py)
notebooks/04_retrieval_augmented_agent.ipynb Phases 6–9 agent eval (regenerate: scripts/build_notebook_p4.py)
scripts/
  download_data.py    downloads the Kaggle dataset into data/
  build_notebook.py   regenerates the Phase-1 notebook from cells (deterministic)
  build_notebook_p2.py  regenerates the Phase-2 notebook from cells (deterministic)
  golden_annotations.py, build_golden_eval.py, build_golden_pool.py   Phase-3 builders
  dev_annotations.py, build_dev_pool.py, build_dev_eval.py            Phase-4 dev-set builders
  run_baselines.py      trains/evaluates baselines -> reports/ (deterministic)
  evaluate_baselines.py Phase-4D segmentation/calibration/error analysis -> reports/
  build_retrieval_index.py  Phase-5 index build (tfidf/bm25/dense/hybrid x msg/ctx)
  prepare_relevance_benchmark.py  Phase-5 labelled retrieval pool (30q/150pairs, seed 42)
  evaluate_retrieval.py Phase-5 retrieval benchmark -> reports/retrieval_results.*
  evaluate_retrieval_scaling.py  10k/50k/100k/full nested subsets -> reports/
  build_weak_intents.py  LogReg intent proxy over 201,741 cases (analysis only)
  build_error_analysis.py  -> reports/retrieval_error_analysis.* (131 categorized failures)
  train_agent_intent.py  Phase-6 intent models (weak-201k + dev-140) -> data/retrieval/models/
  calibrate_escalation.py  Phase-7 dev-only escalation floor -> papers-frozen JSON
  evaluate_agent.py  Phase-8 final harness -> reports/agent_results.* + figures
  analyze_agent_errors.py  Phase-9 per-golden-row failure taxonomy -> reports/agent_error_analysis.*
  build_escalation_benchmark.py  Phase A: labels TSV -> data/golden/escalation_gold.jsonl (200 rows)
  evaluate_escalation.py     Phase A: agent decision vs human-expected -> reports/escalation_gold_results.*
  agent_chat.py  interactive console conversation with the loaded agent (/topk, /inspect)
  evaluate_judge_agreement.py  LLM-judge vs human agreement study -> reports/judge_agreement.*
  build_notebook_p4.py  regenerates the Phases 6–9 notebook
  check_llm_providers.py no-cost LLM health check (optional)
src/
  config.py           project-relative paths & seed
  data_io.py          loaders (parquet cache)
  threads.py          reply-graph / conversation-reconstruction helpers
  amazon.py           AmazonHelp conversation-aware customer-corpus builder
  baselines.py        dev/golden loaders, conversation-safe split, leakage guards, metrics
  evaluation.py       Phase-3 evaluation helpers
  retrieval/          Phase-5 retrievers (corpus, text, tfidf, bm25, dense, hybrid, language)
  agent/              Phases 6–9 agent (intent.py, draft.py, escalation.py, agent.py)
  llm/                optional resilient OpenAI+Gemini provider layer (auto-fallback)
tests/
  test_escalation_benchmark.py  Phase A benchmark integrity (no network)
  test_judge_agreement.py    Phase B judge-metric + rubric-gate tests (no network)
  test_llm_providers.py  mocked unit tests (no API credits)
  test_retrieval_lexical.py / test_retrieval_dense.py  retrieval unit tests (no network)
  test_agent.py        agent drafting/escalation/pipeline unit tests (no network)
requirements.txt
README.md
```

## Reproducibility & correctness

- All sampling uses a fixed seed (`config.RANDOM_SEED = 42`).
- `threads.assign_conversation_ids` labels every tweet with a
  `conversation_id` (union-find over parent edges) and is deterministic and
  idempotent. It is defensive against missing parents, self-loops, cycles and
  malformed ids.
- No statistic is hardcoded; every number is computed from the live data.
  Uncertain interpretations are labelled as *inference* in the notebook.

## Later phases (not yet implemented)

- Phase 3 — Golden evaluation set ✅ *(done, §7)*
- Phase 4 — Deterministic baselines + evaluation ✅ *(done, §8)*
- Phase 5 — Historical-corpus retrieval (RAG memory) ✅ *(done, §9)*
- Phase 6 — Response generation (grounded in retrieved resolutions) ✅ *(done, §10)*
- Phase 7 — Escalation policy (auto-handle vs escalate, confidence-based) ✅ *(done, §10)*
- Phase 8 — Final evaluation harness + LLM judge on the golden benchmark ✅ *(done, §10)*
- Phase 9 — Failure analysis & report ✅ *(done, §10)*
- Phase A — Escalation decision benchmark (human labels + eval) ✅ *(done, §10)*

**Future work (out of scope here):** LLM-labelled weak intents (requires a
provider key; would target the intent ceiling, the dominant failure mode),
tuning hybrid alphas on a dev pool, re-weighting escalation by intent-level
risk rules (the top Phase-A finding), and an LLM-judged drafting-quality sweep.

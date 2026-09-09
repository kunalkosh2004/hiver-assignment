# Hiver SDE-Intern — AI Support Agent (Phase 2: AmazonHelp Intent Taxonomy)

Take-home assignment: build an AI customer-support agent that classifies intents,
drafts replies grounded in a brand's historical resolutions, and decides
auto-handle vs. escalate. This repo currently contains **Phase 1 — Dataset
Forensics** (understanding the dataset and picking a brand) and **Phase 2 —
AmazonHelp support-intent discovery**. No agent is built yet, by design (see
phase plan at the bottom).

This README is the **single progress & results document**: it summarises what
we did, every headline number from the executed notebooks, and the discovered
intent taxonomy for Phase 2.

---

## 1. Progress so far (completed)

| Step | Status | Deliverable |
| ---- | ------ | ----------- |
| Phase 1 — Dataset forensics | ✅ Done | `notebooks/01_dataset_forensics.ipynb` (executed, all outputs saved) |
| Dataset downloaded to `data/` | ✅ Done | `scripts/download_data.py` |
| Reusable helpers | ✅ Done | `src/{config,data_io,threads}.py` |
| Brand choice (recommendation) | ✅ Done | **AmazonHelp** (rationale in §5) |
| Phase 2 — AmazonHelp intent discovery | ✅ Done | `notebooks/02_amazon_intent_discovery.ipynb` + `config/amazon_intents.yaml` |
| Phase 3+ (agent, RAG, eval) | ⏳ Not started | next |

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

## Repo layout

```
data/            git-ignored (raw csv + parquet caches + large Phase-2 artifacts)
config/amazon_intents.yaml       the discovered Phase-2 intent taxonomy
notebooks/01_dataset_forensics.ipynb   the Phase-1 deliverable
notebooks/02_amazon_intent_discovery.ipynb   the Phase-2 deliverable (intent taxonomy)
scripts/
  download_data.py    downloads the Kaggle dataset into data/
  build_notebook.py   regenerates the Phase-1 notebook from cells (deterministic)
  build_notebook_p2.py  regenerates the Phase-2 notebook from cells (deterministic)
src/
  config.py           project-relative paths & seed
  data_io.py          loaders (parquet cache)
  threads.py          reply-graph / conversation-reconstruction helpers
  amazon.py           AmazonHelp conversation-aware customer-corpus builder
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

3. Golden evaluation set
4. Baselines
5. Retrieval (RAG)
6. Response generation
7. Escalation policy
8. Evaluation harness + LLM judge
9. Failure analysis & report

# Hiver SDE-Intern — AI Support Agent (Phase 1)

Take-home assignment: build an AI customer-support agent that classifies intents,
drafts replies grounded in a brand's historical resolutions, and decides
auto-handle vs. escalate. This repo currently contains **Phase 1 — Dataset
Forensics** only (understanding the dataset and picking a brand). No agent is
built yet by design.

## Dataset

**Customer Support on Twitter** (Kaggle: `thoughtvector/customer-support-on-twitter`).

- ~2.81 M tweets, 7 columns
- 108 named brand support accounts; ~0.8 M conversations
- Effective time window is **Oct–Dec 2017** (see the notebook's temporal-artifact note)

## Reproduce Phase 1 (under 15 minutes)

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

## Repo layout

```
data/            git-ignored (raw csv + parquet caches)
notebooks/01_dataset_forensics.ipynb   the Phase-1 deliverable
scripts/
  download_data.py    downloads the Kaggle dataset into data/
  build_notebook.py   regenerates the notebook from cells (deterministic)
src/
  config.py           project-relative paths & seed
  data_io.py          loaders (parquet cache)
  threads.py          reply-graph / conversation-reconstruction helpers
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

2. Brand + conversation-corpus selection & intent discovery
3. Golden evaluation set
4. Baselines
5. Retrieval (RAG)
6. Response generation
7. Escalation policy
8. Evaluation harness + LLM judge
9. Failure analysis & report

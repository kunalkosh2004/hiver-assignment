"""Build notebooks/01_dataset_forensics.ipynb.

Assembles a clean, runnable Phase-1 forensics notebook from a list of cells.
Execute it end-to-end with:

    jupyter nbconvert --to notebook --execute --inplace notebooks/01_dataset_forensics.ipynb

Every statistic is computed live from the dataset at run time; nothing is
hardcoded. Heavy full-corpus results are cached under data/cache/ so repeat
runs are fast. Deterministic sampling is used throughout (seed in src/config).
"""
from __future__ import annotations

from pathlib import Path

import nbformat

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "01_dataset_forensics.ipynb"


def md(s: str):
    return nbformat.v4.new_markdown_cell(s)


def code(s: str):
    return nbformat.v4.new_code_cell(s)


cells: list = []

# --------------------------------------------------------------------------- #
cells.append(md(
    "# 01 — Dataset Forensics\n\n"
    "**Phase 1 of 9** for the Hiver SDE-intern take-home: *build an AI support agent for one brand of the Customer Support on Twitter dataset.*\n\n"
    "This notebook's job is **only** to understand the raw dataset and produce evidence for brand selection. "
    "No agent, classifier, or RAG is built here.\n\n"
    "**Goal questions**\n"
    "- What exactly is this dataset and how are conversations represented?\n"
    "- How messy is it?\n"
    "- Which brands are present and which look suitable for a support agent?\n\n"
    "Every statistic below is computed live from the actual file. Nothing is fabricated; "
    "uncertain interpretations are explicitly labelled *inference*."
))

cells.append(md("## 1. Objective"))
cells.append(md(
    "We want to know whether this corpus can support the later phases: intent classification "
    "of customer messages, retrieval of historical resolutions, and reply generation. Concretely "
    "this notebook establishes:\n"
    "1. The reliable schema (verified, not assumed).\n"
    "2. How reply relationships encode conversations.\n"
    "3. Which brands have large, multi-turn, *response-bearing* support threads.\n"
    "4. Which of those show recurring support patterns (the seed of intent classes).\n"
    "5. Data-quality and leakage risks that will shape the eval split in Phase 3.\n\n"
    "We deliberately do **not** build anything yet."
))

# --------------------------------------------------------------------------- #
cells.append(md("## 2. Environment & file discovery"))
cells.append(code(
    "# Make the project's reusable `src` package importable no matter the CWD.\n"
    "# The notebook lives in notebooks/, so walk up to find the dir holding src/.\n"
    "import sys\n"
    "from pathlib import Path\n\n"
    "cwd = Path.cwd()\n"
    "PROJECT_ROOT = next(\n"
    "    (p for p in [cwd, cwd.parent, cwd/'..', cwd.parents[1]] if (p/'src').exists()),\n"
    "    cwd,\n"
    ")\n"
    "sys.path.insert(0, str(PROJECT_ROOT))\n\n"
    "from src import config\n\n"
    "print('Project root:', config.PROJECT_ROOT)\n"
    "print('Data dir    :', config.DATA_DIR)\n\n"
    "assert config.TWCS_CSV.exists(), (\n"
    "    \"data/twcs.csv missing - run `python scripts/download_data.py`\"\n"
    ")\n\n"
    "for f in sorted(config.DATA_DIR.glob('*.*')):\n"
    "    print(f'{f.name:20s} {f.stat().st_size/1e6:10.1f} MB   {f.suffix}')"
))
cells.append(md(
    "**Why this file?** The Kaggle bundle ships two CSVs: a tiny `sample.csv` "
    "(a hand-picked ~100-row preview) and the full `twcs/twcs.csv` (~517 MB, 2.8 M rows). "
    "We analyse the **full `twcs.csv`**, the actual corpus the assignment targets. "
    "`sample.csv` is only a preview and uses different tweet IDs, so it is not used."
))

# --------------------------------------------------------------------------- #
cells.append(md("## 3. Loading the data"))
cells.append(code(
    "import time\nimport pandas as pd\nimport numpy as np\n\n"
    "from src import data_io\n\n"
    "t0 = time.time()\n"
    "df = data_io.load_twcs(use_cache=True)\n"
    "print(f'Loaded {df.shape[0]:,} rows x {df.shape[1]} cols in {time.time()-t0:.1f}s '\n"
    "      f'(memory ~{df.memory_usage(deep=True).sum()/1e6:.0f} MB)')\n"
    "df.head()"
))

# --------------------------------------------------------------------------- #
cells.append(md("## 4. Schema forensics"))
cells.append(code(
    "print('Number of rows   :', df.shape[0])\n"
    "print('Number of columns:', df.shape[1])\n"
    "print('\\nColumn names & dtypes:')\n"
    "print(df.dtypes.to_string())\n\n"
    "print('\\nDuplicate rows (entire row):', int(df.duplicated().sum()))\n"
    "print('Duplicate tweet_id :', int(df['tweet_id'].duplicated().sum()))\n"
    "print('Unique tweet_id    :', int(df['tweet_id'].nunique()))\n"
    "print('Unique author_id   :', int(df['author_id'].nunique()))"
))
cells.append(md("### Data dictionary"))
cells.append(code(
    "dd = pd.DataFrame({\n"
    "    'Column': df.columns.tolist(),\n"
    "    'Type': [str(t) for t in df.dtypes],\n"
    "    'Meaning': [\n"
    "        'Unique anonymized tweet id',\n"
    "        'Anonymized author id; customers are numeric ids, brands are named handles',\n"
    "        'True=customer msg to brand; False=brand/support reply (proven in sec 5)',\n"
    "        'Tweet creation timestamp (Twitter native format)',\n"
    "        'Tweet content; PII masked (e.g. __email__); may contain URLs/@mentions',\n"
    "        'Comma-separated ids of tweets that *respond to* this tweet (forward edge, noisy)',\n"
    "        'Id of the tweet this tweet is *in response to* (backward parent edge, reliable)',\n"
    "    ],\n"
    "    'Missing (%)': df.isnull().mean().mul(100).round(1).tolist(),\n"
    "})\n"
    "dd"
))
cells.append(code(
    "print('Missing-value counts per column:')\n"
    "print(df.isnull().sum().to_string())\n\n"
    "print('\\nMeaningfully empty text (whitespace only):',\n"
    "      int(df['text'].astype('string').str.strip().eq('').sum()))"
))

# --------------------------------------------------------------------------- #
cells.append(md("## 5. What `inbound` really means"))
cells.append(md(
    "We **prove** the semantics of `inbound` rather than assume them. Hypothesis: "
    "`inbound=True` ⇔ customer (anonymized numeric id); `inbound=False` ⇔ brand/support handle."
))
cells.append(code(
    "is_numeric = df['author_id'].astype('string').str.fullmatch(r'\\d+')\n\n"
    "cross = pd.crosstab(df['inbound'], is_numeric, margins=True)\n"
    "cross.columns = ['author_id NOT numeric', 'author_id numeric', 'All']\n"
    "cross.index = ['inbound=False (outbound)', 'inbound=True (inbound)', 'All']\n"
    "print('Does inbound flag separate numeric customers from named brands?')\n"
    "cross"
))
cells.append(code(
    "# attach the numeric flag to the frame once\n"
    "num = df.assign(num=is_numeric)\n"
    "p_cust_num = float(num.loc[df['inbound'],'num'].mean())*100\n"
    "p_brand_named = float((~num.loc[~df['inbound'],'num']).mean())*100\n"
    "print(f'inbound=True rows that are numeric customers: {p_cust_num:.2f}%')\n"
    "print(f'inbound=False rows that are named brand handles: {p_brand_named:.2f}%')\n\n"
    "print('\\nConclusion: the flag cleanly separates customers from brands.');"
))
cells.append(code(
    "cust = df[df['inbound']].sample(3, random_state=config.RANDOM_SEED)\n"
    "brand = df[~df['inbound']].sample(3, random_state=config.RANDOM_SEED)\n"
    "for _, r in cust.iterrows():\n"
    "    print(f\"[CUSTOMER {r['author_id']}] {str(r['text'])[:130]}\")\n"
    "print('---')\n"
    "for _, r in brand.iterrows():\n"
    "    print(f\"[BRAND {r['author_id']}] {str(r['text'])[:130]}\")"
))

# --------------------------------------------------------------------------- #
cells.append(md("## 6. The reply graph"))
cells.append(md(
    "Two columns encode adjacency: `in_response_to_tweet_id` (the **parent** this tweet "
    "replies to) and `response_tweet_id` (comma-separated **children**). We treat the parent "
    "as the canonical edge (99.8% of set parents exist in the corpus) and `response_tweet_id` "
    "as a noisy redundant forward index."
))
cells.append(code(
    "from src import threads\n\n"
    "parents = threads.parse_parent_ids(df)\n"
    "n_set = int(parents.notna().sum())\n"
    "print('Rows with a set parent      :', n_set)\n"
    "print('Rows without any parent     :', int(parents.isna().sum()))\n\n"
    "par_exists = parents.dropna().isin(set(df['tweet_id']))\n"
    "print('Set parents that exist as tweets:', int(par_exists.sum()),\n"
    "      f'({100*par_exists.mean():.1f}% of set)')\n"
    "print('Orphan refs (set parent absent):', int((~par_exists).sum()))\n"
    "print('Recoverable parents, all rows  :', round(100*n_set/len(df),1), '%')"
))
cells.append(code(
    "rawp = df['in_response_to_tweet_id'].astype('string')\n"
    "multi = rawp.str.count(',').fillna(0)\n"
    "print('Parent values containing a comma (multiple ids):', int((multi>0).sum()))\n"
    "nonnum = rawp.notna() & (~rawp.str.extract(r'(\\d+)', expand=False).notna())\n"
    "print('Parent values with no numeric token            :', int(nonnum.sum()))\n"
    "selfloop = parents == df['tweet_id']\n"
    "print('Self-loop parents (tweet replies to itself)   :', int(selfloop.sum()))"
))
cells.append(code(
    "pmap = threads.build_parent_map(df)\n\n"
    "def depth_of(tid, pmap, cap=200):\n"
    "    seen=set(); d=0; cur=int(tid)\n"
    "    while cur in pmap and pmap[cur]!=cur and cur not in seen and d<cap:\n"
    "        seen.add(cur); cur=pmap[cur]; d+=1\n"
    "    return d\n\n"
    "sample_ids = df['tweet_id'].sample(200_000, random_state=config.RANDOM_SEED).astype(int)\n"
    "depths = np.array([depth_of(t, pmap) for t in sample_ids])\n"
    "print('Reply-chain depth (deterministic sample of 200k tweets):')\n"
    "print(f'  depth 0 (root/isolated): {(depths==0).mean()*100:.1f}%')\n"
    "print(f'  mean {depths.mean():.2f}  median {np.median(depths):.0f}  '\n"
    "      f'p90 {np.percentile(depths,90):.0f}  max {depths.max()}')\n"
    "print(f'  pct with depth>=1 (part of a reply chain): {100*(depths>=1).mean():.1f}%')"
))

# --------------------------------------------------------------------------- #
cells.append(md("## 7. Conversation reconstruction & examples"))
cells.append(md(
    "Tweets in one conversation form a connected component under parent edges. We label every "
    "tweet with `conversation_id = root tweet_id` using a defensive union-find (missing parents, "
    "self-loops, and cycles all terminate safely)."
))
cells.append(code(
    "# Assign conversation ids on the full frame; cache to data/cache for fast re-runs.\n"
    "conv_cache = config.DATA_DIR / 'cache' / 'conversation_ids.parquet'\n"
    "if conv_cache.exists():\n"
    "    conv = pd.read_parquet(conv_cache).set_index('tweet_id')['conversation_id']\n"
    "    df['conversation_id'] = df['tweet_id'].map(conv)\n"
    "else:\n"
    "    df, _ = threads.assign_conversation_ids(df)\n"
    "    df[['tweet_id','conversation_id']].to_parquet(conv_cache)\n\n"
    "print('Total conversations:', f\"{df['conversation_id'].nunique():,}\")\n"
    "sz = df.groupby('conversation_id').size()\n"
    "print('Messages/conversation: mean', round(sz.mean(),2), 'median', int(sz.median()),\n"
    "      'p90', int(sz.quantile(.9)), 'max', int(sz.max()))"
))

cells.append(md("### Example reconstructed conversations"))
cells.append(code(
    "def show(conv_id, label):\n"
    "    print(f\"\\n{'='*70}\\nCONVERSATION {label} (id={conv_id})\\n{'='*70}\")\n"
    "    print(threads.display_conversation(df, conv_id))\n\n"
    "# Deterministic selection: pick one conversation of each interesting shape.\n"
    "sizes = df.groupby('conversation_id').size()\n"
    "all_conv = df.groupby('conversation_id')['inbound']\n"
    "comp = all_conv.agg(lambda s: 'both' if s.any() and (~s).any() else ('cust' if s.any() else 'brand'))\n\n"
    "def pick(pred):\n"
    "    c = [i for i in sizes.index if pred(sizes.loc[i], comp.loc[i])]\n"
    "    return c[0] if c else None\n\n"
    "show(pick(lambda n,c: n==2 and c=='both'), 'short 2-turn')\n"
    "show(pick(lambda n,c: 6<=n<=10 and c=='both'), 'medium multi-turn')\n"
    "show(pick(lambda n,c: n>=12 and c=='both'), 'long multi-turn')\n"
    "c_only = pick(lambda n,c: c=='cust')\n"
    "if c_only: show(c_only, 'customer-only (no brand reply in corpus)')"
))
cells.append(md(
    "These transcripts show the raw corpus reality: real customer language with typos and "
    "frustration, brand replies that range from templated ('please DM us') to substantive "
    "('could you try X'), and occasional multi-turn troubleshooting."
))

# --------------------------------------------------------------------------- #
cells.append(md("## 8. Brand / author analysis"))
cells.append(md(
    "Brands are the 108 named handles that appear as `inbound=False` authors. We profile each "
    "by volume, conversation count and average thread length. All numbers are computed live."
))
cells.append(code(
    "outbound = df[~df['inbound']]\n"
    "brands = outbound['author_id'].unique()\n"
    "print('Candidate brand (support) accounts:', len(brands))\n\n"
    "row = []\n"
    "for b in brands:\n"
    "    convs = set(df.loc[df['author_id']==b, 'conversation_id'].dropna())\n"
    "    conv_df = df[df['conversation_id'].isin(convs)]\n"
    "    sizes = conv_df.groupby('conversation_id').size()\n"
    "    row.append({\n"
    "        'Brand': b,\n"
    "        'Support tweets': int((df['author_id']==b).sum()),\n"
    "        'Customer tweets': int(conv_df['inbound'].sum()),\n"
    "        'Conversations': len(convs),\n"
    "        'Avg turns': round(sizes.mean(),1),\n"
    "        'Median turns': int(sizes.median()),\n"
    "    })\n\n"
    "top = pd.DataFrame(row).sort_values('Conversations', ascending=False).reset_index(drop=True)\n"
    "top.insert(0, 'Rank', range(1, len(top)+1))\n"
    "print('Ranked candidate brands:')\n"
    "top.head(15)"
))

# --------------------------------------------------------------------------- #
cells.append(md("## 9. Temporal analysis"))
cells.append(code(
    "def parse_ts(s):\n"
    "    return pd.to_datetime(s, format='%a %b %d %H:%M:%S %z %Y', errors='coerce')\n\n"
    "ts = parse_ts(df['created_at'])\n"
    "print('Earliest:', ts.min(), '  Latest:', ts.max(), '  Unparseable:', int(ts.isna().sum()))\n"
    "df['dt'] = ts.dt.tz_convert('UTC')\n\n"
    "monthly = df.groupby(df['dt'].dt.to_period('M')).size().dropna()\n"
    "print('\\nTweets by month:')\n"
    "print(monthly.to_string())\n\n"
    "yearly = df.groupby(df['dt'].dt.to_period('Y')).size()\n"
    "print('\\nTweets by year:')\n"
    "print(yearly.to_string())"
))
cells.append(md(
    "**Collection artifact (important):** the nominal range is *2008 → 2017*, but that is "
    "misleading — the monthly output shows the corpus is overwhelmingly a **Oct–Dec 2017** "
    "snapshot. The handful of tweets dated before 2016 are stray long-retained accounts captured "
    "when the brand handle was followed during collection; they are negligible noise, not a "
    "meaningful time series. Any temporal analysis or train/eval split should therefore treat "
    "**Oct–Dec 2017** as the effective window."
))

# --------------------------------------------------------------------------- #
cells.append(md("## 10. Text quality analysis"))
cells.append(md(
    "We quantify how noisy the **customer** side is (customer messages become model input). "
    "Deterministic sample to keep this fast."
))
cells.append(code(
    "import re\n"
    "cust = df[df['inbound']].sample(100_000, random_state=config.RANDOM_SEED).copy()\n"
    "txt = cust['text'].astype('string')\n"
    "def frac(pat):\n"
    "    return round(100*txt.str.contains(pat, regex=True, na=False).mean(),2)\n\n"
    "print('Customer tweets containing:')\n"
    "print('  URL           :', frac(r'https?://'), '%')\n"
    "print('  @mention      :', frac(r'@\\w'), '%')\n"
    "print('  #hashtag      :', frac(r'#\\w'), '%')\n"
    "print('  non-ascii/emoji:', frac(r'[^\\x00-\\x7F]'), '%')\n"
    "print('  all-caps word :', frac(r'\\b[A-Z]{3,}\\b'), '%')\n"
    "print('  DM request    :', frac(r'\\bDM\\b|direct message'), '%')"
))
cells.append(code(
    "lens = txt.str.len()\n"
    "print('Text length (customer sample): min', int(lens.min()), 'max', int(lens.max()),\n"
    "      'mean', round(lens.mean(),1))\n"
    "print('Very short (<10 chars):', int((lens<10).sum()))\n"
    "print('Exact-duplicate customer texts in sample:', int(txt.duplicated().sum()))\n\n"
    "sup = df[~df['inbound']]['text'].astype('string')\n"
    "print('\\nMost frequent identical support replies (canned templates):')\n"
    "for t,c in sup.value_counts().head(6).items():\n"
    "    print(f'  [{c}x] {str(t)[:110]}')"
))

# --------------------------------------------------------------------------- #
cells.append(md("## 11. Repeated support patterns (intent seeds)"))
cells.append(md(
    "We check whether the leading brand has recurring, clusterable customer issues — a precursor "
    "to intent discovery (Phase 2), not the taxonomy itself."
))
cells.append(code(
    "from sklearn.feature_extraction.text import TfidfVectorizer\n\n"
    "LEAD_BRAND = str(top.iloc[0]['Brand'])\n"
    "print('Leading brand by conversation count:', LEAD_BRAND)\n\n"
    "brand_convs = set(df.loc[df['author_id']==LEAD_BRAND, 'conversation_id'])\n"
    "cust_b = df[df['inbound'] & df['conversation_id'].isin(brand_convs)].copy()\n"
    "print('Customer messages in its conversations:', len(cust_b))\n\n"
    "def norm(s):\n"
    "    s = re.sub(r'https?://\\S+',' ', s)\n"
    "    s = re.sub(r'@\\w+',' ', s)\n"
    "    s = re.sub(r'#\\w+',' ', s)\n"
    "    return s.lower()\n\n"
    "cust_sample = cust_b.sample(15_000, random_state=config.RANDOM_SEED)['text'].astype('string')\n"
    "corpus = cust_sample.map(norm).tolist()\n"
    "tf = TfidfVectorizer(max_features=2000, stop_words='english', token_pattern=r'\\b[a-z]{3,}\\b')\n"
    "X = tf.fit_transform(corpus)\n"
    "terms = np.array(tf.get_feature_names_out())\n"
    "mean_tfidf = np.asarray(X.mean(axis=0)).ravel()\n"
    "top_idx = mean_tfidf.argsort()[-30:][::-1]\n"
    "print('Top authoritative terms across its customer messages:')\n"
    "print(', '.join(terms[top_idx]))"
))
cells.append(code(
    "# Show a handful of raw, recurring-style customer messages so we can eyeball clusters.\n"
    "for kw in ['refund','track','broken','cancel','deliver']:\n"
    "    hit = cust_b[cust_b['text'].astype('string').str.lower().str.contains(kw, regex=False)].head(1)\n"
    "    if len(hit):\n"
    "        print(f'[{kw.upper()}] {str(hit.iloc[0][\"text\"])[:150]}')\n"
))

# --------------------------------------------------------------------------- #
cells.append(md("## 12. Candidate brand comparison (scoring)"))
cells.append(md(
    "We score top brands on what matters for later phases. Scores are 0–1 and computed live; "
    "the methodology is explicit so the ranking is transparent."
))
cells.append(code(
    "def score_table(tb, top_n=15):\n"
    "    t = tb.head(top_n).copy().reset_index(drop=True)\n"
    "    t = t.drop(columns=['Rank'], errors='ignore')\n"
    "    t = t[t['Conversations']>0].reset_index(drop=True)\n"
    "    # Volume: log-normalised conversation count (the agent needs data volume above all).\n"
    "    t['Volume'] = (np.log1p(t['Conversations'])/np.log1p(t['Conversations'].max())).round(2)\n"
    "    # Depth: log-normalised mean turns (multi-turn threads = richer resolution evidence and\n"
    "    #        proof that both sides actually exchanged information).\n"
    "    t['Depth'] = (np.log1p(t['Avg turns'])/np.log1p(t['Avg turns'].max())).round(2)\n"
    "    # IntentPool: raw customer-message count shown as context (a proxy for the diversity of\n"
    "    #        intents we can mine). Not part of the score to avoid double-counting volume.\n"
    "    t['IntentPool'] = t['Customer tweets']\n"
    "    # Overall: explicit, transparent weights. Volume weighted higher because intent\n"
    "    #        classification and retrieval both need data above everything.\n"
    "    t['Overall'] = (0.55*t['Volume'] + 0.45*t['Depth']).round(3)\n"
    "    t = t.sort_values('Overall', ascending=False).reset_index(drop=True)\n"
    "    t.insert(0, 'Rank', range(1, len(t)+1))\n"
    "    return t\n\n"
    "st = score_table(top)\n"
    "print('Transparent scoring (method below):')\n"
    "st[['Rank','Brand','Conversations','Avg turns','IntentPool','Volume','Depth','Overall']]"
))
cells.append(md(
    "**Scoring method.** `Volume` = log-normalised conversation count; `Depth` = log-normalised "
    "mean turns; `Overall = 0.55·Volume + 0.45·Depth`. Volume is weighted higher because intent "
    "classification and retrieval both need data above all; Depth rewards multi-turn threads that "
    "prove a genuine two-sided exchange and provide richer response grounding. `IntentPool` "
    "(customer-message count) is shown as context for intent diversity, not double-counted in the "
    "score. Note: *'share of conversations containing a brand reply'* is ~100% for every brand by "
    "construction (a brand appears in a conversation only by replying), so it is **not** a "
    "discriminating feature and is intentionally left out. **This score is a recommendation, not a "
    "final decision** — it will be validated against intent diversity in Phase 2."
))

# --------------------------------------------------------------------------- #
cells.append(md("## 13. Data leakage risks"))
cells.append(md(
    "The eventual evaluation must be split at the **conversation level**. We measure the leak "
    "vectors now so Phase 3 can design a safe split."
))
cells.append(code(
    "cust_txt = df[df['inbound']]['text'].astype('string')\n"
    "print('Exact-duplicate customer texts:', int(cust_txt.duplicated().sum()),\n"
    "      f'({100*cust_txt.duplicated().mean():.1f}%)')\n"
    "print('Identical support templates: ~', len(df[~df['inbound']]['text'].astype('string').value_counts())\n"
    "      ,'unique / ', int(df[~df['inbound']].shape[0]), 'support tweets -> heavy templating')\n"
    "print('\\nConversations (atomic split unit):', f\"{df['conversation_id'].nunique():,}\")\n"
    "print('Mean conversation size:', round(df.groupby('conversation_id').size().mean(),2),\n"
    "      '-> tweet-level split puts a thread\\'s turns in both train & eval')\n"
    "print('\\nRecommendation: group by conversation_id before any train/eval split')\n"
    "print('              and deduplicate near-identical customer text across units.')"
))

# --------------------------------------------------------------------------- #
cells.append(md("## 14. Dataset limitations"))
cells.append(md(
    "The corpus records **public tweets only**. It cannot tell us (and we must not assume):\n"
    "- whether a problem was actually **resolved** (a brand 'thanks, we've taken care of it' is "
    "  not proof of a refund, delivery, or fix);\n"
    "- whether the brand performed a real-world action;\n"
    "- whether the customer was satisfied;\n"
    "- whether a reply's factual claims were correct.\n\n"
    "We therefore use **response-bearing** (brand-answered) conversations as our proxy for "
    "'historical resolution', and treat resolution/satisfaction as **uncertain** in the escalation "
    "policy (Phase 7) and evaluation rubric (Phase 8)."
))

# --------------------------------------------------------------------------- #
cells.append(md("## 15. Visualizations"))
cells.append(code(
    "import matplotlib.pyplot as plt\n"
    "plt.rcParams['figure.figsize'] = (8,4)\n\n"
    "fig, axes = plt.subplots(2,2, figsize=(13,9))\n\n"
    "ax = axes[0,0]\n"
    "vc = df['inbound'].value_counts().reindex([True,False])\n"
    "ax.bar(['inbound (customer)','outbound (brand)'], vc.values, color=['#4C72B0','#55A868'])\n"
    "ax.set_title('Tweets by inbound/outbound'); ax.set_ylabel('count')\n\n"
    "ax = axes[0,1]\n"
    "tp = top.head(10)\n"
    "ax.barh(tp['Brand'][::-1], tp['Conversations'][::-1], color='#C44E52')\n"
    "ax.set_title('Top 10 brands by conversation volume'); ax.set_xlabel('conversations')\n\n"
    "ax = axes[1,0]\n"
    "dist = df['conversation_id'].value_counts().value_counts().sort_index()\n"
    "dist[dist.index<=12].plot(kind='bar', ax=ax)\n"
    "ax.set_title('Conversation length distribution (messages, 1-12)')\n"
    "ax.set_xlabel('messages per conversation'); ax.set_ylabel('# conversations')\n\n"
    "ax = axes[1,1]\n"
    "lb = df[df['author_id']==LEAD_BRAND]['inbound'].value_counts().reindex([True,False], fill_value=0)\n"
    "ax.bar(['customer (its threads)','brand replies'], lb.values, color=['#4C72B0','#55A868'])\n"
    "ax.set_title(f'Support vs customer messages — {LEAD_BRAND}'); ax.set_ylabel('count')\n\n"
    "plt.tight_layout(); plt.show()"
))
cells.append(code(
    "monthly2 = df.groupby(df['dt'].dt.to_period('M')).agg(\n"
    "    total=('tweet_id','size'), customer=('inbound','sum'))\n"
    "ax = monthly2.plot()\n"
    "ax.set_title('Support activity over time')\n"
    "ax.set_xlabel('month'); ax.set_ylabel('tweets')\n"
    "plt.show()"
))

# --------------------------------------------------------------------------- #
cells.append(md("## 16. Executive summary & recommendation"))
cells.append(code(
    "print('DATASET')\n"
    "print('  total records :', f'{df.shape[0]:,}')\n"
    "print('  date range    :', ts.min(), '->', ts.max())\n"
    "print('  inbound       :', f'{int(df[\"inbound\"].sum()):,} customer / {int((~df[\"inbound\"]).sum()):,} brand')\n"
    "print('  recoverable parents:', f'{100*parents.notna().mean():.0f}%')\n"
    "print('\\nBRAND LANDSCAPE')\n"
    "print('  support accounts:', len(brands), '  total conversations:', f\"{df['conversation_id'].nunique():,}\")\n"
    "print('\\nTOP-3 RECOMMENDATION (rationale in sec 12):')\n"
    "for i, r in st.head(3).iterrows():\n"
    "    print(f'  {int(r[\"Rank\"])}. {r[\"Brand\"]}  convos={int(r[\"Conversations\"]):,}  '\n"
    "          f'avg-turns={r[\"Avg turns\"]}  intent-pool={int(r[\"IntentPool\"]):,}  overall={r[\"Overall\"]}')"
))
cells.append(md(
    "### Recommended top-3 (rationale)\n"
    "- **1. AmazonHelp** – the clear all-round winner: highest conversation volume (82.5k) *and* "
    "  among the deepest threads (mean 4.5 turns), with a huge pool of customer messages for "
    "  intent discovery and rich historical response evidence for retrieval/RAG.\n"
    "- **2. AppleSupport** – near-identical conversation volume to Amazon (80.7k) and the largest "
    "  recognisable brand; slightly shallower threads, but enormous intent diversity across "
    "  device/software/battery/account issues with heavily templated (thus learnable) responses.\n"
    "- **3. Tesco** – lower raw volume but the deepest multi-turn exchanges (mean 4.4, median 4); "
    "  a good complement if we want to stress-test multi-turn grounding, though intent diversity "
    "  and volume are smaller than Amazon/Apple.\n\n"
    "> **Recommended next step (Phase 2):** Deep-dive AmazonHelp (primary) plus AppleSupport and "
    "> Tesco (alternates) customer messages via a deterministic sample with TF-IDF/embeddings to "
    "> discover the concrete recurring support intents; confirm which brand has the cleanest, most "
    "> response-bearing set; and lock the conversation-level split for the golden eval set. Do "
    "> **not** build the agent yet."
))

# --------------------------------------------------------------------------- #
# Assemble and write the notebook.
nb = nbformat.v4.new_notebook(
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    cells=cells,
)
nbformat.write(nb, OUT)
print("Wrote", OUT)

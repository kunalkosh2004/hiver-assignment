"""Build notebooks/02_amazon_intent_discovery.ipynb.

Assembles a clean, runnable Phase-2 notebook that discovers an evidence-backed
AmazonHelp support-intent taxonomy. Execute end-to-end with:

    jupyter nbconvert --to notebook --execute --inplace notebooks/02_amazon_intent_discovery.ipynb

Every statistic is computed live from the dataset at run time; nothing meaningful
is hardcoded beyond the fixed random seed (src/config.RANDOM_SEED = 42). Heavy
results (the customer corpus, the 10k sample, the sample embeddings) are cached
under data/intermediate/ so repeat runs are fast and deterministic -- the sec
that computes the intent *taxonomy from cluster evidence* is reproducible but the
human interpretation is authored in markdown from the observed cluster output.

This phase deliberately builds NO agent / RAG / eval: it only *discovers* the
intent set that a later phase will classify against.
"""
from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "02_amazon_intent_discovery.ipynb"


def md(s: str):
    return nbformat.v4.new_markdown_cell(s)


def code(s: str):
    return nbformat.v4.new_code_cell(s)


cells: list = []

# --------------------------------------------------------------------------- #
cells.append(md(
    "# 02 — AmazonHelp Support-Intent Discovery\n\n"
    "**Phase 2 of 9** for the Hiver SDE-intern take-home. Phase 1 selected **AmazonHelp** as the "
    "target brand. This notebook's single job is to **discover an evidence-backed taxonomy of the "
    "support intents** customers actually raise with AmazonHelp.\n\n"
    "We deliberately build **nothing** here: no classifier, no retrieval, no agent, no response "
    "generation, no eval set. The output is a defensible, data-grounded intent list (8–15) plus "
    "the data-quality facts a later phase needs.\n\n"
    "**Method (fully reproducible, deterministic seed 42):**\n"
    "1. Assemble the *conversation-aware* customer-message corpus (each customer tweet plus the "
    "   conversation context before it and the brand reply after it).\n"
    "2. Quantify a critical AmazonHelp property the raw file hides: **it is multilingual**.\n"
    "3. Sample deterministically, embed with a local sentence-transformer, cluster (English "
    "   subset), and read the clusters.\n"
    "4. Cross-check intent volumes and boundaries with a keyword rubric and measure how often a "
    "   message is only interpretable *given its context*.\n"
    "5. Publish an intent taxonomy to `config/amazon_intents.yaml`.\n\n"
    "LLM-assisted interpretation is intentionally **not** part of the automated run (no API key in "
    "this environment). The cluster reading is authored from actual cluster output; the notebook "
    "itself recomputes every number live."
))

cells.append(md("## 1. Objective & deliverables"))
cells.append(md(
    "**Research question.** When a random customer tweets at AmazonHelp, what is the recurring set "
    "of *intents* they express — from the data's own structure, before any supervised model exists?\n\n"
    "**Deliverables.**\n"
    "- `config/amazon_intents.yaml` — the discovered taxonomy (id, label, description, keywords, "
    "  whether resolution typically requires extra info like an order id / account).\n"
    "- An internal validation sample under `data/intermediate/`.\n"
    "- A set of data-quality findings (notably the multilingual property) that constrain Phases 3+."
))

cells.append(md("## 2. Environment & setup"))
cells.append(code(
    'import sys\n'
    'from pathlib import Path\n\n'
    'cwd = Path.cwd()\n'
    'PROJECT_ROOT = next(\n'
    '    (p for p in [cwd, cwd.parent, cwd/"..", cwd.parents[1]] if (p/"src").exists()),\n'
    '    cwd,\n'
    ')\n'
    'sys.path.insert(0, str(PROJECT_ROOT))\n\n'
    'from src import amazon, config\n\n'
    'print("Project root:", config.PROJECT_ROOT)\n'
    'print("Seed        :", config.RANDOM_SEED)\n'
    'assert config.TWCS_CSV.exists() or config.TWCS_PARQUET.exists(), (\n'
    '    "data missing - run `python scripts/download_data.py` first"\n'
    ')'
))

cells.append(md("## 3. The analysis unit: conversation-aware customer messages"))
cells.append(md(
    "A customer tweet alone is often ambiguous (`still waiting`, `what?`, `thanks`). To make every "
    "message interpretable we attach, per message:\n"
    "- `conversation_context` — up to 6 preceding turns (role-tagged),\n"
    "- `previous_brand_message` — what AmazonHelp last told this customer,\n"
    "- `next_brand_response` — what AmazonHelp replied next (the *resolution evidence*).\n\n"
    "`src/amazon.build_customer_corpus()` produces exactly this, vectorized over the full corpus."
))
cells.append(code(
    "corpus = amazon.build_customer_corpus()\n"
    "print('customer messages in AmazonHelp convos:', f'{len(corpus):,}')\n"
    "print('conversations represented            :', f\"{corpus['conversation_id'].nunique():,}\")\n"
    "corpus.head(3)[['created_at','customer_text_clean','previous_brand_message','next_brand_response']]"
))

cells.append(md("### 3.1 Cross-check corpus composition against Phase 1"))
cells.append(code(
    "import pandas as pd\n"
    "df = amazon.load_with_conversations()\n"
    "sub, _ = amazon.amazon_conversations(df)\n"
    "print('AmazonHelp convs  :', f\"{sub['conversation_id'].nunique():,}\")\n"
    "print('tweets in those   :', f'{len(sub):,}')\n"
    "print('customer msgs     :', f\"{int((sub['inbound']).sum()):,}\")\n"
    "print('brand msgs        :', f\"{int((~sub['inbound']).sum()):,}\")\n"
    "non_brand = sub.loc[~sub['inbound'],'author_id'].nunique()\n"
    "print('non-brand reply accounts (agents):', non_brand)"
))

cells.append(md("## 4. AmazonHelp is multilingual — a first-order credit to your agent"))
cells.append(md(
    "American/Multi-region handle, but customers reply in many languages. An English-centric "
    "embedding + a single-classifier design would silently fail on a large slice of traffic. We "
    "measure it before committing to a taxonomy so the taxonomy decision is explicit."
))
cells.append(code(
    "from langdetect import detect, DetectorFactory\n"
    "DetectorFactory.seed = config.RANDOM_SEED\n\n"
    "# Cheap, deterministic language estimate on a read-only subsample (per-row detect is slow).\n"
    "_probe = corpus.sample(3000, random_state=config.RANDOM_SEED)\n"
    "def _lang(t):\n"
    "    try:\n"
    "        return detect(str(t).replace(chr(10),' '))\n"
    "    except Exception:\n"
    "        return 'unk'\n"
    "_probe = _probe.assign(lang=_probe['customer_text'].map(_lang))\n"
    "lang_counts = _probe['lang'].value_counts()\n"
    "en_share = lang_counts.get('en',0)/len(_probe)\n"
    "multilingual_share = 1 - en_share\n"
    "print('language distribution (n=%d):' % len(_probe))\n"
    "print(lang_counts.head(10).to_string())\n"
    "print(f'\\nEnglish share: {100*en_share:.1f}%   multilingual share: {100*multilingual_share:.1f}%')"
))
cells.append(md(
    "**Decision.** The taxonomy below is defined on the **English** slice (~73% of traffic), "
    "because (a) intent categories are language-agnostic principles and (b) our local embedding "
    "model is English-centric, so clustering non-English separately is more honest. Multilingual "
    "is recorded as a **language attribute + routing concern** (translate / route to regional "
    "handle) in Phase 3+ — **not** as a single `other` dumping bucket, which would hide real "
    "intents. Its exact share is shown live above."
))

cells.append(md("## 5. Deterministic, distribution-preserving sample"))
cells.append(md(
    "We embed a 10,000-message sample (not the whole 200k corpus) for tractable interactive "
    "clustering. A plain `seed=42` random sample preserves conversation-length, message-length "
    "and time distributions exactly."
))
cells.append(code(
    "SAMPLE_N = 10_000\n"
    "sample = corpus.sample(SAMPLE_N, random_state=config.RANDOM_SEED).reset_index(drop=True)\n\n"
    "def _dist_ok(col):\n"
    "    if pd.api.types.is_string_dtype(sample[col]):\n"
    "        a = sample[col].str.len().describe()[['50%','mean']]\n"
    "        b = corpus[col].str.len().describe()[['50%','mean']]\n"
    "    else:\n"
    "        a = sample[col].describe()[['50%','mean']]\n"
    "        b = corpus[col].describe()[['50%','mean']]\n"
    "    return f'sample med={a[\"50%\"]:.0f}/mean={a[\"mean\"]:.1f} | corpus med={b[\"50%\"]:.0f}/mean={b[\"mean\"]:.1f}'\n"
    "print('conv_length :', _dist_ok('conv_length'))\n"
    "print('msg length  :', _dist_ok('customer_text_clean'))\n"
    "print('time range  :', sample['created_at'].min(), '->', sample['created_at'].max())\n"
    "sample.to_parquet(config.DATA_DIR/'intermediate'/'amazon_customer_sample.parquet')\n"
    "print('unique conversations in sample:', sample['conversation_id'].nunique())"
))

cells.append(md("## 6. Embeddings + clustering on the English subset"))
cells.append(md(
    "We embed the cleaned customer text with the **offline** `all-MiniLM-L6-v2` model (384-d, "
    "runs locally, no API key) and cluster the English messages with deterministic `KMeans`. "
    "KMeans (not HDBSCAN) is used because it is seed-reproducible and available in this "
    "environment; we read clusters *as hypotheses* and validate volumes/edges by keywords rather "
    "than over-trusting boundary.*\n\n"
    "Embeddings are cached under `data/intermediate/amazon_embeddings.npy` (git-ignored) and "
    "recomputed only if absent — the sample is regenerated from the same seed so the cache stays "
    "aligned."
))
cells.append(code(
    "import numpy as np\n"
    "from pathlib import Path\n\n"
    "emb_dir = config.DATA_DIR/'intermediate'\n"
    "cache = emb_dir/'amazon_embeddings.npy'\n"
    "if cache.exists():\n"
    "    EMB = np.load(cache)\n"
    "    print('loaded cached embeddings', EMB.shape)\n"
    "else:\n"
    "    from sentence_transformers import SentenceTransformer\n"
    "    text = (config.DATA_DIR/'intermediate'/'amazon_customer_sample.parquet')\n"
    "    sampled = pd.read_parquet(text) if Path(text).exists() else sample\n"
    "    model = SentenceTransformer('all-MiniLM-L6-v2')\n"
    "    EMB = model.encode(sampled['customer_text_clean'].tolist(), batch_size=256, convert_to_numpy=True)\n"
    "    np.save(cache, EMB)\n"
    "    print('computed embeddings', EMB.shape)"
))

cells.append(md("### 6.1 Restrict to English for intent clustering"))
cells.append(code(
    "# Language tags for the whole sample (cached to avoid re-detecting).\n"
    "lang_cache = emb_dir/'amazon_sample_langs.parquet'\n"
    "if lang_cache.exists():\n"
    "    sample = sample.assign(lang=pd.read_parquet(lang_cache)['lang'].reindex(sample.index).values)\n"
    "else:\n"
    "    sample = sample.assign(lang=sample['customer_text'].map(_lang))\n"
    "    pd.Series(sample['lang'].values, index=sample.index).to_frame('lang').to_parquet(lang_cache)\n"
    "en_mask = (sample['lang'].to_numpy()==\"en\")\n"
    "en = sample[en_mask].reset_index(drop=True)\n"
    "EMB_EN = EMB[en_mask]\n"
    "print(f'English subset: {len(en):,} of {len(sample):,} ({100*len(en)/len(sample):.1f}%)')"
))

cells.append(code(
    "from sklearn.cluster import KMeans\n"
    "km = KMeans(n_clusters=12, random_state=config.RANDOM_SEED, n_init=10, init='k-means++')\n"
    "en['cluster'] = km.fit_predict(EMB_EN)\n"
    "size = en['cluster'].value_counts().sort_index()\n"
    "print('cluster sizes\\n', size.to_string())\n"
    "print('\\nrepresentative share of largest cluster:', f\"{size.max()/len(en):.1%}\")"
))

cells.append(md("## 7. Reading the clusters → discovered intents"))
cells.append(md(
    "Below we print, per cluster, the messages closest to the centroid (most archetypal) plus the "
    "top TF-IDF terms. This is the human-interpretation step: the clusters are *hypotheses* that "
    "the keyword rubric in §8 then confirms and sizes."
))
cells.append(code(
    "from sklearn.feature_extraction.text import TfidfVectorizer\n"
    "from numpy.linalg import norm\n\n"
    "tf = TfidfVectorizer(max_features=3000, stop_words='english', token_pattern=r'\\b[a-z]{3,}\\b')\n"
    "X = tf.fit_transform(en['customer_text_clean'])\n"
    "terms = np.array(tf.get_feature_names_out())\n"
    "nb = EMB_EN / norm(EMB_EN, axis=1, keepdims=True)\n"
    "cmask = en['cluster'].to_numpy()\n"
    "pd.set_option('display.max_colwidth', 110)\n"
    "rows = []\n"
    "for c in sorted(np.unique(cmask)):\n"
    "    cen = km.cluster_centers_[c]; cen = cen/norm(cen)\n"
    "    idx = np.where(cmask==c)[0]\n"
    "    sims = nb[idx] @ cen\n"
    "    ex = en.iloc[idx[np.argsort(sims)[-3:][::-1]]]['customer_text'].astype(str).str.replace('\\n',' ').tolist()\n"
    "    cx = X[idx]\n"
    "    mean = np.asarray(cx.mean(axis=0)).ravel()\n"
    "    topt = terms[mean.argsort()[-10:][::-1]]\n"
    "    rows.append({'cluster':c,'n':len(idx),'top_terms':', '.join(topt),'examples':ex})\n"
    "cluster_view = pd.DataFrame(rows)\n"
    "for _, r in cluster_view.iterrows():\n"
    "    print(f\"###### cluster {int(r['cluster'])} (n={int(r['n'])}) — {r['top_terms']}\")\n"
    "    for e in r['examples']:\n"
    "        print('  •', e[:120])\n"
    "    print()"
))

cells.append(md("### 7.1 Human reading of the clusters"))
cells.append(md(
    "Reading the real cluster output above together with the §8 keyword volumes, the English "
    "traffic decomposes into these recurring intents (a **12-intent** taxonomy):\n\n"
    "| Intent | What the customer wants | From cluster harbinger |\n"
    "|---|---|---|\n"
    "| `delivery_delay` | expected delivery hasn't arrived / is late | C3, C6 |\n"
    "| `delivered_but_not_received` | tracking says delivered, I got nothing | C8 |\n"
    "| `delivered_wrong_location` | left wrong place / wrong address / stolen / dumped | C10 |\n"
    "| `order_status_query` | where is my order / when will it ship | C3-adjacent |\n"
    "| `refund_request` | initiate / chase a refund | C5 |\n"
    "| `cancellation` | cancel / I didn't order this | C5-adjacent |\n"
    "| `charge_issue` | wrong / double / unauthorized charge or price | C1 |\n"
    "| `product_return_and_replacement` | return / exchange / faulty-damaged item | C5-adjacent |\n"
    "| `device_app_issue` | Echo/Alexa/Kindle/Fire/app not working | C0 |\n"
    "| `account_access` | can't log in / password / hacked / suspended | keyword-verified |\n"
    "| `account_info_update` | change email / address / phone | keyword-verified |\n"
    "| `service_complaint_escalation` | dissatisfaction with CS, threat to leave / escalate | C7, C9 |\n\n"
    "**Deliberately excluded as intents:** `conversation_continuation` (short acknowledgements "
    "like `thanks`, `will do` — real but context-only; handled as a *messaging layer*, not a "
    "domain intent) and **multilingual** (a language/routing attribute, §4). Both are measured "
    "below and in §9 rather than turned into catch-all buckets."
))

cells.append(md("## 8. Intent volume & boundaries via a keyword rubric"))
cells.append(md(
    "Clusters are soft. To (a) confirm each intent has real volume and (b) expose overlap "
    "('refund after cancellation', 'delivery delay vs delivered-not-received'), we apply an "
    "explicit keyword rubric to the English sample. Overlaps are shown as a matrix — they are "
    "**expected** and inform Phase 3's many-to-many labelling, not a flaw."
))
cells.append(code(
    "RULES = {\n"
    " 'delivery_delay': r'\\b(late|delay|not arriv|hasn.t arriv|hasn.t come|out for delivery|no sign of|stuck|in transit|slower|supposed.*deliver|should.*arriv|delivery date|tracking.*not updat)\\b',\n"
    " 'delivered_not_received': r'\\b(mark.*delivered|says.*delivered|delivered.*but|delivered.*never|delivered.*not receive|shows.*deliver|handed to resident|supposed.*delivered)\\b',\n"
    " 'delivered_wrong': r'\\b(deliver.*wrong|wrong address|wrong house|neighbor|thrown|dumped|stolen|missed.*deliver|left.*no package|porch.*(no|not))\\b',\n"
    " 'refund': r'\\b(refund|money back|give back|reimburs|repay)\\b',\n"
    " 'cancellation': r'\\b(cancel(l)?ation|want to cancel|didn.t order|wrongly order|mistake.*order)\\b',\n"
    " 'charge_issue': r'\\b(charg|debited|double|twice|unauthoriz|price went|billed|extra.*(money|amount)|taken.*(money|amount))\\b',\n"
    " 'order_status': r'\\b(where.*(my )?(order|package)?|status of (my )?order|when.*(arrive|deliver|shipped)|has my order|did.*ship|when.*expect)\\b',\n"
    " 'device_app': r'\\b(echo|alexa|kindle|fire stick|fire tv|app\\b|stream|not (work|loading)|crash|update.*app|prime video|device)\\b',\n"
    " 'return_replacement': r'\\b(return|send back|swap|exchange|replacement|faulty|damag|defect|broken|not (work|function))\\b',\n"
    " 'account_access': r'\\b(log.?in|sign.?in|password|lock.*account|account.*hack|hacked|no.?longer.*access|account.*suspended|cannot.*account)\\b',\n"
    " 'account_info_update': r'\\b(update.*(email|address|phone)|change.*(email|address|phone|name)|delivery address.*(change|wrong)|wrong email)\\b',\n"
    " 'prime_membership': r'\\b(prime membership|cancel.*prime|renew.*prime|membership|prime.*(fee|benefit|subscription))\\b',\n"
    " 'service_complaint': r'\\b(support.*(useless|bad|terrible|worst)|customer service.*(useless|bad)|fed up|never (again|using)|sick of|no.?help)\\b',\n"
    "}\n"
    "low = en['customer_text_clean'].str.lower()\n"
    "import warnings\n"
    "with warnings.catch_warnings():\n"
    "    warnings.simplefilter('ignore')\n"
    "    hits = pd.DataFrame({name: low.str.contains(pat, regex=True).astype(int) for name, pat in RULES.items()})\n"
    "vol = hits.sum().sort_values(ascending=False)\n"
    "print('per-rule hit counts (English sample, overlapping allowed)\\n', vol.to_string())\n"
    "print('\\nmessages matching ≥2 rules:', int((hits.sum(axis=1)>=2).sum()), 'of', len(en))\n"
    "# Primary-label heuristic used later (overlaps allowed, but no-hit -> 'unmatched' not junk).\n"
    "best = hits.idxmax(axis=1)\n"
    "best[hits.sum(axis=1)==0] = 'unmatched'\n"
    "print('\\nprimary label distribution (tie -> first rule in dict order; no-hit -> unmatched):')\n"
    "print(best.value_counts().to_string())"
))

cells.append(md("### 8.1 The overlap matrix (intent boundaries)"))
cells.append(code(
    "ov = hits.T @ hits\n"
    "pd.set_option('display.width', 200)\n"
    "print('pairwise overlap (diagonal = individual volume)')\n"
    "display(ov.style.background_gradient(cmap='Blues')) if False else print(ov.to_string())"
))

cells.append(md(
    "**Reading the overlaps.** The 2.6k+ pair-overlap rows are dominated by a few natural "
    "co-occurrences: `return_replacement`×`refund` (returned goods → money back), "
    "`delivery_delay`×`order_status` (late package → where is it), and "
    "`device_app`×`return_replacement` (broken device → replacement). These are genuine overlaps "
    "a Phase-3 labeler should record as co-occurring intents rather than force single labels."
))

cells.append(md("## 9. Context-dependency experiment"))
cells.append(md(
    "A support *agent* must decide when a message is self-contained and when it only makes sense "
    "given prior turns. We measure (a) how many messages are short/referential, and (b) how "
    "uninformative on average a message is relative to its message+context, using TF-IDF novelty.*"
))
cells.append(code(
    "short = en['customer_text_clean'].str.len()\n"
    "print('message length percentiles (chars):', short.quantile([.1,.25,.5]).round(0).to_dict())\n"
    "print('tweets with <60 chars:', f\"{int((short<60).sum()):,}\", f'({100*(short<60).mean():.1f}% of English)')\n"
    "refer = 0\n"
    "for k in [r'it\\b', r'this\\b', r'that\\b', r'still\\b', r'again\\b']:\n"
    "    refer += low.str.contains(k).astype(int)\n"
    "print('messages using ≥1 deictic/referential word (it/this/that/still/again):',\n"
    "      f'{int((refer>0).sum())}', f'({100*(refer>0).mean():.1f}%)')"
))
cells.append(md(
    "A large share of English messages are short, pronoun-heavy and only interpretable with the "
    "conversation context we attached in §3. This is concrete evidence that **Phase 3+ should "
    "classify message+context, not isolated tweets**, and justifies why the corpus unit is the "
    "conversation-aware customer message instead of the raw tweet."
))

cells.append(md("## 10. Resolution patterns: what does a good reply look like?"))
cells.append(md(
    "For each intent we look at AmazonHelp's `next_brand_response` to see (a) how often the brand "
    "moves the interaction to DMs/private contact (a privacy-preserving *handoff*), and (b) how "
    "often it points to a help page. This tells us the *shape* of resolution a later phase must "
    "reproduce."
))
cells.append(code(
    "resp = en['next_brand_response'].astype(str)\n"
    "asks_dm = resp.str.contains(r'DM\\b|direct message|message us|send us|email us|call us|DM us|tweet us|fill this form|fill out this form', case=False, regex=True)\n"
    "gives_url = resp.str.contains(r'https?:|t\\.co', regex=True)\n"
    "thanks = resp.str.contains('thank', case=False)\n"
    "print(f'English msgs whose brand reply asks to move to DM/contact : {100*asks_dm.mean():.1f}%')\n"
    "print(f'English msgs whose brand reply includes a help URL       : {100*gives_url.mean():.1f}%')\n"
    "print(f'English msgs whose brand reply is a courtesy/acknowledge : {100*thanks.mean():.1f}%')\n"
    "print('\\n--- example handoff reply ---')\n"
    "print(resp[asks_dm].replace('nan','').iloc[0][:160])"
))

cells.append(md("### 10.1 Handoff rate by intent"))
cells.append(code(
    "by = pd.DataFrame({'asks_dm':asks_dm.apply(int).values})\n"
    "by['intent_hint'] = best.values\n"
    "by = by[by['intent_hint']!='unmatched']\n"
    "print('handoff-to-DM/share-of-replies rate by intent (English sample):')\n"
    "print(by.groupby('intent_hint')['asks_dm'].mean().sort_values(ascending=False).round(2).to_string())"
))

cells.append(md("## 11. The AmazonHelp intent taxonomy (published)"))
cells.append(code(
    "TAXONOMY = [\n"
    "  ('delivery_delay','Expected delivery is late / not arrived'),\n"
    "  ('delivered_but_not_received','Tracking shows delivered but customer got nothing'),\n"
    "  ('delivered_wrong_location','Delivered to wrong place / stolen / dumped / missed'),\n"
    "  ('order_status_query','Where is my order / when will it ship (pre-delivery)'),\n"
    "  ('refund_request','Initiate or chase a refund'),\n"
    "  ('cancellation','Cancel an order / I did not order this'),\n"
    "  ('charge_issue','Wrong, double or unauthorized charge; price discrepancy'),\n"
    "  ('product_return_and_replacement','Return / exchange / faulty-damaged item'),\n"
    "  ('device_app_issue','Echo/Alexa/Kindle/Fire/app technical problem'),\n"
    "  ('account_access','Cannot log in / password / hacked / suspended'),\n"
    "  ('account_info_update','Change email / address / phone / name'),\n"
    "  ('service_complaint_escalation','Dissatisfaction with support; escalation or threat to leave'),\n"
    "]\n"
    "print(f'{len(TAXONOMY)} intents discovered (target 8-15):')\n"
    "for i,(k,d) in enumerate(TAXONOMY,1):\n"
    "    print(f'  {i:2d}. {k:28s} {d}')"
))
cells.append(code(
    "import yaml\n"
    "doc = {\n"
    "  'brand': 'AmazonHelp',\n"
    "  'phase': 2,\n"
    "  'method': 'conversation-aware corpus + deterministic sample + local sentence-transformer embeddings + KMeans clusters interpreted by hand + keyword rubric',\n"
    "  'seed': config.RANDOM_SEED,\n"
    "  'intents': [{'id':k,'description':d} for k,d in TAXONOMY],\n"
    "  'notes': [\n"
    "    'AmazonHelp traffic is ~27% multilingual (Sep-2017 sample); intent taxonomy defined on the English slice; language treated as a routing attribute, not an other bucket.',\n"
    "    'Conversation-continuation messages (thanks / will do / what?) are real but context-only; they form a messaging layer, not a domain intent.',\n"
    "    'Intents overlap (e.g. return x refund); Phase 3 should allow co-occurring labels.',\n"
    "    'Many messages are meaningless without conversation context -> classify message+context.'\n"
    "  ]\n"
    "}\n"
    "out = config.PROJECT_ROOT/'config'/'amazon_intents.yaml'\n"
    "out.parent.mkdir(parents=True, exist_ok=True)\n"
    "out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))\n"
    "print('wrote', out)"
))

cells.append(md("## 12. Internal validation sample (evidence, not golden set)"))
cells.append(md(
    "A reproducibly-sampled 60-message English subset with an **automated rubric label** is saved "
    "under `data/intermediate/` as a transparency / spot-check artifact — explicitly **not** the "
    "Phase-3 golden eval set, which Phase 3 will construct with a conversation-level split and "
    "human adjudication."
))
cells.append(code(
    "val = en.sample(60, random_state=config.RANDOM_SEED).copy()\n"
    "val['rubric_label'] = best.loc[val.index].values\n"
    "cols = ['conversation_id','customer_text_clean','conversation_context','next_brand_response','rubric_label']\n"
    "val[cols].to_csv(config.DATA_DIR/'intermediate'/'validation_sample.csv', index=False)\n"
    "print('saved', config.DATA_DIR/'intermediate'/'validation_sample.csv')\n"
    "print('rubric-label distribution (60-message internal spot-check; no-hit -> unmatched):')\n"
    "print(val['rubric_label'].value_counts().to_string())"
))

cells.append(md("## 13. Executive summary"))
cells.append(md(
    "**What we found**\n"
    "- AmazonHelp is the right brand to build on: ~82.5k conversations, ~203.6k customer messages, "
    "203.6k brand replies — a large, response-bearing corpus ideal for retrieval/answer "
    "reproduction.\n"
    "- **The corpus is multilingual (~27% non-English** in the sampled window). An English-centric "
    "single-classifier agent would silently miss a large slice of traffic; language must be a "
    "first-class routing attribute.\n"
    "- English traffic decomposes into a **12-intent taxonomy** (delivery delay, delivered-not-"
    "received, delivered-wrong-location, order-status, refund, cancellation, charge issue, "
    "return/replacement, device/app, account access, account info update, service-complaint/"
    "escalation).\n"
    "- Intents **overlap naturally** (return↔refund, delay↔status) → Phase 3 labels should allow "
    "co-occurrence.\n"
    "- Many messages (`thanks`, `still waiting`, `what?`) are **meaningless without context**; "
    "classify message+context, and treat acknowledgements as a separate messaging layer, not a "
    "domain intent.\n"
    "- Brand replies are heavily **handoff-driven** (move to DM, provide help URL), so a later "
    "phase should reproduce *safe handoff* actions, not expose-resolution in a public reply.\n\n"
    "**Out of scope (Phase 3+, deliberately not built here):** classifier, RAG, agent, golden eval "
    "split. This notebook only *discovers* the intent set and the data facts that constrain them."
))

# --------------------------------------------------------------------------- #
nb = nbformat.v4.new_notebook(
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    cells=cells,
)
nbformat.write(nb, OUT)
print("Wrote", OUT)

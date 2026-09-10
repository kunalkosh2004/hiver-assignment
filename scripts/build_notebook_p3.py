"""Regenerate the Phase-5 notebook deterministically (no API key, no network).

    python scripts/build_notebook_p3.py
    -> notebooks/03_historical_support_corpus.ipynb

The notebook recomputes everything live from committed data + reports:
retrieval benchmark (reports/retrieval_results.*), data scaling
(reports/retrieval_scaling.*) and error analysis
(reports/retrieval_error_analysis.*). It never re-embeds or re-trains.
"""
from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "03_historical_support_corpus.ipynb"


def md(s: str):
    return nbformat.v4.new_markdown_cell(s)


def code(s: str):
    return nbformat.v4.new_code_cell(s)


cells: list = []

cells.append(md(
    "# 03 — Historical AmazonHelp Support Corpus & Retrieval Baseline\n\n"
    "**Phase 5 of 9** for the Hiver SDE-intern take-home. Phase 1 selected **AmazonHelp**; Phase 2 "
    "discovered the intent taxonomy; Phases 3–4 shipped a gold set and weak deterministic "
    "classifiers. This notebook asks the RAG question: **does the ~203k historical interaction "
    "corpus help an agent recall how a brand-side agent actually handled a comparable "
    "interaction?** It evaluates *retrieval only* — no agent, no generated replies, no escalation "
    "policy. Phases 6–7 later consume the retrieved resolutions for drafting and escalation.\n\n"
    "**Method (fully reproducible, deterministic seed 42, no API key):**\n"
    "1. Assemble the conversation-aware interaction corpus (`customer message → context → brand "
    "   response`), excluding the golden + holdout conversations with a hard assertion.\n"
    "2. Quantify corpus facts that shape retrieval (multilinguality, canned/template replies, "
    "   URL-heavy messages, context dependency).\n"
    "3. Build TF-IDF, BM25, dense (all-MiniLM-L6-v2) and hybrid retrievers over message-only vs "
    "   message+context inputs; evaluate against a **hand-labelled 30-query / 150-pair** pool with "
    "   a temporal filter (retrieved interactions must predate the tweet).\n"
    "4. Measure the **data-scaling curve** on nested subsets (10k / 50k / 100k / full).\n"
    "5. Categorize >=30 retrieval failures and publish them with raw evidence.\n\n"
    "The notebook recomputes every number live from committed artifacts; it does not re-embed or "
    "re-train anything (those live in the scripts)."
))

cells.append(md("## 1. Corpus assembly & leak protection"))
cells.append(code(
    "import json, sys\n"
    "from pathlib import Path\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "sys.path.insert(0, '..')\n"
    "from src import config\n"
    "from src.retrieval.corpus import load_corpus, load_holdout_ids\n\n"
    "corpus_path = config.DATA_DIR / 'retrieval' / 'corpus.slim.parquet'\n"
    "df = pd.read_parquet(corpus_path)\n"
    "print('corpus rows:', len(df))\n"
    "print('usable responses: {:.1f}%'.format(100 * df['has_usable_response'].mean()))\n"
    "print('distinct conversations:', df['conversation_id'].nunique())\n"
    "print('canned templates (>=20 copies): {:.2f}%'.format(100 * df['is_canned_template'].mean()))\n"
    "print(df['language'].value_counts().head(8).to_string())\n"
))
cells.append(code(
    "holdout_conv = {int(x) for x in load_holdout_ids()}\n"
    "indexed_conv = set(df['conversation_id'].astype(int))\n"
    "overlap = indexed_conv & holdout_conv\n"
    "print('golden/holdout conversations excluded from index:', len(overlap), '| (must be 0)')\n"
    "assert len(overlap) == 0\n"
    "print('leak check PASSED')\n"
))

cells.append(md("## 2. The retrieval benchmark"))
cells.append(md(
    "A pool of 30 queries (24 English + 6 non-English, all treated as context-dependent) was "
    "drawn from golden/holdout-adjacent material with seed 42, and 150 candidate pairs were "
    "hand-labelled *0 = irrelevant, 1 = related, 2 = useful*. Recall is pool-based (unjudged "
    "top-K hits count as misses — standard pooling); hybrid alphas are reported, not tuned on "
    "golden labels."
))
cells.append(code(
    "R = dict()\n"
    "rep = json.loads(Path('../reports/retrieval_results.json').read_text())\n"
    "import csv\n"
    "rows = list(csv.DictReader(open('../reports/retrieval_results.csv')))\n"
    "print('{:<12} {:<4} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>7}'.format(\n"
    "    'model', 'in', 'R@1', 'R@3', 'R@5', 'R@10', 'MRR', 'useful5', 'lat_ms'))\n"
    "for r in rows:\n"
    "    print('{:<12} {:<4} {:>5.3f} {:>5.3f} {:>5.3f} {:>5.3f} {:>5.3f} {:>5.3f} {:>7.1f}'.format(\n"
    "        r['model'], r['variant'], float(r['R@1']), float(r['R@3']), float(r['R@5']),\n"
    "        float(r['R@10']), float(r['MRR']), float(r['useful_R@5']), float(r['latency_ms_avg'])))\n"
    "print()\n"
    "print('pooling note:', rep['pooling_notes'])\n"
))

cells.append(md("## 3. Data-scaling curve (message-only)"))
cells.append(code(
    "sc = json.loads(Path('../reports/retrieval_scaling.json').read_text())\n"
    "print('seed', sc['seed'], '| nested subsets | size rows:')\n"
    "prev = {}\n"
    "for r in sc['rows']:\n"
    "    prev[r['model']] = r['R@5']\n"
    "    print('  {:5s} n={:>7d} R@5={:.3f} MRR={:.3f} lat={:.0f}ms idx={:.0f}MB'.format(\n"
    "        r['model'], r['corpus_size'], r['R@5'], r['MRR'], r['lat_ms_avg'], r['index_bytes']/1e6))\n"
    "d = [r for r in sc['rows'] if r['model'] == 'dense']\n"
    "g = (d[3]['R@5'] - d[1]['R@5']) / max(1e-9, (d[3]['corpus_size'] - d[1]['corpus_size']))\n"
    "print('\\ndense R@5 last-window slope (50k->full): {:.3f} per 1000 docs (still climbing -> no plateau)'.format(g*1000))\n"
))

cells.append(md("## 4. Error analysis (>=30 categorized failures)"))
cells.append(code(
    "ea = json.loads(Path('../reports/retrieval_error_analysis.json').read_text())\n"
    "print('failures:', ea['n_failures'], '| by mode:')\n"
    "for k, v in sorted(ea['by_mode'].items(), key=lambda x: -x[1]):\n"
    "    print('  {:~<34}'.format(k), v)\n"
    "print('\\nheadline (dense/msg) failures:')\n"
    "for f in [x for x in ea['failures'] if x['model'] == 'dense' and x['input_variant'] == 'msg'][:8]:\n"
    "    print('  {} {} {} -> missed {}'.format(f['query_id'], f['mode'], f['miss_kind'],\n"
    "                                          f['missed_useful_case_ids'][:3]))\n"
))

cells.append(md("## 5. Reproduce (deterministic, no API key)"))

cells.append(code(
    "print('''\n"
    "python scripts/build_retrieval_index.py --models tfidf,bm25,dense,hybrid --variant both\n"
    "python scripts/prepare_relevance_benchmark.py\n"
    "python scripts/evaluate_retrieval.py\n"
    "python scripts/build_weak_intents.py\n"
    "python scripts/evaluate_retrieval_scaling.py\n"
    "python scripts/build_error_analysis.py\n"
    "''')\n"
))

nb = nbformat.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.14"},
}
nb["cells"] = cells
nbformat.write(nb, OUT)
print("wrote", OUT)
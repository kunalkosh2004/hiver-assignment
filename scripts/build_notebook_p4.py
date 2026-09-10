"""Regenerate the Phase 6-9 agent notebook deterministically (no API key).

    python scripts/build_notebook_p4.py
    -> notebooks/04_retrieval_augmented_agent.ipynb

The notebook recomputes everything live from committed reports + figures:
intent-model facts (reports/agent_intent_models.json), final agent evaluation
(reports/agent_results.{json,csv}), escalation sensitivity, grounded drafting,
error analysis (reports/agent_error_analysis.json) and sample drafts
(reports/agent_rows.json, gitignored regenerable artifact). It never retrains
or re-embeds.
"""
from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "04_retrieval_augmented_agent.ipynb"


def md(s: str):
    return nbformat.v4.new_markdown_cell(s)


def code(s: str):
    return nbformat.v4.new_code_cell(s)


cells: list = []

cells.append(md(
    "# 04 — Retrieval-Augmented Support Agent (Phases 6–8)\n\n"
    "**Phases 6–8 of 9** for the Hiver SDE-intern take-home. Phase 5 built the historical retrieval "
    "memory (RAG recall memory). This notebook closes the agent loop end-to-end:\n\n"
    "1. **Phase 6 — drafting** (`src/agent/`): intent classifier + dense retrieval "
    "   (message-only, temporal filter) + deterministic *grounded* reply drafting "
    "   (never fabricates order numbers / links / amounts).\n"
    "2. **Phase 7 — escalation** (`src/agent/escalation.py`): a deterministic confidence-based "
    "   policy, calibrated on the **dev** set only (golden stays pristine) and frozen thereafter.\n"
    "3. **Phase 8 — final evaluation on the golden benchmark** (used ONLY here): the intent-classifier "
    "   learning curve (majority 0 → dev-140 human → weak-201k proxy labels; LLM-labelled data is "
    "   future work), escalation behavior + sensitivity to the confidence floor, grounded-drafting "
    "   quality, and an optional LLM judge (disabled when no provider key is set).\n"
    "4. **Phase 9 — failure analysis** (reports/agent_error_analysis.*) categorizes every golden row.\n\n"
    "Golden-benchmark hygiene: the golden/holdout conversations were excluded from the retrieval index "
    "(hard assertion), the intent models never see golden rows, the escalation floor is calibrated on "
    "dev only, and every agent run applies the temporal filter (retrieved resolutions must predate the "
    "customer's message)."
))

cells.append(md("## 1. Agent intent models (offline, deterministic)"))
cells.append(code(
    "import json, csv, sys\n"
    "from pathlib import Path\n"
    "sys.path.insert(0, '..')\n\n"
    "im = json.loads(Path('../reports/agent_intent_models.json').read_text())\n"
    "for name, m in im.items():\n"
    "    print('[{}] source={} n_train={} classes_n={}'.format(\n"
    "        name, m.get('source'), m.get('n_train'), m.get('classes_n')))\n"
    "    if 'class_counts' in m:\n"
    "        top = sorted(m['class_counts'].items(), key=lambda x: -x[1])[:4]\n"
    "        print('    top weak classes:', ', '.join('{}={:,}'.format(k, v) for k, v in top))\n"
    "print('\\nweak-model sanity on dev/validation (proxy labels, human gold):')\n"
    "print('  dev-140 validation macro-F1:', im['dev']['validation_macro_f1'])\n"
    "print('  weak-201k on same dev/validation:', im['weak'].get('dev_validation_proxy_macro_f1'))\n"
))

cells.append(md(
    "## 2. Final evaluation on the golden benchmark (200 rows)\n\n"
    "**Learning curve.** Three models, same golden targets. Note the label-space mismatch: the "
    "weak-201k model has 12 proxy classes (no `none`), dev-140 has 14 human classes — so each model "
    "is also scored on the subset of golden rows whose true intent is inside its label space."
))
cells.append(code(
    "ag = json.loads(Path('../reports/agent_results.json').read_text())\n"
    "print('{:<16} {:>7} {:>8} {:>8}  (golden acc / macro-F1)'.format('model', 'n_train', 'acc', 'F1'))\n"
    "for pt in ag['intent_learning_curve']:\n"
    "    print('{:<16} {:>7} {:>8.3f} {:>8.3f}'.format(\n"
    "        pt['label'], pt['n_train'], pt['accuracy'], pt['macro_f1']))\n"
    "print('\\nOn label-space shared subsets only:')\n"
    "for pt in ag['intent_learning_curve']:\n"
    "    s = pt.get('on_shared_subset')\n"
    "    if s:\n"
    "        print('  {:<16} n={:>3} acc={:.3f} F1={:.3f}'.format(\n"
    "            pt['label'], s['n'], s['accuracy'], s['macro_f1']))\n"
    "print('\\nDrafting (deterministic, grounded):')\n"
    "d = ag['drafting']\n"
    "print('  auto rows reuse retrieved resolution words: {:>5.1f}%'.format(100*d['resolution_reuse_rate_auto']))\n"
    "print('  avg cue overlap with golden human reply (url/email/apology/ask/contact): {:.2f}'.format(d['avg_cue_overlap']))\n"
    "print('  empty drafts:', d['empty_drafts'], '| avg chars:', d['avg_draft_chars'])\n"
    "print('\\nLLM judge:', ag['llm_judge'])\n"
))
cells.append(code(
    "from IPython.display import Image\n"
    "Image(filename='../reports/figures/agent_learning_curve.png')\n"
))

cells.append(md(
    "## 3. Escalation policy (Phase 7, dev-calibrated floor)\n\n"
    "Escalation fires on catch-all intents (`none`/`other_unclear`, no usable reply in top-3, or "
    "combined confidence below a **dev-calibrated** floor (`comb_low`, frozen at 0.8775). The same "
    "policy is run over the golden rows without any retuning; the table below adds a *sensitivity* "
    "analysis across floors to show how the policy's surface responds."
))
cells.append(code(
    "esc = ag['escalation']\n"
    "print('escalation on golden:', esc['escalate'], '/', ag['n_golden'],\n"
    "      '({:.1f}%)'.format(100*esc['escalation_rate']))\n"
    "print('active comb_low floor:', esc['active_comb_low'])\n"
    "print('escalates by true intent:', esc['by_true_intent'])\n"
    "print('\\nSensitivity: escalate count when the combined-confidence floor moves')\n"
    "for k, v in esc['sensitivity_to_comb_low'].items():\n"
    "    print('  comb_low={:>5}: escalate={:>3} ({:.1f}%)'.format(k, v['escalate'], 100*v['rate']))\n"
))
cells.append(code(
    "Image(filename='../reports/figures/agent_escalation_calibration.png')\n"
))

cells.append(md("## 4. Error analysis (Phase 9, per-golden-row taxonomy)"))
cells.append(code(
    "ea = json.loads(Path('../reports/agent_error_analysis.json').read_text())\n"
    "print('golden rows:', ea['n_golden'], '| failure rows:', ea['n_failure_rows'],\n"
    "      '| success rows:', ea['n_success_rows'])\n"
    "print('\\ncategory counts:')\n"
    "for k, v in ea['category_counts'].items():\n"
    "    print('  {:<28}'.format(k), v)\n"
    "print('\\nmost common code combinations:')\n"
    "for k, v in list(ea['co_occurrence'].items())[:5]:\n"
    "    print('  {:<46}'.format(k), v)\n"
))

cells.append(md("## 5. Sample drafts (deterministic, grounded)"))
cells.append(code(
    "rows = json.loads(Path('../reports/agent_rows.json').read_text())\n"
    "if not rows:\n"
    "    print('reports/agent_rows.json missing — run:')\n"
    "    print('  python scripts/evaluate_agent.py --dump-rows reports/agent_rows.json')\n"
    "else:\n"
    "    auto = [r for r in rows if r['action'] == 'auto_handle'][:3]\n"
    "    esc_rows = [r for r in rows if r['action'] == 'escalate'][:2]\n"
    "    for r in auto:\n"
    "        print('--- AUTO {} [{} -> {}] conf={}'.format(\n"
    "            r['example_id'], r['true_intent'], r['intent'], r['action_confidence']))\n"
    "        print(r['draft'][:260])\n"
    "        print()\n"
    "    for r in esc_rows:\n"
    "        print('--- ESCALATE {} [{} -> {}] {}'.format(\n"
    "            r['example_id'], r['true_intent'], r['intent'], r['action_reasons'][:1]))\n"
    "        print(r['draft'][:200])\n"
    "        print()\n"
))

cells.append(md("## 6. Reproduce (deterministic, no API key)"))

cells.append(code(
    "print('''\n"
    "python scripts/train_agent_intent.py        # weak-201k + dev-140 models\n"
    "python scripts/calibrate_escalation.py      # dev-only confidence floor\n"
    "python scripts/evaluate_agent.py --dump-rows reports/agent_rows.json\n"
    "python scripts/analyze_agent_errors.py reports/agent_rows.json\n"
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
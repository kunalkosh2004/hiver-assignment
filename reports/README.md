# Phase-4 Baseline & Evaluation Artifacts

All outputs below are **deterministic** (seed 42, offline) and regenerable with:

```bash
python scripts/run_baselines.py       # trains + evaluates baselines
python scripts/evaluate_baselines.py  # segmentation / calibration / error analysis
```

The golden benchmark is used only for evaluation; no LLM is invoked here.

## Files

| File | Contents |
|------|----------|
| `baseline_results.csv` | One row per (model, input, subset): accuracy / macro-F1 / weighted-F1 / macro-P / macro-R. Subsets: `test`, `val`, `golden_all`, `golden_rep`, `golden_chal`. |
| `baseline_results.json` | Machine-readable metrics + split report (sizes, label distributions, leakage counts) + per-intent P/R/F1 for every (model, input, subset). |
| `baseline_predictions.csv` | Every prediction for every (model, input, subset): gold, predicted, correct, language, difficulty, context dependency, message type, and (logistic only) confidence + predicted probability. |
| `figures/confusion_{model}_{input}.png` | Confusion matrices on the internal test set (logistic + SVM, message-only + message+context). |
| `segmentation.json` | Macro F1 broken down by difficulty / language / context_dependency / message_type, on golden_all and internal test. |
| `calibration.json` | Logistic confidence quantile buckets on the golden benchmark (confidence never exceeds ~0.25 here, so buckets are relative). |
| `error_analysis.json` | **All** golden misclassifications for logistic/msg-only (154), logistic/msg+ctx (151), and svm/msg-only (151), with message text and features. |
| `analysis_4D.md` | The full Phase-4D write-up: headline table, split config, segmentation, calibration, ≥20 error samples + confusion patterns, and the *dev benchmark vs golden benchmark* distinction section. |
| `llm_analysis/` | (Optional, only when an LLM is configured) classified results that are explicitly **NOT ground truth** and never feed the deterministic baselines. |

## Reading the numbers honestly

- Headline golden **macro F1 ≈ 0.12–0.16** — low because of 14 imbalanced
  classes, 140 training rows, short/noisy multilingual tweets, and a dev→golden
  distribution shift. This is the reproducible baseline an LLM agent must beat.
- Confidence is **not calibrated**: even the most-confident logistic outputs sit
  near 0.2 and are only ~as accurate as the least-confident ones — do not use
  logistic `max_prob` for reliable escalation decisions as-is.
- `none` (acknowledgements/chat filler) dominates the dev set but not golden;
  quotes from `analysis_4D.md` §6 explain why that is expected, not a bug.
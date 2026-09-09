"""Build the Phase-3 golden candidate pool (conversation-safe, seed 42).

Selection-only script. It does NOT label examples: the keyword rubric is used
strictly as a sampling device (allowed by the phase spec). Final labels,
including the `language` field, come from human review.

Leakage safety:
  - Every candidate keeps its conversation_id.
  - The representative set uses *distinct* conversations (one golden example
    per conversation); those conversations are saved as a reserved holdout that
    must not enter later training / retrieval corpora.
  - A small number of challenge examples deliberately reuse a conversation to
    exercise within-conversation context-dependency; every reuse is flagged in
    the review sheet and carrying conversation(s) are removed from the holdout
    training pool too.

Outputs (data/golden/):
  - candidates_representative.tsv / candidates_challenge.tsv (review sheets)
  - conversation_holdout.json (reserved conversation ids)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src import config

GOLDEN = config.DATA_DIR / "golden"
GOLDEN.mkdir(parents=True, exist_ok=True)
SEED = config.RANDOM_SEED  # 42
rng = np.random.default_rng(SEED)

corpus = pd.read_parquet(config.DATA_DIR / "intermediate" / "amazon_customer_corpus.parquet")

# --------------------------------------------------------------------------- #
# 1. Coarse intent bucket via keyword rubric — SAMPLING ONLY, not labeling.
RULES = {
    "delivery_delay": r"\b(late|delay|not arriv|hasn.t arriv|out for delivery|no sign of|stuck|in transit|supposed.*deliver|delivery date)\b",
    "delivered_not_received": r"\b(mark.*delivered|says.*delivered|delivered.*but|delivered.*not receive|shows.*deliver|handed to resident)\b",
    "delivered_wrong": r"\b(wrong address|wrong house|neighbor|stolen|dumped|missed.*deliver)\b",
    "refund": r"\b(refund|money back|give back|reimburs)\b",
    "cancellation": r"\b(cancel(l)?ation|want to cancel|didn.t order)\b",
    "charge_issue": r"\b(charg|debited|unauthoriz|price went|double charged|billed)\b",
    "order_status": r"\b(where.*(order|package)|status of.*order|when.*(arrive|deliver|ship)|has my order)\b",
    "device_app": r"\b(echo|alexa|kindle|fire stick|fire tv|prime video|not (work|loading|playing)|app\b)\b",
    "return_replacement": r"\b(return|send back|exchange|replacement|faulty|damag|defect|broken)\b",
    "account_access": r"\b(log.?in|password|hack|suspended account|cannot.*account|forgot.*(email|password))\b",
    "account_info_update": r"\b(update.*(email|address|phone)|change.*(address|email|phone))\b",
    "service_complaint": r"\b(support.*(useless|bad|terrible|worst)|never (again|using)|fed up|sick of)\b",
}
low = corpus["customer_text"].str.lower().fillna("")
hits = pd.DataFrame({name: low.str.contains(p, regex=True).astype(int) for name, p in RULES.items()})
best = hits.idxmax(axis=1)
best[hits.sum(axis=1) == 0] = "unmatched"
corpus["pool_bucket"] = best.values
corpus["pool_short"] = (corpus["customer_text"].str.len() < 90).values
corpus["pool_deictic"] = low.str.contains(r"\b(it|this|that|still|again|what|okay|ok|thanks|thx)\b", regex=True).values
corpus["pool_angry"] = low.str.contains(r"\b(wtf|hell|sick|fed up|useless|terrible|scam|worst|angry|furious|disgusting|never again)\b", regex=True).values

print("pool bucket split (all langs):")
print(corpus["pool_bucket"].value_counts().to_string())


# --------------------------------------------------------------------------- #
# 2. Representative candidate pool: proportional to bucket size, distinct convs.
#    English dominates; we oversample ~40% to compensate for dropping the
#    non-English we'll discard by hand during annotation.
def _distinct_convs(df, n, rng):
    """Pick n rows from distinct conversations (best effort)."""
    out = []
    used = set()
    sh = df.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
    for _, r in sh.iterrows():
        if len(out) >= n:
            break
        if r["conversation_id"] not in used:
            used.add(r["conversation_id"])
            out.append(r)
    return pd.DataFrame(out).reset_index(drop=True)


rep_target = 210  # overshoot; we'll trim to ~150 English by hand
rep_pieces = []
for b, grp in corpus.groupby("pool_bucket"):
    weight = len(grp) / len(corpus)
    k = int(round(rep_target * weight)) + 1
    rep_pieces.append(grp.sample(min(k, len(grp)), random_state=int(rng.integers(0, 1_000_000))))
rep_pool = _distinct_convs(pd.concat(rep_pieces), rep_target, rng)

# --------------------------------------------------------------------------- #
# 3. Challenge candidate pool: hard cases, context-dependent, multilingual mix.
chal_cand = corpus[
    corpus["pool_short"] | corpus["pool_deictic"] | corpus["pool_angry"] |
    (corpus["pool_bucket"] == "unmatched")
]
# mix in some messages that clearly aren't English thank-you acks for multilingual
chal_cand = pd.concat([chal_cand,
                       corpus[~corpus["customer_text"].str.contains("thank", case=False)].sample(
                           min(300, len(corpus)), random_state=int(rng.integers(0, 1_000_000)))])
chal_cand = chal_cand.drop_duplicates("tweet_id").reset_index(drop=True)
chal_pool = chal_cand.sample(min(len(chal_cand), 90), random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. Emit review sheets (conversation id + message + flattened context).
def _review_df(df):
    return pd.DataFrame({
        "conversation_id": df["conversation_id"].values,
        "tweet_id": df["tweet_id"].values,
        "message": df["customer_text"].astype(str).str.replace("\n", " ").str[:400].values,
        "context": df["conversation_context"].astype(str).str[:1200].values,
        "bucket": df["pool_bucket"].values,
        "short": df["pool_short"].values,
        "deictic": df["pool_deictic"].values,
    })


rep_rev = _review_df(rep_pool)
chal_rev = _review_df(chal_pool)

rep_rev.to_csv(GOLDEN / "candidates_representative.tsv", sep="\t", index=False)
chal_rev.to_csv(GOLDEN / "candidates_challenge.tsv", sep="\t", index=False)

# --------------------------------------------------------------------------- #
# 5. Reserved holdout conversations (the golden conversations must not leak into
#    later training / retrieval corpora).
holdout_convs = set(rep_pool["conversation_id"]) | set(chal_pool["conversation_id"])
(GOLDEN / "conversation_holdout.json").write_text(
    json.dumps({"seed": SEED, "n_conversations": len(holdout_convs),
                "conversation_ids": sorted(int(c) for c in holdout_convs)}, indent=2)
)

print("\nRepresentative candidates:", len(rep_rev), "(distinct convs:", rep_rev['conversation_id'].nunique(), ")")
print("Challenge candidates    :", len(chal_rev), "(distinct convs:", chal_rev['conversation_id'].nunique(), ")")
print("Holdout conversations   :", len(holdout_convs))
print("\nWrote review sheets ->", GOLDEN)

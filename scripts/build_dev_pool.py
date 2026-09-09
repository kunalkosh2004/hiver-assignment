"""Assemble the Phase-4 developer-set review sheet (hand-annotation input).

The developer set is the human-labelled TRAIN/VAL/TEST source for Phase-4
baselines. It is conversation-disjoint from the 200-example golden benchmark.

Composition (200):
  - 100 "screened" conversations: real AmazonHelp conversations already reviewed
    in Phase 3 candidate pools but NOT selected for the golden benchmark.
  - 100 "sampled" conversations: real conversations drawn deterministically
    (seed 42) from the corpus, excluding golden + screened, conversation-distinct.

Outputs:
  - data/golden/dev_sampled_100.json : the 100 sampled conversation ids (seed 42)
  - data/golden/dev_ids.json         : conversation_id -> reviewed tweet_id map (200)
  - data/golden/dev_review_200.txt   : human review sheet (message + context)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys_path_hack = ROOT.__str__()
CORPUS = ROOT / "data" / "intermediate" / "amazon_customer_corpus.parquet"
GOLD = ROOT / "data" / "golden"

GOLDEN_IDS = set(
    json.loads(l)["conversation_id"] for l in (GOLD / "amazon_golden_eval.jsonl").open()
)
REP_SHEET = pd.read_csv(GOLD / "candidates_representative.tsv", sep="\t")
CHAL_SHEET = pd.read_csv(GOLD / "candidates_challenge.tsv", sep="\t")
SCREENED = pd.concat([REP_SHEET, CHAL_SHEET])


def main() -> None:
    corpus = pd.read_parquet(CORPUS)
    golden = GOLDEN_IDS
    screened = SCREENED

    # --- 100 screened-but-not-golden (use pool message/context, exact reviewed turn) ---
    screened_dev = screened[~screened["conversation_id"].isin(golden)]
    sampled_manifest = GOLD / "dev_sampled_100.json"

    # --- 100 sampled conversations (seed 42), disjoint from golden + screened ---
    if sampled_manifest.exists():
        sampled = json.loads(sampled_manifest.read_text())["conversation_ids"]
    else:
        used = golden | set(screened_dev["conversation_id"])
        avail = sorted(set(corpus["conversation_id"].unique()) - used)
        rng = np.random.default_rng(42)
        rng.shuffle(avail)
        sampled = [int(c) for c in avail[:100]]
        sampled_manifest.write_text(
            json.dumps({"n": len(sampled), "seed": 42, "conversation_ids": sampled}, indent=2)
        )

    corpus = corpus.sort_values(["conversation_id", "created_at", "tweet_id"]).reset_index(drop=True)
    by_tweet = corpus.drop_duplicates("tweet_id").set_index("tweet_id", drop=False)

    items: list[tuple[int, int, str, str, str]] = []

    # screened ones: exact reviewed message
    for cid in sorted(set(screened_dev["conversation_id"])):
        row = screened_dev[screened_dev["conversation_id"] == cid].iloc[0]
        items.append((int(cid), int(row["tweet_id"]), str(row["message"]), str(row["context"]), "screened"))

    # sampled ones: choose first meaningful customer tweet + its context
    for cid in sorted(sampled):
        sub = corpus[corpus["conversation_id"] == cid]
        pick = sub.iloc[0]
        for _, r in sub.iterrows():
            t = str(r.get("customer_text", ""))
            if len(t.strip()) > 2:
                pick = r
                break
        tid = int(pick["tweet_id"])
        msg = str(pick.get("customer_text_clean") or pick.get("customer_text") or "")
        ctx = str(pick["conversation_context"]) if isinstance(pick["conversation_context"], str) else ""
        items.append((int(cid), tid, msg, ctx[:1500], "sampled"))

    assert len(items) == 200, f"expected 200 dev candidates, got {len(items)}"

    dev_ids = {c: t for c, t, _, _, _ in items}
    assert len(dev_ids) == 200, "dup conversation ids"
    assert set(dev_ids) & golden == set(), "leak: dev conv overlaps golden"
    (GOLD / "dev_ids.json").write_text(json.dumps({int(k): int(v) for k, v in dev_ids.items()}, indent=2))

    lines = []
    for cid, tid, msg, ctx, src in sorted(items):
        lines.append(f"### CID={cid} TWEET={tid} SRC={src}")
        lines.append(f"MSG: {msg}")
        lines.append(f"CTX: {ctx}")
        lines.append("")
    (GOLD / "dev_review_200.txt").write_text("\n".join(lines))
    print(f"wrote dev review sheet ({len(items)} candidates) -> {GOLD}/dev_review_200.txt")
    print(f"wrote dev_ids.json -> {GOLD}/dev_ids.json")


if __name__ == "__main__":
    main()

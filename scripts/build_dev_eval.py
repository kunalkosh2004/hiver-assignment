"""Build the Phase-4 developer set (labelled) from hand-authored annotations.

Reads `scripts/dev_annotations.py` (DEV dict keyed by conversation_id), joins
the exact reviewed tweet from the corpus (via data/golden/dev_ids.json
conversation_id -> tweet_id), and writes:

  data/golden/amazon_dev_set.jsonl

Each record: conversation_id, tweet_id, current_customer_message,
conversation_context (parsed role/text), language, primary_intent,
secondary_intents, message_type, context_dependency, difficulty,
label_confidence, annotation_note.

No `next_brand_response` is included (leakage-free input for classifier training).
Conversation-level disjointness from the golden benchmark is asserted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import dev_annotations as DA  # noqa: E402

DATA_DIR = ROOT / "data"
CORPUS = DATA_DIR / "intermediate" / "amazon_customer_corpus.parquet"
GOLD = DATA_DIR / "golden"
OUT = GOLD / "amazon_dev_set.jsonl"
TAXONOMY = ROOT / "config" / "amazon_intents.yaml"

INTENT_IDS = {i["id"] for i in yaml.safe_load(TAXONOMY.read_text())["intents"]}


def main():
    corpus = pd.read_parquet(CORPUS)
    dev_ids = {int(k): int(v) for k, v in json.loads((GOLD / "dev_ids.json").read_text()).items()}
    golden_ids = {json.loads(l)["conversation_id"] for l in (GOLD / "amazon_golden_eval.jsonl").open()}

    # leak guard: dev conversations must be disjoint from golden conversations
    assert set(dev_ids).isdisjoint(golden_ids), "dev set overlaps golden benchmark"

    corpus["tweet_id_int"] = corpus["tweet_id"].astype("int64")
    by_tweet = corpus.drop_duplicates("tweet_id_int").set_index("tweet_id_int", drop=False)

    records = []
    for cid, lab in sorted(DA.DEV.items()):
        tid = dev_ids[int(cid)]
        row = by_tweet.loc[tid]
        message = str(row["customer_text_clean"])
        assert message and message.strip(), f"empty message @ {cid}"

        context = []
        ctx_raw = row["conversation_context"]
        if isinstance(ctx_raw, str) and ctx_raw.strip():
            for part in ctx_raw.split("|"):
                part = part.strip()
                if part.upper().startswith("CUSTOMER:"):
                    context.append({"role": "customer", "text": part[len("CUSTOMER:"):].strip()})
                elif part.upper().startswith("BRAND:"):
                    context.append({"role": "brand", "text": part[len("BRAND:"):].strip()})

        records.append({
            "conversation_id": int(cid),
            "tweet_id": str(tid),
            "current_customer_message": message,
            "conversation_context": context,
            "language": lab.get("lang", "unknown"),
            "message_type": lab["mtype"],
            "context_dependency": lab["ctx"],
            "primary_intent": lab["primary"],
            "secondary_intents": lab["secondary"],
            "difficulty": lab["difficulty"],
            "label_confidence": lab["conf"],
            "annotation_note": lab.get("note", ""),
        })

    with OUT.open("w") as f:
        for rec in sorted(records, key=lambda r: r["conversation_id"]):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote {OUT} ({len(records)} labelled dev examples)")
    return records


if __name__ == "__main__":
    main()

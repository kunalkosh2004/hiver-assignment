"""Build the Phase-3 AmazonHelp golden evaluation set from hand-authored labels.

Reads `scripts/golden_annotations.py` (REPRESENTATIVE / CHALLENGE dicts keyed by
conversation_id), joins real corpus text, and writes:
  data/golden/amazon_golden_eval.jsonl           (current message + context, NO next_brand_response)
  data/golden/amazon_golden_eval_reference.jsonl (adds next_brand_response for inspection)
  data/golden/amazon_golden_annotation.csv       (human-review ledger)

Hard validation guarantees:
  - exactly 150 representative + 50 challenge
  - every labelled id exists in the corpus and is unique
  - no conversation_id appears in both splits
  - all primary intents are in the frozen taxonomy (or none/other_unclear)
  - conversation-overlap safety vs the training holdout
"""

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import golden_annotations as GA  # noqa: E402

DATA_DIR = ROOT / "data"
CORPUS = DATA_DIR / "intermediate" / "amazon_customer_corpus.parquet"
HOLDOUT = DATA_DIR / "golden" / "conversation_holdout.json"
OUT_EVAL = DATA_DIR / "golden" / "amazon_golden_eval.jsonl"
OUT_REF = DATA_DIR / "golden" / "amazon_golden_eval_reference.jsonl"
OUT_CSV = DATA_DIR / "golden" / "amazon_golden_annotation.csv"
TAXONOMY = ROOT / "config" / "amazon_intents.yaml"

INTENT_IDS = {i["id"] for i in yaml.safe_load(TAXONOMY.read_text())["intents"]}
TARGET_REP, TARGET_CHAL = 150, 50


def rows_from_dict(d, split):
    rows = []
    for cid, lab in d.items():
        rows.append({"conversation_id": cid, "split": split, **lab})
    return rows


def main():
    corpus = pd.read_parquet(CORPUS)
    holdout = set(json.loads(HOLDOUT.read_text()))

    # Map each annotated conversation_id to the EXACT tweet_id that was hand
    # reviewed (a conversation has many tweets; labeling was done on one turn).
    rep_sheet = pd.read_csv(DATA_DIR / "golden" / "candidates_representative.tsv", sep="\t")
    chal_sheet = pd.read_csv(DATA_DIR / "golden" / "candidates_challenge.tsv", sep="\t")
    sheet = pd.concat([rep_sheet, chal_sheet])
    conv_to_tweet = {int(c): int(t) for c, t in zip(sheet["conversation_id"], sheet["tweet_id"])}
    corpus["tweet_id_int"] = corpus["tweet_id"].astype("int64")
    by_tweet = corpus.drop_duplicates("tweet_id_int").set_index("tweet_id_int", drop=False)

    reps = rows_from_dict(GA.REPRESENTATIVE, "representative")
    chals = rows_from_dict(GA.CHALLENGE, "challenge")

    # ---- hard count / id validation -------------------------------------
    assert len(reps) == TARGET_REP, f"representative count {len(reps)} != {TARGET_REP}"
    assert len(chals) == TARGET_CHAL, f"challenge count {len(chals)} != {TARGET_CHAL}"

    all_rows = reps + chals
    ids = [r["conversation_id"] for r in all_rows]
    assert len(ids) == len(set(ids)), "duplicate conversation_id in golden set"

    rep_ids = {r["conversation_id"] for r in reps}
    chal_ids = {r["conversation_id"] for r in chals}
    assert rep_ids.isdisjoint(chal_ids), "conversation appears in both splits"

    missing = []
    for r in all_rows:
        cid = r["conversation_id"]
        tid = conv_to_tweet.get(cid)
        if tid is None or tid not in by_tweet.index:
            missing.append(cid)
    assert not missing, f"no reviewed tweet found for conversation_ids: {missing}"

    # ---- taxonomy / schema validation -----------------------------------
    VALID_MTYPE = {"support_request","follow_up","clarification","acknowledgement","complaint","unknown"}
    VALID_CTX = {"self_contained","context_helpful","context_required"}
    VALID_DIFF = {"easy","medium","hard"}
    VALID_CONF = {"high","medium","low"}
    for r in all_rows:
        p = r["primary"]
        assert p in INTENT_IDS or p in ("none", "other_unclear"), f"bad primary {p} @ {r['conversation_id']}"
        assert r["mtype"] in VALID_MTYPE
        assert r["ctx"] in VALID_CTX
        assert r["difficulty"] in VALID_DIFF
        assert r["conf"] in VALID_CONF
        assert r["secondary"] == [] or all(s in INTENT_IDS for s in r["secondary"]), \
            f"bad secondary @ {r['conversation_id']}"
        if r["conf"] == "low" or r["primary"] == "other_unclear":
            assert r.get("note"), f"low-confidence/unclear needs note @ {r['conversation_id']}"

    # ---- split conversation overlap metric ------------------------------
    from src.evaluation import assert_no_conversation_overlap
    train_set = set()
    assert_no_conversation_overlap(all_rows, train_set, allow_overlap=True)

    # ---- build records ----------------------------------------------------
    eval_records, ref_records, csv_rows = [], [], []
    for r in sorted(all_rows, key=lambda x: (x["split"], x["primary"])):
        cid = r["conversation_id"]
        tid = conv_to_tweet[cid]
        row = by_tweet.loc[tid]
        message = str(row["customer_text_clean"])
        assert message and message.strip(), f"empty message @ {cid}"
        context = []  # parse flattened CUSTOMER|BRAND context in order
        ctx_raw = row["conversation_context"]
        if isinstance(ctx_raw, str) and ctx_raw.strip():
            for part in ctx_raw.split("|"):
                part = part.strip()
                if part.upper().startswith("CUSTOMER:"):
                    context.append({"role": "customer", "text": part[len("CUSTOMER:"):].strip()})
                elif part.upper().startswith("BRAND:"):
                    context.append({"role": "brand", "text": part[len("BRAND:"):].strip()})
        holdout_flag = cid in holdout

        rec = {
            "example_id": f"{r['split']}-{cid}",
            "conversation_id": cid,
            "current_customer_message": message,
            "conversation_context": context,
            "language": r.get("lang", "unknown"),
            "message_type": r["mtype"],
            "context_dependency": r["ctx"],
            "primary_intent": r["primary"],
            "secondary_intents": r["secondary"],
            "difficulty": r["difficulty"],
            "label_confidence": r["conf"],
            "annotation_note": r.get("note", ""),
            "split": r["split"],
        }
        eval_records.append(rec)

        ref_rec = dict(rec)
        ref_rec["next_brand_response"] = row["next_brand_response"] if pd.notna(row["next_brand_response"]) else None
        ref_rec.pop("annotation_note", None)
        ref_rec["annotation_note"] = r.get("note", "")
        ref_rec["in_conversation_holdout"] = holdout_flag
        ref_rec["tweet_id"] = str(row["tweet_id"]) if pd.notna(row["tweet_id"]) else ""
        ref_rec["created_at"] = str(row["created_at"])
        ref_rec["pos_in_conv"] = int(row["pos_in_conv"]) if pd.notna(row["pos_in_conv"]) else None
        ref_rec["conv_length"] = int(row["conv_length"]) if pd.notna(row["conv_length"]) else None
        ref_rec["customer_text"] = str(row["customer_text"])
        ref_rec["previous_brand_message"] = str(row["previous_brand_message"]) if pd.notna(row["previous_brand_message"]) else None
        ref_rec["bucket"] = r.get("bucket", "")
        ref_rec.pop("current_customer_message", None)
        ref_rec["current_customer_message"] = str(row["customer_text_clean"])
        ref_rec.pop("conversation_context", None)
        ref_rec["conversation_context"] = context

        ref_records.append(ref_rec)

        csv_rows.append({
            "example_id": rec["example_id"],
            "conversation_id": cid,
            "split": r["split"],
            "tweet_id": str(row["tweet_id"]) if pd.notna(row["tweet_id"]) else "",
            "language": r.get("lang", "unknown"),
            "primary_intent": r["primary"],
            "secondary_intents": "+".join(r["secondary"]),
            "message_type": r["mtype"],
            "context_dependency": r["ctx"],
            "difficulty": r["difficulty"],
            "label_confidence": r["conf"],
            "annotation_note": r.get("note", ""),
        })

    with OUT_EVAL.open("w") as f:
        for rec in eval_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with OUT_REF.open("w") as f:
        for rec in ref_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False)

    print(f"wrote {OUT_EVAL.name} ({len(eval_records)}), {OUT_REF.name} ({len(ref_records)}), "
          f"{OUT_CSV.name} ({len(csv_rows)})")
    return eval_records


if __name__ == "__main__":
    main()

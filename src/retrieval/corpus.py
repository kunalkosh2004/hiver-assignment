"""Historical AmazonHelp support-interaction corpus builder.

A **historical case** is the atomic retrieval unit:

    customer_message -> conversation_context (prior turns only)
                       -> brand_response (the response that FOLLOWED it)

Total corpus is ~203k AmazonHelp customer messages. The Phase-3 golden
benchmark (and the larger reserved holdout it sits inside) is **never** indexed:
all 300 holdout conversation ids are excluded and `assert_no_golden_overlap`
fails loudly if a golden conversation ever leaks back in.

Response-level "resolution" is never claimed — a case proves that a response
occurred, not that the issue was successfully fixed.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

from .. import amazon, config, data_io
from .base import RetrievalHit
from .language import detect_language

CANNED_TEMPLATE_THRESHOLD = 20

# Response-pattern features ------------------------------------------------- #
DM_RE = re.compile(
    r"\b(dm us|dm me|direct message|send us a (direct )?message|pm us|"
    r"private message|message us|message our|write to us|write us|"
    r"reach out to us|reach out to our|inbox us|contact us on (twitter|dm)"
    r")\b",
    re.IGNORECASE,
)
HELP_CTX_RE = re.compile(
    r"help|troubleshoot|article|support page|/gp/help|customer service page", re.IGNORECASE
)
SUPPORT_LINK_CTX_RE = re.compile(
    r"reach (out )?(to )?us|contact us|here:|for assistance|more (info|information)|"
    r"take a (further )?look|live support|phone or chat|send us", re.IGNORECASE
)
ANY_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
TRACKING_RE = re.compile(
    r"\b(tracking|track your|tracking id|track number)\b|"
    r"track\.amazon|/gp/css/order|orderstatus|delivery-track", re.IGNORECASE
)
PHONE_RE = re.compile(
    r"\+?1[-. ]?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}|"
    r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b", re.IGNORECASE
)
INSTRUCTION_RE = re.compile(
    r"\b(please|kindly|could you|can you|you can|we ask|we recommend|we suggest|"
    r"reach out|contact us|click|go to|visit|select|choose|update your|verify|"
    r"sign in|sign into|let us know|check your)\b", re.IGNORECASE
)
APOLOGY_RE = re.compile(
    r"\b(sorry|apologize|apologise|apologies|apologetic|regret|most sincerely)\b",
    re.IGNORECASE,
)
ACK_RE = re.compile(
    r"\b(thank you for|thanks for|thank you for reaching|we understand|"
    r"we appreciate|we'?re glad|we are glad|noted|well received|"
    r"thanks for updating)\b", re.IGNORECASE
)
ACCOUNT_PRIVACY_RE = re.compile(
    r"\b(account|password|two[- ]step|verification|verify your identity|"
    r"security code|sign[- ]in|personal information|privacy)\b", re.IGNORECASE
)
RESOLVED_CLAIM_RE = re.compile(
    r"\b(resolved|fixed the issue|your issue has been (resolved|fixed)|"
    r"we have (resolved|fixed))\b", re.IGNORECASE
)


def response_pattern_features(response: str | None) -> dict:
    if not response or not str(response).strip():
        return {
            "has_response": False, "asks_dm": False, "contains_any_url": False,
            "contains_help_url": False, "contains_support_link": False,
            "contains_tracking_url": False, "contains_phone": False,
            "contains_instruction": False, "contains_apology": False,
            "contains_acknowledgement": False, "account_or_privacy": False,
            "claims_resolved": False,
        }
    resp = str(response)
    has_url = bool(ANY_URL_RE.search(resp))
    return {
        "has_response": True,
        "asks_dm": bool(DM_RE.search(resp)),
        "contains_any_url": has_url,
        "contains_help_url": has_url and bool(HELP_CTX_RE.search(resp)),
        "contains_support_link": has_url and bool(SUPPORT_LINK_CTX_RE.search(resp)),
        "contains_tracking_url": bool(TRACKING_RE.search(resp)),
        "contains_phone": bool(PHONE_RE.search(resp)),
        "contains_instruction": bool(INSTRUCTION_RE.search(resp)),
        "contains_apology": bool(APOLOGY_RE.search(resp)),
        "contains_acknowledgement": bool(ACK_RE.search(resp)),
        "account_or_privacy": bool(ACCOUNT_PRIVACY_RE.search(resp)),
        "claims_resolved": bool(RESOLVED_CLAIM_RE.search(resp)),
    }


def canonical_template(response: str) -> str:
    t = (response or "").lower()
    t = re.sub(r"https?://\S+", "URL", t)
    t = re.sub(r"@\w+", "@USER", t)
    t = re.sub(r"\d[\d.,-]*", "#", t)
    t = re.sub(r"[^\w\s@#-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_context(context: str | None) -> list[dict]:
    if not context or not str(context).strip():
        return []
    out = []
    for part in str(context).split("|"):
        part = part.strip()
        if part.upper().startswith("CUSTOMER:"):
            out.append({"role": "customer", "text": part[len("CUSTOMER:"):].strip()})
        elif part.upper().startswith("BRAND:"):
            out.append({"role": "brand", "text": part[len("BRAND:"):].strip()})
    return out


def _next_brand_turn_series(sub: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Vectorized nearest-following brand turn (text + created_at) per row.

    `sub` is sorted ascending by (conversation_id, created_at, tweet_id).
    Reversing the frame and forward-filling within each conversation gives, for
    every row, the nearest brand turn that in the ORIGINAL order occurs strictly
    after it (a brand row gets its own text, which we ignore for customer rows).
    """
    isb = (~sub["inbound"]).to_numpy(dtype=bool)
    bt = pd.Series(np.where(isb, sub["text"].astype(str), np.nan), index=sub.index, dtype="object")
    tst = pd.Series(np.where(isb, sub["created_at"].astype(str), np.nan), index=sub.index, dtype="object")

    rev_idx = sub.index[::-1]
    tmp = pd.DataFrame({
        "conversation_id": sub.loc[rev_idx, "conversation_id"].to_numpy(),
        "bt": bt.reindex(rev_idx).to_numpy(),
        "tst": tst.reindex(rev_idx).to_numpy(),
    })
    filled_bt = tmp.groupby("conversation_id")["bt"].transform("ffill").to_numpy()
    filled_ts = tmp.groupby("conversation_id")["tst"].transform("ffill").to_numpy()
    next_text = pd.Series(filled_bt, index=rev_idx).loc[sub.index]
    next_ts = pd.Series(filled_ts, index=rev_idx).loc[sub.index]
    return next_text, next_ts


def build_cases(
    corpus: pd.DataFrame,
    holdout_ids: set[int],
    twcs: pd.DataFrame | None = None,
) -> list[dict]:
    """Build cleaned historical cases from the customer corpus.

    Excludes every conversation in `holdout_ids` (which contains the golden
    benchmark). Response timestamps are recomputed from the full conversation
    graph and cross-validated against the corpus's `next_brand_response`.
    """
    if twcs is None:
        df = amazon.load_with_conversations()
        sub, _ = amazon.amazon_conversations(df)
    else:
        sub = twcs[twcs["conversation_id"].isin(corpus["conversation_id"])].copy()

    sub = sub.sort_values(["conversation_id", "created_at", "tweet_id"]).reset_index(drop=True)
    sub["_pos"] = sub.groupby("conversation_id").cumcount()
    sub["_is_customer"] = sub["inbound"]

    resp_text, resp_ts = _next_brand_turn_series(sub)
    sub["_next_resp"] = resp_text
    sub["_next_resp_ts"] = resp_ts

    cust = sub[sub["_is_customer"]].copy()
    cust["_resp_ts"] = cust["_next_resp_ts"]

    # Cross-validate against the prebuilt corpus (same conversation graph).
    merged = corpus.merge(
        cust[["tweet_id", "_next_resp", "_resp_ts"]], on="tweet_id", how="left"
    )

    def _norm(x) -> str:
        return "" if pd.isna(x) or x is None else str(x).strip()

    mism = merged.apply(
        lambda r: bool(r["next_brand_response"])
        and _norm(r["next_brand_response"]) != _norm(r["_next_resp"]),
        axis=1,
    )
    if mism.sum():
        raise RuntimeError(f"response recomputation mismatch on {int(mism.sum())} rows")
    resp_ts_map = dict(zip(merged["tweet_id"].astype(str), merged["_resp_ts"]))

    cases = []
    resp_counts = Counter()
    for rec in corpus.to_dict("records"):
        cid = int(rec["conversation_id"])
        if cid in holdout_ids:
            continue
        msg = str(rec["customer_text_clean"])
        context_list = parse_context(rec.get("conversation_context"))
        raw_resp = rec.get("next_brand_response")
        response = "" if raw_resp is None or pd.isna(raw_resp) else str(raw_resp).strip()
        has_resp = bool(response)
        response_ts_raw = resp_ts_map.get(str(rec["tweet_id"]))
        response_ts = "" if pd.isna(response_ts_raw) else str(response_ts_raw)
        features = response_pattern_features(response if has_resp else None)
        template = canonical_template(response) if has_resp else ""
        cases.append({
            "case_id": str(rec["tweet_id"]),
            "conversation_id": cid,
            "position_in_conversation": int(rec["pos_in_conv"]),
            "conv_length": int(rec["conv_length"]),
            "customer_timestamp": str(rec["created_at"]),
            "response_timestamp": response_ts or None,
            "language": detect_language(msg),
            "customer_message": msg,
            "raw_customer_message": str(rec.get("customer_text") or msg),
            "conversation_context": context_list,
            "brand_response": response if has_resp else None,
            "has_usable_response": has_resp,
            "template_key": template or None,
            "metadata": features,
        })
        if has_resp:
            resp_counts[template] += 1

    for c in cases:
        if c["template_key"]:
            c["template_count"] = resp_counts[c["template_key"]]
            c["is_canned_template"] = resp_counts[c["template_key"]] >= CANNED_TEMPLATE_THRESHOLD
        else:
            c["template_count"] = 0
            c["is_canned_template"] = False
    return cases


def save_slim_corpus(cases: list[dict], out_path: str) -> None:
    """Persist a fast-to-reload copy of the case list.

    `doc_msg`/`doc_ctx` are the pre-serialized Experiment A/B strings, so a
    reloading process never re-walks 200k contexts. `conversation_context` and
    `metadata` are stored as JSON strings and re-parsed on load.
    """
    from .text import document_text

    df = pd.DataFrame({
        "case_id": [c["case_id"] for c in cases],
        "conversation_id": [int(c["conversation_id"]) for c in cases],
        "position_in_conversation": [int(c["position_in_conversation"]) for c in cases],
        "conv_length": [int(c["conv_length"]) for c in cases],
        "customer_timestamp": [c["customer_timestamp"] for c in cases],
        "response_timestamp": [c.get("response_timestamp") for c in cases],
        "language": [c.get("language", "") for c in cases],
        "customer_message": [c["customer_message"] for c in cases],
        "conversation_context": [json.dumps(c.get("conversation_context") or [], ensure_ascii=False) for c in cases],
        "brand_response": [c.get("brand_response") for c in cases],
        "has_usable_response": [bool(c["has_usable_response"]) for c in cases],
        "is_canned_template": [bool(c.get("is_canned_template")) for c in cases],
        "template_count": [int(c.get("template_count", 0)) for c in cases],
        "metadata": [
            json.dumps({k: bool(v) for k, v in (c.get("metadata") or {}).items()}, ensure_ascii=False)
            for c in cases
        ],
        "doc_msg": [document_text(c["customer_message"], None, False) for c in cases],
        "doc_ctx": [
            document_text(c["customer_message"], c.get("conversation_context"), True) for c in cases
        ],
    })
    df.to_parquet(out_path, index=False)


def _cases_from_slim(df: pd.DataFrame) -> list[dict]:
    cases = []
    for row in df.to_dict("records"):
        features = json.loads(row["metadata"] or "{}")
        for k, v in row.items():
            if k in ("has_usable_response", "is_canned_template"):
                row[k] = bool(v)
            elif k in ("template_count", "position_in_conversation", "conv_length"):
                row[k] = int(v)
        row["metadata"] = {k: bool(v) for k, v in features.items()}
        row["conversation_context"] = json.loads(row["conversation_context"] or "[]")
        row["__doc_msg"] = row["doc_msg"]
        row["__doc_ctx"] = row["doc_ctx"]
        cases.append(row)
    return cases


def load_corpus(path: str | None = None, prefer_slim: bool = True) -> list[dict]:
    p = path or str(config.DATA_DIR / "retrieval" / "corpus.jsonl")
    slim = str(config.DATA_DIR / "retrieval" / "corpus.slim.parquet")
    if prefer_slim and os.path.exists(slim):
        return _cases_from_slim(pd.read_parquet(slim))
    with open(p) as fh:
        return [json.loads(line) for line in fh]


class CaseStore:
    """Index of historical cases + deterministic text construction per variant."""

    def __init__(self, cases: list[dict], include_context: bool):
        self.cases = cases
        self.include_context = include_context
        self._by_id = {c["case_id"]: i for i, c in enumerate(cases)}
        self._texts: list[str] | None = None

    def n(self) -> int:
        return len(self.cases)

    def texts(self) -> list[str]:
        if self._texts is None:
            from .text import document_text

            cached = "__doc_ctx" if self.include_context else "__doc_msg"
            if self.cases and cached in self.cases[0]:
                self._texts = [c[cached] for c in self.cases]
            else:
                self._texts = [
                    document_text(
                        c["customer_message"], c.get("conversation_context"), self.include_context
                    )
                    for c in self.cases
                ]
        return self._texts

    def query_text(self, query: str, context: list[dict] | str | None) -> str:
        from .text import document_text

        return document_text(query, context, self.include_context)

    def hit(self, i: int, score: float, timestamp_cutoff: str | None = None) -> "RetrievalHit":
        c = self.cases[i]
        return RetrievalHit(
            case_id=c["case_id"],
            score=score,
            customer_message=c["customer_message"],
            brand_response=c.get("brand_response"),
            conversation_context=c.get("conversation_context") or [],
            language=c.get("language", ""),
            customer_timestamp=c.get("customer_timestamp", ""),
            metadata=c.get("metadata") or {},
        )

    def filter_mask(
        self, timestamp_cutoff: str | None = None, exclude_conversation_ids: set[int] | None = None
    ) -> np.ndarray:
        mask = np.ones(self.n(), dtype=bool)
        if timestamp_cutoff:
            ts = np.array([c.get("customer_timestamp", "") for c in self.cases], dtype="object")
            mask &= ts <= timestamp_cutoff
        if exclude_conversation_ids:
            cids = np.array([int(c["conversation_id"]) for c in self.cases])
            mask &= ~np.isin(cids, list(exclude_conversation_ids))
        return mask


def assert_no_golden_overlap(indexed_conversation_ids, golden: pd.DataFrame | None = None) -> None:
    """Fail loudly if any indexed conversation belongs to the golden benchmark."""
    from .. import config

    if golden is None:
        if (config.DATA_DIR / "golden" / "amazon_golden_eval.jsonl").exists():
            golden_ids = {
                json.loads(l)["conversation_id"]
                for l in (config.DATA_DIR / "golden" / "amazon_golden_eval.jsonl").open()
            }
        else:
            golden_ids = set()
    else:
        golden_ids = set(golden["conversation_id"])
    overlap = set(indexed_conversation_ids) & golden_ids
    if overlap:
        raise RuntimeError(
            f"GOLDEN LEAK: {len(overlap)} golden conversation(s) entered the "
            f"retrieval corpus, e.g. {sorted(overlap)[:5]}. Index is corrupt."
        )


def load_holdout_ids() -> set[int]:
    from .. import config

    p = config.DATA_DIR / "golden" / "conversation_holdout.json"
    return set(int(x) for x in json.loads(p.read_text())["conversation_ids"])


def forensics(cases: list[dict]) -> dict:
    rows = pd.DataFrame(cases)
    n = len(rows)
    has_resp = rows["has_usable_response"]
    with_resp = rows[has_resp]

    def rate(key: str) -> float:
        if len(with_resp) == 0:
            return 0.0
        return round(100 * float(with_resp["metadata"].map(lambda m: bool(m.get(key, False))).mean()), 2)

    stats = {
        "total_cases": n,
        "cases_with_usable_response": int(has_resp.sum()),
        "pct_with_usable_response": round(100 * float(has_resp.mean()), 2),
        "distinct_conversations": int(rows["conversation_id"].nunique()),
        "avg_conv_length": round(float(rows["conv_length"].mean()), 2),
        "median_conv_length": int(rows["conv_length"].median()),
        "avg_response_chars": round(float(with_resp["brand_response"].str.len().mean()), 1),
        "median_response_chars": int(with_resp["brand_response"].str.len().median()),
        "pct_first_message": round(100 * float(rows["position_in_conversation"].eq(0).mean()), 2),
        "language_dist": {
            k: int(v) for k, v in rows["language"].value_counts().head(12).items()
        },
        "unique_customer_messages": int(rows["customer_message"].nunique()),
        "unique_responses": int(with_resp["brand_response"].nunique()),
        "unique_exact_pairs": int(with_resp.groupby(["customer_message", "brand_response"]).ngroups),
        "canned_template_rate": round(100 * float(with_resp["is_canned_template"].mean()), 2),
        "low_content_rate": round(
            100 * float(rows["customer_message"].str.replace("@USER", "").str.strip().str.len().lt(8).mean()), 2
        ),
        "top_dup_messages": {
            k: int(v) for k, v in rows["customer_message"].value_counts().head(6).items()
        },
        "feature_rates": {
            k: rate(k)
            for k in [
                "asks_dm", "contains_any_url", "contains_help_url",
                "contains_support_link", "contains_tracking_url",
                "contains_phone", "contains_instruction",
                "contains_apology", "contains_acknowledgement",
                "account_or_privacy", "claims_resolved",
            ]
        },
    }
    return stats
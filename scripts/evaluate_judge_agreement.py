"""LLM-judge vs human agreement study (deliverable: evidence the judge is trustworthy).

For a sample of golden rows we judge TWO texts with the live LLM (Gemini,
OpenAI fallback), twice for the human reference reply (self-consistency):

  1. the human-written reference reply (`golden_reply`),
  2. the agent's draft for the same query.

We then report, on the *judged* rows (rate-limited skips are excluded):

  * ``judge_human_helpful_rate``      — judge sanity: how often a human-written
                                       resolution is judged helpful (should be
                                       high; low values indict the judge)
  * ``judge_draft_{helpful,grounded,...}`` — same rubric applied to agent drafts
  * ``human_draft_helpful_gap``       — helpful_rate(human) - helpful_rate(draft)
                                       (positive = judge prefers the human reply)
  * ``repeat_self_agreement``         — % of identical decisions across the two
                                       repeated judgments of each human reply
  * ``deterministic_agreement``       — 2x2 + Cohen's kappa between the judge's
                                       ``grounded_in_retrieval`` and a
                                       reference-derived oracle (token overlap
                                       >= 2 between draft and human reply),
                                       a human-ground-truth-adjacent signal.

Usage:
    python scripts/evaluate_judge_agreement.py                # sample=15
    python scripts/evaluate_judge_agreement.py --sample 25
    Python scripts/evaluate_judge_agreement.py --dump-rows reports/judge_agreement_rows.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "helpful": {"type": "boolean"},
        "grounded_in_retrieval": {"type": "boolean"},
        "hallucination_free": {"type": "boolean"},
        "notes": {"type": "string"},
    },
}


def _bool(j: dict, *keys: str) -> bool:
    for k in keys:
        if isinstance(j.get(k), bool):
            return j[k]
    score = j.get("score")
    if isinstance(score, (int, float)):
        return bool(score >= (j.get("max_score") or 100) * 0.6)
    return False


def _judge(router, system_prompt: str, user_prompt: str) -> dict | None:
    from src.llm import LLMError  # noqa: PLC0415
    try:
        res = router.complete_structured(
            system_prompt=system_prompt, user_prompt=user_prompt,
            response_schema=JUDGE_SCHEMA)
    except LLMError as exc:
        print(f"  judge skip: {str(exc)[:160]}", flush=True)
        return None
    time.sleep(11)  # stay under Gemini free-tier (~5 req/min)
    return res.parsed or None


def _evid(evidence: list) -> str:
    parts = [h.get("brand_response") for h in evidence if isinstance((v := h.get("brand_response")), str) and v]
    return " || ".join(parts)[:1400]


def run(sample: int, seed: int) -> dict:
    from src.llm import make_router  # noqa: PLC0415
    router = make_router()
    if not router.enabled():
        return {"status": "disabled", "note": "no LLM provider key set"}

    rows = json.loads((ROOT / "reports" / "agent_rows.json").read_text())
    rows = [r for r in rows if isinstance(r.get("golden_reply"), str) and r["golden_reply"].strip()]
    picked = random.Random(seed).sample(rows, min(sample, len(rows)))

    judged_rows = []
    for r in picked:
        user_human = (
            f"Customer message:\n{r['query']}\n\n"
            f"SUPPORT REPLY written by a human specialist:\n{r['golden_reply']}\n\n"
            "Grade the human reply. Output ONLY the schema fields: "
            "helpful (did it resolve/target the concern?), "
            "grounded_in_retrieval (is it consistent with the customer's facts?), "
            "hallucination_free (does it invent order ids, promises, fees?), "
            "notes (2-4 words).")
        a = _judge(router, "You grade a support reply. Reply with JSON only.", user_human)
        b = _judge(router, "You grade a support reply. Reply with JSON only.", user_human)  # repeat

        user_draft = (
            f"Customer message:\n{r['query']}\n\n"
            f"DRAFT reply (agent):\n{r['draft']}\n\n"
            "Retrieved historical resolutions:\n" + _evid(r["evidence"]) + "\n\n"
            "Grade the DRAFT. Output ONLY the schema fields: helpful, "
            "grounded_in_retrieval (uses the retrieved resolutions / customer facts), "
            "hallucination_free, notes.")
        d = _judge(router, "You grade a support agent draft. Reply with JSON only.", user_draft)

        if a is None and b is None and d is None:
            continue
        judged_rows.append({
            "example_id": r["example_id"],
            "language": r.get("language"),
            "human_helpful": a["helpful"] if a else None,
            "human_helpful_repeat": b["helpful"] if b else None,
            "draft_helpful": d["helpful"] if d else None,
            "draft_grounded": d["grounded_in_retrieval"] if d else None,
            "draft_hallucination_free": d["hallucination_free"] if d else None,
            "draft_hallucinates": (not d["hallucination_free"]) if d else None,
            "det_grounded": _deterministic_grounded(r["draft"], r["golden_reply"]),
        })
        print(f"  ok {judged_rows[-1]['example_id']}", flush=True)

    summary = _summarize(judged_rows)
    summary.update({
        "status": "ran",
        "sample_requested": sample,
        "rows_judged": len(judged_rows),
        "rows_available": len(rows),
        "provider": router.last_attempted[0] if router.last_attempted else [],
    })
    return {"summary": summary, "rows": judged_rows}


def _deterministic_grounded(draft: str, golden_reply: str) -> bool:
    def _tokens(s: str) -> set[str]:
        return {t for t in s.lower().split() if t.isalnum() and len(t) > 3}
    return len(_tokens(draft) & _tokens(golden_reply)) >= 2


def _cohen_kappa(obs: list[tuple[bool, bool]]) -> float | None:
    n = len(obs)
    if n == 0:
        return None
    a = sum(1 for x, y in obs if x and y)          # both yes
    d = sum(1 for x, y in obs if not x and not y)  # both no
    agree = (a + d) / n
    p1 = sum(1 for x, _ in obs if x) / n
    p2 = sum(1 for _, y in obs if y) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    if pe == 1.0:
        return None
    return round((agree - pe) / (1 - pe), 4)


def _summarize(judged: list[dict]) -> dict:
    n = len(judged)
    if n == 0:
        return {"note": "no rows judged (rate limit)"}

    def rate(key: str) -> float | None:
        vals = [x[key] for x in judged if x.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    repeated = [(x["human_helpful"], x["human_helpful_repeat"]) for x in judged
                if x.get("human_helpful") is not None and x.get("human_helpful_repeat") is not None]

    det = [(x["det_grounded"], x["draft_grounded"]) for x in judged
           if x.get("draft_grounded") is not None]

    summary = {
        "n_rows_judged": n,
        "judge_human_helpful_rate": rate("human_helpful"),
        "judge_repeat_self_agreement": (
            round(sum(1 for x, y in repeated if x == y) / max(1, len(repeated)), 4) if repeated else None),
        "repeat_pairs": len(repeated),
        "judge_draft_helpful_rate": rate("draft_helpful"),
        "judge_draft_grounded_rate": rate("draft_grounded"),
        "judge_draft_hallucination_free_rate": rate("draft_hallucination_free"),
        "human_draft_helpful_gap": (
            round((rate("human_helpful") or 0.0) - (rate("draft_helpful") or 0.0), 4)),
        "judge_vs_deterministic_grounded_kappa": _cohen_kappa(det),
        "judge_vs_deterministic_pairs": len(det),
        "det_grounded_rate": round(sum(1 for x, _ in det if x) / max(1, len(det)), 4) if det else None,
        "sampling_note": "reference-grounded 'det_grounded' is token-overlap>=2 between agent draft and the "
                         "human reference reply; 'judge_vs_deterministic_*' compares the LLM judge's "
                         "grounded_in_retrieval against it on the same rows (a human-ground-truth proxy, "
                         "since per-row draft quality labels are not otherwise available).",
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump-rows")
    args = ap.parse_args()

    out = run(args.sample, args.seed)
    if out.get("status") == "disabled":
        print("LLM disabled — set GEMINI_API_KEY/OPENAI_API_KEY (see .env.example).")
        return 1

    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    summary = out["summary"]
    (report_dir / "judge_agreement.json").write_text(json.dumps(summary, indent=2))
    with (report_dir / "judge_agreement.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("\nJudge agreement summary")
    for k, v in summary.items():
        print(f"  {k:44}: {v}")

    if args.dump_rows:
        Path(args.dump_rows).write_text(json.dumps(out["rows"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
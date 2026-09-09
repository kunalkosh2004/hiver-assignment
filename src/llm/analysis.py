"""Optional LLM-assisted classification job.

The outputs are BEST-EFFORT LLM judgements, stored separately under
`reports/llm_analysis/`, explicitly marked as NOT ground truth. This module is
never required for the deterministic baselines and makes no network calls
unless a router is enabled AND `max_calls_per_run` allows.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .base import LLMError
from .router import LLMRouter
from .schemas import CLASSIFY_SYSTEM_PROMPT, INTENT_SCHEMA

DEFAULT_MAX_CALLS_PER_RUN = 200


def classify_requests(
    router: LLMRouter,
    items: list[dict],
    *,
    max_calls_per_run: int = DEFAULT_MAX_CALLS_PER_RUN,
    out_dir: Path,
    temperature: float = 0.0,
    timeout: float = 30.0,
    sleep_between: float = 0.0,
) -> dict:
    """Classify a list of `{prompt: str}` items, storing results on disk.

    Never exceeds `max_calls_per_run`. Returns a summary dict; parity with
    deterministic labels is NOT implied (LLM-assisted, not ground truth).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not router.enabled():
        return {"enabled": False, "reason": "no LLM provider configured", "n": 0}

    results = []
    calls = 0
    t0 = time.time()
    for it in items:
        if calls >= max_calls_per_run:
            break
        prompt = it["prompt"]
        calls += 1
        try:
            resp = router.complete_structured(
                CLASSIFY_SYSTEM_PROMPT, prompt, INTENT_SCHEMA,
                temperature=temperature, timeout=timeout)
            results.append({
                "id": it.get("id", calls),
                "status": "ok",
                "provider": resp.provider,
                "parsed": resp.parsed,
            })
        except (LLMError, Exception) as exc:  # noqa: BLE001
            results.append({"id": it.get("id", calls), "status": "error", "error": str(exc)})
        if sleep_between:
            time.sleep(sleep_between)

    manifest = {
        "generated_by": "src.llm.analysis.classify_requests",
        "marker": "LLM-ASSISTED OUTPUT — NOT GROUND TRUTH; NOT used to train deterministic baselines",
        "n_items": len(items),
        "n_calls": calls,
        "n_ok": sum(1 for r in results if r["status"] == "ok"),
        "providers_used": sorted({r.get("provider") for r in results if r.get("provider")}),
        "elapsed_s": round(time.time() - t0, 2),
        "results": results,
    }
    (out_dir / "llm_classification_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    return {
        "enabled": True,
        "n_calls": calls,
        "n_ok": manifest["n_ok"],
        "manifest": str(out_dir / "llm_classification_manifest.json"),
    }
"""Time the deterministic offline reproduction end-to-end (Phase D deliverable).

Runs the exact README "Reproduce offline" commands against warm caches and
writes reports/reproduction_timing.json so the <=15-min claim is measured, not
assumed. Uses ``./.venv/bin/python``; long-running and safe to re-run.

Usage:
    python scripts/time_reproduction.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = root_venv = None
AGENT_RESULTS = ROOT / "reports" / "agent_results.json"


def run(step: str, *args: str) -> float:
    started = time.time()
    print(f"[{step}] running {args} ...", flush=True)
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT,
                   capture_output=True, text=True)
    elapsed = time.time() - started
    print(f"[{step}] {elapsed:.1f}s", flush=True)
    return round(elapsed, 2)


def main() -> int:
    global PY
    PY = [sys.executable]
    backup = AGENT_RESULTS.with_suffix(".json.timing_backup")
    if AGENT_RESULTS.exists():
        shutil.copy2(AGENT_RESULTS, backup)
    steps = [
        ("train_agent_intent", "scripts/train_agent_intent.py"),
        ("calibrate_escalation", "scripts/calibrate_escalation.py"),
        ("evaluate_agent", "scripts/evaluate_agent.py", "--dump-rows", "reports/agent_rows.json", "--llm-sample", "0"),
        ("analyze_agent_errors", "scripts/analyze_agent_errors.py", "reports/agent_rows.json"),
        ("build_escalation_benchmark", "scripts/build_escalation_benchmark.py"),
        ("evaluate_escalation", "scripts/evaluate_escalation.py", "reports/agent_rows.json"),
        ("consolidate_benchmark", "scripts/consolidate_benchmark.py"),
    ]
    timings = {}
    try:
        for step, script, *extra in steps:
            timings[step] = run(step, script, *extra)
    finally:
        if backup.exists():
            shutil.copy2(backup, AGENT_RESULTS)   # restore committed report
            backup.unlink()

    started = time.time()
    print("[unit_tests] running discover ...", flush=True)
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                   check=True, cwd=ROOT, capture_output=True, text=True)
    timings["unit_tests"] = round(time.time() - started, 2)

    out = {
        "env": {"cwd": str(ROOT), "interpreter": sys.executable},
        "cache_note": ("warm caches (data/retrieval/* and data/*.parquet prebuilt); "
                       "a cold run additionally pays the corpus-index build (build_retrieval_index.py) "
                       "and parquet cache build, documented separately."),
        "steps_seconds": timings,
        "total_seconds": round(sum(timings.values()), 2),
        "total_minutes": round(sum(timings.values()) / 60, 2),
    }
    (ROOT / "reports" / "reproduction_timing.json").write_text(json.dumps(out, indent=2))
    print(f"\nTOTAL offline reproduction (warm): {out['total_minutes']} min "
          f"({out['total_seconds']} s); steps: {timings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
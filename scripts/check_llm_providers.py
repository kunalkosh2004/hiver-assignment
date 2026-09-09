"""Optional LLM provider health check (no API credits consumed).

Usage:
    python scripts/check_llm_providers.py
    GEMINI_API_KEY=... python scripts/check_llm_providers.py

Reports which providers are configured (key present + package importable).
Does NOT call any model — you must opt into that separately. Non-zero exit if
both providers are unusable.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm import make_router  # noqa: E402


def main() -> int:
    router = make_router()
    print("LLM router config")
    print(f"  primary  : {type(router.primary).__name__}")
    print(f"  fallback : {type(router.fallback).__name__}")
    print(f"  enabled  : {router.enabled()}")
    for name in ("GEMINI_API_KEY", "OPENAI_API_KEY"):
        print(f"  {name:14}: {'SET' if os.getenv(name) else 'not set'}")
    for prov in (router.primary, router.fallback):
        print(f"  {prov.name:<10}: available={prov.available()}")
    if not router.enabled():
        print("\nLLM disabled — set GEMINI_API_KEY and/or OPENAI_API_KEY (see .env.example).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
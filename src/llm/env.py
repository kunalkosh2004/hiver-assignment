"""Tiny dependency-free .env loader.

Loads KEY=VALUE pairs from the project-root `.env` into ``os.environ``
(only if not already exported), no python-dotenv required. Comments and blank
lines are skipped; values are used verbatim after stripping one layer of
single/double quotes.
"""
from __future__ import annotations

import os
from pathlib import Path

from .. import config


def load_env(path: Path | str | None = None) -> dict[str, str]:
    path = Path(path) if path is not None else config.PROJECT_ROOT / ".env"
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded
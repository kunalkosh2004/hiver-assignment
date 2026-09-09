"""Project-wide configuration for the Hiver AI support agent assignment."""
from __future__ import annotations

from pathlib import Path

# Project root = one level up from the src/ package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw data location (project-relative, not machine-specific).
DATA_DIR = PROJECT_ROOT / "data"
TWCS_CSV = DATA_DIR / "twcs.csv"

# A prepared cache of the raw CSV (faster repeat loads). Built on first use.
TWCS_PARQUET = DATA_DIR / "twcs.parquet"

# Deterministic sampling seed used anywhere sampling is required.
RANDOM_SEED = 42


def data_path(name: str) -> Path:
    """Return an absolute path under the project data/ directory."""
    return DATA_DIR / name

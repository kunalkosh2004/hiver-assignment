"""Loading helpers for the Customer Support on Twitter dataset.

Prefers a parquet cache for fast repeat loads; falls back to a chunked CSV
read if the cache is absent. Column dtypes are fixed to be stable across
platforms so downstream joins never silently change types.
"""
from __future__ import annotations

import pandas as pd

from . import config


DTYPES: dict[str, str] = {
    "tweet_id": "int64",
    "author_id": "string",
    "inbound": "bool",
    "created_at": "string",
    "text": "string",
    "response_tweet_id": "string",  # comma-separated list; keep as string
    "in_response_to_tweet_id": "string",  # may be missing
}


def load_twcs(use_cache: bool = True, columns: list[str] | None = None) -> pd.DataFrame:
    """Load the full tweet corpus.

    Parameters
    ----------
    use_cache:
        If True (default), build/use a parquet cache in ``data/``. This makes
        repeat loads fast without re-parsing the ~500 MB CSV.
    columns:
        Optional subset of columns to load, to reduce memory for aggregations.

    If the primary CSV is missing, raise a clear error pointing at the
    download script rather than failing cryptically.
    """
    if not config.TWCS_CSV.exists():
        raise FileNotFoundError(
            "data/twcs.csv is missing. Run `python scripts/download_data.py` "
            "first to download the Kaggle dataset."
        )

    if use_cache:
        if config.TWCS_PARQUET.exists():
            return _read_cache(columns)
        _build_cache(columns)

    return pd.read_csv(config.TWCS_CSV, dtype=DTYPES, usecols=columns)


def _build_cache(columns: list[str] | None = None) -> None:
    """Read the CSV in chunks and write a parquet cache once."""
    reader = pd.read_csv(
        config.TWCS_CSV, dtype=DTYPES, chunksize=500_000, usecols=columns
    )
    parts = [chunk for chunk in reader]
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(config.TWCS_PARQUET)


def _read_cache(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(config.TWCS_PARQUET, columns=columns)

"""Download the Customer Support on Twitter dataset into the project data/ dir.

Usage:
    python scripts/download_data.py

This downloads the Kaggle dataset (thoughtvector/customer-support-on-twitter)
into a temp cache via kagglehub, then copies the main CSV into this project's
``data/`` directory so that all further analysis uses a project-relative path.

The dataset is ~500 MB; it is git-ignored so it is not committed to the repo.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

DATASET = "thoughtvector/customer-support-on-twitter"
MAIN_FILE = "twcs/twcs.csv"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Kaggle dataset: {DATASET}")
    src_dir = Path(kagglehub.dataset_download(DATASET))
    src_file = src_dir / MAIN_FILE

    if not src_file.exists():
        raise FileNotFoundError(f"Expected file not found in cache: {src_file}")

    dest = DATA_DIR / "twcs.csv"
    print(f"Copying {src_file} -> {dest}")
    shutil.copyfile(src_file, dest)
    print(f"Done. Data available at {dest}")


if __name__ == "__main__":
    main()

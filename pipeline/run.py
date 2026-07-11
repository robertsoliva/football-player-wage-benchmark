"""Entry point: download → clean → store."""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

from pipeline.ingestion import download_salary_data
from pipeline.cleaning import load_and_clean
from pipeline import storage


def run(force_download: bool = False) -> None:
    raw_path = download_salary_data(force=force_download)
    df = load_and_clean(raw_path)
    storage.load(df, raw_path)
    print(f"Pipeline complete. {len(df)} rows available in wages.db.")


if __name__ == "__main__":
    force = "--force" in sys.argv
    run(force_download=force)

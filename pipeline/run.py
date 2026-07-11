"""Entry point: fetch → clean → store."""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

from pipeline.ingestion import fetch_salary_data
from pipeline.cleaning import clean
from pipeline import storage


def run(force: bool = False) -> None:
    raw_df, source_label = fetch_salary_data(force=force)
    df = clean(raw_df, source_label)
    storage.load(df, source_label=source_label)
    print(f"Pipeline complete. Source: {source_label}. {len(df)} rows in wages.db.")


if __name__ == "__main__":
    force = "--force" in sys.argv
    run(force=force)

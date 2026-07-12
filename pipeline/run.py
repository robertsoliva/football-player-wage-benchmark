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
from algorithm.benchmark import _load_ss_data, _enrich_with_market_values


def run(force: bool = False) -> None:
    raw_df, source_label = fetch_salary_data(force=force)
    df = clean(raw_df, source_label)
    ss_df = _load_ss_data()
    df = _enrich_with_market_values(df, ss_df)
    storage.load(df, source_label=source_label)
    matched = (df["market_value_eur"] > 0).sum()
    print(
        f"Pipeline complete. Source: {source_label}. {len(df)} rows in wages.db "
        f"({matched} enriched with market values)."
    )


if __name__ == "__main__":
    force = "--force" in sys.argv
    run(force=force)

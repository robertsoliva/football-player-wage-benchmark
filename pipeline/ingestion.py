"""
Orchestrates salary data acquisition:
  1. Attempt Capology scrape (real-world estimates).
  2. If that fails, fall back to EA FC 25 dataset via Kaggle.

The caller receives a DataFrame in the canonical salary schema regardless
of which source was used. A 'source' column records the origin.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
KAGGLE_DATASET = "stefanoleone992/ea-sports-fc-25-complete-player-dataset"


class IngestionError(Exception):
    pass


def fetch_salary_data(force: bool = False) -> tuple[pd.DataFrame, str]:
    """
    Return (df, source_label) where source_label is 'capology' or 'ea_fc25'.

    Args:
        force: Bypass all caches and re-download from source.
    """
    from pipeline.scraping import scrape_all_leagues, ScrapingError

    try:
        df = scrape_all_leagues(force=force)
        logger.info("Using Capology as salary data source.")
        return df, "capology"
    except ScrapingError as exc:
        logger.warning(
            "Capology scrape failed (%s). Falling back to EA FC 25 dataset.", exc
        )

    df = _download_ea_fc25(force=force)
    return df, "ea_fc25"


# ── EA FC 25 fallback ─────────────────────────────────────────────────────────

def _download_ea_fc25(force: bool = False) -> pd.DataFrame:
    target = RAW_DIR / "ea_fc25_players.csv"
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if target.exists() and not force:
        logger.info("EA FC 25 raw file already present, skipping download.")
        return pd.read_csv(target, low_memory=False)

    logger.info("Downloading EA FC 25 dataset from Kaggle…")
    try:
        import kagglehub
        dataset_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as exc:
        raise IngestionError(
            f"EA FC 25 fallback also failed — cannot download from Kaggle: {exc}. "
            "Ensure ~/.kaggle/kaggle.json exists or set KAGGLE_USERNAME + KAGGLE_KEY."
        ) from exc

    candidates = sorted(dataset_path.rglob("male_players*.csv"))
    if not candidates:
        candidates = sorted(dataset_path.rglob("*.csv"))
    if not candidates:
        raise IngestionError(f"No CSV found in downloaded Kaggle dataset at {dataset_path}")

    source = max(candidates, key=lambda p: p.stat().st_size)
    import shutil
    shutil.copy(source, target)
    logger.info("EA FC 25 raw data saved to %s", target)
    return pd.read_csv(target, low_memory=False)

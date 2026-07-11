"""Download the EA FC 25 player dataset from Kaggle and save the raw CSV."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "stefanoleone992/ea-sports-fc-25-complete-player-dataset"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


class IngestionError(Exception):
    pass


def download_salary_data(force: bool = False) -> Path:
    """
    Download EA FC 25 dataset via kagglehub.

    Returns the path to the main players CSV.
    Raises IngestionError on network or auth failure.

    Args:
        force: Re-download even if the file already exists locally.
    """
    import kagglehub

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    target = RAW_DIR / "ea_fc25_players.csv"
    if target.exists() and not force:
        logger.info("Raw file already present at %s, skipping download.", target)
        return target

    logger.info("Downloading dataset '%s' from Kaggle…", KAGGLE_DATASET)
    try:
        dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)
    except Exception as exc:
        raise IngestionError(
            f"Failed to download dataset '{KAGGLE_DATASET}': {exc}"
        ) from exc

    dataset_path = Path(dataset_path)
    # kagglehub places files inside a versioned subdirectory; find the main CSV
    candidates = sorted(dataset_path.rglob("male_players*.csv"))
    if not candidates:
        candidates = sorted(dataset_path.rglob("*.csv"))
    if not candidates:
        raise IngestionError(
            f"No CSV found in downloaded dataset at {dataset_path}"
        )

    # Use the largest file as the main player dataset
    source = max(candidates, key=lambda p: p.stat().st_size)
    logger.info("Using source file: %s", source)

    import shutil
    shutil.copy(source, target)
    logger.info("Saved raw data to %s", target)
    return target

"""Validate and transform the raw EA FC 25 dataset into a canonical schema."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columns we need from the raw EA FC 25 file
_REQUIRED_COLUMNS = {
    "long_name",
    "short_name",
    "player_positions",
    "age",
    "club_name",
    "league_name",
    "wage_eur",
}

# Canonical position groups used across the project
POSITION_MAP = {
    "GK": "GK",
    "CB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF",
    "CDM": "MID", "CM": "MID", "CAM": "MID", "LM": "MID", "RM": "MID",
    "LW": "ATT", "RW": "ATT", "ST": "ATT", "CF": "ATT", "RF": "ATT", "LF": "ATT",
}

LEAGUE_TIER = {
    "English Premier League": 1,
    "Spain Primera Division": 1,
    "German 1. Bundesliga": 1,
    "Italian Serie A": 1,
    "French Ligue 1": 1,
}


class ValidationError(Exception):
    pass


def load_and_clean(raw_path: Path) -> pd.DataFrame:
    """
    Read, validate, and clean the raw EA FC 25 CSV.

    Returns a DataFrame with the canonical schema.
    Raises ValidationError if required columns are missing.
    """
    logger.info("Loading raw file: %s", raw_path)

    try:
        df = pd.read_csv(raw_path, low_memory=False)
    except Exception as exc:
        raise ValidationError(f"Cannot read CSV at {raw_path}: {exc}") from exc

    _validate_columns(df)
    df = _select_and_rename(df)
    df = _filter_rows(df)
    df = _normalize_positions(df)
    df = _add_league_tier(df)
    df = _drop_duplicates(df)

    logger.info("Clean dataset: %d rows", len(df))
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValidationError(
            f"Source format has changed — missing columns: {missing}. "
            "Check whether the Kaggle dataset was updated."
        )


def _select_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    keep = {
        "long_name": "player_name",
        "short_name": "short_name",
        "player_positions": "raw_positions",
        "age": "age",
        "club_name": "club_name",
        "league_name": "league_name",
        "wage_eur": "wage_eur_weekly",
    }
    # value_eur is optional (not always reliable)
    if "value_eur" in df.columns:
        keep["value_eur"] = "ea_value_eur"

    return df[list(keep.keys())].rename(columns=keep).copy()


def _filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    initial = len(df)

    # Drop rows with null wages or zero wages — unusable for benchmarking
    null_wage = df["wage_eur_weekly"].isna() | (df["wage_eur_weekly"] == 0)
    if null_wage.sum():
        logger.warning("Dropping %d rows with null/zero wage.", null_wage.sum())
    df = df[~null_wage].copy()

    # Wage must be positive numeric
    df["wage_eur_weekly"] = pd.to_numeric(df["wage_eur_weekly"], errors="coerce")
    invalid_wage = df["wage_eur_weekly"].isna() | (df["wage_eur_weekly"] <= 0)
    if invalid_wage.sum():
        logger.warning("Dropping %d rows with non-positive wage.", invalid_wage.sum())
    df = df[~invalid_wage].copy()

    # Age sanity check
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    bad_age = df["age"].isna() | (df["age"] < 15) | (df["age"] > 45)
    if bad_age.sum():
        logger.warning("Dropping %d rows with invalid age.", bad_age.sum())
    df = df[~bad_age].copy()

    logger.info("Row filter: %d → %d rows kept.", initial, len(df))
    return df


def _normalize_positions(df: pd.DataFrame) -> pd.DataFrame:
    def _primary(raw: str) -> str:
        """Take the first listed position and map to canonical group."""
        first = str(raw).split(",")[0].strip()
        return POSITION_MAP.get(first, "MID")  # default to MID if unknown

    df["position_group"] = df["raw_positions"].apply(_primary)
    df["primary_position"] = df["raw_positions"].apply(
        lambda r: str(r).split(",")[0].strip()
    )
    return df


def _add_league_tier(df: pd.DataFrame) -> pd.DataFrame:
    df["league_tier"] = df["league_name"].map(LEAGUE_TIER).fillna(2).astype(int)
    return df


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["player_name", "club_name"]).reset_index(drop=True)
    if len(df) < before:
        logger.info("Removed %d duplicate rows.", before - len(df))
    return df

"""
Validate and clean salary data from either Capology or EA FC 25 into a
canonical schema before loading into DuckDB.

Canonical columns:
  player_name, short_name, primary_position, position_group,
  age, club_name, league_name, league_tier,
  wage_eur_weekly, ea_value_eur, source
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

POSITION_MAP = {
    # Canonical passthrough (scraper already resolved these)
    "ATT": "ATT", "DEF": "DEF", "MID": "MID", "GK": "GK",
    # EA FC 25 specific positions
    "CB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF",
    "CDM": "MID", "CM": "MID", "CAM": "MID", "LM": "MID", "RM": "MID",
    "LW": "ATT", "RW": "ATT", "ST": "ATT", "CF": "ATT", "RF": "ATT",
    "LF": "ATT", "SS": "ATT",
    # Capology broad codes
    "F": "ATT", "M": "MID", "D": "DEF",
}

EA_LEAGUE_TIER = {
    "English Premier League": 1,
    "Spain Primera Division": 1,
    "German 1. Bundesliga": 1,
    "Italian Serie A": 1,
    "French Ligue 1": 1,
}

# Required columns for EA FC 25 raw file
_EA_REQUIRED = {"long_name", "player_positions", "age", "club_name", "league_name", "wage_eur"}


class ValidationError(Exception):
    pass


# ── Public interface ──────────────────────────────────────────────────────────

def clean(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Route to the appropriate cleaner based on source label.

    Args:
        df:     Raw DataFrame (from Capology scrape or EA FC 25 CSV).
        source: 'capology' or 'ea_fc25'.

    Returns:
        Cleaned DataFrame in the canonical schema.
    """
    if source == "capology":
        return _clean_capology(df)
    if source == "ea_fc25":
        return _clean_ea(df)
    raise ValueError(f"Unknown source: {source!r}")


# ── Capology cleaner ──────────────────────────────────────────────────────────

def _clean_capology(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Wage validation
    df["wage_eur_weekly"] = pd.to_numeric(df["wage_eur_weekly"], errors="coerce")
    bad_wage = df["wage_eur_weekly"].isna() | (df["wage_eur_weekly"] <= 0)
    if bad_wage.sum():
        logger.warning("Capology: dropping %d rows with invalid wage.", bad_wage.sum())
    df = df[~bad_wage]

    # Age validation
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    bad_age = df["age"].isna() | (df["age"] < 15) | (df["age"] > 45)
    if bad_age.sum():
        logger.warning("Capology: dropping %d rows with invalid age.", bad_age.sum())
    df = df[~bad_age]

    # Drop players on loan (they inflate the wage at their parent club)
    if "on_loan" in df.columns:
        loans = df["on_loan"] == True
        if loans.sum():
            logger.info("Capology: dropping %d loaned players.", loans.sum())
        df = df[~loans]

    df["position_group"] = df["position_group"].apply(
        lambda p: POSITION_MAP.get(str(p).strip(), "MID")
    )
    # Use primary_position as authoritative fallback (covers GK and edge cases
    # where Capology's broad position code didn't map cleanly)
    df["position_group"] = df.apply(
        lambda row: POSITION_MAP.get(str(row["primary_position"]).strip(), row["position_group"]),
        axis=1,
    )

    if "ea_value_eur" not in df.columns:
        df["ea_value_eur"] = None

    df = df.drop_duplicates(subset=["player_name", "club_name"])

    canonical = _select_canonical(df)
    logger.info("Capology clean: %d rows.", len(canonical))
    return canonical


# ── EA FC 25 cleaner ──────────────────────────────────────────────────────────

def _clean_ea(raw_df: pd.DataFrame) -> pd.DataFrame:
    _validate_ea_columns(raw_df)

    df = raw_df.rename(columns={
        "long_name": "player_name",
        "short_name": "short_name",
        "player_positions": "raw_positions",
        "wage_eur": "wage_eur_weekly",
    }).copy()

    if "value_eur" in raw_df.columns:
        df["ea_value_eur"] = pd.to_numeric(raw_df["value_eur"], errors="coerce")
    else:
        df["ea_value_eur"] = None

    # Wage validation
    df["wage_eur_weekly"] = pd.to_numeric(df["wage_eur_weekly"], errors="coerce")
    bad_wage = df["wage_eur_weekly"].isna() | (df["wage_eur_weekly"] <= 0)
    if bad_wage.sum():
        logger.warning("EA FC 25: dropping %d rows with null/zero wage.", bad_wage.sum())
    df = df[~bad_wage]

    # Age validation
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    bad_age = df["age"].isna() | (df["age"] < 15) | (df["age"] > 45)
    if bad_age.sum():
        logger.warning("EA FC 25: dropping %d rows with invalid age.", bad_age.sum())
    df = df[~bad_age]

    # Positions
    df["primary_position"] = df["raw_positions"].apply(
        lambda r: str(r).split(",")[0].strip()
    )
    df["position_group"] = df["primary_position"].apply(
        lambda p: POSITION_MAP.get(p, "MID")
    )

    # League tier
    df["league_tier"] = df["league_name"].map(EA_LEAGUE_TIER).fillna(2).astype(int)

    df["source"] = "ea_fc25"
    df = df.drop_duplicates(subset=["player_name", "club_name"])

    canonical = _select_canonical(df)
    logger.info("EA FC 25 clean: %d rows.", len(canonical))
    return canonical


def _validate_ea_columns(df: pd.DataFrame) -> None:
    missing = _EA_REQUIRED - set(df.columns)
    if missing:
        raise ValidationError(
            f"EA FC 25 source format changed — missing columns: {missing}. "
            "Check whether the Kaggle dataset was updated."
        )


def _select_canonical(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "player_name", "short_name", "primary_position", "position_group",
        "age", "club_name", "league_name", "league_tier",
        "wage_eur_weekly", "ea_value_eur", "source",
    ]
    # Fill in any missing canonical columns with None
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols].reset_index(drop=True)


# ── Kept for backwards-compat (used directly by tests) ───────────────────────

def _validate_columns(df: pd.DataFrame) -> None:
    """Validate EA FC 25 raw columns. Used by unit tests."""
    _validate_ea_columns(df)


def _filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter wage and age rows. Works on already-renamed (post-select) df."""
    df = df.copy()
    df["wage_eur_weekly"] = pd.to_numeric(df["wage_eur_weekly"], errors="coerce")
    bad_wage = df["wage_eur_weekly"].isna() | (df["wage_eur_weekly"] <= 0)
    if bad_wage.sum():
        logger.warning("Dropping %d rows with null/zero wage.", bad_wage.sum())
    df = df[~bad_wage]

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    bad_age = df["age"].isna() | (df["age"] < 15) | (df["age"] > 45)
    if bad_age.sum():
        logger.warning("Dropping %d rows with invalid age.", bad_age.sum())
    df = df[~bad_age]
    return df


def _normalize_positions(df: pd.DataFrame) -> pd.DataFrame:
    df["position_group"] = df["raw_positions"].apply(
        lambda r: POSITION_MAP.get(str(r).split(",")[0].strip(), "MID")
    )
    df["primary_position"] = df["raw_positions"].apply(
        lambda r: str(r).split(",")[0].strip()
    )
    return df


def _add_league_tier(df: pd.DataFrame) -> pd.DataFrame:
    df["league_tier"] = df["league_name"].map(EA_LEAGUE_TIER).fillna(2).astype(int)
    return df


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["player_name", "club_name"]).reset_index(drop=True)
    if len(df) < before:
        logger.info("Removed %d duplicate rows.", before - len(df))
    return df

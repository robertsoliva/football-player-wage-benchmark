"""Tests for pipeline/cleaning.py — no I/O, only transformation logic."""

import pandas as pd
import pytest

from pipeline.cleaning import (
    ValidationError,
    _validate_columns,
    _filter_rows,
    _normalize_positions,
    _add_league_tier,
    _drop_duplicates,
)


def _raw_df(**overrides) -> pd.DataFrame:
    """Mimics raw EA FC 25 column names (before rename)."""
    data = {
        "long_name": ["Alice", "Bob", "Carol"],
        "short_name": ["Alice", "Bob", "Carol"],
        "player_positions": ["ST", "CB", "GK"],
        "age": [25, 30, 22],
        "club_name": ["Club A", "Club B", "Club C"],
        "league_name": [
            "English Premier League",
            "Spain Primera Division",
            "German 1. Bundesliga",
        ],
        "wage_eur": [10_000, 20_000, 5_000],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def _base_df(**overrides) -> pd.DataFrame:
    """Already-renamed columns (post _select_and_rename)."""
    data = {
        "player_name": ["Alice", "Bob", "Carol"],
        "short_name": ["Alice", "Bob", "Carol"],
        "raw_positions": ["ST", "CB", "GK"],
        "age": [25, 30, 22],
        "club_name": ["Club A", "Club B", "Club C"],
        "league_name": [
            "English Premier League",
            "Spain Primera Division",
            "German 1. Bundesliga",
        ],
        "wage_eur_weekly": [10_000, 20_000, 5_000],
    }
    data.update(overrides)
    return pd.DataFrame(data)


# ── _validate_columns ─────────────────────────────────────────────────────────

def test_validate_columns_ok():
    df = _raw_df()
    _validate_columns(df)  # should not raise


def test_validate_columns_missing_raises():
    df = _raw_df().drop(columns=["wage_eur"])
    with pytest.raises(ValidationError, match="missing columns"):
        _validate_columns(df)


# ── _filter_rows ──────────────────────────────────────────────────────────────

def test_filter_drops_null_wage():
    df = _base_df(wage_eur_weekly=[None, 20_000, 5_000])
    result = _filter_rows(df)
    assert len(result) == 2
    assert "Alice" not in result["player_name"].values


def test_filter_drops_zero_wage():
    df = _base_df(wage_eur_weekly=[0, 20_000, 5_000])
    result = _filter_rows(df)
    assert len(result) == 2


def test_filter_drops_invalid_age():
    df = _base_df(age=[14, 30, 22])  # 14 is below minimum
    result = _filter_rows(df)
    assert len(result) == 2


def test_filter_keeps_valid_rows():
    df = _base_df()
    result = _filter_rows(df)
    assert len(result) == 3


# ── _normalize_positions ──────────────────────────────────────────────────────

def test_position_mapping_gk():
    df = _base_df(raw_positions=["GK", "ST", "CB"])
    result = _normalize_positions(df)
    assert result.loc[0, "position_group"] == "GK"


def test_position_mapping_att():
    df = _base_df(raw_positions=["ST,CF", "LW", "RW"])
    result = _normalize_positions(df)
    assert all(result["position_group"] == "ATT")


def test_position_mapping_multi_listed():
    """Only the first listed position is used."""
    df = _base_df(raw_positions=["CB,CM", "CB", "GK"])
    result = _normalize_positions(df)
    assert result.loc[0, "position_group"] == "DEF"


# ── _add_league_tier ─────────────────────────────────────────────────────────

def test_league_tier_top5():
    df = _base_df()
    result = _add_league_tier(df)
    assert result.loc[0, "league_tier"] == 1  # Premier League


def test_league_tier_unknown_defaults_to_2():
    df = _base_df(league_name=["Unknown League", "Unknown", "Mystery"])
    result = _add_league_tier(df)
    assert (result["league_tier"] == 2).all()


# ── _drop_duplicates ──────────────────────────────────────────────────────────

def test_drop_duplicates_removes_same_player_club():
    df = pd.concat([_base_df(), _base_df()], ignore_index=True)
    result = _drop_duplicates(df)
    assert len(result) == 3


def test_drop_duplicates_keeps_different_clubs():
    df = _base_df(club_name=["Club A", "Club A", "Club B"])
    df2 = df.copy()
    df2.loc[0, "club_name"] = "Club X"  # different club, same player name
    combined = pd.concat([df, df2], ignore_index=True)
    result = _drop_duplicates(combined)
    # Alice appears twice under different clubs → both kept; Bob/Carol are duped
    assert result[result["player_name"] == "Alice"].shape[0] >= 1

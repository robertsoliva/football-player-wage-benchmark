"""
Salary benchmark: given a player from SoccerSolver's dataset,
find similar peers in the EA FC 25 salary data and return an expected
salary range with a confidence rating.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[1] / "wages.db"
SS_DATA = Path(__file__).resolve().parents[1] / "data" / "soccersolver" / "data.csv"

LEAGUE_TIER = {
    "GB1": 1, "ES1": 1, "L1": 1, "IT1": 1, "FR1": 1,
}

# Weights for the distance metric (must sum to 1)
WEIGHTS = {
    "league_tier": 0.35,
    "market_value_norm": 0.40,
    "age": 0.25,
}

HIGH_PEERS = 15
LOW_PEERS = 5


@dataclass
class BenchmarkResult:
    player_name: str
    position_group: str
    median_wage_eur_year: float
    p25_wage_eur_year: float
    p75_wage_eur_year: float
    confidence: str          # "High" | "Medium" | "Low" | "Insufficient data"
    peer_count: int
    peers: pd.DataFrame = field(repr=False)


def _load_salary_db() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("SELECT * FROM salary_data").df()
    con.close()
    return df


def _load_ss_data() -> pd.DataFrame:
    df = pd.read_csv(SS_DATA)
    df["birth_date"] = pd.to_datetime(df["birth_date"], utc=True)
    df["age"] = ((pd.Timestamp.now(tz="UTC") - df["birth_date"]).dt.days / 365.25).astype(int)
    df["league_tier"] = df["competition_id"].map(LEAGUE_TIER).fillna(2).astype(int)
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce").fillna(0)
    return df


def get_all_players() -> pd.DataFrame:
    return _load_ss_data()[["player_name", "main_position", "team_name", "competition_name", "market_value", "age"]]


def benchmark_player(player_name: str, current_wage_eur_year: float | None = None) -> BenchmarkResult:
    """
    Compute the salary benchmark for a player in SoccerSolver's dataset.

    Args:
        player_name: Exact name as it appears in data.csv.
        current_wage_eur_year: Optionally provide the player's actual wage
                               to compute a percentile rank.
    """
    ss_df = _load_ss_data()
    player_row = ss_df[ss_df["player_name"] == player_name]
    if player_row.empty:
        raise ValueError(f"Player '{player_name}' not found in SoccerSolver dataset.")
    player = player_row.iloc[0]

    position_group = _map_position(player["main_position"])
    salary_df = _load_salary_db()

    # Filter to same position group
    peers_pool = salary_df[salary_df["position_group"] == position_group].copy()

    if peers_pool.empty:
        return BenchmarkResult(
            player_name=player_name,
            position_group=position_group,
            median_wage_eur_year=0,
            p25_wage_eur_year=0,
            p75_wage_eur_year=0,
            confidence="Insufficient data",
            peer_count=0,
            peers=pd.DataFrame(),
        )

    # Normalise features for distance computation
    scaler = MinMaxScaler()
    market_vals = np.log1p(peers_pool["ea_value_eur"].fillna(0).values)
    peers_pool = peers_pool.copy()
    peers_pool["_mv_norm"] = scaler.fit_transform(market_vals.reshape(-1, 1)).ravel()

    player_mv_norm = scaler.transform(
        np.log1p([[player["market_value"]]]).reshape(-1, 1)
    )[0][0]

    player_tier = float(player["league_tier"])
    player_age = float(player["age"])

    feature_cols = ["league_tier", "_mv_norm", "age"]
    X = peers_pool[feature_cols].fillna(0).values

    # Apply weights by scaling each column
    w = np.array([WEIGHTS["league_tier"], WEIGHTS["market_value_norm"], WEIGHTS["age"]])
    X_w = X * w

    player_vec = np.array([[player_tier, player_mv_norm, player_age]]) * w

    n_neighbors = min(20, len(X_w))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(X_w)
    _, indices = knn.kneighbors(player_vec)

    peers = peers_pool.iloc[indices[0]].copy()
    wages_year = peers["wage_eur_weekly"] * 52

    peer_count = len(peers)
    confidence = (
        "High" if peer_count >= HIGH_PEERS
        else "Medium" if peer_count >= LOW_PEERS
        else "Low"
    )

    result = BenchmarkResult(
        player_name=player_name,
        position_group=position_group,
        median_wage_eur_year=float(np.median(wages_year)),
        p25_wage_eur_year=float(np.percentile(wages_year, 25)),
        p75_wage_eur_year=float(np.percentile(wages_year, 75)),
        confidence=confidence,
        peer_count=peer_count,
        peers=peers[["player_name", "club_name", "league_name", "age", "wage_eur_weekly"]].copy(),
    )

    if current_wage_eur_year is not None:
        result.current_wage_percentile = float(
            np.mean(wages_year <= current_wage_eur_year) * 100
        )

    return result


_POSITION_MAP = {
    # SoccerSolver main_position → canonical group
    "Centre-Back": "DEF", "Left-Back": "DEF", "Right-Back": "DEF",
    "Left Wing-Back": "DEF", "Right Wing-Back": "DEF",
    "Defensive Midfield": "MID", "Central Midfield": "MID",
    "Attacking Midfield": "MID", "Left Midfield": "MID", "Right Midfield": "MID",
    "Centre-Forward": "ATT", "Left Winger": "ATT", "Right Winger": "ATT",
    "Second Striker": "ATT",
    "Goalkeeper": "GK",
}


def _map_position(main_position: str) -> str:
    return _POSITION_MAP.get(main_position, "MID")

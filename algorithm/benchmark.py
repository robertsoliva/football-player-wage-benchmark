"""
Salary benchmark: given a player from SoccerSolver's dataset,
find similar peers in the salary database and return an expected
salary range with a confidence rating.

KNN features (all normalised to [0, 1] before weighting):
  - market_value   (weight 0.40) — from SoccerSolver; enriched via fuzzy match
  - league_tier    (weight 0.35) — top-5 league = 1, others = 2
  - age            (weight 0.25) — career-stage proxy

Peers are filtered to the same position group before KNN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from rapidfuzz import process as fuzz_process, fuzz, distance as fuzz_distance
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[1] / "wages.db"
SS_DATA = Path(__file__).resolve().parents[1] / "data" / "soccersolver" / "data.csv"

LEAGUE_TIER = {
    "GB1": 1, "ES1": 1, "L1": 1, "IT1": 1, "FR1": 1,
}

WEIGHTS = {
    "market_value": 0.40,
    "league_tier":  0.35,
    "age":          0.25,
}

HIGH_PEERS = 15
LOW_PEERS = 5
FUZZY_THRESHOLD = 80   # minimum name-match score (0–100) to accept enrichment


@dataclass
class BenchmarkResult:
    player_name: str
    position_group: str
    median_wage_eur_year: float
    p25_wage_eur_year: float
    p75_wage_eur_year: float
    confidence: str           # "High" | "Medium" | "Low" | "Insufficient data"
    peer_count: int
    peers: pd.DataFrame = field(repr=False)
    current_wage_percentile: float | None = None


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_salary_db() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("SELECT * FROM salary_data").df()
    con.close()
    return df


def _load_ss_data() -> pd.DataFrame:
    df = pd.read_csv(SS_DATA)
    df["birth_date"] = pd.to_datetime(df["birth_date"], utc=True, errors="coerce")
    age_days = (pd.Timestamp.now(tz="UTC") - df["birth_date"]).dt.days
    df["age"] = (age_days / 365.25).fillna(25).astype(int)
    df["league_tier"] = df["competition_id"].map(LEAGUE_TIER).fillna(2).astype(int)
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce").fillna(0)
    return df


def get_all_players() -> pd.DataFrame:
    return _load_ss_data()[
        ["player_name", "main_position", "team_name", "competition_name", "market_value", "age"]
    ]


# ── Market-value enrichment via fuzzy match ───────────────────────────────────

def load_enriched_salary_db() -> pd.DataFrame:
    """Load salary data (market_value_eur pre-computed by the pipeline)."""
    return _load_salary_db()


def _enrich_with_market_values(salary_df: pd.DataFrame, ss_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each player in salary_df, try to find a matching player in ss_df by
    fuzzy name match. Populates 'market_value_eur'.

    Uses vectorized cdist calls per tier (all-in-C, no per-row GIL churn)
    which is safe to call from non-main threads (e.g. Streamlit's runner).
    """
    import numpy as np

    salary_df = salary_df.copy()
    market_values = np.zeros(len(salary_df), dtype=float)

    # Process each tier separately so the matrix stays small
    for tier in salary_df["league_tier"].unique():
        sal_mask = salary_df["league_tier"] == tier
        ss_mask = ss_df["league_tier"] == tier

        sal_names = salary_df.loc[sal_mask, "player_name"].tolist()
        ss_sub = ss_df[ss_mask]
        ss_names = ss_sub["player_name"].tolist()
        ss_vals = ss_sub["market_value"].to_numpy()

        if not sal_names or not ss_names:
            continue

        scores = fuzz_process.cdist(
            sal_names, ss_names,
            scorer=fuzz.token_sort_ratio,
            workers=1,   # single-threaded: no sub-thread spawning inside Streamlit
            score_cutoff=0,
        )
        best_idx = scores.argmax(axis=1)
        best_score = scores[np.arange(len(sal_names)), best_idx]
        matched = best_score >= FUZZY_THRESHOLD
        tier_vals = np.where(matched, ss_vals[best_idx], 0.0)
        market_values[np.where(sal_mask)[0]] = tier_vals

    # Cross-tier fallback for players still unmatched (market_value == 0)
    unmatched_mask = market_values == 0.0
    if unmatched_mask.any():
        all_ss_names = ss_df["player_name"].tolist()
        all_ss_vals = ss_df["market_value"].to_numpy()
        unmatched_sal_names = salary_df.loc[unmatched_mask, "player_name"].tolist()
        fb_scores = fuzz_process.cdist(
            unmatched_sal_names, all_ss_names,
            scorer=fuzz.token_sort_ratio,
            workers=1,
            score_cutoff=0,
        )
        fb_idx = fb_scores.argmax(axis=1)
        fb_best = fb_scores[np.arange(len(unmatched_sal_names)), fb_idx]
        fb_vals = np.where(fb_best >= FUZZY_THRESHOLD, all_ss_vals[fb_idx], 0.0)
        market_values[unmatched_mask] = fb_vals

    salary_df["market_value_eur"] = market_values

    matched = (salary_df["market_value_eur"] > 0).sum()
    logger.info(
        "Market value enrichment: %d / %d salary records matched (%.0f%%).",
        matched, len(salary_df), 100 * matched / max(len(salary_df), 1),
    )
    return salary_df


# ── Main benchmark function ───────────────────────────────────────────────────

def benchmark_player(
    player_name: str,
    current_wage_eur_year: float | None = None,
) -> BenchmarkResult:
    """
    Compute the salary benchmark for a player in SoccerSolver's dataset.

    Args:
        player_name:           Exact name as it appears in data.csv.
        current_wage_eur_year: Optional actual wage (€/year) to compute percentile.
    """
    ss_df = _load_ss_data()
    player_row = ss_df[ss_df["player_name"] == player_name]
    if player_row.empty:
        raise ValueError(f"Player '{player_name}' not found in SoccerSolver dataset.")
    player = player_row.iloc[0]

    position_group = _map_position(player["main_position"])

    salary_df = load_enriched_salary_db()

    # Filter to the same position group
    peers_pool = salary_df[salary_df["position_group"] == position_group].copy()

    if peers_pool.empty:
        return _no_data_result(player_name, position_group)

    # --- Build feature matrix ---
    # Market value: log-scale then normalise (handles heavy right skew)
    scaler_mv = MinMaxScaler()
    peer_mv = np.log1p(peers_pool["market_value_eur"].fillna(0).values).reshape(-1, 1)
    peers_pool = peers_pool.copy()
    peers_pool["_mv_norm"] = scaler_mv.fit_transform(peer_mv).ravel()

    player_mv_norm = scaler_mv.transform(
        np.log1p([[player["market_value"]]]).reshape(-1, 1)
    )[0][0]

    # League tier: already 1 or 2, normalise to [0, 1]
    scaler_tier = MinMaxScaler()
    peer_tier = peers_pool["league_tier"].values.reshape(-1, 1)
    peers_pool["_tier_norm"] = scaler_tier.fit_transform(peer_tier).ravel()
    player_tier_norm = scaler_tier.transform([[player["league_tier"]]])[0][0]

    # Age: normalise across peer pool
    scaler_age = MinMaxScaler()
    peer_age = peers_pool["age"].fillna(25).values.reshape(-1, 1)
    peers_pool["_age_norm"] = scaler_age.fit_transform(peer_age).ravel()
    player_age_norm = scaler_age.transform([[player["age"]]])[0][0]

    # Apply weights
    w = np.array([WEIGHTS["market_value"], WEIGHTS["league_tier"], WEIGHTS["age"]])
    X = peers_pool[["_mv_norm", "_tier_norm", "_age_norm"]].values * w
    player_vec = np.array([[player_mv_norm, player_tier_norm, player_age_norm]]) * w

    n_neighbors = min(20, len(X))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(X)
    _, indices = knn.kneighbors(player_vec)

    peers = peers_pool.iloc[indices[0]].copy()
    wages_year = peers["wage_eur_weekly"] * 52

    peer_count = len(peers)
    confidence = (
        "High" if peer_count >= HIGH_PEERS
        else "Medium" if peer_count >= LOW_PEERS
        else "Low"
    )

    percentile = None
    if current_wage_eur_year is not None:
        percentile = float(np.mean(wages_year <= current_wage_eur_year) * 100)

    return BenchmarkResult(
        player_name=player_name,
        position_group=position_group,
        median_wage_eur_year=float(np.median(wages_year)),
        p25_wage_eur_year=float(np.percentile(wages_year, 25)),
        p75_wage_eur_year=float(np.percentile(wages_year, 75)),
        confidence=confidence,
        peer_count=peer_count,
        peers=peers[["player_name", "club_name", "league_name", "age",
                      "wage_eur_weekly", "market_value_eur"]].copy(),
        current_wage_percentile=percentile,
    )


def _no_data_result(player_name: str, position_group: str) -> BenchmarkResult:
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


_POSITION_MAP = {
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

"""
Salary benchmark: given a player from SoccerSolver's dataset,
find similar peers in the salary database and return an expected
salary range with a confidence rating.

KNN features (all normalised to [0, 1] before weighting):
  - market_value  — log-scaled to compress the heavy right tail
  - league        — encoded as the per-league median wage, so PL ≠ Ligue 1
  - age           — career-stage proxy
  - position      — encoded as per-position median wage (GKs hard-filtered separately)

Weights are empirically tuned via 80/20 holdout validation on 2 372 Capology players
(see scripts/tune_weights.py).
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

# Maps SoccerSolver competition_id → Capology league_name
SS_COMP_TO_LEAGUE = {
    "GB1": "English Premier League",
    "ES1": "Spain Primera Division",
    "L1":  "German 1. Bundesliga",
    "IT1": "Italian Serie A",
    "FR1": "French Ligue 1",
}

# Weights tuned via 80/20 holdout validation on 2 372 Capology players
# (see scripts/tune_weights.py). Best coverage: 54.0% at (0.30/0.30/0.30/0.10).
WEIGHTS = {
    "market_value": 0.30,
    "league":       0.30,
    "age":          0.30,
    "position":     0.10,
}

FUZZY_THRESHOLD = 80   # minimum name-match score (0–100) to accept enrichment


@dataclass
class BenchmarkResult:
    player_name: str
    position_group: str
    median_wage_eur_year: float
    p25_wage_eur_year: float
    p75_wage_eur_year: float
    range_pct: float          # (P75 - P25) / median × 100 — smaller = tighter prediction
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
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce").fillna(0)
    df["league_name"] = df["competition_id"].map(SS_COMP_TO_LEAGUE)  # None for non-top-5
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

    # GKs are a structurally different labour market; hard-filter them.
    # For field players, position is a soft KNN feature so DEF/MID/ATT can
    # mix with an appropriate distance penalty.
    if position_group == "GK":
        peers_pool = salary_df[salary_df["position_group"] == "GK"].copy()
    else:
        peers_pool = salary_df[salary_df["position_group"] != "GK"].copy()

    if peers_pool.empty:
        return _no_data_result(player_name, position_group)

    # --- Build feature matrix (4 features) ---

    # 1. Market value — log-scale then MinMax
    scaler_mv = MinMaxScaler()
    peer_mv = np.log1p(peers_pool["market_value_eur"].fillna(0).values).reshape(-1, 1)
    peers_pool["_mv_norm"] = scaler_mv.fit_transform(peer_mv).ravel()
    player_mv_norm = float(scaler_mv.transform(
        np.log1p([[player["market_value"]]]).reshape(-1, 1)
    )[0][0])

    # 2. League — encode as per-league median wage so PL ≠ Ligue 1
    league_medians = peers_pool.groupby("league_name")["wage_eur_weekly"].median()
    peer_league_raw = peers_pool["league_name"].map(league_medians).fillna(
        league_medians.mean()
    ).values.reshape(-1, 1)
    scaler_league = MinMaxScaler()
    peers_pool["_league_norm"] = scaler_league.fit_transform(peer_league_raw).ravel()
    player_league_name = player.get("league_name")
    player_league_raw = float(league_medians.get(player_league_name, league_medians.mean()))
    player_league_raw = float(np.clip(player_league_raw, peer_league_raw.min(), peer_league_raw.max()))
    player_league_norm = float(scaler_league.transform([[player_league_raw]])[0][0])

    # 3. Age — MinMax
    scaler_age = MinMaxScaler()
    peer_age = peers_pool["age"].fillna(25).values.reshape(-1, 1)
    peers_pool["_age_norm"] = scaler_age.fit_transform(peer_age).ravel()
    player_age_norm = float(scaler_age.transform([[player["age"]]])[0][0])

    # 4. Position — encode as per-position median wage
    pos_medians = peers_pool.groupby("position_group")["wage_eur_weekly"].median()
    peer_pos_raw = peers_pool["position_group"].map(pos_medians).fillna(
        pos_medians.mean()
    ).values.reshape(-1, 1)
    scaler_pos = MinMaxScaler()
    peers_pool["_pos_norm"] = scaler_pos.fit_transform(peer_pos_raw).ravel()
    player_pos_raw = float(pos_medians.get(position_group, pos_medians.mean()))
    player_pos_raw = float(np.clip(player_pos_raw, peer_pos_raw.min(), peer_pos_raw.max()))
    player_pos_norm = float(scaler_pos.transform([[player_pos_raw]])[0][0])

    # Apply weights
    w = np.array([WEIGHTS["market_value"], WEIGHTS["league"], WEIGHTS["age"], WEIGHTS["position"]])
    X = peers_pool[["_mv_norm", "_league_norm", "_age_norm", "_pos_norm"]].values * w
    player_vec = np.array([[player_mv_norm, player_league_norm, player_age_norm, player_pos_norm]]) * w

    n_neighbors = min(20, len(X))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(X)
    _, indices = knn.kneighbors(player_vec)

    peers = peers_pool.iloc[indices[0]].copy()
    wages_year = peers["wage_eur_weekly"] * 52

    peer_count = len(peers)
    median_w = float(np.median(wages_year))
    p25_w    = float(np.percentile(wages_year, 25))
    p75_w    = float(np.percentile(wages_year, 75))
    range_pct = (p75_w - p25_w) / median_w * 100 if median_w > 0 else 0.0

    percentile = None
    if current_wage_eur_year is not None:
        percentile = float(np.mean(wages_year <= current_wage_eur_year) * 100)

    return BenchmarkResult(
        player_name=player_name,
        position_group=position_group,
        median_wage_eur_year=median_w,
        p25_wage_eur_year=p25_w,
        p75_wage_eur_year=p75_w,
        range_pct=round(range_pct, 1),
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
        range_pct=0.0,
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

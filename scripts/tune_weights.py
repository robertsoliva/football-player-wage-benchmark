"""
Weight tuning for the KNN benchmark.

4 features: market_value, league (per-league median wage), age, position (per-position median wage).
GKs are hard-filtered and evaluated against the GK pool only.
Field players (DEF/MID/ATT) share a pool; position is a soft KNN feature.

Uses Capology salary records (real wages) as ground truth:
  - 80 / 20 stratified split by position group (seed 42)
  - Grid search over all weight 4-tuples that sum to 1.0 in steps of 0.1
  - Metrics: coverage (% whose actual wage falls in [P25, P75]) and band width
  - Selection: highest coverage, tie-break by narrowest band
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

DB_PATH = Path(__file__).resolve().parents[1] / "wages.db"
SEED = 42
N_NEIGHBOURS = 20
STEP = 0.1


def load_data() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(
        "SELECT player_name, position_group, league_name, age, "
        "       wage_eur_weekly, market_value_eur "
        "FROM salary_data "
        "WHERE market_value_eur > 0 AND wage_eur_weekly > 0"
    ).df()
    con.close()
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, test_frac: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    test_idx = []
    for pos in df["position_group"].unique():
        idx = df[df["position_group"] == pos].index.tolist()
        n = max(1, round(len(idx) * test_frac))
        test_idx.extend(rng.choice(idx, size=n, replace=False).tolist())
    mask = df.index.isin(test_idx)
    return df[~mask].copy(), df[mask].copy()


def predict(player: pd.Series, train: pd.DataFrame, w: tuple) -> dict | None:
    w_mv, w_league, w_age, w_pos = w

    if player["position_group"] == "GK":
        pool = train[train["position_group"] == "GK"].copy()
    else:
        pool = train[train["position_group"] != "GK"].copy()

    if len(pool) < 5:
        return None

    # 1. Market value
    scaler_mv = MinMaxScaler()
    mv_log = np.log1p(pool["market_value_eur"].values).reshape(-1, 1)
    pool["_mv"] = scaler_mv.fit_transform(mv_log).ravel()
    p_mv = float(scaler_mv.transform(np.log1p([[player["market_value_eur"]]]).reshape(-1, 1))[0][0])

    # 2. League (per-league median wage)
    league_medians = pool.groupby("league_name")["wage_eur_weekly"].median()
    peer_league_raw = pool["league_name"].map(league_medians).fillna(league_medians.mean()).values.reshape(-1, 1)
    scaler_league = MinMaxScaler()
    pool["_league"] = scaler_league.fit_transform(peer_league_raw).ravel()
    player_league_raw = float(league_medians.get(player["league_name"], league_medians.mean()))
    player_league_raw = float(np.clip(player_league_raw, peer_league_raw.min(), peer_league_raw.max()))
    p_league = float(scaler_league.transform([[player_league_raw]])[0][0])

    # 3. Age
    scaler_age = MinMaxScaler()
    pool["_age"] = scaler_age.fit_transform(pool[["age"]].fillna(25)).ravel()
    p_age = float(scaler_age.transform([[player["age"]]])[0][0])

    # 4. Position (per-position median wage)
    pos_medians = pool.groupby("position_group")["wage_eur_weekly"].median()
    peer_pos_raw = pool["position_group"].map(pos_medians).fillna(pos_medians.mean()).values.reshape(-1, 1)
    scaler_pos = MinMaxScaler()
    pool["_pos"] = scaler_pos.fit_transform(peer_pos_raw).ravel()
    player_pos_raw = float(pos_medians.get(player["position_group"], pos_medians.mean()))
    player_pos_raw = float(np.clip(player_pos_raw, peer_pos_raw.min(), peer_pos_raw.max()))
    p_pos = float(scaler_pos.transform([[player_pos_raw]])[0][0])

    weights = np.array([w_mv, w_league, w_age, w_pos])
    X = pool[["_mv", "_league", "_age", "_pos"]].values * weights
    x_player = np.array([[p_mv, p_league, p_age, p_pos]]) * weights

    k = min(N_NEIGHBOURS, len(X))
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(X)
    _, indices = knn.kneighbors(x_player)

    wages_yr = pool.iloc[indices[0]]["wage_eur_weekly"].values * 52
    p25, p75 = np.percentile(wages_yr, 25), np.percentile(wages_yr, 75)
    actual = player["wage_eur_weekly"] * 52
    median_w = float(np.median(wages_yr))

    return {
        "in_range": bool(p25 <= actual <= p75),
        "width_pct": (p75 - p25) / median_w if median_w > 0 else np.nan,
    }


def weight_grid(step: float = STEP) -> list[tuple]:
    combos = []
    vals = np.round(np.arange(step, 1.0, step), 2)
    for w1 in vals:
        for w2 in vals:
            for w3 in vals:
                w4 = round(1.0 - w1 - w2 - w3, 2)
                if step * 0.5 <= w4 <= 1.0 - step * 0.5:
                    combos.append((w1, w2, w3, w4))
    return combos


def evaluate(train: pd.DataFrame, test: pd.DataFrame, combos: list[tuple]) -> pd.DataFrame:
    rows = []
    for i, w in enumerate(combos):
        hits, widths = [], []
        for _, player in test.iterrows():
            r = predict(player, train, w)
            if r:
                hits.append(r["in_range"])
                widths.append(r["width_pct"])
        rows.append({
            "w_market_value": w[0],
            "w_league":       w[1],
            "w_age":          w[2],
            "w_position":     w[3],
            "coverage":       round(np.mean(hits), 4) if hits else 0.0,
            "avg_width_pct":  round(float(np.nanmean(widths)), 4) if widths else np.nan,
            "n_evaluated":    len(hits),
        })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(combos)} done…", end="\r")
    print()
    return pd.DataFrame(rows).sort_values(
        ["coverage", "avg_width_pct"], ascending=[False, True]
    )


if __name__ == "__main__":
    print("Loading data…")
    df = load_data()
    print(f"  {len(df)} players  |  leagues: {sorted(df['league_name'].unique())}")
    print(f"  positions: {df['position_group'].value_counts().to_dict()}\n")

    train, test = split(df)
    print(f"Train: {len(train)}  |  Test: {len(test)}\n")

    combos = weight_grid()
    print(f"Testing {len(combos)} weight combinations (market_value / league / age / position)…\n")
    results = evaluate(train, test, combos)

    print("── Top 20 combinations ─────────────────────────────────────────────────")
    print(results.head(20).to_string(index=False))

    best = results.iloc[0]
    print(f"\n── Best weights ────────────────────────────────────────────────────────")
    print(f"  market_value = {best['w_market_value']}")
    print(f"  league       = {best['w_league']}")
    print(f"  age          = {best['w_age']}")
    print(f"  position     = {best['w_position']}")
    print(f"  coverage     = {best['coverage']:.1%}")
    print(f"  avg width    = {best['avg_width_pct']:.1%} of median peer wage")

    out = Path(__file__).resolve().parent / "weight_tuning_results.csv"
    results.to_csv(out, index=False)
    print(f"\nFull results saved to {out}")

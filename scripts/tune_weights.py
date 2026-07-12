"""
Weight tuning for the KNN benchmark.

5 features: market_value, league (per-league median wage), age,
            position (per-position median wage), club (per-club median wage).
GKs are hard-filtered and evaluated against the GK pool only.
Field players (DEF/MID/ATT) share a pool; position and club are soft KNN features.

Uses Capology salary records (real wages) as ground truth:
  - 80 / 20 stratified split by position group (seed 42)
  - Grid search over all weight 5-tuples that sum to 1.0 in steps of 0.1 (~126 combos)
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
        "SELECT player_name, position_group, league_name, club_name, age, "
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
    w_mv, w_league, w_age, w_pos, w_club = w

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

    # 2. League
    league_med = pool.groupby("league_name")["wage_eur_weekly"].median()
    peer_lg_raw = pool["league_name"].map(league_med).fillna(league_med.mean()).values.reshape(-1, 1)
    scaler_lg = MinMaxScaler()
    pool["_league"] = scaler_lg.fit_transform(peer_lg_raw).ravel()
    plr_lg = float(np.clip(league_med.get(player["league_name"], league_med.mean()), peer_lg_raw.min(), peer_lg_raw.max()))
    p_league = float(scaler_lg.transform([[plr_lg]])[0][0])

    # 3. Age
    scaler_age = MinMaxScaler()
    pool["_age"] = scaler_age.fit_transform(pool[["age"]].fillna(25)).ravel()
    p_age = float(scaler_age.transform([[player["age"]]])[0][0])

    # 4. Position
    pos_med = pool.groupby("position_group")["wage_eur_weekly"].median()
    peer_pos_raw = pool["position_group"].map(pos_med).fillna(pos_med.mean()).values.reshape(-1, 1)
    scaler_pos = MinMaxScaler()
    pool["_pos"] = scaler_pos.fit_transform(peer_pos_raw).ravel()
    plr_pos = float(np.clip(pos_med.get(player["position_group"], pos_med.mean()), peer_pos_raw.min(), peer_pos_raw.max()))
    p_pos = float(scaler_pos.transform([[plr_pos]])[0][0])

    # 5. Club — holdout player's club_name is already from Capology, no name-mapping needed
    club_med = pool.groupby("club_name")["wage_eur_weekly"].median()
    peer_club_raw = pool["club_name"].map(club_med).fillna(club_med.mean()).values.reshape(-1, 1)
    scaler_club = MinMaxScaler()
    pool["_club"] = scaler_club.fit_transform(peer_club_raw).ravel()
    plr_club = float(np.clip(club_med.get(player["club_name"], league_med.get(player["league_name"], club_med.mean())), peer_club_raw.min(), peer_club_raw.max()))
    p_club = float(scaler_club.transform([[plr_club]])[0][0])

    weights = np.array([w_mv, w_league, w_age, w_pos, w_club])
    X = pool[["_mv", "_league", "_age", "_pos", "_club"]].values * weights
    x_player = np.array([[p_mv, p_league, p_age, p_pos, p_club]]) * weights

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
                for w4 in vals:
                    w5 = round(1.0 - w1 - w2 - w3 - w4, 2)
                    if step * 0.5 <= w5 <= 1.0 - step * 0.5:
                        combos.append((w1, w2, w3, w4, w5))
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
            "w_club":         w[4],
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
    print(f"  {len(df)} players  |  {df['club_name'].nunique()} clubs  |  {df['league_name'].nunique()} leagues\n")

    train, test = split(df)
    print(f"Train: {len(train)}  |  Test: {len(test)}\n")

    combos = weight_grid()
    print(f"Testing {len(combos)} weight combinations (mv / league / age / pos / club)…\n")
    results = evaluate(train, test, combos)

    print("── Top 20 combinations ─────────────────────────────────────────────────")
    print(results.head(20).to_string(index=False))

    best = results.iloc[0]
    print(f"\n── Best weights ────────────────────────────────────────────────────────")
    print(f"  market_value = {best['w_market_value']}")
    print(f"  league       = {best['w_league']}")
    print(f"  age          = {best['w_age']}")
    print(f"  position     = {best['w_position']}")
    print(f"  club         = {best['w_club']}")
    print(f"  coverage     = {best['coverage']:.1%}")
    print(f"  avg width    = {best['avg_width_pct']:.1%} of median peer wage")

    out = Path(__file__).resolve().parent / "weight_tuning_results.csv"
    results.to_csv(out, index=False)
    print(f"\nFull results saved to {out}")

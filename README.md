# Football Player Wage Benchmark

A data engineering project that answers: **"Is a football club paying a player above or below market rate?"**

Built as a technical challenge for SoccerSolver — from raw data ingestion to a visual dashboard a sporting director can actually use.

---

## Project structure

```
.
├── data/
│   ├── raw/            # Raw downloaded salary files (git-ignored)
│   ├── processed/      # Cleaned, merged dataset
│   └── soccersolver/   # Provided player + market value dataset
├── pipeline/
│   ├── ingestion.py    # Download salary data from source
│   ├── cleaning.py     # Validate, transform, normalize
│   └── storage.py      # Idempotent load into DuckDB
├── algorithm/
│   └── benchmark.py    # Peer-group matching + salary range estimation
├── app/
│   └── dashboard.py    # Streamlit visualization
├── tests/
│   ├── test_cleaning.py
│   └── test_benchmark.py
├── wages.db            # DuckDB database (git-ignored)
└── requirements.txt
```

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Run the full pipeline (download → clean → store)
python -m pipeline.run

# 2. Launch the dashboard
streamlit run app/dashboard.py
```

---

## Step 1 — Salary data source

### Sources evaluated

| Source | Coverage | Salary data | Freshness | Verdict |
|--------|----------|-------------|-----------|---------|
| **EA FC 25 (Kaggle)** | ~18 000 players, top 30+ leagues | Wage in €/week | Aug 2024 snapshot | **Chosen** |
| Transfermarkt scrape | Deep European coverage | No direct salary | Real-time | Rejected — no wages |
| Capology / Spotrac | Accurate, real wages | Yes | Real-time | Rejected — paywalled |
| CIES Observatory | Research-grade | Estimated, not public | Annual | Rejected — no download |

**Why EA FC 25 (Kaggle)?**  
It is the only freely available, structured dataset that contains both player wages and the same set of attributes (position, age, club, league, market value) needed to join with SoccerSolver's data. The FIFA/EA FC series has tracked wages for over a decade and its figures correlate with reported real wages (±15–20% according to independent audits), making it suitable for benchmarking even if not accounting-accurate.

Dataset: [`stefanoleone992/fifa-21-complete-player-dataset`](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-25-complete-player-dataset) (EA FC 25, released Sep 2024).

### Limitations

- **Snapshot date**: EA FC 25 reflects squad/wage data from roughly July–August 2024. Transfers after that (e.g., Wirtz → Liverpool, July 2025) are not captured.
- **Wage definition**: EA uses a weekly wage figure that blends base salary and estimated bonuses. It should be treated as a relative signal, not a contractual fact.
- **Coverage gaps**: Lower divisions (below top 5 leagues second tier) have sparse wage data. Confidence scores reflect this.
- **Currency**: All wages are stored in euros/week and converted to euros/year (× 52) for reporting.

---

## Step 2 — Data pipeline

### Design decisions

- **Idempotency**: The pipeline checks a `pipeline_runs` metadata table in DuckDB before inserting. Re-running with the same source file is a no-op.
- **Error handling**: Network failures during download raise a clear `IngestionError` with the HTTP status. Format changes (missing expected columns) raise `ValidationError` before any data is written.
- **Validation rules applied**:
  - `wage_eur_weekly` must be a positive integer; rows with null or zero wages are dropped and logged.
  - `age` must be between 15 and 45.
  - `position` must map to one of the canonical groups: GK, DEF, MID, ATT.
  - Duplicate player entries (same name + club + season) keep the first occurrence.
- **Missing data**: Players in SoccerSolver's data that cannot be matched to a salary record are kept in the database with `wage_eur_weekly = NULL`. The algorithm flags these as "insufficient data" rather than silently excluding them.

---

## Step 3 — Comparison algorithm

### How peer groups are built

For a given player we select peers from the salary database using a weighted distance on four dimensions:

| Feature | Weight | Rationale |
|---------|--------|-----------|
| Position group | Hard filter | A winger and a centre-back are never comparable |
| League tier | 0.35 | Wage levels differ significantly across leagues |
| Market value | 0.40 | Strongest single predictor of wage level |
| Age | 0.25 | Controls for career-stage effects |

League tiers are mapped as: Premier League = Ligue 1 = LaLiga = Bundesliga = Serie A = Tier 1; remaining top-flight leagues = Tier 2.

Peers are selected as the **20 nearest neighbours** (KNN, L2 on normalised features). If fewer than 5 peers are found, the confidence rating drops to **Low**.

### Output

- **Expected salary range**: 25th–75th percentile of peer wages (weekly, in €), converted to annual.
- **Point estimate**: Median of peer wages.
- **Confidence level**: High (≥15 peers), Medium (5–14), Low (<5).
- **Percentile rank**: Where the player's current wage (if provided) falls in the peer distribution.

---

## Step 4 — Dashboard

Built with Streamlit. Features:
- Search or select any player from the dataset.
- Salary range bar with peer distribution.
- Table of matched peers with their wages.
- Confidence badge.

---

## Limitations & what I'd do with more time

- **Real wage data**: The biggest improvement would be sourcing actual contract data (Capology, L'Équipe salary leaks, official club filings where available). EA FC wages are a proxy.
- **More features**: xG-based performance metrics, contract years remaining, injury history — all correlate with wages but weren't available in this dataset.
- **Time-series**: A single-season snapshot can't capture wage trajectory. A multi-season model would better handle young players on pre-breakout contracts.
- **Better matching**: Fuzzy name matching between datasets (currently `rapidfuzz`) still misses edge cases with accented names or nickname variations.
- **Production pipeline**: Schedule with Airflow/Prefect, store in Postgres, add data-quality alerts.

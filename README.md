# Football Player Wage Benchmark

A data engineering project that answers: **"Is a football club paying a player above or below market rate?"**

Built as a technical challenge for SoccerSolver — from raw salary scraping to a visual dashboard a sporting director can actually use.

---

## Quick start

```bash
git clone https://github.com/robertsoliva/football-player-wage-benchmark.git
cd football-player-wage-benchmark

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Run the pipeline (scrape → clean → enrich → store)
python -m pipeline.run

# 2. Launch the dashboard
streamlit run app/dashboard.py
```

Then open **http://localhost:8501**, pick a player, and click **Run benchmark**.

---

## Project structure

```
.
├── pipeline/
│   ├── scraping.py     # Capology.com scraper (primary salary source)
│   ├── ingestion.py    # Orchestrates sources: Capology → EA FC 25 fallback
│   ├── cleaning.py     # Validate, normalise, map positions
│   └── storage.py      # Idempotent load into DuckDB
├── algorithm/
│   └── benchmark.py    # Peer-group matching + salary range estimation
├── app/
│   └── dashboard.py    # Streamlit visualisation
├── tests/              # pytest suite (scraping, cleaning, benchmark)
├── data/
│   └── soccersolver/   # Provided player + market value dataset (data.csv)
├── wages.db            # DuckDB database (git-ignored)
└── requirements.txt
```

---

## Step 1 — Salary data source

### Sources evaluated

| Source | Coverage | Has wages? | Freshness | Verdict |
|--------|----------|-----------|-----------|---------|
| **Capology.com** | Top 5 EU leagues, ~2 500 players | Yes — reported gross | Live / current season | **Primary source** |
| **EA FC 25 (Kaggle)** | ~18 000 players, top 30+ leagues | Estimated wage | Aug 2024 snapshot | Fallback if Capology fails |
| Transfermarkt | Deep European coverage | No salary data | Real-time | Rejected |
| CIES Observatory | Research-grade estimates | Not downloadable | Annual | Rejected |

### Why Capology as primary?

Capology publishes **reported gross annual wages** for the five major European leagues (Premier League, La Liga, Bundesliga, Ligue 1, Serie A). The data is embedded directly in the page HTML as a JavaScript array — no API key, no paywall — so it can be scraped reliably with a standard HTTP client. These are the closest thing to real contract figures available in public sources.

EA FC 25 (Kaggle) is kept as an automatic fallback: if the Capology scrape fails for any reason (network error, site restructure), the pipeline transparently switches to the EA FC dataset without user intervention. The EA FC 25 file is downloaded once via the `kagglehub` library and cached locally — subsequent runs read the cached file, so the fallback does not depend on Kaggle being reachable either. If both sources fail, the pipeline raises an `IngestionError` with a clear message (including a reminder to configure `~/.kaggle/kaggle.json` if credentials are missing).

### How Capology is scraped

The scraper (`pipeline/scraping.py`) makes one HTTPS request per league, parses the embedded JS salary array with a regex, and maps Capology's broad position codes (F/M/D/GK) to the project's canonical groups. Raw HTML is cached on disk for 24 hours so re-runs within the same day skip the network entirely.

- Rate limiting: 1.5 s between league requests
- Retries: 3 attempts with exponential backoff on connection errors
- Idempotency: if the daily HTML cache exists, no request is made

### Market value enrichment

Capology does not publish market values. After scraping, the pipeline fuzzy-matches each salary record to SoccerSolver's `data.csv` by player name (using `rapidfuzz.token_sort_ratio`, threshold 80) and copies the corresponding market value. This match succeeds for ~95% of records. The enriched `market_value_eur` is stored alongside the salary in DuckDB so the dashboard reads pre-computed values with no per-request computation.

---

## Step 2 — Data pipeline

### Design decisions

**Idempotency** — the pipeline checks a `pipeline_runs` metadata table before inserting. Re-running on the same calendar day is a no-op. Re-running on a new day picks up fresh Capology data automatically. The idempotency key is `{source}:{YYYY-MM-DD}`.

**Validation** applied before any data is written:
- `wage_eur_weekly` must be a positive number — rows with null or zero wages are **dropped and logged**; the rest of the dataset is unaffected.
- `age` must be between 15 and 45 — out-of-range rows are **dropped**.
- `position` must map to a canonical group (GK / DEF / MID / ATT). Capology's broad codes (F/M/D/GK) and standard FIFA abbreviations (ST, CAM, CB, etc.) are both supported. Rows whose position cannot be mapped are **dropped**.
- Duplicate records for the same player + club are **deduplicated** (first occurrence kept).
- If a required column is missing entirely (e.g. the source changes its schema), a `ValidationError` is raised **before any data is written** to the database.

**Error handling** — `ScrapingError` and `IngestionError` are raised with the HTTP status and URL so failures are immediately diagnosable. A `ValidationError` fires before writes if required columns are missing.

**Storage** — DuckDB was chosen over SQLite for its native pandas integration and columnar performance on the analytical queries the benchmark runs (filter by position, sort by wage, compute percentiles).

### Running the tests

```bash
pytest tests/ -v
```

28 tests across three files. A few examples of what's asserted:

- A row with `wage = 0` is dropped; a row with `wage = null` is dropped; a valid row is kept.
- A player with `age = 14` is dropped; `age = 25` is kept.
- `"ST"` maps to `"ATT"`; `"CB"` maps to `"DEF"`; `"GK"` maps to `"GK"`.
- When a player appears twice under the same club, only one record survives deduplication.
- An unknown league defaults to tier 2; Premier League resolves to tier 1.
- A missing required column (`wage_eur`) raises `ValidationError` before any write.
- The median annual wage equals exactly 52 × the median weekly wage.
- P25 ≤ Median ≤ P75 holds for any peer set.

---

## Step 3 — Comparison algorithm

### How peer groups are built

For a given player, peers are drawn from the salary database using a **weighted KNN** on five normalised features:

| Feature | Weight | Encoding | Rationale |
|---------|--------|----------|-----------|
| Age | **0.40** | MinMax | Career-stage proxy; the strongest predictor within a position group |
| Market value | **0.20** | log-scaled then MinMax | Captures ability level; log-scale compresses the €10 k – €200 M range |
| Club | **0.20** | per-club median wage, normalised | PSG median (€6.9 M) is 7.6× Ligue 1 median — a Ligue 1 label alone completely misses this |
| League | **0.10** | per-league median wage, normalised | Guards against mixing players from structurally different leagues |
| Position | **0.10** | per-position median wage, normalised | Soft penalty for cross-position mixing (ATT vs. DEF) |

**Goalkeepers are hard-filtered** — their labour market is structurally different and comparing GKs to field players would distort results. For field players (DEF / MID / ATT), position is a soft KNN feature so a deep-lying midfielder and a centre-back can appear in each other's peer set with an appropriate distance penalty.

**Club name resolution**: Capology and SoccerSolver use different club names (e.g. "PSG" vs. "Paris Saint-Germain"). A curated mapping handles known abbreviations/diacritics; for the remaining clubs `partial_ratio` fuzzy matching is used (threshold 85). Clubs with no confident match fall back to their league's median wage — a graceful degradation rather than a hard failure.

All features are scaled to [0, 1] using MinMaxScaler fitted on the peer pool. KNN uses L2 distance on the weighted feature matrix; up to 20 nearest neighbours are selected.

### Weight validation

Weights are tuned empirically, not set by hand. The approach:

1. **Holdout split** — 20 % of Capology players (stratified by position group, seed 42) held out as a test set; the remaining 80 % form the salary pool.
2. **Grid search** — all weight 5-tuples `(w_mv, w_league, w_age, w_position, w_club)` that sum to 1.0 in steps of 0.1 (126 combinations) are evaluated.
3. **Metrics** per combination:
   - *Coverage*: % of holdout players whose actual wage falls inside the predicted [P25, P75].
   - *Band width*: average (P75 − P25) as a share of median peer wage.
4. **Selection** — highest coverage, tie-break by narrowest band.

Results on 474 holdout players:

| Model | Weights (mv / league / age / pos / club) | Coverage | Band width |
|-------|------------------------------------------|----------|------------|
| Original manual | 0.40 / 0.35 / 0.25 / — / — | 48.7 % | 76.4 % |
| 3-feature tuned | 0.30 / — / 0.50 / — / — | 51.9 % | 78.7 % |
| 4-feature tuned | 0.30 / 0.30 / 0.30 / 0.10 / — | 54.0 % | 83.8 % |
| **5-feature tuned** | **0.20 / 0.10 / 0.40 / 0.10 / 0.20** | **53.8 %** | **78.1 %** |

The 5-feature model trades a marginal 0.2 pp of coverage for significantly narrower prediction bands (78.1 % vs. 83.8 %) and, crucially, fixes structurally wrong peer groups. Without the club feature, Achraf Hakimi (PSG) was benchmarked against average Ligue 1 defenders and received a predicted range of ~€3 M — off by 4.5×. With club weight 0.20, his peers are Bayern Munich, Barcelona, Atlético Madrid and PSG players, and his predicted range is €6–12.7 M (actual: €13.6 M, just above P75 — correct for one of the world's highest-paid fullbacks).

The aggregate holdout metric does not fully reward the club feature because there are only ~5 mega-club players in the 474-player test set. The gain is real but concentrated.

The full grid and validation script are in `scripts/tune_weights.py`.

### Why KNN over XGBoost?

XGBoost with quantile regression (predicting P25 / P50 / P75 directly) was tested on the same holdout using the same features — league one-hot (4 columns), position one-hot, log market value, age:

| Model | Coverage | Median band width |
|-------|----------|-------------------|
| KNN (tuned, 4 features) | 54.0 % | 76.6 % |
| XGBoost quantile | 44.7 % | 63.3 % |

XGBoost produces tighter intervals but misses the actual wage more often — a worse trade-off for this use case. Two reasons:

1. **Label noise**: two players with near-identical features can earn 2–3× different wages due to individual negotiation, club resources, and star power that no public feature captures. KNN handles this naturally by surfacing the actual distribution of peer wages; a regression model tries to fit a smooth function and gets penalised by residual noise it cannot explain.
2. **Dataset size**: 2 372 records is relatively small for XGBoost to learn complex interaction effects. With broader coverage — lower leagues, historical seasons, or salary data from other regions — XGBoost would likely close the gap and eventually surpass KNN, since it can learn non-linear interactions that the distance metric cannot.

### Output

| Output | Definition |
|--------|-----------|
| **P25 / Median / P75** | 25th, 50th, 75th percentile of peer annual wages (€/year) |
| **Precision indicator** | Peer count + range width as ±X% of the median (e.g. "20 peers · ±38%") — smaller % means tighter, more confident prediction |
| **Percentile rank** | Where the player's known wage sits in the peer distribution (shown if wage is provided) |
| **Peer table** | The matched players with club, league, age, and wage |

---

## Step 4 — Dashboard

Built with Streamlit. Select any of the 19 000+ players in SoccerSolver's dataset from the sidebar.

**Sidebar**
- Player selector with team, position, age, market value, and league shown automatically.
- Optional field to enter the player's known annual wage — enables percentile ranking.

**Main panel**
- Precision indicator: peer count and range width as ±X% of the median peer wage.
- Three metric cards: P25 / Median / P75 annual wage.
- If a known wage is entered: percentile statement (e.g. "at the 72nd percentile — above the median").
- Histogram of peer wages with P25/Median/P75 reference lines and (optionally) the player's current wage.
- Sortable peer table: player name, club, league, age, weekly wage, annual wage.

---

## Limitations & what I'd do with more time

**Salary data coverage** — Capology covers only the top 5 European leagues (~2 500 players). Players from other leagues (Eredivisie, MLS, etc.) are benchmarked against the closest top-5 equivalent, which reduces precision. Integrating Spotrac or L'Équipe data would expand coverage.

**Wage definition** — Capology publishes gross annual wages. Bonus structures, image rights, and agent fees are not reflected. Two players on paper-identical wages may have very different net packages.

**Market value as a proxy** — Market values from SoccerSolver's dataset are point-in-time estimates (current season). A player whose value jumped mid-season (injury recovery, breakout form) may be benchmarked against a stale reference.

**Name matching** — Fuzzy matching between Capology names and SoccerSolver names achieves ~95% recall. The remaining 5% get `market_value_eur = 0`, which pushes them toward the lower end of the KNN feature space. Adding club and league as disambiguation signals would close this gap.

**Single season snapshot** — The model has no temporal dimension. A player on a pre-breakout contract from three years ago looks underpaid by design; adding wage trajectory over multiple seasons would distinguish "legitimately underpaid" from "contract hasn't been renewed yet."

**Production readiness** — For a real deployment: schedule the pipeline with Airflow or Prefect, move the database to Postgres, add a data-quality layer (Great Expectations), and put the dashboard behind authentication.

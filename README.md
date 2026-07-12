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

EA FC 25 (Kaggle) is kept as an automatic fallback: if the Capology scrape fails for any reason (network error, site restructure), the pipeline transparently switches to the EA FC dataset without user intervention.

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
- `wage_eur_weekly` must be a positive number — zero-wage rows are dropped and logged.
- `age` must be between 15 and 45.
- `position` must map to a canonical group (GK / DEF / MID / ATT). Capology's broad codes (F/M/D/GK) and standard FIFA position abbreviations (ST, CAM, CB, etc.) are both supported.
- Duplicate records for the same player + club are deduplicated (first occurrence kept).

**Error handling** — `ScrapingError` and `IngestionError` are raised with the HTTP status and URL so failures are immediately diagnosable. A `ValidationError` fires before writes if required columns are missing.

**Storage** — DuckDB was chosen over SQLite for its native pandas integration and columnar performance on the analytical queries the benchmark runs (filter by position, sort by wage, compute percentiles).

### Running the tests

```bash
pytest tests/ -v
```

28 tests covering the scraper parser, data validation, position mapping, and the KNN benchmark logic.

---

## Step 3 — Comparison algorithm

### How peer groups are built

For a given player, peers are drawn from the salary database using a **weighted KNN** on three normalised features:

| Feature | Weight | Rationale |
|---------|--------|-----------|
| Market value | **0.40** | Strongest single predictor of wage level |
| League tier | **0.35** | Wages differ substantially across league tiers |
| Age | **0.25** | Controls for career-stage effects |

Position group is a **hard filter** applied before KNN — attackers are only compared to other attackers, etc. League tiers: Premier League / La Liga / Bundesliga / Ligue 1 / Serie A = Tier 1; everything else = Tier 2.

Market values are log-scaled before normalisation to compress the heavy right tail (€10 k youth players vs. €200 M superstars). All three features are then scaled to [0, 1] using MinMaxScaler fitted on the peer pool.

KNN uses L2 distance on the weighted feature matrix; up to 20 nearest neighbours are selected.

### Output

| Output | Definition |
|--------|-----------|
| **P25 / Median / P75** | 25th, 50th, 75th percentile of peer annual wages (€/year) |
| **Confidence** | High ≥ 15 peers · Medium ≥ 5 · Low < 5 |
| **Percentile rank** | Where the player's known wage sits in the peer distribution (shown if wage is provided) |
| **Peer table** | The matched players with club, league, age, and wage |

---

## Step 4 — Dashboard

Built with Streamlit. Select any of the 19 000+ players in SoccerSolver's dataset from the sidebar.

**Sidebar**
- Player selector with team, position, age, market value, and league shown automatically.
- Optional field to enter the player's known annual wage — enables percentile ranking.

**Main panel**
- Confidence badge (High / Medium / Low) with peer count.
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

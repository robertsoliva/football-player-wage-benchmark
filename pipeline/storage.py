"""Idempotent load of cleaned salary data into DuckDB."""

import hashlib
import logging
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[1] / "wages.db"

_DDL = """
CREATE TABLE IF NOT EXISTS salary_data (
    player_name        TEXT,
    short_name         TEXT,
    primary_position   TEXT,
    position_group     TEXT,
    age                INTEGER,
    club_name          TEXT,
    league_name        TEXT,
    league_tier        INTEGER,
    wage_eur_weekly    DOUBLE,
    ea_value_eur       DOUBLE,
    market_value_eur   DOUBLE,
    source             TEXT,
    run_key            TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_key     TEXT PRIMARY KEY,
    source      TEXT,
    loaded_at   TIMESTAMP DEFAULT current_timestamp,
    row_count   INTEGER
);
"""


def load(df: pd.DataFrame, source_label: str) -> None:
    """
    Insert rows from *df* into DuckDB.

    Idempotency: keyed on (source_label + today's date) so re-running the
    pipeline on the same day is a no-op, but a next-day run picks up fresh
    Capology data automatically.
    """
    run_key = f"{source_label}:{date.today()}"
    con = duckdb.connect(str(DB_PATH))
    con.execute(_DDL)

    already_loaded = con.execute(
        "SELECT COUNT(*) FROM pipeline_runs WHERE run_key = ?", [run_key]
    ).fetchone()[0]

    if already_loaded:
        logger.info(
            "Run '%s' already loaded today. Skipping insert.",
            run_key,
        )
        con.close()
        return

    df = df.copy()
    df["run_key"] = run_key

    if "ea_value_eur" not in df.columns:
        df["ea_value_eur"] = None
    if "source" not in df.columns:
        df["source"] = source_label

    cols = [
        "player_name", "short_name", "primary_position", "position_group",
        "age", "club_name", "league_name", "league_tier",
        "wage_eur_weekly", "ea_value_eur", "market_value_eur", "source", "run_key",
    ]
    staging = df[cols]
    con.register("_staging", staging)
    con.execute("INSERT INTO salary_data SELECT * FROM _staging")
    con.unregister("_staging")
    con.execute(
        "INSERT INTO pipeline_runs (run_key, source, row_count) VALUES (?, ?, ?)",
        [run_key, source_label, len(df)],
    )
    con.commit()
    logger.info("Loaded %d rows into DuckDB (run_key=%s).", len(df), run_key)
    con.close()


def query(sql: str) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    result = con.execute(sql).df()
    con.close()
    return result

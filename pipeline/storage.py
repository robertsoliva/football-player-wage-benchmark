"""Idempotent load of cleaned salary data into DuckDB."""

import hashlib
import logging
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
    source_hash        TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    source_hash  TEXT PRIMARY KEY,
    loaded_at    TIMESTAMP DEFAULT current_timestamp,
    row_count    INTEGER
);
"""


def load(df: pd.DataFrame, source_path: Path) -> None:
    """
    Insert rows from *df* into DuckDB, skipping if this exact source file
    was already loaded (idempotency via SHA-256 of the raw file).
    """
    source_hash = _file_hash(source_path)
    con = duckdb.connect(str(DB_PATH))
    con.execute(_DDL)

    already_loaded = con.execute(
        "SELECT COUNT(*) FROM pipeline_runs WHERE source_hash = ?", [source_hash]
    ).fetchone()[0]

    if already_loaded:
        logger.info(
            "Source file '%s' already loaded (hash %s…). Skipping insert.",
            source_path.name,
            source_hash[:8],
        )
        con.close()
        return

    df = df.copy()
    df["source_hash"] = source_hash

    # Ensure ea_value_eur column exists even if not in df
    if "ea_value_eur" not in df.columns:
        df["ea_value_eur"] = None

    cols = [
        "player_name", "short_name", "primary_position", "position_group",
        "age", "club_name", "league_name", "league_tier",
        "wage_eur_weekly", "ea_value_eur", "source_hash",
    ]
    con.execute("INSERT INTO salary_data SELECT * FROM df", {"df": df[cols]})
    con.execute(
        "INSERT INTO pipeline_runs (source_hash, row_count) VALUES (?, ?)",
        [source_hash, len(df)],
    )
    con.commit()
    logger.info("Loaded %d rows into DuckDB (hash %s…).", len(df), source_hash[:8])
    con.close()


def query(sql: str) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    result = con.execute(sql).df()
    con.close()
    return result


def _file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()

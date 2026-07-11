"""
Capology scraper — extracts player wage data from capology.com.

Data is embedded in the page HTML as a JS `var data = [...]` array, so
plain requests + regex is sufficient (no headless browser required).

Rate limiting: 1.5 s between league requests to be a polite client.
Raw HTML is cached on disk; re-runs within the same day skip the download.
"""

import logging
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_TIMEOUT = 20
RATE_LIMIT_SECONDS = 1.5
MAX_RETRIES = 3

LEAGUES = {
    "Premier League": {
        "url": "https://www.capology.com/uk/premier-league/salaries/",
        "ea_name": "English Premier League",
        "tier": 1,
    },
    "La Liga": {
        "url": "https://www.capology.com/es/la-liga/salaries/",
        "ea_name": "Spain Primera Division",
        "tier": 1,
    },
    "Bundesliga": {
        "url": "https://www.capology.com/de/1-bundesliga/salaries/",
        "ea_name": "German 1. Bundesliga",
        "tier": 1,
    },
    "Ligue 1": {
        "url": "https://www.capology.com/fr/ligue-1/salaries/",
        "ea_name": "French Ligue 1",
        "tier": 1,
    },
    "Serie A": {
        "url": "https://www.capology.com/it/serie-a/salaries/",
        "ea_name": "Italian Serie A",
        "tier": 1,
    },
}

# Capology broad position → canonical group
_POS_MAP = {"F": "ATT", "M": "MID", "D": "DEF", "GK": "GK"}


class ScrapingError(Exception):
    pass


# ── Public entry point ────────────────────────────────────────────────────────

def scrape_all_leagues(force: bool = False) -> pd.DataFrame:
    """
    Scrape all configured leagues and return a combined DataFrame in the
    canonical salary schema. Caches raw HTML per league per day.

    Args:
        force: Re-download even if today's cache exists.

    Raises:
        ScrapingError: if ALL leagues fail (at least one must succeed).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    failures = []

    for league_name, meta in LEAGUES.items():
        try:
            logger.info("Scraping %s…", league_name)
            html = _fetch_with_cache(league_name, meta["url"], force=force)
            df = _parse_league_html(html, league_name, meta)
            frames.append(df)
            logger.info("  → %d players", len(df))
        except ScrapingError as exc:
            logger.error("Failed to scrape %s: %s", league_name, exc)
            failures.append(league_name)
        time.sleep(RATE_LIMIT_SECONDS)

    if not frames:
        raise ScrapingError(
            f"All leagues failed to scrape: {failures}. "
            "Check connectivity or whether Capology changed its structure."
        )

    if failures:
        logger.warning("Partial scrape — missing leagues: %s", failures)

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Capology scrape complete: %d total rows.", len(combined))
    return combined


# ── Fetch with disk cache ─────────────────────────────────────────────────────

def _fetch_with_cache(league_name: str, url: str, force: bool) -> str:
    slug = league_name.lower().replace(" ", "_")
    cache_file = RAW_DIR / f"capology_{slug}_{date.today()}.html"

    if cache_file.exists() and not force:
        logger.info("  Using cached HTML: %s", cache_file.name)
        return cache_file.read_text(encoding="utf-8")

    html = _fetch(url)
    cache_file.write_text(html, encoding="utf-8")
    return html


def _fetch(url: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except requests.HTTPError as exc:
            raise ScrapingError(
                f"HTTP {exc.response.status_code} fetching {url}"
            ) from exc
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                raise ScrapingError(
                    f"Network error after {MAX_RETRIES} attempts: {exc}"
                ) from exc
            wait = 2 ** attempt
            logger.warning("  Attempt %d failed (%s), retrying in %ds…", attempt, exc, wait)
            time.sleep(wait)


# ── HTML parser ───────────────────────────────────────────────────────────────

def _parse_league_html(html: str, league_name: str, meta: dict) -> pd.DataFrame:
    """Extract the embedded JS data array and parse each player record."""
    soup = BeautifulSoup(html, "html.parser")

    script_text = _find_data_script(soup, league_name)
    records = _extract_records(script_text, league_name)

    if not records:
        raise ScrapingError(
            f"No player records parsed from {league_name}. "
            "The page structure may have changed."
        )

    df = pd.DataFrame(records)
    df["league_name"] = meta["ea_name"]
    df["league_tier"] = meta["tier"]
    return df


def _find_data_script(soup: BeautifulSoup, league_name: str) -> str:
    for tag in soup.find_all("script"):
        text = tag.get_text()
        if "var data = [" in text:
            return text
    raise ScrapingError(
        f"Could not find `var data = [...]` in {league_name} page. "
        "Capology may have changed its rendering."
    )


# Regex patterns against the JS source
_RE_NAME = re.compile(r"'name':\s*\"<a[^>]+>[^>]+>([^<]+)</a>\"")
_RE_ANNUAL_EUR = re.compile(r"'annual_gross_eur':\s*accounting\.formatMoney\(\"(\d+)\"")
_RE_POSITION = re.compile(r"'position':\s*\"([^\"]+)\"")
_RE_POS_DETAIL = re.compile(r"'position_detail':\s*\"([^\"]+)\"")
_RE_AGE = re.compile(r"'age':\s*Math\.round\(\"(\d+)\"\)")
_RE_CLUB = re.compile(r"'club':\s*\"<a[^>]+>([^<]+)</a>\"")
_RE_COUNTRY = re.compile(r"'country':\s*\"([^\"]+)\"")
_RE_ACTIVE = re.compile(r"'active':\s*\"([^\"]+)\"")
_RE_LOAN = re.compile(r"'loan':\s*\"([^\"]+)\"")


def _extract_records(script: str, league_name: str) -> list[dict]:
    """Split the data array into individual player blocks and parse each."""
    # Isolate the array body between `var data = [` and `];`
    start = script.index("var data = [") + len("var data = [")
    end = script.index("];", start)
    array_body = script[start:end]

    # Split on `},{` to get individual player dict strings
    blocks = re.split(r"\},\s*\{", array_body)

    records = []
    for i, block in enumerate(blocks):
        try:
            record = _parse_block(block)
            if record:
                records.append(record)
        except Exception as exc:
            logger.debug("Skipping block %d in %s: %s", i, league_name, exc)

    return records


def _parse_block(block: str) -> dict | None:
    name = _first(re.search(_RE_NAME, block))
    annual_eur = _first(re.search(_RE_ANNUAL_EUR, block))
    position = _first(re.search(_RE_POSITION, block))
    pos_detail = _first(re.search(_RE_POS_DETAIL, block))
    age = _first(re.search(_RE_AGE, block))
    club = _first(re.search(_RE_CLUB, block))
    country = _first(re.search(_RE_COUNTRY, block))
    active = _first(re.search(_RE_ACTIVE, block))
    loan = _first(re.search(_RE_LOAN, block))

    if not name or not annual_eur:
        return None

    wage_eur_annual = float(annual_eur)
    if wage_eur_annual <= 0:
        return None

    return {
        "player_name": name.strip(),
        "short_name": name.strip(),
        "primary_position": pos_detail or position or "",
        "position_group": _POS_MAP.get(position, "MID"),
        "age": int(age) if age else None,
        "club_name": club.strip() if club else "",
        "country": country or "",
        "wage_eur_weekly": round(wage_eur_annual / 52, 2),
        "ea_value_eur": None,
        "active": active == "True",
        "on_loan": loan == "True",
        "source": "capology",
    }


def _first(match: re.Match | None) -> str | None:
    return match.group(1) if match else None

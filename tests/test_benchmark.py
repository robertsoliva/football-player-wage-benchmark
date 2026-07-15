"""Tests for algorithm/benchmark.py — pure logic, no DB."""

import numpy as np
import pandas as pd
import pytest

from algorithm.benchmark import _map_position, BenchmarkResult


# ── _map_position ─────────────────────────────────────────────────────────────

def test_map_position_known():
    assert _map_position("Centre-Back") == "DEF"
    assert _map_position("Centre-Forward") == "ATT"
    assert _map_position("Central Midfield") == "MID"
    assert _map_position("Goalkeeper") == "GK"


def test_map_position_unknown_defaults_to_mid():
    assert _map_position("Unknown Role") == "MID"


def test_map_position_winger():
    assert _map_position("Left Winger") == "ATT"
    assert _map_position("Right Winger") == "ATT"


# ── BenchmarkResult ───────────────────────────────────────────────────────────

def _make_result(peer_count: int, wages: list[float]) -> BenchmarkResult:
    peers = pd.DataFrame({
        "player_name": [f"P{i}" for i in range(peer_count)],
        "club_name": ["Club"] * peer_count,
        "league_name": ["League"] * peer_count,
        "age": [25] * peer_count,
        "wage_eur_weekly": wages,
    })
    wages_year = np.array(wages) * 52
    median_w = float(np.median(wages_year))
    p25_w = float(np.percentile(wages_year, 25))
    p75_w = float(np.percentile(wages_year, 75))
    return BenchmarkResult(
        player_name="Test Player",
        position_group="ATT",
        median_wage_eur_year=median_w,
        p25_wage_eur_year=p25_w,
        p75_wage_eur_year=p75_w,
        range_pct=round((p75_w - p25_w) / median_w * 100, 1) if median_w > 0 else 0.0,
        peer_count=peer_count,
        peers=peers,
    )


def test_result_range_ordering():
    result = _make_result(20, [1000 * i for i in range(1, 21)])
    assert result.p25_wage_eur_year <= result.median_wage_eur_year <= result.p75_wage_eur_year


def test_result_range_pct_non_negative():
    result = _make_result(20, [1000 * i for i in range(1, 21)])
    assert result.range_pct >= 0


def test_result_precision_tighter_for_uniform_wages():
    uniform = _make_result(10, [5000] * 10)
    spread = _make_result(10, [1000 * i for i in range(1, 11)])
    assert uniform.range_pct < spread.range_pct


def test_result_annual_wage_is_52x_weekly():
    wages_weekly = [10_000, 20_000, 30_000, 40_000, 50_000]
    result = _make_result(5, wages_weekly)
    expected_median = np.median(wages_weekly) * 52
    assert abs(result.median_wage_eur_year - expected_median) < 0.01

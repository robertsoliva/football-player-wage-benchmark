"""Tests for the Capology HTML parser — no network calls."""

from pipeline.scraping import _parse_block, _extract_records


# ── _parse_block ──────────────────────────────────────────────────────────────

def _make_block(
    name="Test Player",
    annual_eur="5200000",
    position="F",
    pos_detail="CF",
    age="25",
    club="Test FC",
    country="Spain",
    active="True",
    loan="False",
) -> str:
    return f"""
        'name': "<a class='firstcol' href='/player/test/'><img src='flag.svg'>{ name}</a>",
        'annual_gross_eur': accounting.formatMoney("{annual_eur}", "€ ", 0),
        'position': "{position}",
        'position_detail': "{pos_detail}",
        'age': Math.round("{age}"),
        'club': "<a class='firstcol' href='/club/test/'>{ club}</a>",
        'country': "{country}",
        'active': "{active}",
        'loan': "{loan}",
    """


def test_parse_block_returns_record():
    result = _parse_block(_make_block())
    assert result is not None
    assert result["player_name"] == "Test Player"
    assert result["wage_eur_weekly"] == round(5_200_000 / 52, 2)
    assert result["position_group"] == "ATT"
    assert result["age"] == 25
    assert result["club_name"] == "Test FC"
    assert result["source"] == "capology"


def test_parse_block_position_mapping():
    assert _parse_block(_make_block(position="D"))["position_group"] == "DEF"
    assert _parse_block(_make_block(position="M"))["position_group"] == "MID"
    assert _parse_block(_make_block(position="GK"))["position_group"] == "GK"


def test_parse_block_zero_wage_returns_none():
    assert _parse_block(_make_block(annual_eur="0")) is None


def test_parse_block_missing_name_returns_none():
    block = "'annual_gross_eur': accounting.formatMoney(\"5200000\", \"€ \", 0),"
    assert _parse_block(block) is None


def test_parse_block_loan_flag():
    result = _parse_block(_make_block(loan="True"))
    assert result["on_loan"] is True


# ── _extract_records ──────────────────────────────────────────────────────────

def test_extract_records_two_players():
    block1 = _make_block(name="Alice", annual_eur="5200000")
    block2 = _make_block(name="Bob", annual_eur="2600000", position="D")
    script = f"var data = [{{{block1}}},{{{block2}}}];"
    records = _extract_records(script, "Test League")
    assert len(records) == 2
    names = {r["player_name"] for r in records}
    assert "Alice" in names and "Bob" in names


def test_extract_records_skips_invalid():
    valid = _make_block(name="Valid Player", annual_eur="1000000")
    invalid = "'name': \"<a>No wage here</a>\","
    script = f"var data = [{{{valid}}},{{{invalid}}}];"
    records = _extract_records(script, "Test League")
    assert len(records) == 1
    assert records[0]["player_name"] == "Valid Player"

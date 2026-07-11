"""Streamlit dashboard for the football wage benchmark."""

import sys
from pathlib import Path

# Make sure imports resolve whether run as `streamlit run app/dashboard.py`
# or from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import altair as alt

from algorithm.benchmark import benchmark_player, get_all_players, BenchmarkResult

st.set_page_config(
    page_title="Football Wage Benchmark",
    page_icon="⚽",
    layout="centered",
)

st.title("⚽ Football Player Wage Benchmark")
st.caption("Is this player paid above or below market rate?")

# ── Sidebar: player selector ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Select a player")

    @st.cache_data
    def load_players():
        return get_all_players().sort_values("player_name")

    players_df = load_players()
    player_names = players_df["player_name"].tolist()

    selected_name = st.selectbox("Player", player_names)

    player_info = players_df[players_df["player_name"] == selected_name].iloc[0]
    st.markdown(f"**Team:** {player_info['team_name']}")
    st.markdown(f"**Position:** {player_info['main_position']}")
    st.markdown(f"**Age:** {player_info['age']}")
    st.markdown(f"**Market value:** €{player_info['market_value']:,.0f}")
    st.markdown(f"**League:** {player_info['competition_name']}")

    st.divider()
    current_wage_input = st.number_input(
        "Known annual wage (€/year, optional)",
        min_value=0,
        value=0,
        step=100_000,
        help="Enter the player's actual wage if known to see where they rank.",
    )
    current_wage = current_wage_input if current_wage_input > 0 else None

    run_btn = st.button("Run benchmark", type="primary", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────
if not run_btn:
    st.info("Select a player in the sidebar and click **Run benchmark**.")
    st.stop()

with st.spinner("Computing peer group…"):
    try:
        result: BenchmarkResult = benchmark_player(selected_name, current_wage)
    except FileNotFoundError:
        st.error(
            "Salary database not found. "
            "Run `python -m pipeline.run` first to build it."
        )
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

# ── Confidence badge ──────────────────────────────────────────────────────────
badge_color = {"High": "green", "Medium": "orange", "Low": "red"}.get(
    result.confidence, "grey"
)
st.markdown(
    f"**Confidence:** :{badge_color}[{result.confidence}]  "
    f"({result.peer_count} comparable peers found)"
)

if result.confidence == "Insufficient data":
    st.warning("Not enough peers to compute a reliable benchmark for this player.")
    st.stop()

# ── Salary range ──────────────────────────────────────────────────────────────
st.subheader("Expected salary range")

col1, col2, col3 = st.columns(3)
col1.metric("25th percentile", f"€{result.p25_wage_eur_year:,.0f} / yr")
col2.metric("Median (estimate)", f"€{result.median_wage_eur_year:,.0f} / yr")
col3.metric("75th percentile", f"€{result.p75_wage_eur_year:,.0f} / yr")

if current_wage:
    pct = getattr(result, "current_wage_percentile", None)
    if pct is not None:
        label = "above" if pct >= 50 else "below"
        st.info(
            f"With a reported wage of **€{current_wage:,.0f}/yr**, "
            f"this player is at the **{pct:.0f}th percentile** — "
            f"**{label} the median** of their peer group."
        )

# ── Peer distribution chart ───────────────────────────────────────────────────
st.subheader("Peer wage distribution")

peers = result.peers.copy()
peers["wage_eur_year"] = peers["wage_eur_weekly"] * 52

range_df = pd.DataFrame({
    "label": ["P25", "Median", "P75"],
    "value": [result.p25_wage_eur_year, result.median_wage_eur_year, result.p75_wage_eur_year],
})

hist = (
    alt.Chart(peers)
    .mark_bar(color="#4a90d9", opacity=0.7)
    .encode(
        alt.X("wage_eur_year:Q", bin=alt.Bin(maxbins=15), title="Annual wage (€)"),
        alt.Y("count()", title="Number of peers"),
    )
)

rules = (
    alt.Chart(range_df)
    .mark_rule(strokeDash=[6, 3])
    .encode(
        x=alt.X("value:Q"),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(
                domain=["P25", "Median", "P75"],
                range=["#f0a500", "#e63946", "#f0a500"],
            ),
        ),
        size=alt.value(2),
    )
)

if current_wage:
    player_line = (
        alt.Chart(pd.DataFrame({"value": [current_wage], "label": ["Current wage"]}))
        .mark_rule(color="green", strokeDash=[4, 2], size=2)
        .encode(x="value:Q")
    )
    chart = hist + rules + player_line
else:
    chart = hist + rules

st.altair_chart(chart, use_container_width=True)

# ── Peer table ────────────────────────────────────────────────────────────────
st.subheader(f"Peer players used ({result.peer_count})")

display = peers.rename(columns={
    "player_name": "Player",
    "club_name": "Club",
    "league_name": "League",
    "age": "Age",
    "wage_eur_weekly": "Wage (€/week)",
    "wage_eur_year": "Wage (€/year)",
})

st.dataframe(
    display[["Player", "Club", "League", "Age", "Wage (€/week)", "Wage (€/year)"]]
    .sort_values("Wage (€/year)", ascending=False)
    .reset_index(drop=True),
    use_container_width=True,
)

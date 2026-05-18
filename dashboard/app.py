# =============================================================
# IPL ANALYTICS — WEEK 4: STREAMLIT DASHBOARD
# Run: streamlit run dashboard/app.py
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Analytics Engine",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d1117; }
  [data-testid="stSidebar"]          { background: #161b22; }
  h1, h2, h3                         { color: #f97316 !important; }
  .metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 16px; margin: 6px 0;
  }
  .stMetric label { color: #8b949e !important; }
  .stMetric [data-testid="stMetricValue"] { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(__file__) + "/../data"
    bat  = pd.read_csv(f"{base}/batting_with_clusters.csv")
    bowl = pd.read_csv(f"{base}/bowling_features.csv")
    mat  = pd.read_csv(f"{base}/matches_clean.csv",
                       parse_dates=["date"])
    mb   = pd.read_csv(f"{base}/match_batting.csv")
    return bat, bowl, mat, mb

@st.cache_resource
def load_models():
    base = os.path.dirname(__file__) + "/../src"
    rf   = joblib.load(f"{base}/batting_runs_model.pkl")
    km   = joblib.load(f"{base}/archetype_kmeans.pkl")
    sc   = joblib.load(f"{base}/cluster_scaler.pkl")
    gb   = joblib.load(f"{base}/match_outcome_model.pkl")
    return rf, km, sc, gb

bat, bowl, matches, mb = load_data()
rf_model, km_model, cluster_scaler, outcome_model = load_models()

ARCHETYPE_NAMES = {
    0: "Powerplay Dominator",
    1: "Finisher / Death Specialist",
    2: "Anchor / Run Machine",
    3: "Explosive Hitter",
    4: "Utility Batter",
}
bat["archetype"] = bat["cluster"].map(ARCHETYPE_NAMES).fillna("Unknown")

# ─────────────────────────────────────────────────────────────
# SIDEBAR NAV
# ─────────────────────────────────────────────────────────────
st.sidebar.title("🏏 IPL Analytics")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Player Explorer", "Team Builder (XI)", "Match Predictor"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Built with Python · scikit-learn · Streamlit")


# ═════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("IPL Player Performance Analytics Engine")
    st.markdown("**End-to-end ML project** — EDA · Feature Engineering · Clustering · Prediction")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players (Batters)",  f"{len(bat):,}")
    c2.metric("Players (Bowlers)",  f"{len(bowl):,}")
    c3.metric("Total Matches",       f"{len(matches):,}")
    c4.metric("Seasons Covered",     f"{matches['season'].nunique()}")

    st.markdown("---")

    # Top run-scorers
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 15 Run Scorers")
        top_bat = bat.nlargest(15, "total_runs")
        fig = px.bar(top_bat, x="total_runs", y="batter",
                     orientation="h", color="pvs_batting",
                     color_continuous_scale="Oranges",
                     labels={"total_runs": "Total Runs", "batter": ""},
                     template="plotly_dark")
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                          coloraxis_showscale=False, yaxis={"autorange": "reversed"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 15 Wicket Takers")
        top_bowl = bowl.nlargest(15, "total_wickets")
        fig = px.bar(top_bowl, x="total_wickets", y="bowler",
                     orientation="h", color="pvs_bowling",
                     color_continuous_scale="Greens",
                     labels={"total_wickets": "Total Wickets", "bowler": ""},
                     template="plotly_dark")
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                          coloraxis_showscale=False, yaxis={"autorange": "reversed"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Player Archetypes (Clustering)")
    fig = px.scatter(
        bat.dropna(subset=["cluster"]),
        x="avg_runs", y="strike_rate",
        color="archetype", size="total_runs",
        hover_name="batter", size_max=30,
        color_discrete_sequence=px.colors.qualitative.Vivid,
        template="plotly_dark",
        labels={"avg_runs": "Batting Average", "strike_rate": "Strike Rate"},
    )
    fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE 2 — PLAYER EXPLORER
# ═════════════════════════════════════════════════════════════
elif page == "Player Explorer":
    st.title("Player Explorer")

    mode = st.radio("Select type", ["Batter", "Bowler"], horizontal=True)

    if mode == "Batter":
        player = st.selectbox("Choose a batter",
                              sorted(bat["batter"].unique()))
        row = bat[bat["batter"] == player].iloc[0]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Runs",    f"{int(row['total_runs']):,}")
        c2.metric("Innings",       f"{int(row['innings'])}")
        c3.metric("Batting Avg",   f"{row['avg_runs']:.1f}")
        c4.metric("Strike Rate",   f"{row['strike_rate']:.1f}")
        c5.metric("PVS Score",     f"{row['pvs_batting']:.1f}/100")

        st.info(f"**Archetype:** {row['archetype']}")

        # Form trend
        player_hist = mb[mb["batter"] == player].copy()
        if len(player_hist) > 0:
            player_hist = player_hist.merge(
                matches[["id", "date"]].rename(columns={"id": "match_id"}),
                on="match_id", how="left"
            ).sort_values("date")
            player_hist["rolling10"] = player_hist["runs"].rolling(10, min_periods=3).mean()

            st.subheader("Innings-by-Innings Form")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(range(len(player_hist))),
                                  y=player_hist["runs"],
                                  marker_color="#30363d", name="Runs"))
            fig.add_trace(go.Scatter(x=list(range(len(player_hist))),
                                      y=player_hist["rolling10"],
                                      line=dict(color="#f97316", width=2),
                                      name="10-inning avg"))
            fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                              font_color="#e6edf3", xaxis_title="Innings",
                              yaxis_title="Runs")
            st.plotly_chart(fig, use_container_width=True)

    else:
        player = st.selectbox("Choose a bowler",
                              sorted(bowl["bowler"].unique()))
        row = bowl[bowl["bowler"] == player].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Wickets", f"{int(row['total_wickets'])}")
        c2.metric("Economy",       f"{row['economy']:.2f}")
        c3.metric("Bowling Avg",   f"{row['bowling_avg']:.1f}" if not pd.isna(row['bowling_avg']) else "—")
        c4.metric("PVS Score",     f"{row['pvs_bowling']:.1f}/100")


# ═════════════════════════════════════════════════════════════
# PAGE 3 — TEAM BUILDER
# ═════════════════════════════════════════════════════════════
elif page == "Team Builder (XI)":
    st.title("Optimal XI Builder")
    st.markdown("Select players to compare and build your best playing XI.")

    tab1, tab2 = st.tabs(["Pick Players", "Team Summary"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Batters by PVS")
            n = st.slider("Show top N batters", 10, 50, 20)
            top = bat.nlargest(n, "pvs_batting")[
                ["batter", "total_runs", "avg_runs", "strike_rate",
                 "pvs_batting", "archetype"]
            ]
            st.dataframe(top.reset_index(drop=True), use_container_width=True)

        with col2:
            st.subheader("Top Bowlers by PVS")
            n2 = st.slider("Show top N bowlers", 10, 50, 20)
            topb = bowl.nlargest(n2, "pvs_bowling")[
                ["bowler", "total_wickets", "economy", "bowling_avg", "pvs_bowling"]
            ]
            st.dataframe(topb.reset_index(drop=True), use_container_width=True)

    with tab2:
        st.subheader("Build Your XI")
        selected_bat  = st.multiselect("Pick batters (6–7)", sorted(bat["batter"].unique()),
                                        max_selections=7)
        selected_bowl = st.multiselect("Pick bowlers (4–5)", sorted(bowl["bowler"].unique()),
                                        max_selections=5)

        if len(selected_bat) + len(selected_bowl) == 11:
            st.success("✅ Full XI selected!")
            total_pvs_bat  = bat[bat["batter"].isin(selected_bat)]["pvs_batting"].sum()
            total_pvs_bowl = bowl[bowl["bowler"].isin(selected_bowl)]["pvs_bowling"].sum()
            st.metric("Combined Team Strength",
                      f"{(total_pvs_bat + total_pvs_bowl) / 11:.1f} avg PVS")
        else:
            st.warning(f"Select exactly 11 players ({len(selected_bat)+len(selected_bowl)}/11 chosen)")


# ═════════════════════════════════════════════════════════════
# PAGE 4 — MATCH PREDICTOR
# ═════════════════════════════════════════════════════════════
elif page == "Match Predictor":
    st.title("Match Outcome Predictor")
    st.markdown("Predict which team is more likely to win based on match conditions.")

    teams = sorted(matches["team1"].dropna().unique())

    c1, c2 = st.columns(2)
    with c1:
        team1 = st.selectbox("Team 1", teams, index=0)
    with c2:
        team2 = st.selectbox("Team 2", [t for t in teams if t != team1], index=0)

    toss_winner = st.radio("Toss won by", [team1, team2], horizontal=True)
    toss_dec    = st.radio("Toss decision", ["bat", "field"], horizontal=True)
    season      = st.slider("Season", 2008, 2024, 2023)

    if st.button("Predict Winner 🏏", type="primary"):
        team_enc = {t: i for i, t in enumerate(matches["team1"].dropna().unique())}
        t1_enc   = team_enc.get(team1, -1)
        t2_enc   = team_enc.get(team2, -1)
        toss_bat = 1 if toss_dec == "bat" else 0
        s_norm   = (season - 2008) / (2024 - 2008)

        X = pd.DataFrame([[toss_bat, s_norm, t1_enc, t2_enc]],
                          columns=["toss_win_bat", "season_norm",
                                   "team1_enc", "team2_enc"])
        prob = outcome_model.predict_proba(X)[0]

        st.markdown("---")
        col1, col2 = st.columns(2)
        col1.metric(f"{team1} Win Probability", f"{prob[1]*100:.1f}%")
        col2.metric(f"{team2} Win Probability", f"{prob[0]*100:.1f}%")

        fig = go.Figure(go.Bar(
            x=[team1, team2],
            y=[prob[1]*100, prob[0]*100],
            marker_color=["#f97316", "#58a6ff"],
            text=[f"{prob[1]*100:.1f}%", f"{prob[0]*100:.1f}%"],
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font_color="#e6edf3", yaxis_title="Win %",
            yaxis_range=[0, 100], showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

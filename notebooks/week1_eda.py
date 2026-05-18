# =============================================================
# IPL PLAYER PERFORMANCE PREDICTION — WEEK 1: EDA & DATA PREP
# =============================================================
# Dataset: Download from Kaggle
#   → https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020
#   → Place 'matches.csv' and 'deliveries.csv' in the ../data/ folder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Plotting style ──────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#0d1117",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#e6edf3",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.8,
    "font.family":      "monospace",
})
ACCENT   = "#f97316"   # orange
BLUE     = "#58a6ff"
GREEN    = "#3fb950"


# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  IPL Analytics — Week 1: Data Loading & EDA")
print("=" * 60)

matches    = pd.read_csv("C:/Users/sidha/ipl-analytics/data/matches.csv")
deliveries = pd.read_csv("C:/Users/sidha/ipl-analytics/data/deliveries.csv")

print(f"\n✅ matches.csv    : {matches.shape[0]:,} rows × {matches.shape[1]} cols")
print(f"✅ deliveries.csv : {deliveries.shape[0]:,} rows × {deliveries.shape[1]} cols")

print("\n── matches columns ──")
print(matches.columns.tolist())
print("\n── deliveries columns ──")
print(deliveries.columns.tolist())


# ─────────────────────────────────────────────────────────────
# 2. BASIC CLEANING
# ─────────────────────────────────────────────────────────────
print("\n\n[1/5] Cleaning data...")

# ── matches ──
matches["date"] = pd.to_datetime(matches["date"])
matches["season"] = matches["date"].dt.year

# Standardise team names (some changed over the years)
team_rename = {
    "Delhi Daredevils":      "Delhi Capitals",
    "Deccan Chargers":       "Sunrisers Hyderabad",
    "Rising Pune Supergiant":"Rising Pune Supergiants",
    "Kings XI Punjab":       "Punjab Kings",
}
for col in ["team1", "team2", "winner", "toss_winner"]:
    if col in matches.columns:
        matches[col] = matches[col].replace(team_rename)

# ── deliveries ──
# Total runs per ball (bat + extras)
deliveries["total_runs"] = deliveries["total_runs"].fillna(0)

# Legal deliveries (not wides/no-balls for economy calc)
deliveries["is_legal"] = ~deliveries["extras_type"].isin(["wides", "noballs"])

null_pct = (deliveries.isnull().sum() / len(deliveries) * 100).round(2)
print("Null % per column (deliveries):")
print(null_pct[null_pct > 0].to_string())
print("\n✅ Cleaning done.")


# ─────────────────────────────────────────────────────────────
# 3. BATTING AGGREGATIONS
# ─────────────────────────────────────────────────────────────
print("\n[2/5] Computing batting stats...")

batting = (
    deliveries
    .groupby(["match_id", "batting_team", "batter"])
    .agg(
        runs        = ("batsman_runs", "sum"),
        balls_faced = ("is_legal",     "sum"),
        fours       = ("batsman_runs", lambda x: (x == 4).sum()),
        sixes       = ("batsman_runs", lambda x: (x == 6).sum()),
    )
    .reset_index()
)

# Career batting summary
career_bat = (
    batting
    .groupby("batter")
    .agg(
        innings     = ("match_id",    "nunique"),
        total_runs  = ("runs",        "sum"),
        total_balls = ("balls_faced", "sum"),
        avg_runs    = ("runs",        "mean"),
        fours       = ("fours",       "sum"),
        sixes       = ("sixes",       "sum"),
        fifties     = ("runs",        lambda x: (x >= 50).sum()),
        hundreds    = ("runs",        lambda x: (x >= 100).sum()),
    )
    .reset_index()
)

career_bat["strike_rate"] = (career_bat["total_runs"] / career_bat["total_balls"] * 100).round(2)
career_bat["avg_runs"]    = career_bat["avg_runs"].round(2)

# Filter to meaningful sample (≥20 innings)
career_bat = career_bat[career_bat["innings"] >= 20].copy()
print(f"  Batters with ≥20 innings: {len(career_bat)}")


# ─────────────────────────────────────────────────────────────
# 4. BOWLING AGGREGATIONS
# ─────────────────────────────────────────────────────────────
print("\n[3/5] Computing bowling stats...")

bowling = (
    deliveries
    .groupby(["match_id", "bowling_team", "bowler"])
    .agg(
        runs_conceded = ("total_runs",     "sum"),
        wickets       = ("is_wicket",      "sum"),
        legal_balls   = ("is_legal",       "sum"),
    )
    .reset_index()
)
bowling["overs"] = bowling["legal_balls"] / 6

career_bowl = (
    bowling
    .groupby("bowler")
    .agg(
        matches       = ("match_id",      "nunique"),
        total_wickets = ("wickets",       "sum"),
        total_runs    = ("runs_conceded", "sum"),
        total_overs   = ("overs",         "sum"),
        avg_wickets   = ("wickets",       "mean"),
    )
    .reset_index()
)

career_bowl["economy"]      = (career_bowl["total_runs"] / career_bowl["total_overs"]).round(2)
career_bowl["bowling_avg"]  = np.where(
    career_bowl["total_wickets"] > 0,
    (career_bowl["total_runs"] / career_bowl["total_wickets"]).round(2),
    np.nan
)

career_bowl = career_bowl[career_bowl["matches"] >= 15].copy()
print(f"  Bowlers with ≥15 matches: {len(career_bowl)}")


# ─────────────────────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────────────────────
print("\n[4/5] Generating plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("IPL Analytics Dashboard — EDA Overview", fontsize=16,
             fontweight="bold", color=ACCENT, y=1.01)


# ── Plot 1: Top 10 run-scorers ──────────────────────────────
ax = axes[0, 0]
top_bat = career_bat.nlargest(10, "total_runs")
bars = ax.barh(top_bat["batter"], top_bat["total_runs"], color=BLUE)
ax.bar_label(bars, fmt="%,.0f", padding=4, fontsize=8, color="#e6edf3")
ax.set_title("Top 10 Run Scorers (Career)", color=ACCENT)
ax.set_xlabel("Total Runs")
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.4)


# ── Plot 2: Strike Rate vs Avg Runs (scatter) ───────────────
ax = axes[0, 1]
sc = ax.scatter(
    career_bat["avg_runs"], career_bat["strike_rate"],
    c=career_bat["total_runs"], cmap="YlOrRd",
    alpha=0.7, s=60, edgecolors="none"
)
plt.colorbar(sc, ax=ax, label="Total Runs")
ax.set_title("Strike Rate vs Batting Average", color=ACCENT)
ax.set_xlabel("Batting Average (runs/innings)")
ax.set_ylabel("Strike Rate")
ax.axhline(130, color=GREEN,  lw=0.8, ls="--", alpha=0.6, label="SR 130")
ax.axvline(25,  color=ACCENT, lw=0.8, ls="--", alpha=0.6, label="Avg 25")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)


# ── Plot 3: Top 10 wicket-takers ────────────────────────────
ax = axes[0, 2]
top_bowl = career_bowl.nlargest(10, "total_wickets")
bars = ax.barh(top_bowl["bowler"], top_bowl["total_wickets"], color=GREEN)
ax.bar_label(bars, padding=4, fontsize=8, color="#e6edf3")
ax.set_title("Top 10 Wicket Takers (Career)", color=ACCENT)
ax.set_xlabel("Total Wickets")
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.4)


# ── Plot 4: Matches per season ──────────────────────────────
ax = axes[1, 0]
season_counts = matches.groupby("season").size()
ax.plot(season_counts.index, season_counts.values,
        color=ACCENT, marker="o", linewidth=2, markersize=6)
ax.fill_between(season_counts.index, season_counts.values, alpha=0.15, color=ACCENT)
ax.set_title("Matches Played per Season", color=ACCENT)
ax.set_xlabel("Season")
ax.set_ylabel("Number of Matches")
ax.grid(alpha=0.3)


# ── Plot 5: Toss decision distribution ──────────────────────
ax = axes[1, 1]
toss_dec = matches["toss_decision"].value_counts()
wedge_props = {"linewidth": 2, "edgecolor": "#0d1117"}
ax.pie(toss_dec.values, labels=toss_dec.index, autopct="%1.1f%%",
       colors=[BLUE, GREEN], wedgeprops=wedge_props,
       textprops={"color": "#e6edf3"})
ax.set_title("Toss Decision Distribution", color=ACCENT)


# ── Plot 6: Economy vs Wickets (bowlers) ────────────────────
ax = axes[1, 2]
sc2 = ax.scatter(
    career_bowl["economy"], career_bowl["total_wickets"],
    c=career_bowl["bowling_avg"], cmap="RdYlGn_r",
    alpha=0.7, s=60, edgecolors="none"
)
plt.colorbar(sc2, ax=ax, label="Bowling Avg")
ax.set_title("Economy vs Total Wickets", color=ACCENT)
ax.set_xlabel("Economy Rate (runs/over)")
ax.set_ylabel("Total Wickets")
ax.axvline(7.5, color=ACCENT, lw=0.8, ls="--", alpha=0.6, label="Economy 7.5")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)


plt.tight_layout()
plt.savefig("C:/Users/sidha/ipl-analytics/data/eda_overview.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
print("  Saved → data/eda_overview.png")
plt.show()


# ─────────────────────────────────────────────────────────────
# 6. SAVE CLEAN DATA
# ─────────────────────────────────────────────────────────────
print("\n[5/5] Saving processed data...")

career_bat.to_csv("C:/Users/sidha/ipl-analytics/data/career_batting.csv",  index=False)
career_bowl.to_csv("C:/Users/sidha/ipl-analytics/data/career_bowling.csv", index=False)
batting.to_csv("C:/Users/sidha/ipl-analytics/data/match_batting.csv",       index=False)
matches.to_csv("C:/Users/sidha/ipl-analytics/data/matches_clean.csv",        index=False)

print("  Saved → data/career_batting.csv")
print("  Saved → data/career_bowling.csv")
print("  Saved → data/match_batting.csv")
print("  Saved → data/matches_clean.csv")

print("\n" + "=" * 60)
print("  Week 1 Complete! 🏏")
print("  Next → Week 2: Feature Engineering & Player Value Score")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# QUICK STATS PRINTOUT
# ─────────────────────────────────────────────────────────────
print("\n── Top 5 Batters by Total Runs ──")
print(career_bat.nlargest(5, "total_runs")[
    ["batter","innings","total_runs","avg_runs","strike_rate","fifties","hundreds"]
].to_string(index=False))

print("\n── Top 5 Bowlers by Wickets ──")
print(career_bowl.nlargest(5, "total_wickets")[
    ["bowler","matches","total_wickets","economy","bowling_avg"]
].to_string(index=False))

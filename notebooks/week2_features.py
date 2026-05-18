# =============================================================
# IPL ANALYTICS — WEEK 2: FEATURE ENGINEERING
# Player Value Score + Rolling Stats + Phase Analysis
# =============================================================
# Run AFTER week1_eda.py has saved the cleaned CSVs.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#0d1117",
    "axes.edgecolor": "#30363d",   "axes.labelcolor": "#e6edf3",
    "xtick.color": "#8b949e",      "ytick.color": "#8b949e",
    "text.color": "#e6edf3",       "grid.color": "#21262d",
    "font.family": "monospace",
})
ACCENT, BLUE, GREEN = "#f97316", "#58a6ff", "#3fb950"

print("=" * 60)
print("  IPL Analytics — Week 2: Feature Engineering")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────
deliveries  = pd.read_csv("C:/Users/sidha/ipl-analytics/data/deliveries.csv")
matches     = pd.read_csv("C:/Users/sidha/ipl-analytics/data/matches_clean.csv", parse_dates=["date"])
career_bat  = pd.read_csv("C:/Users/sidha/ipl-analytics/data/career_batting.csv")
career_bowl = pd.read_csv("C:/Users/sidha/ipl-analytics/data/career_bowling.csv")

deliveries["is_legal"] = ~deliveries["extras_type"].isin(["wides", "noballs"])
deliveries["over_num"] = deliveries["over"]   # 0-indexed over number (0–19)

# ─────────────────────────────────────────────────────────────
# 1. PHASE-BASED STATS (Powerplay / Middle / Death)
# ─────────────────────────────────────────────────────────────
print("\n[1/4] Computing phase-based batting stats...")

def phase(over):
    if over <= 5:   return "Powerplay"
    if over <= 14:  return "Middle"
    return "Death"

deliveries["phase"] = deliveries["over_num"].apply(phase)

phase_bat = (
    deliveries.groupby(["batter", "phase"])
    .agg(
        runs   = ("batsman_runs", "sum"),
        balls  = ("is_legal",     "sum"),
    )
    .reset_index()
)
phase_bat["strike_rate"] = (phase_bat["runs"] / phase_bat["balls"] * 100).round(2)

# Pivot so each phase is a feature column
phase_pivot = phase_bat.pivot_table(
    index="batter", columns="phase",
    values=["runs", "strike_rate"], aggfunc="sum"
).fillna(0)
phase_pivot.columns = ["_".join(c) for c in phase_pivot.columns]
phase_pivot = phase_pivot.reset_index()

# ─────────────────────────────────────────────────────────────
# 2. ROLLING FORM (last 10 innings)
# ─────────────────────────────────────────────────────────────
print("[2/4] Computing rolling form (last 10 innings)...")

match_bat = pd.read_csv("C:/Users/sidha/ipl-analytics/data/match_batting.csv")
match_bat = match_bat.merge(
    matches[["id", "date", "season"]].rename(columns={"id": "match_id"}),
    on="match_id", how="left"
).sort_values(["batter", "date"])

match_bat["rolling_runs_10"] = (
    match_bat.groupby("batter")["runs"]
    .transform(lambda x: x.rolling(10, min_periods=3).mean())
    .round(2)
)
match_bat["rolling_sr_10"] = (
    match_bat.groupby("batter")["balls_faced"]
    .transform(lambda x: x.rolling(10, min_periods=3).mean())
)

# Most recent form per player
recent_form = (
    match_bat.groupby("batter")
    .last()
    .reset_index()[["batter", "rolling_runs_10"]]
    .rename(columns={"rolling_runs_10": "form_avg_last10"})
)

# ─────────────────────────────────────────────────────────────
# 3. VENUE & OPPOSITION PERFORMANCE
# ─────────────────────────────────────────────────────────────
print("[3/4] Computing venue performance...")

del_with_venue = deliveries.merge(
    matches[["id", "venue"]].rename(columns={"id": "match_id"}),
    on="match_id", how="left"
)

venue_bat = (
    del_with_venue.groupby(["batter", "venue"])
    .agg(runs=("batsman_runs", "sum"), balls=("is_legal", "sum"))
    .reset_index()
)
venue_bat["venue_sr"] = (venue_bat["runs"] / venue_bat["balls"] * 100).round(2)

best_venue = (
    venue_bat[venue_bat["balls"] >= 30]
    .sort_values("venue_sr", ascending=False)
    .groupby("batter")
    .first()
    .reset_index()[["batter", "venue", "runs", "venue_sr"]]
    .rename(columns={"venue": "best_venue", "runs": "best_venue_runs",
                     "venue_sr": "best_venue_sr"})
)

# ─────────────────────────────────────────────────────────────
# 4. PLAYER VALUE SCORE (PVS)
# ─────────────────────────────────────────────────────────────
print("[4/4] Computing Player Value Score (PVS)...")

# ── Batting PVS ─────────────────────────────────────────────
# Combine career stats + phase + form
bat_features = (
    career_bat
    .merge(phase_pivot,  on="batter", how="left")
    .merge(recent_form,  on="batter", how="left")
    .merge(best_venue,   on="batter", how="left")
    .fillna(0)
)

def minmax(s):
    """Normalise a series to [0, 1]."""
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)

# Weighted formula — tweak weights to experiment!
bat_features["pvs_batting"] = (
    0.30 * minmax(bat_features["avg_runs"])       +   # consistency
    0.25 * minmax(bat_features["strike_rate"])    +   # aggression
    0.20 * minmax(bat_features["total_runs"])     +   # volume
    0.15 * minmax(bat_features["form_avg_last10"].fillna(bat_features["avg_runs"])) +  # recent form
    0.05 * minmax(bat_features["sixes"])          +   # six-hitting
    0.05 * minmax(bat_features["fifties"] + bat_features["hundreds"] * 2)  # milestones
)
bat_features["pvs_batting"] = (bat_features["pvs_batting"] * 100).round(2)

# ── Bowling PVS ─────────────────────────────────────────────
bowl_features = career_bowl.copy()
bowl_features["pvs_bowling"] = (
    0.40 * minmax(bowl_features["total_wickets"]) +
    0.35 * (1 - minmax(bowl_features["economy"]))  +   # lower is better
    0.25 * (1 - minmax(bowl_features["bowling_avg"].fillna(50)))
)
bowl_features["pvs_bowling"] = (bowl_features["pvs_bowling"] * 100).round(2)

# ── Save ────────────────────────────────────────────────────
bat_features.to_csv("C:/Users/sidha/ipl-analytics/data/batting_features.csv",   index=False)
bowl_features.to_csv("C:/Users/sidha/ipl-analytics/data/bowling_features.csv",  index=False)

print("\n── Top 10 Batters by Player Value Score ──")
print(
    bat_features.nlargest(10, "pvs_batting")[
        ["batter", "total_runs", "avg_runs", "strike_rate",
         "form_avg_last10", "pvs_batting"]
    ].to_string(index=False)
)

print("\n── Top 10 Bowlers by Player Value Score ──")
print(
    bowl_features.nlargest(10, "pvs_bowling")[
        ["bowler", "total_wickets", "economy", "bowling_avg", "pvs_bowling"]
    ].to_string(index=False)
)

# ── Quick plot: PVS distribution ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Player Value Score (PVS) Distribution", fontsize=14,
             fontweight="bold", color=ACCENT)

for ax, df, col, title, color in [
    (axes[0], bat_features,  "pvs_batting",  "Batters",  BLUE),
    (axes[1], bowl_features, "pvs_bowling",  "Bowlers",  GREEN),
]:
    ax.hist(df[col], bins=25, color=color, alpha=0.8, edgecolor="#0d1117")
    ax.axvline(df[col].mean(), color=ACCENT, lw=1.5, ls="--",
               label=f"Mean {df[col].mean():.1f}")
    ax.set_title(f"PVS Distribution — {title}", color=ACCENT)
    ax.set_xlabel("Player Value Score")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("C:/Users/sidha/ipl-analytics/data/pvs_distribution.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
print("\n  Saved → data/pvs_distribution.png")

print("\n" + "=" * 60)
print("  Week 2 Complete! 🏏")
print("  Next → Week 3: ML Models (regression + clustering)")
print("=" * 60)

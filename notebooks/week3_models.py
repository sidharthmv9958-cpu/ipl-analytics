# =============================================================
# IPL ANALYTICS — WEEK 3: ML MODELS
# Regression (predict runs) + Clustering (player archetypes)
# + Match outcome classifier
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.ensemble         import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model     import Ridge
from sklearn.cluster          import KMeans
from sklearn.metrics          import (mean_absolute_error, r2_score,
                                      classification_report, silhouette_score)
from sklearn.impute            import SimpleImputer

plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#0d1117",
    "axes.edgecolor": "#30363d",   "axes.labelcolor": "#e6edf3",
    "xtick.color": "#8b949e",      "ytick.color": "#8b949e",
    "text.color": "#e6edf3",       "grid.color": "#21262d",
    "font.family": "monospace",
})
ACCENT, BLUE, GREEN = "#f97316", "#58a6ff", "#3fb950"
CLUSTER_COLORS = ["#f97316", "#58a6ff", "#3fb950", "#a371f7", "#f85149"]

print("=" * 60)
print("  IPL Analytics — Week 3: ML Modelling")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# LOAD FEATURES
# ─────────────────────────────────────────────────────────────
bat  = pd.read_csv("C:/Users/sidha/ipl-analytics/data/batting_features.csv")
bowl = pd.read_csv("C:/Users/sidha/ipl-analytics/data/bowling_features.csv")

# ─────────────────────────────────────────────────────────────
# MODEL A — PREDICT BATTING RUNS (Regression)
# ─────────────────────────────────────────────────────────────
print("\n[1/3] Training batting runs predictor...")

BAT_FEATURES = [
    "innings", "total_balls", "fours", "sixes",
    "fifties", "hundreds", "strike_rate",
    "form_avg_last10",
]
# fill missing cols with 0
for c in BAT_FEATURES:
    if c not in bat.columns:
        bat[c] = 0

X_bat = bat[BAT_FEATURES].copy()
y_bat = bat["total_runs"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X_bat, y_bat, test_size=0.2, random_state=42
)

rf_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("model",   RandomForestRegressor(n_estimators=200, random_state=42,
                                       max_depth=8, n_jobs=-1)),
])
rf_pipe.fit(X_train, y_train)
y_pred = rf_pipe.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
cv  = cross_val_score(rf_pipe, X_bat, y_bat, cv=5, scoring="r2").mean()

print(f"  MAE        : {mae:.1f} runs")
print(f"  R² (test)  : {r2:.3f}")
print(f"  R² (CV-5)  : {cv:.3f}")

# Feature importance
fi = pd.Series(
    rf_pipe.named_steps["model"].feature_importances_, index=BAT_FEATURES
).sort_values(ascending=False)

# ─────────────────────────────────────────────────────────────
# MODEL B — PLAYER ARCHETYPES (Clustering)
# ─────────────────────────────────────────────────────────────
print("\n[2/3] Clustering player archetypes...")

CLUSTER_FEATURES = ["avg_runs", "strike_rate", "pvs_batting",
                    "sixes", "fifties", "hundreds"]
for c in CLUSTER_FEATURES:
    if c not in bat.columns:
        bat[c] = 0

X_cluster = bat[CLUSTER_FEATURES].dropna()
valid_idx  = X_cluster.index

scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X_cluster)

# Elbow method to pick k
inertias = []
sil_scores = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_sc)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_sc, labels))

best_k = 5
print(f"  Best k (silhouette) : {best_k}")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
bat.loc[valid_idx, "cluster"] = km_final.fit_predict(X_sc)

# Name clusters by dominant stats
cluster_summary = (
    bat.loc[valid_idx]
    .groupby("cluster")[CLUSTER_FEATURES]
    .mean()
    .round(2)
)
print("\nCluster Centroids:")
print(cluster_summary.to_string())

ARCHETYPE_NAMES = {
    0: "Powerplay Dominator",
    1: "Finisher / Death Specialist",
    2: "Anchor / Run Machine",
    3: "Explosive Hitter",
    4: "Utility Batter",
}
bat["archetype"] = bat["cluster"].map(ARCHETYPE_NAMES).fillna("Unknown")

# ─────────────────────────────────────────────────────────────
# MODEL C — MATCH OUTCOME CLASSIFIER
# ─────────────────────────────────────────────────────────────
print("\n[3/3] Training match outcome classifier...")

matches = pd.read_csv("C:/Users/sidha/ipl-analytics/data/matches_clean.csv")

# Simple features: toss, venue encoding, season
matches["toss_win_bat"] = (matches["toss_decision"] == "bat").astype(int)
matches["home_win"]     = (matches["toss_winner"] == matches["winner"]).astype(int)
matches["season_norm"]  = (matches["season"] - matches["season"].min()) / \
                          (matches["season"].max() - matches["season"].min())

# Encode teams
team_enc = {t: i for i, t in enumerate(matches["team1"].unique())}
matches["team1_enc"] = matches["team1"].map(team_enc).fillna(-1)
matches["team2_enc"] = matches["team2"].map(team_enc).fillna(-1)
matches["winner_enc"]= matches["winner"].map(team_enc).fillna(-1)

# target: did team1 win?
matches["team1_won"] = (matches["winner"] == matches["team1"]).astype(int)

MC_FEATURES = ["toss_win_bat", "season_norm", "team1_enc", "team2_enc"]
mc_data = matches[MC_FEATURES + ["team1_won"]].dropna()

X_mc = mc_data[MC_FEATURES]
y_mc = mc_data["team1_won"]

X_tr, X_te, y_tr, y_te = train_test_split(X_mc, y_mc, test_size=0.2, random_state=42)

gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                 max_depth=3, random_state=42)
gb.fit(X_tr, y_tr)
print("\nMatch Outcome Classifier Report:")
print(classification_report(y_te, gb.predict(X_te)))

# ─────────────────────────────────────────────────────────────
# SAVE MODELS & DATA
# ─────────────────────────────────────────────────────────────
import os
os.makedirs("C:/Users/sidha/ipl-analytics/src", exist_ok=True)

joblib.dump(rf_pipe,  "C:/Users/sidha/ipl-analytics/src/batting_runs_model.pkl")
joblib.dump(km_final, "C:/Users/sidha/ipl-analytics/src/archetype_kmeans.pkl")
joblib.dump(scaler,   "C:/Users/sidha/ipl-analytics/src/cluster_scaler.pkl")
joblib.dump(gb,       "C:/Users/sidha/ipl-analytics/src/match_outcome_model.pkl")
bat.to_csv("C:/Users/sidha/ipl-analytics/data/batting_with_clusters.csv", index=False)

print("\n  Models saved → src/")

# ─────────────────────────────────────────────────────────────
# VISUALISE
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Week 3 — Model Results", fontsize=14,
             fontweight="bold", color=ACCENT)

# Plot 1: Actual vs Predicted
ax = axes[0]
ax.scatter(y_test, y_pred, alpha=0.5, color=BLUE, s=30, edgecolors="none")
mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
ax.plot([mn, mx], [mn, mx], color=ACCENT, lw=1.5, ls="--", label="Perfect")
ax.set_title(f"Batting Runs: Actual vs Predicted\nR²={r2:.3f}  MAE={mae:.0f}",
             color=ACCENT)
ax.set_xlabel("Actual Runs")
ax.set_ylabel("Predicted Runs")
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Elbow / Silhouette
ax = axes[1]
ax2 = ax.twinx()
ax.plot(list(K_range), inertias, color=BLUE, marker="o", label="Inertia")
ax2.plot(list(K_range), sil_scores, color=GREEN, marker="s", label="Silhouette")
ax.axvline(best_k, color=ACCENT, ls="--", lw=1.5, label=f"Best k={best_k}")
ax.set_title("Elbow & Silhouette Curve", color=ACCENT)
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia", color=BLUE)
ax2.set_ylabel("Silhouette Score", color=GREEN)
ax.legend(loc="upper right")
ax.grid(alpha=0.3)

# Plot 3: Player Archetypes scatter
ax = axes[2]
for cid in sorted(bat["cluster"].dropna().unique()):
    sub = bat[bat["cluster"] == cid]
    label = ARCHETYPE_NAMES.get(int(cid), f"Cluster {int(cid)}")
    ax.scatter(sub["avg_runs"], sub["strike_rate"],
               label=label, alpha=0.7, s=40, edgecolors="none",
               color=CLUSTER_COLORS[int(cid) % len(CLUSTER_COLORS)])
ax.set_title("Player Archetypes", color=ACCENT)
ax.set_xlabel("Batting Average")
ax.set_ylabel("Strike Rate")
ax.legend(fontsize=7, loc="upper left")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("C:/Users/sidha/ipl-analytics/src/week3_models.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
print("  Saved → data/week3_models.png")

print("\n" + "=" * 60)
print("  Week 3 Complete! 🏏")
print("  Next → Week 4: Streamlit Dashboard")
print("=" * 60)

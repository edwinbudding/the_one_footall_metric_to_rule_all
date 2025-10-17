# ======================================
# TDM - Part 4: Apply Ridge Weights & Validate vs Defensive DVOA
# ======================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

BASE = "/Users/anokhpalakurthi/Downloads/"
TDM_PATH = BASE + "TDM_Base_Weighted.csv"
PLAYERAGG_PATH = BASE + "TDM_Base_PlayerAgg.csv"
DVOA_PATH = BASE + "Defensive DVOA.csv"
WEIGHTS_PATH = BASE + "TDM_Calibrated_Weights_SplitPhase.csv"

# Load
tdm = pd.read_csv(TDM_PATH)
dvoa = pd.read_csv(DVOA_PATH)
weights = pd.read_csv(WEIGHTS_PATH)

# Team Identifiers
if "Team" not in tdm.columns:
    playeragg = pd.read_csv(PLAYERAGG_PATH)[["Player", "PrimaryTeam"]]
    tdm = tdm.merge(playeragg, on="Player", how="left")
    tdm.rename(columns={"PrimaryTeam": "Team"}, inplace=True)

# Clean DVOA
dvoa.columns = dvoa.columns.str.strip().str.upper()
dvoa.rename(columns={
    "TEAM": "Team",
    "DVOA": "DefensiveDVOA",
    "PASS": "PassDefenseDVOA",
    "RUSH": "RushDefenseDVOA"
}, inplace=True)
dvoa = dvoa[["Team", "DefensiveDVOA", "PassDefenseDVOA", "RushDefenseDVOA"]]

# Standardize Names
fix = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "LA": "LAR"}
tdm["Team"] = tdm["Team"].replace(fix)
dvoa["Team"] = dvoa["Team"].replace(fix)

# Ridge Weights
beta_passdef = weights.query("Phase == 'PassDef'").set_index("Metric")["Ridge"].to_dict()
beta_rushdef = weights.query("Phase == 'RushDef'").set_index("Metric")["Ridge"].to_dict()
print("\nLoaded Ridge Weights")
print("PassDef →", beta_passdef)
print("RushDef →", beta_rushdef)

# Calculate TDM Components
for c in ["PassRushScore", "CoverageScore", "RunDefenseScore"]:
    tdm[c] = tdm.get(c, 0.0)

tdm["PassDef_TDM"] = (
    beta_passdef.get("PassRushScore", 0) * tdm["PassRushScore"] +
    beta_passdef.get("CoverageScore", 0) * tdm["CoverageScore"]
)
tdm["RushDef_TDM"] = (
    beta_rushdef.get("RunDefenseScore", 0) * tdm["RunDefenseScore"] +
    beta_rushdef.get("PassRushScore", 0) * tdm["PassRushScore"]
)
tdm["TotalTDM"] = tdm["PassDef_TDM"] + tdm["RushDef_TDM"]

# Slight weights
PASSRUSH_W, COVERAGE_W, RUNDEF_W = 1.2, 0.9, 1.0
tdm["PassDef_TDM_Adj"] = (
    PASSRUSH_W * beta_passdef.get("PassRushScore", 0) * tdm["PassRushScore"] +
    COVERAGE_W * beta_passdef.get("CoverageScore", 0) * tdm["CoverageScore"]
)
tdm["RushDef_TDM_Adj"] = (
    RUNDEF_W * beta_rushdef.get("RunDefenseScore", 0) * tdm["RunDefenseScore"] +
    PASSRUSH_W * beta_rushdef.get("PassRushScore", 0) * tdm["PassRushScore"]
)
tdm["TotalTDM_Adj"] = tdm["PassDef_TDM_Adj"] + tdm["RushDef_TDM_Adj"]

# Winsorize extreme outliers
for col in ["PassDef_TDM_Adj", "RushDef_TDM_Adj", "TotalTDM_Adj"]:
    q1, q99 = tdm[col].quantile([0.01, 0.99])
    tdm[col] = tdm[col].clip(q1, q99)

# Aggregate by Team
agg_cols = ["PassDef_TDM", "RushDef_TDM", "TotalTDM", "TotalTDM_Adj"]
team = tdm.groupby("Team", as_index=False)[agg_cols].mean()
team = team.merge(dvoa, on="Team", how="inner")
print(f"\nAggregated to {len(team)} teams (expected 32)")

# Correation Diagnostics
corrs = {
    "PassDef_TDM vs PassDVOA": -team["PassDef_TDM"].corr(team["PassDefenseDVOA"]),
    "RushDef_TDM vs RushDVOA": -team["RushDef_TDM"].corr(team["RushDefenseDVOA"]),
    "TotalTDM vs DVOA":       -team["TotalTDM"].corr(team["DefensiveDVOA"]),
    "TotalTDM_Adj vs DVOA":   -team["TotalTDM_Adj"].corr(team["DefensiveDVOA"]),
}
print("\nCorrelations (inverted so ↑ = better defense)")
for k, v in corrs.items():
    print(f"{k:30s}: {v:.3f}")

# Visualization
team_sorted = team.sort_values("TotalTDM_Adj", ascending=False)

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "axes.edgecolor": "0.3",
    "axes.linewidth": 0.8,
    "axes.labelweight": "semibold",
    "axes.titleweight": "bold",
    "font.size": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white"
})

fig, ax1 = plt.subplots(figsize=(13, 6))

# Bars for TDM
bar_color = "#264653"
sns.barplot(
    data=team_sorted, x="Team", y="TotalTDM_Adj",
    color=bar_color, alpha=0.95, ax=ax1
)
ax1.set_ylabel("Total Defensive Metric (Adj.)", fontsize=12, labelpad=10, weight="semibold")
ax1.set_xlabel("")
ax1.grid(axis="y", linestyle="--", alpha=0.4)

# Line for DVOA
ax2 = ax1.twinx()
line_color = "#E76F51"
sns.lineplot(
    data=team_sorted, x="Team", y="DefensiveDVOA",
    color=line_color, marker="o", markersize=5, linewidth=2, ax=ax2
)
ax2.set_ylabel("Defensive DVOA", fontsize=12, color=line_color, weight="semibold")
ax2.tick_params(axis="y", colors=line_color)
ax2.grid(False)

# Title & layout
plt.title("Team-Level Defensive TDM vs Defensive DVOA (2024)", fontsize=14, weight="bold", pad=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(BASE + "TDM_TeamRank_vs_DefDVOA.png", dpi=400)
plt.show()

# Export
team.to_csv(BASE + "TDM_Team_Aggregates.csv", index=False)
print(f"Exported: {BASE}TDM_Team_Aggregates.csv") 
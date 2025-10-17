# ======================================
# TDM - Part 5: Player Leaderboard
# ======================================

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

BASE = "/Users/anokhpalakurthi/Downloads/"
TDM_PATH   = BASE + "TDM_Base_Weighted.csv"                 
WEIGHTS_SP = BASE + "TDM_Calibrated_Weights_SplitPhase.csv" 

# Load
tdm = pd.read_csv(TDM_PATH)
wts = pd.read_csv(WEIGHTS_SP)

# Ridge coef mapping
def coef_map(phase, needed):
    sub = wts[(wts["Phase"] == phase) & (wts["Metric"].isin(needed))]
    m = dict(zip(sub["Metric"], sub["Ridge"]))
    for k in needed:
        m.setdefault(k, 0.0)
    return m

pass_need = ["PassRushScore", "CoverageScore"]
rush_need = ["RunDefenseScore", "PassRushScore"]

beta_pass = coef_map("PassDef", pass_need)
beta_rush = coef_map("RushDef", rush_need)

print("\nWeight summaries (ridge core):")
print("Pass weights:", beta_pass)
print("Rush weights:", beta_rush)

# Ensure scores exist
for c in ["PassRushScore", "CoverageScore", "RunDefenseScore"]:
    if c not in tdm.columns:
        tdm[c] = 0.0

# Pure ridge core
tdm["PassDef_TDM_core"] = (
    beta_pass["PassRushScore"] * tdm["PassRushScore"] +
    beta_pass["CoverageScore"]  * tdm["CoverageScore"]
)
tdm["RushDef_TDM_core"] = (
    beta_rush["RunDefenseScore"] * tdm["RunDefenseScore"] +
    beta_rush["PassRushScore"]   * tdm["PassRushScore"]
)
tdm["TotalTDM_core"] = tdm["PassDef_TDM_core"] + tdm["RushDef_TDM_core"]

# Phase weighting
PASS_W = 1.20
RUSH_W = 0.80

tdm["PassDef_TDM"] = PASS_W * tdm["PassDef_TDM_core"]
tdm["RushDef_TDM"] = RUSH_W * tdm["RushDef_TDM_core"]
tdm["TotalTDM"]    = tdm["PassDef_TDM"] + tdm["RushDef_TDM"]

# Outlier Control
for col in ["PassDef_TDM", "RushDef_TDM", "TotalTDM"]:
    lo, hi = tdm[col].quantile([0.01, 0.99])
    tdm[col] = tdm[col].clip(lo, hi)

# Calibrate roles
pos_map = {
    "EDGE": "ED", "ED": "ED", "DE": "ED", "OLB": "ED",
    "DT": "DI", "IDL": "DI", "NT": "DI", "DI": "DI",
    "ILB": "LB", "LB": "LB", "MLB": "LB",
    "CB": "CB", "SCB": "CB",
    "S": "S", "SS": "S", "FS": "S"
}
tdm["PositionGroup"] = tdm["Position"].map(pos_map).fillna("Other")

role_mult = {
    # mild trims to trench dominance
    "DI": 0.92,
    "ED": 0.95,
    # slight boost to back-seven
    "LB": 1.03,
    "S":  1.08,
    "CB": 1.12,
    # others neutral
    "Other": 1.00
}
tdm["RoleMult"] = tdm["PositionGroup"].map(role_mult).fillna(1.00)

# Apply role multiplier to final (post-hoc), not to phase components
tdm["TotalTDM_Adjusted"] = tdm["TotalTDM"] * tdm["RoleMult"]


# Leaderboard generation
cols = [
    "Player", "Position", "PositionGroup",
    "TotalTDM_Adjusted", "TotalTDM", "PassDef_TDM", "RushDef_TDM",
    "PassRushScore", "CoverageScore", "RunDefenseScore", "RoleMult", "TotalSnaps"
]

top25 = tdm.sort_values("TotalTDM_Adjusted", ascending=False).head(25)
print("\n🛡️ Top 25 — TotalTDM_Adjusted (Phase-weighted, Role-calibrated):")
print(top25[cols].round(3))

# Positional balance snapshot
pos_summary = (
    tdm.groupby("PositionGroup")["TotalTDM_Adjusted"]
    .agg(["mean", "std", "count"])
    .sort_values("mean", ascending=False)
    .round(3)
)
print("\n📊 Positional Summary (after role calibration):")
print(pos_summary)

# Simple diagnostics for a few DBs (optional)
for name in ["Pat Surtain II", "Sauce Gardner", "Jaire Alexander", "Xavier McKinney"]:
    diag = tdm.loc[tdm["Player"].str.contains(name, case=False, na=False), cols]
    if not diag.empty:
        print(f"\n🔎 Diagnostic: {name}")
        print(diag.round(3))

# Export and visualize
out_csv = BASE + "TDM_Player_Leaderboard_PhaseWeighted_RoleCalibrated.csv"
tdm.to_csv(out_csv, index=False)

viz = top25.sort_values("TotalTDM_Adjusted", ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(
    data=viz,
    y="Player",
    x="TotalTDM_Adjusted",
    hue="PositionGroup",
    dodge=False,
    palette="viridis",
    order=viz["Player"]
)
plt.title("Top 25 Defensive Players — Phase-Weighted, Role-Calibrated (TOM-style)", fontsize=14, weight="bold")
plt.xlabel("Adjusted TDM (post-hoc role calibration)")
plt.ylabel("")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.legend(title="Position Group", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(BASE + "TDM_Top25_PhaseWeighted_RoleCalibrated.png", dpi=300)
plt.show()
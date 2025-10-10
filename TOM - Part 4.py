# ======================================
# UVM - Part 4: Apply Split-Phase Weights & Validate Against DVOA
# ======================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE = "/Users/anokhpalakurthi/Downloads/"
UVM_PATH   = BASE + "Unified_Value_Model_Base.csv"
WEIGHTS_SP = BASE + "UVM_Calibrated_Weights_SplitPhase.csv"
DVOA_PATH  = BASE + "Offensive DVOA.csv"

# ---------- Load ----------
uvm = pd.read_csv(UVM_PATH)
wts = pd.read_csv(WEIGHTS_SP)
dvoa = pd.read_csv(DVOA_PATH)

print("✅ Data loaded.")
print(f"UVM: {uvm.shape}, Weights: {wts.shape}, DVOA: {dvoa.shape}")

# ---------- Clean DVOA ----------
dvoa.columns = dvoa.columns.str.strip().str.upper()
dvoa.rename(columns={"TEAM":"Team","DVOA":"OffensiveDVOA",
                     "PASS":"PassDVOA","RUSH":"RushDVOA"}, inplace=True)
dvoa = dvoa[["Team","OffensiveDVOA","PassDVOA","RushDVOA"]]

# ---------- Normalize team names ----------
fix = {"ARZ":"ARI","BLT":"BAL","CLV":"CLE","HST":"HOU","SF":"SFO","LA":"LAR"}
uvm["Team"] = uvm["Team"].replace(fix).replace({"SFO":"SF"})
dvoa["Team"] = dvoa["Team"].replace({"SFO":"SF"})

# ---------- Build coefficient dicts ----------
def coef_map(phase, needed):
    sub = wts[(wts["Phase"]==phase) & (wts["Metric"].isin(needed))]
    m = dict(zip(sub["Metric"], sub["Ridge"]))
    for k in needed:
        m.setdefault(k, 0.0)
    return m

pass_need = ["AirScore","ReceiveScore","BlockScore"]
rush_need = ["RushScore","BlockScore"]

beta_pass = coef_map("Pass", pass_need)
beta_rush = coef_map("Rush", rush_need)

# ---------- Compute per-player TOM ----------
for c in ["AirScore","RushScore","ReceiveScore","BlockScore"]:
    if c not in uvm.columns: 
        uvm[c] = 0.0

uvm["PassTOM"] = (
    beta_pass["AirScore"]     * uvm["AirScore"] +
    beta_pass["ReceiveScore"] * uvm["ReceiveScore"] +
    beta_pass["BlockScore"]   * uvm["BlockScore"]
)
uvm["RushTOM"] = (
    beta_rush["RushScore"]    * uvm["RushScore"] +
    beta_rush["BlockScore"]   * uvm["BlockScore"]
)
uvm["TotalTOM"] = uvm["PassTOM"] + uvm["RushTOM"]

# =====================================================
# Phase-weighted calibration (post-hoc adjustments)
# =====================================================

PASS_WEIGHT = 1.5
RUSH_WEIGHT = 1.0
NUDGE_AIR   = 1.5
NUDGE_REC   = 0.5

uvm["PassTOM_Adjusted"] = (
    NUDGE_AIR * beta_pass["AirScore"]     * uvm["AirScore"] +
    NUDGE_REC * beta_pass["ReceiveScore"] * uvm["ReceiveScore"] +
               beta_pass["BlockScore"]    * uvm["BlockScore"]
)

uvm["TotalTOM_Adjusted"] = (
    PASS_WEIGHT * uvm["PassTOM_Adjusted"] +
    RUSH_WEIGHT * uvm["RushTOM"]
)

# ---------- Team-level aggregation ----------
team = (
    uvm.groupby("Team", as_index=False)
    [["PassTOM","RushTOM","TotalTOM","TotalTOM_Adjusted"]]
    .sum()
    .merge(dvoa, on="Team", how="inner")
)

# ---------- Correlations ----------
corr_pass = team["PassTOM"].corr(team["PassDVOA"])
corr_rush = team["RushTOM"].corr(team["RushDVOA"])
corr_off  = team["TotalTOM"].corr(team["OffensiveDVOA"])
corr_off_adj = team["TotalTOM_Adjusted"].corr(team["OffensiveDVOA"])

print("\n📈 Correlations:")
print(f"  PassTOM vs Pass DVOA: {corr_pass:.3f}")
print(f"  RushTOM vs Rush DVOA: {corr_rush:.3f}")
print(f"  TotalTOM vs Offensive DVOA: {corr_off:.3f}")
print(f"  TotalTOM_Adjusted vs Offensive DVOA: {corr_off_adj:.3f}")

# ---------- Visualization ----------
plt.figure(figsize=(11,6))
team_sorted = team.sort_values("TotalTOM_Adjusted", ascending=False)

sns.barplot(data=team_sorted, x="Team", y="TotalTOM_Adjusted",
            color="#4B8BBE", alpha=0.8, label="TotalTOM_Adjusted")
sns.lineplot(data=team_sorted, x="Team", y="OffensiveDVOA",
             color="#E06C75", marker="o", label="OffensiveDVOA")

plt.title("Team-Level Comparison: Adjusted TOM vs Offensive DVOA", fontsize=13, weight='bold')
plt.ylabel("Scaled Value")
plt.xlabel("Team (sorted by TotalTOM_Adjusted)")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.legend(frameon=False)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(BASE + "UVM_TeamRank_TOM_vs_DVOA.png", dpi=300)
plt.close()
print("📊 Saved: UVM_TeamRank_TOM_vs_DVOA.png")
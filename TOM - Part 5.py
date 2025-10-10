# ======================================
# UVM - Part 5: Player Leaderboard (Phase-Weighted Calibration)
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
print(f"UVM: {uvm.shape}, Weights: {wts.shape}")

# ---------- Build coefficient dicts ----------
def coef_map(phase, needed):
    sub = wts[(wts["Phase"] == phase) & (wts["Metric"].isin(needed))]
    m = dict(zip(sub["Metric"], sub["Ridge"]))
    for k in needed:
        m.setdefault(k, 0.0)
    return m

pass_need = ["AirScore", "ReceiveScore", "BlockScore"]
rush_need = ["RushScore", "BlockScore"]

beta_pass = coef_map("Pass", pass_need)
beta_rush = coef_map("Rush", rush_need)

print("Pass weights:", beta_pass)
print("Rush weights:", beta_rush)

# ---------- Ensure score cols exist ----------
for c in ["AirScore", "RushScore", "ReceiveScore", "BlockScore"]:
    if c not in uvm.columns:
        uvm[c] = 0.0

# ---------- Compute Split-Phase TOM ----------
uvm["PassTOM"] = (
    beta_pass["AirScore"] * uvm["AirScore"] +
    beta_pass["ReceiveScore"] * uvm["ReceiveScore"] +
    beta_pass["BlockScore"] * uvm["BlockScore"]
)
uvm["RushTOM"] = (
    beta_rush["RushScore"] * uvm["RushScore"] +
    beta_rush["BlockScore"] * uvm["BlockScore"]
)
uvm["TotalTOM"] = uvm["PassTOM"] + uvm["RushTOM"]

# =====================================================
# NEW: Phase weighting (post-hoc calibration)
# =====================================================
USE_TUNER = False
PASS_WEIGHT = 1.50
RUSH_WEIGHT = 1.00

# --- Optional tuning using team-level corr with DVOA ---
if USE_TUNER:
    dvoa = pd.read_csv(DVOA_PATH)
    dvoa.columns = dvoa.columns.str.strip().str.upper()
    dvoa.rename(columns={"TEAM": "Team", "DVOA": "OffensiveDVOA"}, inplace=True)
    fix = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SF": "SFO", "LA": "LAR"}
    uvm["Team"] = uvm["Team"].replace(fix).replace({"SFO": "SF"})
    dvoa["Team"] = dvoa["Team"].replace({"SFO": "SF"})

    team_phase = uvm.groupby("Team", as_index=False)[["PassTOM", "RushTOM"]].sum()
    merged = team_phase.merge(dvoa[["Team", "OffensiveDVOA"]], on="Team", how="inner")

    best = {"pw": None, "rw": None, "corr": -9}
    for pw in np.round(np.arange(1.4, 2.21, 0.05), 2):
        for rw in np.round(np.arange(0.50, 0.91, 0.05), 2):
            if pw / rw < 1.8:
                continue
            total = pw * merged["PassTOM"] + rw * merged["RushTOM"]
            r = total.corr(merged["OffensiveDVOA"])
            if pd.notna(r) and r > best["corr"]:
                best = {"pw": pw, "rw": rw, "corr": r}
    PASS_WEIGHT = best["pw"] if best["pw"] else PASS_WEIGHT
    RUSH_WEIGHT = best["rw"] if best["rw"] else RUSH_WEIGHT
    print(f"🔧 Tuned weights → PASS={PASS_WEIGHT}, RUSH={RUSH_WEIGHT} (corr≈{best['corr']:.3f})")

# --- Radical intra-pass tilt ---
NUDGE_AIR = 1.50
NUDGE_REC = 0.50

uvm["PassTOM_Adjusted"] = (
    NUDGE_AIR * beta_pass["AirScore"] * uvm["AirScore"] +
    NUDGE_REC * beta_pass["ReceiveScore"] * uvm["ReceiveScore"] +
    beta_pass["BlockScore"] * uvm["BlockScore"]
)

uvm["TotalTOM_Adjusted"] = (
    PASS_WEIGHT * uvm["PassTOM_Adjusted"] +
    RUSH_WEIGHT * uvm["RushTOM"]
)

# =====================================================
# ---------- Offense-only filter + optional volume floor ----------
# =====================================================

off_positions = {"QB","HB","RB","FB","WR","TE","T","G","C","LT","LG","RT","RG","OL"}
uvm_off = uvm[uvm["Position"].isin(off_positions)].copy()

APPLY_VOLUME_FLOOR = True
try:
    if APPLY_VOLUME_FLOOR:
        passing   = pd.read_csv(BASE + "Passing_PFF_Clean.csv")[["Player","Team","Position","Dropbacks"]]
        rushing   = pd.read_csv(BASE + "Rushing_PFF_Clean.csv")[["Player","Team","Position","RushAttempts"]]
        receiving = pd.read_csv(BASE + "Receiving_PFF_Clean.csv")[["Player","Team","Position","Targets"]]
        blocking  = pd.read_csv(BASE + "Blocking_PFF_Clean.csv")[["Player","Team","Position","TotalBlockSnaps"]]

        key = ["Player","Team","Position"]
        vol = (
            uvm_off[key].drop_duplicates()
            .merge(passing, on=key, how="left")
            .merge(rushing, on=key, how="left")
            .merge(receiving, on=key, how="left")
            .merge(blocking, on=key, how="left")
        )
        for c in ["Dropbacks", "RushAttempts", "Targets", "TotalBlockSnaps"]:
            if c in vol.columns:
                vol[c] = vol[c].fillna(0)
        uvm_off = uvm_off.merge(vol, on=key, how="left")

        floor = (
            (uvm_off.get("Dropbacks", 0) >= 50)
            | (uvm_off.get("RushAttempts", 0) >= 30)
            | (uvm_off.get("Targets", 0) >= 30)
            | (uvm_off.get("TotalBlockSnaps", 0) >= 200)
        )
        uvm_off = uvm_off[floor].copy()
except Exception as e:
    print(f"(Info) Volume merge/floor skipped: {e}")

# --- QB premium (WAR-style, after volume filter) ---
uvm_off["TotalTOM_Adjusted"] = np.where(
    uvm_off["Position"] == "QB",
    uvm_off["TotalTOM_Adjusted"] * 1.25,
    uvm_off["TotalTOM_Adjusted"]
)

# =====================================================
# ---------- Leaderboards ----------
# =====================================================

top25_total = uvm_off.sort_values("TotalTOM_Adjusted", ascending=False).head(25)
top25_pass  = uvm_off.sort_values("PassTOM_Adjusted", ascending=False).head(25)
top25_rush  = uvm_off.sort_values("RushTOM", ascending=False).head(25)

def show(df, label):
    cols = [
        "Player", "Team", "Position", "TotalTOM_Adjusted", "PassTOM_Adjusted", "RushTOM",
        "AirScore", "ReceiveScore", "RushScore", "BlockScore"
    ]
    extra = [c for c in ["Dropbacks", "RushAttempts", "Targets", "TotalBlockSnaps"] if c in df.columns]
    print(f"\n🏈 Top 25 — {label}:")
    print(df[cols + extra].round(3))

show(top25_total, "TotalTOM_Adjusted (Phase-weighted)")
show(top25_pass,  "PassTOM_Adjusted")
show(top25_rush,  "RushTOM")

# ---------- Diagnostics ----------
print("\n📊 Positional Averages (mean TotalTOM_Adjusted):")
print(uvm_off.groupby("Position")["TotalTOM_Adjusted"].mean().sort_values(ascending=False).round(2))

# ---------- Export ----------
uvm_off.to_csv(BASE + "UVM_Player_Leaderboard_PhaseWeighted.csv", index=False)
top25_total.to_csv(BASE + "UVM_Player_Leaderboard_PhaseWeighted_Top25_Total.csv", index=False)
top25_pass.to_csv(BASE + "UVM_Player_Leaderboard_PhaseWeighted_Top25_Pass.csv", index=False)
top25_rush.to_csv(BASE + "UVM_Player_Leaderboard_PhaseWeighted_Top25_Rush.csv", index=False)
print("\n✅ Exported phase-weighted player leaderboards.")

# Sort descending so highest TOM is on top
viz = top25_total.sort_values("TotalTOM_Adjusted", ascending=False)

plt.figure(figsize=(10,8))
sns.barplot(
    data=viz,
    y="Player",
    x="TotalTOM_Adjusted",
    hue="Position",
    dodge=False,
    palette="coolwarm",
    order=viz["Player"]  # preserve sorted order
)

plt.title("Top 25 Offensive Players in 2024 by Adjusted TOM Score", fontsize=14, weight="bold")
plt.xlabel("Adjusted TOM Score (Phase-weighted)")
plt.ylabel("")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.legend(title="Position", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

plt.savefig(BASE + "UVM_Top25_TotalTOM_BarChart.png", dpi=300)
plt.show()
print("📊 Saved Top 25 TotalTOM visualization.")
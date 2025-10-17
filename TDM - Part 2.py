# ======================================
# TDM - Part 2: Weighted Defensive Domain Scores
# ======================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE = "/Users/anokhpalakurthi/Downloads/"
passrush = pd.read_csv(BASE + "PassRush_PFF_Clean.csv")
coverage = pd.read_csv(BASE + "Coverage_PFF_Clean.csv")
rundef   = pd.read_csv(BASE + "RunDefense_PFF_Clean.csv")

# Detect correct base (prefer PlayerAgg)
if os.path.exists(BASE + "TDM_Base_PlayerAgg.csv"):
    base = pd.read_csv(BASE + "TDM_Base_PlayerAgg.csv")
    key = ["Player", "Position"] 
else:
    base = pd.read_csv(BASE + "TDM_Base_TeamLinked.csv")
    key = ["Player", "Team", "Position"]

# Ensure snap columns present
snap_cols = ["PassRushSnaps", "CoverageSnaps", "RunDefenseSnaps"]
for col in snap_cols:
    if col not in base.columns:
        base[col] = 0.0
base = base[key + snap_cols]

# Z-Score and Directional Normalization
def zscore_cols(df, cols, negate_cols=None):
    df = df.copy()
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df, pd.Series(dtype=float)

    if negate_cols:
        for c in negate_cols:
            if c in df.columns:
                df[c] = -df[c]

    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    comp = df[cols].mean(axis=1)
    return df, comp

# Domain Feature Sets
pr_cols = [
    "Sacks", "Hits", "Hurries", "Pressures",
    "PressureRate", "WinRate", "PRP", "PassRushWins",
    "PFF_PassRushGrade", "PFF_DefenseGrade"
]

cov_cols = [
    "Targets", "ReceptionsAllowed", "YardsAllowed", "TDsAllowed",
    "PasserRatingAllowed", "ForcedIncompletions", "YardsPerTarget",
    "INTs", "PBUs", "PFF_CoverageGrade", "PFF_DefenseGrade"
]
cov_negate = [
    "Targets", "ReceptionsAllowed", "YardsAllowed", "TDsAllowed",
    "PasserRatingAllowed", "YardsPerTarget"
]

run_cols = [
    "Stops", "MissedTackles", "StopPercent", "MissedTackleRate",
    "ForcedFumbles", "Tackles", "PFF_RunDefenseGrade", "PFF_DefenseGrade"
]
run_negate = ["MissedTackles", "MissedTackleRate"]

# Z-Scores
_, pr_comp  = zscore_cols(passrush, pr_cols)
_, cov_comp = zscore_cols(coverage, cov_cols, negate_cols=cov_negate)
_, run_comp = zscore_cols(rundef,   run_cols,  negate_cols=run_negate)

passrush["PassRushScore_raw"]   = pr_comp
coverage["CoverageScore_raw"]   = cov_comp
rundef["RunDefenseScore_raw"]   = run_comp

# Merge Domain Tables
merged = (
    passrush[key + ["PassRushScore_raw"]]
    .merge(coverage[key + ["CoverageScore_raw"]], on=key, how="outer")
    .merge(rundef[key + ["RunDefenseScore_raw"]], on=key, how="outer")
)

# Collapse duplicates
merged = merged.groupby(key, as_index=False).mean(numeric_only=True)

# Bring snaps from base
merged = merged.merge(base, on=key, how="left").fillna(0.0)

# Ensure Defensive Players Only
defensive_positions = ["DL", "DI", "DT", "NT", "EDGE", "ED", "DE",
                        "LB", "ILB", "OLB", "CB", "DB", "S", "FS", "SS"]
pre_ct = len(merged)
merged = merged[merged["Position"].isin(defensive_positions)].copy()
print(f"🧹 Position filter: kept {len(merged)}/{pre_ct} rows (defenders only)")

# Total snaps per player
merged["TotalSnaps"] = merged[["PassRushSnaps","CoverageSnaps","RunDefenseSnaps"]].sum(axis=1)

# Domain Shares
for dom, snap_col in [
    ("PassRush",  "PassRushSnaps"),
    ("Coverage",  "CoverageSnaps"),
    ("RunDefense","RunDefenseSnaps"),
]:
    merged[f"{dom}Weight"] = np.where(merged["TotalSnaps"] > 0,
                                      merged[snap_col] / merged["TotalSnaps"], 0.0)
    merged[f"{dom}Score"]  = merged.get(f"{dom}Score_raw", 0.0) * merged[f"{dom}Weight"]

# Domain Minimums
MIN_PR  = 75
MIN_COV = 150
MIN_RUN = 100

merged.loc[merged["PassRushSnaps"]   < MIN_PR,  "PassRushScore"]  = 0.0
merged.loc[merged["CoverageSnaps"]   < MIN_COV, "CoverageScore"]  = 0.0
merged.loc[merged["RunDefenseSnaps"] < MIN_RUN, "RunDefenseScore"]= 0.0

# Overall defensive snap floor
SNAP_FLOOR = 200
pre_floor = len(merged)
merged = merged[merged["TotalSnaps"] >= SNAP_FLOOR].copy()

# Summary diagnostics
domain_stats = []
for dom in ["PassRush", "Coverage", "RunDefense"]:
    col = f"{dom}Score"
    domain_stats.append({
        "Domain": dom,
        "Mean": merged[col].mean(),
        "Std": merged[col].std(),
        "Min": merged[col].min(),
        "Max": merged[col].max(),
        "NonZeroPlayers": int((merged[col] != 0).sum()),
    })
domain_df = pd.DataFrame(domain_stats)
print("\n🧮 Defensive Domain Score Summary (post-filters):")
print(domain_df.to_string(index=False))


# Last export
out_cols = key + [
    "PassRushSnaps","CoverageSnaps","RunDefenseSnaps","TotalSnaps",
    "PassRushScore","CoverageScore","RunDefenseScore"
]
merged[out_cols].to_csv(BASE + "TDM_Base_Weighted.csv", index=False)
print(f"\n Exported weighted base ({merged.shape[0]} rows): {BASE}TDM_Base_Weighted.csv")

# Vizualization
try:
    plt.figure(figsize=(9, 5))
    sns.violinplot(
        data=merged[["PassRushScore","CoverageScore","RunDefenseScore"]],
        inner="box", cut=0, palette="coolwarm"
    )
    plt.title("Distribution of Weighted Defensive Domain Scores (Player-Normalized, Post-Filters)")
    plt.axhline(0, color="gray", ls="--", lw=1)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("(Info) Visualization skipped:", e)
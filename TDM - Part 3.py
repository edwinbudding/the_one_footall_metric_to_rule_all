# ======================================
# TDM - Part 3: Ridge Modeling vs Defensive DVOA (Aligned & Snap-Weighted)
# ======================================

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE = "/Users/anokhpalakurthi/Downloads/"
TDM_PATH = BASE + "TDM_Base_Weighted.csv"
TEAM_MAP_PATH = BASE + "TDM_Base_TeamLinked.csv"
DVOA_PATH = BASE + "Defensive DVOA.csv"

# Load and Team Map
tdm = pd.read_csv(TDM_PATH)
team_map = pd.read_csv(TEAM_MAP_PATH)[["Player", "Team"]].drop_duplicates()
tdm = tdm.merge(team_map, on="Player", how="left")

# Loading and Cleaning DVOA
dvoa = pd.read_csv(DVOA_PATH)
dvoa.columns = dvoa.columns.str.strip().str.upper()
dvoa.rename(columns={
    "TEAM": "Team",
    "DVOA": "DefensiveDVOA",
    "PASS": "PassDefenseDVOA",
    "RUSH": "RushDefenseDVOA"
}, inplace=True)
dvoa = dvoa[["Team","DefensiveDVOA","PassDefenseDVOA","RushDefenseDVOA"]]

# Normalize abbreviations
fix = {"ARZ":"ARI","BLT":"BAL","CLV":"CLE","HST":"HOU","SF":"SFO","LA":"LAR"}
tdm["Team"] = tdm["Team"].replace(fix).replace({"SFO":"SF"})
dvoa["Team"] = dvoa["Team"].replace(fix).replace({"SFO":"SF"})

# Team-Level Snap Weight Agg
domains = ["PassRushScore","CoverageScore","RunDefenseScore"]

for dom in domains:
    tdm[f"{dom}_Weighted"] = tdm[dom] * tdm["TotalSnaps"]

team = (
    tdm.groupby("Team", as_index=False)
       .agg({f"{d}_Weighted": "sum" for d in domains} | {"TotalSnaps": "sum"})
)

for dom in domains:
    team[dom] = team[f"{dom}_Weighted"] / team["TotalSnaps"]

team = team[["Team"] + domains]

# DVOA Merge
merged = team.merge(dvoa, on="Team", how="inner")
print(f"Merged to {len(merged)} team rows (expected 32).")

# Invert DVOA so higher = better defense
for col in ["DefensiveDVOA","PassDefenseDVOA","RushDefenseDVOA"]:
    merged[col + "_Positive"] = -merged[col]

# Ridge Stuff
def fit_ridge(X_df, y, alphas=np.logspace(-3, 3, 200)):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_df)
    ridge = RidgeCV(alphas=alphas, store_cv_results=True).fit(Xs, y)
    yhat = ridge.predict(Xs)
    r2 = ridge.score(Xs, y)
    mae = float(np.mean(np.abs(y - yhat)))
    coefs = pd.Series(ridge.coef_, index=X_df.columns)
    print(f"Best α = {ridge.alpha_:.5f} | R²={r2:.3f} | MAE={mae:.3f}")
    return coefs, r2, mae, yhat

# Split Phase Ridge
X_pass = merged[["PassRushScore","CoverageScore"]]
y_pass = merged["PassDefenseDVOA_Positive"]
coefs_pass, r2_pass, mae_pass, yhat_pass = fit_ridge(X_pass, y_pass)

X_rush = merged[["RunDefenseScore","PassRushScore"]]
y_rush = merged["RushDefenseDVOA_Positive"]
coefs_rush, r2_rush, mae_rush, yhat_rush = fit_ridge(X_rush, y_rush)

# All-Phase Ridge
X_all = merged[domains]
y_all = merged["DefensiveDVOA_Positive"]
coefs_all, r2_all, mae_all, yhat_all = fit_ridge(X_all, y_all)

# Coefficient Summaries
print("\nSplit-Phase Coefficients")
print("Pass Defense →")
print(coefs_pass.round(3))
print(f"R²={r2_pass:.3f} | MAE={mae_pass:.3f}")

print("\nRush Defense →")
print(coefs_rush.round(3))
print(f"R²={r2_rush:.3f} | MAE={mae_rush:.3f}")

print("\nAll-Phase Ridge Coefficients")
print(coefs_all.round(3))
print(f"R²={r2_all:.3f} | MAE={mae_all:.3f}")

# Visualize Coefficients
try:
    res_df = pd.DataFrame([
        {"Model":"Split - PassDef","Metric":k,"Weight":v} for k,v in coefs_pass.items()
    ] + [
        {"Model":"Split - RushDef","Metric":k,"Weight":v} for k,v in coefs_rush.items()
    ] + [
        {"Model":"All-Phase (Defense)","Metric":k,"Weight":v} for k,v in coefs_all.items()
    ])

    plt.figure(figsize=(12,6))
    sns.barplot(data=res_df, x="Metric", y="Weight", hue="Model", palette="viridis")
    plt.title("Defensive Ridge Coefficients (Split & All-Phase)", fontsize=14, weight="bold")
    plt.axhline(0, color="black", lw=1)
    plt.grid(axis="y", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("(Info) Coefficient viz skipped:", e)

# Predicted vs. Actual Plots
try:
    plt.figure(figsize=(6,5))
    sns.regplot(x=y_all, y=yhat_all, scatter_kws={'s':70}, line_kws={'color':'red'})
    plt.xlabel("Actual Defensive DVOA (+)")
    plt.ylabel("Predicted Defensive DVOA (+)")
    plt.title("Ridge Model Fit – All Defense")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("(Info) Regression fit plot skipped:", e)

# =====================================================
# Multicollinearity Diagnostics
# =====================================================
print("\nMulticollinearity Diagnostics")

X_for_vif = merged[domains]
X_scaled = pd.DataFrame(StandardScaler().fit_transform(X_for_vif), columns=X_for_vif.columns)
vif_df = pd.DataFrame({
    "Variable": X_scaled.columns,
    "VIF": [variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])]
})
print("\nVariance Inflation Factors:")
print(vif_df.round(3))

corr = X_for_vif.corr().round(2)
print("\nCorrelation Matrix:")
print(corr)

try:
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation Matrix of Domain Scores")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("(Info) Correlation heatmap skipped:", e)

# =====================================================
# Export Ridge Weights
# =====================================================
weights_df = pd.concat([
    pd.DataFrame({"Phase":"PassDef","Metric":coefs_pass.index,"Ridge":coefs_pass.values}),
    pd.DataFrame({"Phase":"RushDef","Metric":coefs_rush.index,"Ridge":coefs_rush.values}),
    pd.DataFrame({"Phase":"AllDefense","Metric":coefs_all.index,"Ridge":coefs_all.values})
], ignore_index=True)

weights_df.to_csv(BASE + "TDM_Calibrated_Weights_SplitPhase.csv", index=False)
print(f"\nExported Ridge Weights → {BASE}TDM_Calibrated_Weights_SplitPhase.csv")
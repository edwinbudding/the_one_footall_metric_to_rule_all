# ======================================
# UVM - Part 3: Ridge Modeling vs DVOA
# ======================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# ---------- Paths ----------
BASE = "/Users/anokhpalakurthi/Downloads/"
UVM_PATH = BASE + "Unified_Value_Model_Base.csv"
DVOA_PATH = BASE + "Offensive DVOA.csv"

# ---------- Load ----------
uvm = pd.read_csv(UVM_PATH)
dvoa = pd.read_csv(DVOA_PATH)

print("✅ Loaded:")
print(f"UVM base: {uvm.shape} | DVOA: {dvoa.shape}")

# ---------- Clean DVOA ----------
dvoa.columns = dvoa.columns.str.strip().str.upper()
rename = {"TEAM": "Team", "DVOA": "OffensiveDVOA", "OFF": "OffensiveDVOA",
          "PASS": "PassDVOA", "RUSH": "RushDVOA"}
dvoa.rename(columns=rename, inplace=True)
dvoa = dvoa[["Team", "OffensiveDVOA", "PassDVOA", "RushDVOA"]]

# ---------- Normalize team names ----------
fix = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SF": "SFO", "LA": "LAR"}
uvm["Team"] = uvm["Team"].replace(fix).replace({"SFO": "SF"})
dvoa["Team"] = dvoa["Team"].replace({"SFO": "SF"})

# ---------- Aggregate to team-level ----------
domain_cols = ["AirScore", "RushScore", "ReceiveScore", "BlockScore"]
team = uvm.groupby("Team")[domain_cols].mean().reset_index()
merged = team.merge(dvoa, on="Team", how="inner")
print(f"✅ Merged to teams: {len(merged)} rows ({merged['Team'].nunique()} teams)")

# ---------- Ridge helper ----------
def fit_ridge(X_df, y, alphas=np.logspace(-3, 3, 100)):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    ridge = RidgeCV(alphas=alphas, store_cv_results=True).fit(X_scaled, y)
    yhat = ridge.predict(X_scaled)
    r2 = ridge.score(X_scaled, y)
    mae = float(np.mean(np.abs(y - yhat)))
    coefs = pd.Series(ridge.coef_, index=X_df.columns)
    return ridge, coefs, r2, mae

# ======================================
# 1️⃣ SPLIT-PHASE MODELING (Interpretability)
# ======================================

X_pass = merged[["AirScore", "ReceiveScore", "BlockScore"]]
y_pass = merged["PassDVOA"]
ridge_pass, coefs_pass, r2_pass, mae_pass = fit_ridge(X_pass, y_pass)

X_rush = merged[["RushScore", "BlockScore"]]
y_rush = merged["RushDVOA"]
ridge_rush, coefs_rush, r2_rush, mae_rush = fit_ridge(X_rush, y_rush)

print("\n📊 Split-Phase Ridge Coefficients")
print("PASS →", coefs_pass.round(3), f" | R²={r2_pass:.3f} | MAE={mae_pass:.2f}")
print("RUSH →", coefs_rush.round(3), f" | R²={r2_rush:.3f} | MAE={mae_rush:.2f}")

# ======================================
# 2️⃣ CROSS-PHASE MODELING (Synergy Test)
# ======================================

X_pass_cross = merged[["AirScore", "ReceiveScore", "RushScore", "BlockScore"]]
ridge_pass_cross, coefs_pass_cross, r2_pass_cross, mae_pass_cross = fit_ridge(X_pass_cross, y_pass)

X_rush_cross = merged[["RushScore", "AirScore", "ReceiveScore", "BlockScore"]]
ridge_rush_cross, coefs_rush_cross, r2_rush_cross, mae_rush_cross = fit_ridge(X_rush_cross, y_rush)

print("\n📊 Cross-Phase Ridge Coefficients (Synergy Models)")
print("PASS →", coefs_pass_cross.round(3), f" | R²={r2_pass_cross:.3f} | MAE={mae_pass_cross:.2f}")
print("RUSH →", coefs_rush_cross.round(3), f" | R²={r2_rush_cross:.3f} | MAE={mae_rush_cross:.2f}")

# ======================================
# 3️⃣ ALL-PHASE MODELING (Holistic Offense)
# ======================================

X_all = merged[["AirScore", "ReceiveScore", "RushScore", "BlockScore"]]
ridge_all, coefs_all, r2_all, mae_all = fit_ridge(X_all, merged["OffensiveDVOA"])

print("\n📊 All-Phase Ridge for OffensiveDVOA")
print(coefs_all.round(3))
print(f"R²={r2_all:.3f} | MAE={mae_all:.2f}")

# ======================================
# 4️⃣ VISUALIZATION: Coefficients Comparison
# ======================================
res_df = pd.DataFrame([
    {"Model": "Split - Pass", "Metric": k, "Weight": v} for k, v in coefs_pass.items()
] + [
    {"Model": "Split - Rush", "Metric": k, "Weight": v} for k, v in coefs_rush.items()
] + [
    {"Model": "Cross - Pass", "Metric": k, "Weight": v} for k, v in coefs_pass_cross.items()
] + [
    {"Model": "Cross - Rush", "Metric": k, "Weight": v} for k, v in coefs_rush_cross.items()
] + [
    {"Model": "All-Phase (Offense)", "Metric": k, "Weight": v} for k, v in coefs_all.items()
])

plt.figure(figsize=(12, 6))
sns.barplot(data=res_df, x="Metric", y="Weight", hue="Model", palette="Spectral")
plt.title("Phase and Cross-Phase Ridge Coefficients Across Models", fontsize=14, weight='bold')
plt.ylabel("Coefficient Weight (Importance)")
plt.xlabel("Domain Metric")
plt.axhline(0, color='black', linewidth=1)
plt.legend(title="Model Type", fontsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ======================================
# 5️⃣ MODEL PERFORMANCE SUMMARY
# ======================================
summary = pd.DataFrame([
    ["Split - Pass", r2_pass, mae_pass],
    ["Split - Rush", r2_rush, mae_rush],
    ["Cross - Pass", r2_pass_cross, mae_pass_cross],
    ["Cross - Rush", r2_rush_cross, mae_rush_cross],
    ["All-Phase", r2_all, mae_all]
], columns=["Model", "R²", "MAE"]).round(3)

print("\n📈 Model Performance Summary:")
print(summary)
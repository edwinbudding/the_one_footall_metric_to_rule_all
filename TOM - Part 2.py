# ======================================
# UVM - Part 2: Feature Scaling & Unified Value Merge (Statistical Prep)
# ======================================

import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Load Cleaned Datasets ----------
path_base = "/Users/anokhpalakurthi/Downloads/"

passing   = pd.read_csv(path_base + "Passing_PFF_Clean.csv")
rushing   = pd.read_csv(path_base + "Rushing_PFF_Clean.csv")
receiving = pd.read_csv(path_base + "Receiving_PFF_Clean.csv")
blocking  = pd.read_csv(path_base + "Blocking_PFF_Clean.csv")

print("✅ Datasets successfully loaded.")
print(f"Passing: {passing.shape}, Rushing: {rushing.shape}, Receiving: {receiving.shape}, Blocking: {blocking.shape}")

# ---------- Optional: Clip negative anomalies ----------
for df_ in [passing, rushing, receiving, blocking]:
    for c in df_.select_dtypes(include='number').columns:
        # Keep negative PFF grades but clip other negatives
        if df_[c].min() < 0 and 'grade' not in c.lower():
            df_[c] = df_[c].clip(lower=0)

# ---------- Utility: z-score normalization ----------
def normalize_features(df, feature_cols, new_prefix):
    """Z-score normalize given columns and return average composite score."""
    scaler = StandardScaler()
    df_norm = df.copy()
    valid_cols = [col for col in feature_cols if col in df.columns]
    if not valid_cols:
        print(f"⚠️ No valid columns found for {new_prefix}")
        df_norm[f"{new_prefix}Score"] = 0
        return df_norm
    df_norm[valid_cols] = scaler.fit_transform(df_norm[valid_cols])
    df_norm[f"{new_prefix}Score"] = df_norm[valid_cols].mean(axis=1)
    print(f"✅ {new_prefix} normalized on {len(valid_cols)} features.")
    return df_norm

# ---------- Feature Selection per Domain ----------
# These match your current Part 1 cleaned CSVs

passing_features = [
    'PassingYards', 'PassingTDs', 'INTs', 'CompletionPercent',
    'YardsPerAttempt', 'PFF_PassGrade', 'PFF_OffenseGrade'
]

rushing_features = [
    'RushYards', 'RushTDs', 'YardsAfterContact', 'BreakawayYards',
    'YardsPerAttempt', 'YAC_PerAttempt', 'PFF_RunGrade', 'PFF_OffenseGrade'
]

receiving_features = [
    'ReceivingYards', 'ReceivingTDs', 'ReceivingFirstDowns', 'CatchPercent',
    'YardsPerRouteRun', 'YardsAfterCatch', 'PFF_RouteGrade', 'PFF_OffenseGrade'
]

blocking_features = [
    'TotalBlockSnaps', 'PassBlockSnaps', 'RunBlockSnaps',
    'PressuresAllowed', 'SacksAllowed', 'PassBlockEfficiency',
    'PFF_PassBlockGrade', 'PFF_RunBlockGrade', 'PFF_OffenseGrade'
]

# ---------- Normalize Each Group ----------
passing_norm   = normalize_features(passing,   passing_features,   "Air")
rushing_norm   = normalize_features(rushing,   rushing_features,   "Rush")
receiving_norm = normalize_features(receiving, receiving_features, "Receive")
blocking_norm  = normalize_features(blocking,  blocking_features,  "Block")

# ---------- Merge All Players ----------
def base_merge(df, key_cols):
    return df[key_cols + [col for col in df.columns if "Score" in col]]

key_cols = ["Player", "Team", "Position"]

merged = (
    base_merge(passing_norm, key_cols)
    .merge(base_merge(rushing_norm, key_cols),   on=key_cols, how="outer")
    .merge(base_merge(receiving_norm, key_cols), on=key_cols, how="outer")
    .merge(base_merge(blocking_norm, key_cols),  on=key_cols, how="outer")
)

# ---------- Fill missing domain scores with 0 ----------
merged.fillna(0, inplace=True)

# ---------- Export Clean Unified Dataset (no weighting yet) ----------
out_path = path_base + "Unified_Value_Model_Base.csv"
merged.to_csv(out_path, index=False)
print(f"\n✅ Unified base dataset with domain scores saved to: {out_path}")

# ---------- Diagnostics ----------
print(f"\nSummary Stats (Domain Scores Only):")
print(merged[['AirScore', 'RushScore', 'ReceiveScore', 'BlockScore']].describe())

print(f"\nMerged players: {merged['Player'].nunique()}")
print("\n✅ Ready for statistical weighting calibration in Part 3 (Ridge or OLS).")

# ======================================
#  Domain Distribution Visualization
# ======================================
domain_scores = merged[['AirScore', 'RushScore', 'ReceiveScore', 'BlockScore']]

# Compute means and standard deviations
means = domain_scores.mean()
stds = domain_scores.std()

plt.figure(figsize=(10, 6))
sns.violinplot(data=domain_scores, inner='box', cut=0, palette='muted')
plt.title("Distribution of Standardized Domain Scores (Z-Normalized)", fontsize=14, pad=15)
plt.xlabel("Domain (Phase)", fontsize=12)
plt.ylabel("Standardized Score (Z-Value)", fontsize=12)
plt.axhline(0, color='gray', linestyle='--', linewidth=1)

# Add annotations for mean ± std
for i, col in enumerate(domain_scores.columns):
    mean_val = means[col]
    std_val = stds[col]
    plt.text(
        i, mean_val + 0.25,  # position slightly above the mean
        f"μ={mean_val:.2f}\nσ={std_val:.2f}",
        ha='center', va='bottom', fontsize=10, color='black', fontweight='medium'
    )

plt.tight_layout()
plt.show()

print("\n🔢 Domain Distribution Summary:")
for col in domain_scores.columns:
    print(f"{col:<13} → mean={means[col]:.3f}, std={stds[col]:.3f}, range=({domain_scores[col].min():.2f}, {domain_scores[col].max():.2f})")
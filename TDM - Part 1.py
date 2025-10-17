# ======================================
# TDM - Part 1: Defensive Data Preparation
# ======================================

import pandas as pd
import numpy as np

BASE = "/Users/anokhpalakurthi/Downloads/"

def safe_divide(n, d):
    return np.where(d > 0, n / d, 0)

# Pass Rush
pass_rush = pd.read_csv(BASE + "pass_rush_summary.csv")
keep = [
    "player","team_name","position",
    "snap_counts_pass_rush","sacks","hits","hurries","total_pressures",
    "pass_rush_win_rate","prp","grades_pass_rush_defense","grades_defense",
    "pass_rush_wins"
]
pass_rush = pass_rush[[c for c in keep if c in pass_rush.columns]].rename(columns={
    "player":"Player","team_name":"Team","position":"Position",
    "snap_counts_pass_rush":"PassRushSnaps","sacks":"Sacks","hits":"Hits",
    "hurries":"Hurries","total_pressures":"Pressures",
    "pass_rush_win_rate":"WinRate","prp":"PRP",
    "grades_pass_rush_defense":"PFF_PassRushGrade",
    "grades_defense":"PFF_DefenseGrade","pass_rush_wins":"PassRushWins"
})
pass_rush["PressureRate"] = safe_divide(pass_rush["Pressures"], pass_rush["PassRushSnaps"])
for c in [c for c in pass_rush.columns if "Rate" in c or "PRP" in c]:
    if pass_rush[c].max() > 1: pass_rush[c] /= 100
pass_rush.fillna(0, inplace=True)

# Coverage
coverage = pd.read_csv(BASE + "defense_coverage_summary.csv")
keep = [
    "player","team_name","position","snap_counts_coverage",
    "targets","receptions","yards","touchdowns",
    "qb_rating_against","forced_incompletes",
    "grades_coverage_defense","grades_defense",
    "interceptions","pass_break_ups"
]
coverage = coverage[[c for c in keep if c in coverage.columns]].rename(columns={
    "player":"Player","team_name":"Team","position":"Position",
    "snap_counts_coverage":"CoverageSnaps",
    "targets":"Targets","receptions":"ReceptionsAllowed",
    "yards":"YardsAllowed","touchdowns":"TDsAllowed",
    "qb_rating_against":"PasserRatingAllowed",
    "forced_incompletes":"ForcedIncompletions",
    "grades_coverage_defense":"PFF_CoverageGrade",
    "grades_defense":"PFF_DefenseGrade",
    "interceptions":"INTs","pass_break_ups":"PBUs"
})
coverage["YardsPerTarget"] = safe_divide(coverage["YardsAllowed"], coverage["Targets"])
coverage.fillna(0, inplace=True)

# Run Defense
run = pd.read_csv(BASE + "run_defense_summary.csv")
keep = [
    "player","team_name","position","snap_counts_run",
    "stops","missed_tackles","missed_tackle_rate",
    "stop_percent","grades_run_defense","grades_defense",
    "forced_fumbles","tackles"
]
run = run[[c for c in keep if c in run.columns]].rename(columns={
    "player":"Player","team_name":"Team","position":"Position",
    "snap_counts_run":"RunDefenseSnaps",
    "stops":"Stops","missed_tackles":"MissedTackles",
    "missed_tackle_rate":"MissedTackleRate","stop_percent":"StopPercent",
    "grades_run_defense":"PFF_RunDefenseGrade",
    "grades_defense":"PFF_DefenseGrade",
    "forced_fumbles":"ForcedFumbles","tackles":"Tackles"
})
for c in [c for c in run.columns if "Rate" in c or "Percent" in c]:
    if run[c].max() > 1: run[c] /= 100
run.fillna(0, inplace=True)

# Merge and Clean
tdm = pass_rush.merge(coverage, on=["Player","Team","Position"], how="outer") \
               .merge(run, on=["Player","Team","Position"], how="outer")

for col in ["PFF_DefenseGrade_x","PFF_DefenseGrade_y"]:
    if col in tdm.columns:
        tdm["PFF_DefenseGrade"] = tdm.get("PFF_DefenseGrade",0) + tdm[col].fillna(0)
        tdm.drop(col, axis=1, inplace=True)

for c in ["PassRushSnaps","CoverageSnaps","RunDefenseSnaps"]:
    if c not in tdm.columns: tdm[c] = 0.0
tdm.fillna(0, inplace=True)

# Export versions
# Team Linked
tdm.to_csv(BASE + "TDM_Base_TeamLinked.csv", index=False)

# Player Level
agg_funcs = {
    "PassRushSnaps":"sum","CoverageSnaps":"sum","RunDefenseSnaps":"sum",
    "Pressures":"sum","Sacks":"sum","Hits":"sum","Hurries":"sum",
    "Stops":"sum","ForcedFumbles":"sum","INTs":"sum","PBUs":"sum","Tackles":"sum",
    "PFF_PassRushGrade":"mean","PFF_CoverageGrade":"mean",
    "PFF_RunDefenseGrade":"mean","PFF_DefenseGrade":"mean"
}
player_agg = tdm.groupby(["Player","Position"], as_index=False).agg(agg_funcs)

player_team_stats = (
    tdm.groupby("Player")
    .agg(TeamsPlayed=("Team","nunique"),
         PrimaryTeam=("Team", lambda x: x.value_counts().idxmax() if len(x) else None))
    .reset_index()
)

player_agg = player_agg.merge(player_team_stats, on="Player", how="left")
player_agg.fillna(0, inplace=True)

#Export
player_agg.to_csv(BASE + "TDM_Base_PlayerAgg.csv", index=False)
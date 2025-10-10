import pandas as pd

# ==========================
# PART 1: PASSING (PFF)
# ==========================

file_path = "/Users/anokhpalakurthi/Downloads/Passing (PFF).csv"
df = pd.read_csv(file_path)

keep_cols = [
    'player', 'team_name', 'position', 'dropbacks', 
    'attempts', 'completions', 'yards', 'touchdowns', 
    'interceptions', 'first_downs', 'completion_percent', 'ypa', 
    'pressure_to_sack_rate', 'def_gen_pressures', 'big_time_throws', 
    'turnover_worthy_plays', 'btt_rate', 'twp_rate', 'grades_pass', 
    'grades_offense'
]
df = df[keep_cols]

df.rename(columns={
    'player': 'Player', 'team_name': 'Team', 'position': 'Position',
    'dropbacks': 'Dropbacks', 'attempts': 'Attempts', 'completions': 'Completions',
    'yards': 'PassingYards', 'touchdowns': 'PassingTDs', 'interceptions': 'INTs',
    'first_downs': 'FirstDowns', 'completion_percent': 'CompletionPercent', 
    'ypa': 'YardsPerAttempt', 'pressure_to_sack_rate': 'PressureToSackRate', 
    'def_gen_pressures': 'PressuresFaced', 'big_time_throws': 'BigTimeThrows',
    'turnover_worthy_plays': 'TurnoverWorthyPlays', 'btt_rate': 'BTT_Rate',
    'twp_rate': 'TWP_Rate', 'grades_pass': 'PFF_PassGrade', 'grades_offense': 'PFF_OffenseGrade'
}, inplace=True)

# Cleaning
df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
df.fillna(0, inplace=True)
df['CompletionPercent'] = df['CompletionPercent'].clip(0, 100)
df['YardsPerAttempt'] = df['YardsPerAttempt'].clip(lower=0, upper=20)
for col in df.columns:
    if any(k in col for k in ['Yards', 'TD', 'Rate']) and df[col].min() < 0:
        df[col] = df[col].clip(lower=0)
rate_cols = [c for c in df.columns if 'Rate' in c or 'Percent' in c]
df[rate_cols] = df[rate_cols].apply(lambda x: x / 100 if x.max() > 1 else x)

# ==========================
# PART 2: RUSHING (PFF)
# ==========================
rush_path = "/Users/anokhpalakurthi/Downloads/Rushing (PFF).csv"
rush_df = pd.read_csv(rush_path)
rush_df.columns = rush_df.columns.str.strip().str.lower()

rush_keep_cols = [
    'player', 'team_name', 'position',
    'attempts', 'yards', 'touchdowns', 'first_downs',
    'yards_after_contact', 'breakaway_yards', 'fumbles',
    'grades_run', 'grades_offense'
]
rush_df = rush_df[rush_keep_cols].rename(columns={
    'player': 'Player', 'team_name': 'Team', 'position': 'Position',
    'attempts': 'RushAttempts', 'yards': 'RushYards',
    'touchdowns': 'RushTDs', 'first_downs': 'RushFirstDowns',
    'yards_after_contact': 'YardsAfterContact', 'breakaway_yards': 'BreakawayYards',
    'fumbles': 'Fumbles', 'grades_run': 'PFF_RunGrade', 'grades_offense': 'PFF_OffenseGrade'
})

rush_df['YardsPerAttempt'] = rush_df['RushYards'] / rush_df['RushAttempts']
rush_df['YAC_PerAttempt'] = rush_df['YardsAfterContact'] / rush_df['RushAttempts']
rush_df['ExplosiveRunRate'] = rush_df['BreakawayYards'] / rush_df['RushYards']
rush_df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
rush_df.fillna(0, inplace=True)
for col in rush_df.columns:
    if any(k in col for k in ['Yards', 'TD', 'Rate']) and rush_df[col].min() < 0:
        rush_df[col] = rush_df[col].clip(lower=0)
rate_cols = [c for c in rush_df.columns if 'Rate' in c or 'Percent' in c]
rush_df[rate_cols] = rush_df[rate_cols].apply(lambda x: x / 100 if x.max() > 1 else x)

# ==========================
# PART 3: RECEIVING (PFF)
# ==========================
receive_path = "/Users/anokhpalakurthi/Downloads/Receiving (PFF).csv"
receive_df = pd.read_csv(receive_path)

receive_keep_cols = [
    'player', 'team_name', 'position',
    'targets', 'receptions', 'yards', 'touchdowns', 'first_downs',
    'caught_percent', 'yprr','yards_after_catch', 'avoided_tackles',
    'drop_rate', 'drops', 'contested_targets', 'contested_receptions',
    'pass_block_rate', 'pass_blocks', 'grades_pass_route', 'grades_offense'
]

receive_df.rename(columns={
    'player': 'Player', 'team_name': 'Team', 'position': 'Position',
    'targets': 'Targets', 'receptions': 'Receptions', 'yards': 'ReceivingYards',
    'touchdowns': 'ReceivingTDs', 'first_downs': 'ReceivingFirstDowns',
    'caught_percent': 'CatchPercent', 'yprr': 'YardsPerRouteRun', 'yards_after_catch': 'YardsAfterCatch',
    'avoided_tackles': 'AvoidedTackles', 'drop_rate': 'DropRate', 'drops': 'Drops',
    'contested_targets': 'ContestedTargets', 'contested_receptions': 'ContestedReceptions',
    'pass_block_rate': 'PassBlockRate', 'pass_blocks': 'PassBlocks',
    'grades_pass_route': 'PFF_RouteGrade', 'grades_offense': 'PFF_OffenseGrade'
}, inplace=True)

receive_df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
receive_df.fillna(0, inplace=True)

# Apply abs() only to true yardage metrics
for col in ['ReceivingYards', 'YardsAfterCatch', 'YardsPerRouteRun']:
    if col in receive_df.columns:
        receive_df[col] = receive_df[col].abs()

# Do not abs() AvgDepthTarget — negatives indicate direction
for col in receive_df.columns:
    if any(k in col for k in ['TD', 'Rate']) and receive_df[col].min() < 0:
        receive_df[col] = receive_df[col].clip(lower=0)
rate_cols = [c for c in receive_df.columns if 'Rate' in c or 'Percent' in c]
receive_df[rate_cols] = receive_df[rate_cols].apply(lambda x: x / 100 if x.max() > 1 else x)

# ==========================
# PART 4: BLOCKING (PFF)
# ==========================
block_path = "/Users/anokhpalakurthi/Downloads/Blocking (PFF).csv"
block_df = pd.read_csv(block_path)

block_keep_cols = [
    'player', 'team_name', 'position', 'player_game_count',
    'snap_counts_block', 'snap_counts_pass_block', 'snap_counts_run_block',
    'grades_pass_block', 'grades_run_block', 'grades_offense',
    'pressures_allowed', 'hits_allowed', 'hurries_allowed', 'sacks_allowed',
    'pbe', 'penalties'
]
existing_cols = [c for c in block_keep_cols if c in block_df.columns]
block_df = block_df[existing_cols]

block_df.rename(columns={
    'player': 'Player', 'team_name': 'Team', 'position': 'Position',
    'player_game_count': 'Games', 'snap_counts_block': 'TotalBlockSnaps',
    'snap_counts_pass_block': 'PassBlockSnaps', 'snap_counts_run_block': 'RunBlockSnaps',
    'grades_pass_block': 'PFF_PassBlockGrade', 'grades_run_block': 'PFF_RunBlockGrade',
    'grades_offense': 'PFF_OffenseGrade', 'pressures_allowed': 'PressuresAllowed',
    'hits_allowed': 'HitsAllowed', 'hurries_allowed': 'HurriesAllowed',
    'sacks_allowed': 'SacksAllowed', 'pbe': 'PassBlockEfficiency', 'penalties': 'Penalties'
}, inplace=True)

block_df['PressureRateAllowed'] = block_df['PressuresAllowed'] / block_df['PassBlockSnaps']
block_df['SackRateAllowed'] = block_df['SacksAllowed'] / block_df['PassBlockSnaps']
block_df['PenaltyRate_Block'] = block_df['Penalties'] / block_df['TotalBlockSnaps']
block_df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
block_df.fillna(0, inplace=True)
for col in block_df.columns:
    if any(k in col for k in ['Yards', 'TD', 'Rate']) and block_df[col].min() < 0:
        block_df[col] = block_df[col].clip(lower=0)
rate_cols = [c for c in block_df.columns if 'Rate' in c or 'Percent' in c]
block_df[rate_cols] = block_df[rate_cols].apply(lambda x: x / 100 if x.max() > 1 else x)

# ==========================
# FINAL EXPORTS
# ==========================
df.to_csv("/Users/anokhpalakurthi/Downloads/Passing_PFF_Clean.csv", index=False)
rush_df.to_csv("/Users/anokhpalakurthi/Downloads/Rushing_PFF_Clean.csv", index=False)
receive_df.to_csv("/Users/anokhpalakurthi/Downloads/Receiving_PFF_Clean.csv", index=False)
block_df.to_csv("/Users/anokhpalakurthi/Downloads/Blocking_PFF_Clean.csv", index=False)
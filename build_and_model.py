"""
HIT140 Assessment 3 - Objective 2: Linear Regression
FIFA World Cup 2026

This script builds both required datasets and fits both linear regression
models end to end. Run this file directly to reproduce every number in
the accompanying report.

Data sources:
  - Team metadata (FIFA World Ranking, confederation, prior World Cup
    appearances, qualification date): Wikipedia 2026 FIFA World Cup
    group pages (all 12 groups), each sourced from FIFA's official
    ranking releases.
  - Match fixtures (all 104 matches, dates, stage, scores): FBref.com,
    2026 World Cup Scores & Fixtures page.
  - Squad average age and squad size: FBref.com, 2026 World Cup
    Squad Standard Stats page.

All explanatory variables below are known before kickoff of the match
they describe. No in-match or post-match statistic (shots, cards,
possession, etc.) is used as a predictor in either model.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

TOURNEY_START = pd.Timestamp('2026-06-11')

# ---------------------------------------------------------------
# 1. Load raw inputs
# ---------------------------------------------------------------
meta = pd.read_csv('team_metadata.csv')
poss = pd.read_csv('team_possession.csv')[['Squad', 'PlCount', 'Age']]
fixtures = pd.read_csv('fixtures.csv')
fixtures['date'] = pd.to_datetime(fixtures['date'])

team_feat = meta.merge(poss, left_on='Team', right_on='Squad', how='left')
team_feat['QualDate'] = pd.to_datetime(team_feat['QualDate'])
team_feat = team_feat.set_index('Team')

def get(team, col):
    return team_feat.loc[team, col]

def qual_days(team):
    return (TOURNEY_START - pd.Timestamp(get(team, 'QualDate'))).days

# ---------------------------------------------------------------
# 2. Compute rest days: days since each team's previous match
# ---------------------------------------------------------------
fixtures = fixtures.sort_values('date').reset_index(drop=True)
last_played = {}
rest_home, rest_away = [], []
for _, r in fixtures.iterrows():
    h, a, d = r['home_team'], r['away_team'], r['date']
    rest_home.append((d - last_played[h]).days if h in last_played else np.nan)
    rest_away.append((d - last_played[a]).days if a in last_played else np.nan)
    last_played[h] = d
    last_played[a] = d
fixtures['rest_days_home'] = rest_home
fixtures['rest_days_away'] = rest_away
median_rest = pd.Series(rest_home + rest_away).median()
fixtures['rest_days_home'] = fixtures['rest_days_home'].fillna(median_rest)
fixtures['rest_days_away'] = fixtures['rest_days_away'].fillna(median_rest)

# ---------------------------------------------------------------
# 3. Dataset 2.1 - one row per match (104 rows), predicts goal difference
# ---------------------------------------------------------------
rows = []
for _, r in fixtures.iterrows():
    h, a = r['home_team'], r['away_team']
    rows.append({
        'match_id': r['match_id'], 'date': r['date'],
        'home_team': h, 'away_team': a,
        'rank_diff': get(a, 'Rank_Jun2026') - get(h, 'Rank_Jun2026'),
        'age_diff': get(h, 'Age') - get(a, 'Age'),
        'appearances_diff': get(h, 'FinalsAppearances') - get(a, 'FinalsAppearances'),
        'squadsize_diff': get(h, 'PlCount') - get(a, 'PlCount'),
        'qualdays_diff': qual_days(h) - qual_days(a),
        'rest_days_diff': r['rest_days_home'] - r['rest_days_away'],
        'knockout_stage': r['knockout'],
        'same_confederation': int(get(h, 'Confederation') == get(a, 'Confederation')),
        'goal_difference': r['home_score'] - r['away_score'],
    })
df21 = pd.DataFrame(rows)
df21.to_csv('dataset_2_1_goal_difference.csv', index=False)

# ---------------------------------------------------------------
# 4. Dataset 2.2 - one row per team-per-match (208 rows), predicts goals scored
# ---------------------------------------------------------------
rows2 = []
for _, r in fixtures.iterrows():
    for team, opp, rest_own, gls in [
        (r['home_team'], r['away_team'], r['rest_days_home'], r['home_score']),
        (r['away_team'], r['home_team'], r['rest_days_away'], r['away_score']),
    ]:
        rows2.append({
            'match_id': r['match_id'], 'date': r['date'],
            'team': team, 'opponent': opp,
            'team_rank': get(team, 'Rank_Jun2026'),
            'opponent_rank': get(opp, 'Rank_Jun2026'),
            'team_age': get(team, 'Age'),
            'team_appearances': get(team, 'FinalsAppearances'),
            'team_squadsize': get(team, 'PlCount'),
            'team_qualdays': qual_days(team),
            'rest_days': rest_own,
            'knockout_stage': r['knockout'],
            'goals_scored': gls,
        })
df22 = pd.DataFrame(rows2)
df22.to_csv('dataset_2_2_goals_scored.csv', index=False)

# ---------------------------------------------------------------
# 5. Fit and evaluate Model 2.1
# ---------------------------------------------------------------
print("="*60, "\nMODEL 2.1: Goal Difference\n", "="*60, sep="")
X_cols = ['rank_diff','age_diff','appearances_diff','squadsize_diff',
          'qualdays_diff','rest_days_diff','knockout_stage','same_confederation']
X, y = df21[X_cols], df21['goal_difference']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
m1 = LinearRegression().fit(X_tr, y_tr)
pred_te, pred_tr = m1.predict(X_te), m1.predict(X_tr)
print(f"Train R2: {r2_score(y_tr, pred_tr):.3f}  |  Test R2: {r2_score(y_te, pred_te):.3f}")
print(f"Test MAE: {mean_absolute_error(y_te, pred_te):.3f}  |  Test RMSE: {np.sqrt(mean_squared_error(y_te, pred_te)):.3f}")
for c, coef in zip(X_cols, m1.coef_):
    print(f"  {c:20s} {coef:8.4f}")
print(f"  {'intercept':20s} {m1.intercept_:8.4f}")

# ---------------------------------------------------------------
# 6. Fit and evaluate Model 2.2
# ---------------------------------------------------------------
print("\n" + "="*60, "\nMODEL 2.2: Goals Scored\n", "="*60, sep="")
X2_cols = ['team_rank','opponent_rank','team_age','team_appearances',
           'team_squadsize','team_qualdays','rest_days','knockout_stage']
X2, y2 = df22[X2_cols], df22['goals_scored']
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2, random_state=42)
m2 = LinearRegression().fit(X2_tr, y2_tr)
pred2_te, pred2_tr = m2.predict(X2_te), m2.predict(X2_tr)
print(f"Train R2: {r2_score(y2_tr, pred2_tr):.3f}  |  Test R2: {r2_score(y2_te, pred2_te):.3f}")
print(f"Test MAE: {mean_absolute_error(y2_te, pred2_te):.3f}  |  Test RMSE: {np.sqrt(mean_squared_error(y2_te, pred2_te)):.3f}")
for c, coef in zip(X2_cols, m2.coef_):
    print(f"  {c:20s} {coef:8.4f}")
print(f"  {'intercept':20s} {m2.intercept_:8.4f}")

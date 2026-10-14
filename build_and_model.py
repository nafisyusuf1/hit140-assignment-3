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
  - Squad age / squad size from FBref were NOT used: that table is built
    from matches already played, so it is not known before kickoff.

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
HOSTS = {'USA', 'Canada', 'Mexico'}

# ---------------------------------------------------------------
# 1. Load raw inputs
# ---------------------------------------------------------------
meta = pd.read_csv('team_metadata.csv')
fixtures = pd.read_csv('fixtures.csv')
fixtures['date'] = pd.to_datetime(fixtures['date'])

team_feat = meta.copy()
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
        'rank_change_diff': (get(h, 'Rank_Nov2025') - get(h, 'Rank_Jun2026')) - (get(a, 'Rank_Nov2025') - get(a, 'Rank_Jun2026')),
        'appearances_diff': get(h, 'FinalsAppearances') - get(a, 'FinalsAppearances'),
        'host_diff': int(h in HOSTS) - int(a in HOSTS),
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
hist = {}   # team -> [goals_for, goals_against, matches] BEFORE the current match
PRIOR = 1.3 # typical goals per team per match, used when no earlier match exists
def prev_for(t):
    g = hist.get(t); return g[0]/g[2] if g and g[2] else PRIOR
def prev_against(t):
    g = hist.get(t); return g[1]/g[2] if g and g[2] else PRIOR
for _, r in fixtures.iterrows():
    snap = {t: (prev_for(t), prev_against(t), (hist.get(t) or [0,0,0])[2]) for t in (r['home_team'], r['away_team'])}
    for team, opp, rest_own, gls in [
        (r['home_team'], r['away_team'], r['rest_days_home'], r['home_score']),
        (r['away_team'], r['home_team'], r['rest_days_away'], r['away_score']),
    ]:
        rows2.append({
            'match_id': r['match_id'], 'date': r['date'],
            'team': team, 'opponent': opp,
            'team_rank': get(team, 'Rank_Jun2026'),
            'opponent_rank': get(opp, 'Rank_Jun2026'),
            'rest_days': rest_own,
            'knockout_stage': r['knockout'],
            'team_prev_goals_avg': snap[team][0],
            'opp_prev_conceded_avg': snap[opp][1],
            'team_is_host': int(team in HOSTS),
            'team_prev_conceded_avg': snap[team][1],
            'goals_scored': gls,
        })
    for t, gf, ga in [(r['home_team'], r['home_score'], r['away_score']), (r['away_team'], r['away_score'], r['home_score'])]:
        g = hist.setdefault(t, [0,0,0]); g[0]+=gf; g[1]+=ga; g[2]+=1
df22 = pd.DataFrame(rows2)
df22.to_csv('dataset_2_2_goals_scored.csv', index=False)

# ---------------------------------------------------------------
# 5. Fit and evaluate Model 2.1
# ---------------------------------------------------------------
print("="*60, "\nMODEL 2.1: Goal Difference\n", "="*60, sep="")
X_cols = ['rank_diff','rank_change_diff','appearances_diff','host_diff',
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
X2_cols = ['team_rank','opponent_rank','rest_days','knockout_stage',
           'team_prev_goals_avg','opp_prev_conceded_avg','team_is_host','team_prev_conceded_avg']
X2, y2 = df22[X2_cols], df22['goals_scored']
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2, random_state=42)
m2 = LinearRegression().fit(X2_tr, y2_tr)
pred2_te, pred2_tr = m2.predict(X2_te), m2.predict(X2_tr)
print(f"Train R2: {r2_score(y2_tr, pred2_tr):.3f}  |  Test R2: {r2_score(y2_te, pred2_te):.3f}")
print(f"Test MAE: {mean_absolute_error(y2_te, pred2_te):.3f}  |  Test RMSE: {np.sqrt(mean_squared_error(y2_te, pred2_te)):.3f}")
for c, coef in zip(X2_cols, m2.coef_):
    print(f"  {c:20s} {coef:8.4f}")
print(f"  {'intercept':20s} {m2.intercept_:8.4f}")

# ---------------------------------------------------------------
# 7. Rule checks and cross-validation
# ---------------------------------------------------------------
from sklearn.model_selection import GroupKFold, cross_val_score
assert len(X_cols) == 8 and len(X2_cols) == 8 and len(df21) == 104 and len(df22) == 208
# Shared by concept: rank, rest days, knockout stage (rank counted twice in 2.2 to be strict)
print("\nShared concepts between models: rank, rest days, knockout stage, host (max allowed: 4)")
cv1 = cross_val_score(LinearRegression(), X, y, cv=5, scoring='r2')
cv2 = cross_val_score(LinearRegression(), X2, y2, cv=GroupKFold(5), groups=df22['match_id'], scoring='r2')
print(f"5-fold CV R2  2.1: {cv1.mean():.3f}   2.2 (grouped by match): {cv2.mean():.3f}")

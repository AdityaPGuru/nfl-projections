# NFL projections: QB yards per attempt, fantasy points, betting props, defensive YPA and YPC allowed

Week-ahead NFL models built from public data. The main question is how much of a single game can be predicted before kickoff, and the honest answer for most of these targets is "some, not a lot".

## What is in here

| Script | What it does |
|---|---|
| `fetch_data.py` | Downloads the Kaggle player-game table, nflverse play-by-play and the ESPN scoreboard into `$S` |
| `bakeoff.py` | 18 models (KNN, empirical Bayes, gradient boosting, linear, logistic, ridge, lasso, SVR, random forest, neural net, blends) on QB YPA and tight end stats |
| `fantasy_bet.py` | Fantasy points and betting props, with calibrated over/under probabilities |
| `pbp_feats.py`, `check45.py` | Play-by-play features, and a test of 45 hand-built situational variables |
| `week2.py`, `season_eval.py`, `tracked.py` | 2026 week 2 projections, a full 2025 season check, and why six games cannot rank stats |
| `def_model.py`, `noshrink.py`, `eb_sd.py` | Defense-only models for pass YPA allowed and rush YPC allowed, and shrinkage derived from standard deviations |
| `rush_model.py`, `recency_learned.py`, `l2_eval.py` | Rushing YPC matchup model, a recency half-life learned from training data, and the weekly vector evaluation |

`results/` has the output tables and `results/logs/` has the raw printed output of the main runs.

## Rules the evaluation follows

- Every feature uses only games played before the game being predicted. The Kaggle table ships rolling averages that include the current game, so all lags are rebuilt from the base columns. A perturbation test confirms nothing leaks.
- Out-of-fold validation uses 5 folds scattered across all seasons, grouped by game, so no single season drives the result. A strict forward test (train on the past, predict the next season) is reported next to it.
- Tuning values are learned from training data, never picked by hand. The shrinkage constant is k = (luck sd of one play)^2 / (true team-to-team sd)^2. The recency half-life is chosen by nested forward validation.
- The best model for each target is calibrated on its out-of-fold predictions.

## Main results

| Target | Correlation (out of fold) | Error |
|---|---|---|
| QB yards per attempt, single game | 0.30 | RMSE 1.81, luck floor about 1.78 |
| QB passing yards | 0.59 | RMSE 78 yards |
| QB fantasy points (PPR) | 0.46 | RMSE 10.1 |
| TE receiving yards | 0.53 | RMSE 23.6 yards |
| TE receptions | 0.57 | RMSE 1.79 |
| TE yards per target | 0.10 | close to unpredictable |
| Team pass YPA allowed, full matchup model | 0.32 (0.41 on 2025) | RMSE 1.70 |
| Team pass YPA allowed, defense-only stats | 0.14 | RMSE 1.78 |
| Team rush YPC allowed, matchup model | 0.19 (0.23 on 2025) | RMSE 1.35, per-game miss about 1.07 YPC |

Things that turned out to matter: the Vegas spread and total, the opposing offense, and per-QB EPA and CPOE. Things that did not: narrow situational splits (4th down, first quarter, blowouts), short recent-form windows, and heavy recency weighting of the training data.

A single game carries a lot of luck. One pass attempt has a standard deviation of about 10 yards and one carry about 6.4 yards, so a team's single-game number moves by roughly 30 percent on luck alone. Scoring projections over four to eight game windows cuts the error roughly in half.

The betting probabilities are calibrated, but they have only been tested against a stand-in line (the player's own recent average), not real sportsbook lines. Nothing here shows a betting edge.

## Defensive pass YPA allowed: final stack

| Script | What it does |
|---|---|
| `def_ypa_full_eval.py` | Leakage test, minimum training seasons, every baseline stat with its week-ahead correlation, model ladder |
| `ypa_2022_2025.py` | Best stats on 2022 to 2025, tracked matchups, and how projections behave in the top and bottom 5 and 10 percent of actual YPA |
| `knn_comps.py`, `knn_elite_test.py` | Team-vector KNN (find the most similar past matchups) and a test of whether it helps against elite offenses |
| `full_stack.py`, `best_stack.py` | 21 level-1 models, ridge meta-model, calibration, head-to-head against baselines |
| `stack_ranges.py` | Per-game 5th to 95th percentile ranges and blow-up / shutdown probabilities |

Strictly forward, 2020 to 2025, 3,230 games:

| Model | Correlation | Avg weekly correlation | Per-game L2 |
|---|---|---|---|
| Guess the prior average | 0.02 | 0.00 | 1.791 |
| Defense's raw trailing YPA allowed | 0.12 | 0.12 | 1.788 |
| 23 hand-built defensive variables | 0.11 | 0.11 | 1.778 |
| Ridge, 7 inputs | 0.33 | 0.34 | 1.688 |
| Final stack of 21 models, calibrated | 0.33 | 0.35 | 1.687 |

The stack beats every baseline by about 10 standard errors and ties the 7-input ridge model. Six to eight training seasons is the minimum before forward error stops improving. The team-vector KNN gives sensible comps but adds nothing to the stack, including against elite offenses. Knowing the opposing starting quarterback removes about a third of the bias in games with a new starter and does not change accuracy, because the Vegas line already carries it. Per-game ranges are honest: 80.7 percent of results land inside the 10th to 90th range and 90.3 percent inside the 5th to 95th.

## Running it

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
export S=./data
venv/bin/python fetch_data.py 2026
venv/bin/python pbp_feats.py
venv/bin/python bakeoff.py
venv/bin/python rush_model.py
```

## Data sources

Kaggle `philiphyde1/nfl-stats-1999-2022` (covers 2012 to 2025 despite the name), nflverse play-by-play releases, and the ESPN public scoreboard API for schedules and lines. No raw data is stored in this repository.

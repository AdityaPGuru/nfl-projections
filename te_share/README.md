# TE receiving-yards share: rebuild and fix

Target: per offense team-game, TE receiving yards ÷ all team receiving yards (regular season), same definition as the frozen R baseline.
Data: nflverse play-by-play, weekly rosters, injury reports, PFR snap counts, and nfldata schedule/lines, 2012–2026.

Nothing is fit on 2026. Each forecast season S is fit on 2013..S-1, and 2026 uses a model fit on 2013–2025.
2026 games through week W-1 enter only as pregame features (season-to-date shares, roster, injury report), which follows the same timing rule as the original project.

## Quickstart
Needs Python 3.11 or newer. On a Mac, LightGBM also needs OpenMP: `brew install libomp`.

```
git clone https://github.com/AdityaPGuru/nfl-projections.git
cd nfl-projections/te_share
python -m pip install -r requirements.txt
```

**Run right away (no download).** The built stats table `data/team_games.parquet` is in the repo, covering 2012 through 2026 Week 5 plus the Week 6 schedule:
```
python model.py            # forward-season evaluation, prints every table in this README
python predict_week.py 6   # Week 6 projections -> results/week6_2026_projections.csv
```

**Each new week.** Pull fresh nflverse data, rebuild the table, then project. This takes a few minutes the first time (about 290 MB). After that, only the current season is downloaded again.
```
python fetch_data.py
python build.py
python predict_week.py 7
```
The upcoming week's projections are best after Friday's final injury reports. nflverse usually posts completed games within a day.

**Exact reproduction.** `python fetch_data.py --snapshot` downloads the raw files behind these results (2026 through Week 5) from the [repo release](https://github.com/AdityaPGuru/nfl-projections/releases/tag/te-share-data-2026-10-15). Then `python build.py && python model.py` reproduces the numbers below.

Data comes from [nflverse](https://github.com/nflverse/nflverse-data) (CC-BY 4.0).

## Files
- `fetch_data.py`: downloads raw nflverse data into `data/`
- `build.py`: team-game table and pregame features → `data/team_games.parquet`
- `model.py`: forward-season evaluation of all models, plus diagnostics → `results/forward_predictions.csv`, `results/model_out_v4.txt`
- `predict_week.py N`: 2026 week N projections → `results/weekN_2026_projections.csv`
- `results/2026_wk1-5_projections_vs_actual.csv`: every 2026 game so far, projected vs actual

## Why the old model failed in 2026
1. **League level.** 2025 had the highest league TE share in the data (23.7%). Every model that fits a weight on the league level, or on raw shares, carries that record into 2026 and over-projects. 2026 is running at 22.8%. My first history-trained models were biased +3.6 to +4.2 points at every 2026 week. Predicting each team's share *relative to* last season's league level, and centering roster and snap features on each week's league mean, cut the bias to about +1.5.
2. **Selection.** The 86 terms came from 500-stat screens over the same history they were scored on. Many terms are tiny situational slices (4th-and-short, defense timeout low, goal-to-go), so a 0.43 correlation over the screened history is optimistic. A plain model on a dozen stable inputs scores 0.39–0.40 forward without any screening.
3. **2026 itself isn't a regime break at the team level.** Prior-season TE share still predicts weeks 1–5 about as usual (team-level r 0.39). Week 1 of 2026 was a miss for every model (r ≈ 0). On 32 games the chance spread of r is about ±0.18, and past week-1 r ranged from −0.27 to 0.58.

## Why projections shrank late in the season
The baseline puts fixed weights on season-to-date rates. Early in the season those rates are mostly noise, so the fitted weights are small. Late in the season the rates settle down and spread out less, and the same small weights then squeeze projections toward the middle.
Measured on 2016–2025 forward predictions with the same fixed-weight structure:

| weeks | projection sd | calibration slope (1.0 = right width) |
|---|---|---|
| 1–4 | 0.050 | 0.71 (too wide) |
| 5–8 | 0.049 | 1.03 |
| 9–12 | 0.047 | 1.15 (too narrow) |
| 13–18 | 0.046 | 1.15 (too narrow) |

Fix: empirical-Bayes shares. The posterior is (k·prior + n·season-to-date) / (k + n), with k learned on training data: offense k ≈ 10.5 games, defense k ≈ 89 games, so defense share allowed barely persists. The weight on the current season then grows with games played. On top of that, a calibration whose slope depends on that weight is fit on leave-one-season-out predictions inside the training seasons. Result for the final model (`eb_full_cal`):

| weeks | projection sd | calibration slope |
|---|---|---|
| 1–4 | 0.049 | 0.82 |
| 5–8 | 0.054 | 1.03 |
| 9–12 | 0.055 | 1.06 |
| 13–18 | 0.058 | 0.98 |

## Results (r, RMSE)
| model | forward 2016–2025 | 2026 wk 1–5 (never trained on) |
|---|---|---|
| fixed-weight style (raw rates) | 0.359, 0.1269 | 0.266, 0.1272 |
| EB offense × defense only | 0.379, 0.1258 | 0.270, 0.1273 |
| **EB + personnel + Vegas, calibrated** | **0.395, 0.1250** | **0.308, 0.1244** |

Inputs in the final model: EB offense share, EB defense share allowed, their product, TE1/WR1 out (reserve list or Out/Doubtful on the injury report), lost TE/WR production, TE share of returning receiving production, season-to-date TE target and air-yards shares, TE snap shares, implied team total, spread, home, and this season's league drift.
The noise floor is high. A single game's TE share has an sd of about 0.13, and the best model's RMSE is 0.125.

## Review
A Codex adversarial review (review/report.txt) confirmed the forward fits never touch the test season, the defense and offense season-to-date columns match a brute-force recompute, and the spread sign is right.
It found one look-ahead: receiver positions came from the season-long mode. They now use each week's roster, and the numbers above are from the rerun (the change moved r by about 0.001).
The other leak it found is left as is. Week-level centering of the injury-derived roster features uses every team's report for that week, so a Thursday game can see other teams' later reports. This only shifts a league-wide constant.

## Running the upcoming week
Rerun `python3 build.py && python3 predict_week.py 6` after Friday's final injury reports. As of Thursday, most teams' Out/Doubtful designations for Sunday aren't published yet.

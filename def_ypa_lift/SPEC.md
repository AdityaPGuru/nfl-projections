# Defensive YPA model: exact definitions, sources, and the Week 7 2026 build

This answers the request for the exact definition and source of every term, especially `c_qb_epa`, and for Week 7 values built from completed 2026 Weeks 1–6 plus eligible prior-season data. Every play-derived term uses only those weeks. Schedule fields (lines, rest days, listed starters) come from the Week 7 rows of a schedule snapshot taken 2026-10-23 (section 3 notes).

## Read this first: three corrections

1. **This is not the accepted Week 6 baseline.** That baseline is your R model (`defensive_ypa_old_baseline_lm.R`). It reads cached files (`defensive_ypa_model_data.rds`, `expanded_pregame_features.parquet` and the other `r_ypa_historical_addition_screen` caches) that were never shared, so none of them are used here. Everything below is a separate Python rebuild from nflverse play-by-play, with term names borrowed from your R script. Its values will not match your caches exactly, and nothing here can stand in for them.
2. **There is no 16-term model any more.** The 16 terms in the first version were selected on a feature table with a leak. When a team's usual QB sat out a game, his stats were missing for that game, and the median fill told the model that a backup was playing. The leak is fixed (section 5). Re-running the same selection gives a **14-term** model that uses no QB-starter field, and a **17-term** model that adds `c_qb_epa` and `c_qb_new`.
3. **`c_qb_epa` uses retrospective information** (section 2). Rebuilt from information available before kickoff, it adds nothing. The 14-term model is the one to use for a pregame forecast.

| model | terms | r 2016–25 | r 2023–25 | MAE 2016–25 | r 2026 Wk 1–6 | MAE 2026 Wk 1–6 |
|---|---|---|---|---|---|---|
| **14-term, pregame only (recommended)** | `pruned_terms_noqb.csv` | 0.351 | 0.369 | 1.408 | 0.399 | 1.381 |
| 17-term, adds schedule-starter QB | `pruned_terms.csv` | 0.355 | 0.369 | 1.405 | 0.398 | 1.382 |
| 33 clean terms (your 35 minus wind and temp) | `evaluate.py: BASE33` | 0.340 | 0.367 | 1.414 | | |

These use your protocol: for season s and week w, refit OLS on every row from seasons before s plus weeks before w of season s, fill missing cells with the training medians, and score the raw prediction. The terms were picked on 2016–2022 only, so 2023–2025 and 2026 are honest checks. On those, the 14- and 17-term models are equal.

## 1. Rules shared by every term

**Play set ("dropbacks").** nflverse play-by-play, `season_type == "REG"`, `posteam` and `defteam` not null, `pass_attempt == 1` (sacks and spikes included) and `two_point_attempt != 1`. Columns used: `yards_gained, air_yards, yards_after_catch, epa, yac_epa, success, complete_pass, sack, qb_hit, qb_scramble, pass_touchdown, qtr, down, score_differential, yardline_100, passer_player_id, pass_oe, xpass`. Code: `build.py: load()` and the `db = ...` line in `main()`.

**Target.** Net passing yards per dropback allowed in the game = sum of `yards_gained` over dropbacks ÷ number of dropbacks. Sacks count as attempts and their lost yards are included.

**Pregame window ("ytd").** For a row in season s, week w, a team's or QB's term pools every qualifying play from **the whole regular season s−1** together with **season s weeks 1 to w−1**. Seasons before s−1 are never used. The row's own week is never used. Rates are ratios of pooled sums, not averages of per-game rates. Code: `build.py: window()` and `rate_feats()`.

**Week 7 2026 cutoff.** The window is the full 2025 regular season plus 2026 Weeks 1–6. `FORECAST=2026,7` makes `load()` drop every play with `season*100 + week >= 202607`. That matters because NE @ CHI (Week 7, Oct 22) was already in the downloaded file.

**Situations** (scores are from the offense's view): blowout = |`score_differential`| ≥ 17; neutral = |`score_differential`| ≤ 7; behind the line = `air_yards` ≤ 0; deep = `air_yards` ≥ 20; completion = `complete_pass == 1`; backed up = `yardline_100` ≥ 80. Sacks have no `air_yards`, so they drop out of every air-yards term.

## 2. `c_qb_epa`

| | |
|---|---|
| Formula | sum of `epa` over the QB's dropbacks in the window ÷ number of those dropbacks (a dropback with missing `epa` counts as 0 in the sum and 1 in the count) |
| Unit | expected points added per dropback (sacks included) |
| Which QB | the offense's listed starter in the nflverse schedule: `home_qb_id` when the offense is the home team, else `away_qb_id` |
| Window | that QB's own dropbacks, for any team, in the whole previous regular season plus earlier weeks of this season. Week 7 2026 uses 2025 REG plus 2026 Weeks 1–6. |
| Source | `games.csv` (columns `home_qb_id, away_qb_id, home_team`) and play-by-play (`passer_player_id, epa, pass_attempt, two_point_attempt, season_type, season, week`) |
| Week 7 files | `week7/inputs_2026-10-23/games.csv`, `week7/inputs_2026-10-23/pbp_2026.parquet`, and `pbp_2025.parquet` from the history folder |
| Code | `build.py: main()`, block `C4` (`qs = rate_feats(...)`) |
| Command | see section 6 |

**Retrospective information: yes.** nflverse writes a projected starter into `home_qb_id`/`away_qb_id` a few days before a game and overwrites it with the actual starter afterward. Every historical row the model trains on therefore has the actual starter, while the Week 7 rows have projected starters as of 2026-10-23 20:46 ET. The exception is NE @ CHI: it was played on Oct 22, before the snapshot, so its starter IDs are already the post-game actual starters. The leak test confirms that `c_qb_epa` depends on this field: faking one week's starter IDs changes that week's values.

**How much it matters.** Added alone to the 33 clean terms, `c_qb_epa` raises r 2016–25 by +0.0031. `c_qb_epa_pre` is the same formula with the QB chosen before the game: the offense's leading passer (most dropbacks) in its most recent completed game. Added alone, it changes r by −0.0002. The gain comes from knowing who actually started, which a pregame forecast can't know with certainty.

`c_qb_new` = 1 when the listed starter differs from the offense's top QB (defined below), else 0. It has the same retrospective dependence.

## 3. The 14 terms of the recommended model

All terms are built in `build.py: main()` from the play-by-play columns above. "D window" means the defense's plays faced in the window; "O window" means the offense's plays.

| # | term | exact definition | coef (Week 7 fit) |
|---|---|---|---|
| 1 | `def_blowout_std_air_comp_mean` | sample SD (n−1) of `air_yards` on completions in blowout states, D window | −0.0475 |
| 2 | `def_backedup_ypa_min` | for each game in the D window, mean `yards_gained` on backed-up dropbacks; the term is the minimum of those per-game means | 0.0155 |
| 3 | `def_blowout_air_std` | sample SD of `air_yards` on attempts with recorded `air_yards` (sacks have none) in blowout states, D window | 0.0769 |
| 4 | `def_qb_hit_rate_q4` | sum of `qb_hit` ÷ dropbacks, `qtr == 4`, D window | −1.8715 |
| 5 | `def_epa_neutral_behind_los` | mean `epa` on neutral-state throws behind the line, D window | 0.0642 |
| 6 | `def_td_rate_behind_los` | sum of `pass_touchdown` ÷ throws behind the line, D window | 5.3433 |
| 7 | `def_yac_epa_per_dropback_ytd` | sum of `yac_epa` on completions ÷ all dropbacks, D window | −1.4218 |
| 8 | `def_rest_days` | `away_rest` if the defense is the away team, else `home_rest` (`games.csv`) | −0.0205 |
| 9 | `off_success_rate_ytd` | mean `success` over dropbacks, O window | 3.1898 |
| 10 | `off_top_qb_yac_per_completion_ytd` | top QB = passer with the most dropbacks for this offense in the O window; term = his sum of `yards_after_catch` on completions ÷ his completions, over his own window (any team) | 0.1076 |
| 11 | `off_pass_rate_ytd` | share of plays with `pass_attempt == 1` among plays with `pass_attempt == 1` or `rush_attempt == 1` (two-point tries excluded), O window | 2.7496 |
| 12 | `off_pass_oe_ytd` | mean `pass_oe` over those plays where it is not null, O window | −0.0462 |
| 13 | `off_spread_line` | `spread_line` × (+1 if the offense is home, −1 if away); positive = offense favored (`games.csv`; nflverse `spread_line` > 0 means home favored) | 0.0770 |
| 14 | `total_line` | `total_line` (`games.csv`) | 0.0878 |

Intercept −0.7391. The coefficients come from fitting on every completed row from 2013 through 2026 Week 6 (`week7/week7_coefficients_14.csv`; 17-term version in `week7_coefficients_17.csv`).

The 17-term model adds `def_neut_q2_air_yards` (mean `air_yards` on neutral-state, `qtr == 2` attempts with recorded `air_yards`, D window), `c_qb_epa`, and `c_qb_new`.

Notes:
- `off_spread_line` and `total_line` are the lines in `games.csv` when it was downloaded. Historical rows have nflverse's closing lines. The Week 7 rows have lines as of 2026-10-23 20:46 ET, which can still move before kickoff.
- `def_rest_days` is known from the schedule ahead of time.
- Despite the name, `off_ngs_*` terms in your list are rebuilt here from play-by-play, not Next Gen Stats. Neither appears in the 14- or 17-term model.

## 4. Week 7 2026 values

Built with `FORECAST=2026,7`, so only 2025 REG plus completed 2026 Weeks 1–6 plays enter any play-derived term. Lines, rest, and listed starters come from the 2026-10-23 schedule snapshot; for NE @ CHI (already played) the starter IDs are post-game values, which affects only the 17-term projection. All 14 terms are filled for all 28 rows, so no median fill was needed. Every term value for every row is in `week7/week7_2026_terms_and_projections.csv`.

| game | offense | defense | schedule starter | c_qb_epa | last-game leading passer | c_qb_epa_pre | top QB (window) | proj NY/A, 14-term | proj NY/A, 17-term |
|---|---|---|---|---|---|---|---|---|---|
| 2026_07_CIN_BAL | BAL | CIN | T.Huntley | 0.028 | T.Huntley | 0.028 | L.Jackson | 6.685 | 6.714 |
| 2026_07_CIN_BAL | CIN | BAL | J.Burrow | 0.120 | J.Burrow | 0.120 | J.Burrow | 6.626 | 6.649 |
| 2026_07_CLE_TEN | CLE | TEN | D.Watson | -0.041 | D.Watson | -0.041 | S.Sanders | 5.439 | 5.488 |
| 2026_07_CLE_TEN | TEN | CLE | C.Ward | -0.194 | C.Ward | -0.194 | C.Ward | 5.220 | 5.266 |
| 2026_07_DAL_PHI | DAL | PHI | D.Prescott | 0.154 | D.Prescott | 0.154 | D.Prescott | 6.602 | 6.677 |
| 2026_07_DAL_PHI | PHI | DAL | J.Hurts | 0.021 | T.McKee | -0.118 | J.Hurts | 5.692 | 5.846 |
| 2026_07_DEN_ARI | ARI | DEN | J.Brissett | -0.009 | J.Brissett | -0.009 | J.Brissett | 5.616 | 5.609 |
| 2026_07_DEN_ARI | DEN | ARI | B.Nix | 0.046 | B.Nix | 0.046 | B.Nix | 6.331 | 6.386 |
| 2026_07_GB_DET | DET | GB | J.Goff | 0.204 | J.Goff | 0.204 | J.Goff | 7.574 | 7.500 |
| 2026_07_GB_DET | GB | DET | J.Love | 0.193 | J.Love | 0.193 | J.Love | 6.662 | 6.780 |
| 2026_07_IND_MIN | IND | MIN | D.Jones | 0.041 | D.Jones | 0.041 | D.Jones | 5.101 | 5.186 |
| 2026_07_IND_MIN | MIN | IND | K.Murray | -0.043 | K.Murray | -0.043 | J.McCarthy | 6.281 | 6.231 |
| 2026_07_KC_SEA | KC | SEA | P.Mahomes | 0.138 | P.Mahomes | 0.138 | P.Mahomes | 5.534 | 5.488 |
| 2026_07_KC_SEA | SEA | KC | S.Darnold | 0.057 | S.Darnold | 0.057 | S.Darnold | 6.363 | 6.309 |
| 2026_07_LA_LV | LA | LV | M.Stafford | 0.185 | M.Stafford | 0.185 | M.Stafford | 6.855 | 6.922 |
| 2026_07_LA_LV | LV | LA | K.Cousins | 0.075 | K.Cousins | 0.075 | G.Smith | 6.410 | 6.332 |
| 2026_07_MIA_NYJ | MIA | NYJ | M.Willis | -0.031 | M.Willis | -0.031 | T.Tagovailoa | 5.572 | 5.599 |
| 2026_07_MIA_NYJ | NYJ | MIA | G.Smith | -0.087 | G.Smith | -0.087 | J.Fields | 6.078 | 6.002 |
| 2026_07_NE_CHI | CHI | NE | C.Williams | 0.098 | C.Williams | 0.098 | C.Williams | 7.017 | 6.956 |
| 2026_07_NE_CHI | NE | CHI | D.Maye | 0.248 | D.Maye | 0.248 | D.Maye | 6.442 | 6.457 |
| 2026_07_NYG_HOU | HOU | NYG | C.Stroud | 0.108 | C.Stroud | 0.108 | C.Stroud | 6.764 | 6.795 |
| 2026_07_NYG_HOU | NYG | HOU | J.McCarthy | -0.166 | J.Winston | -0.016 | J.Dart | 5.144 | 5.121 |
| 2026_07_PIT_NO | NO | PIT | T.Shough | -0.025 | T.Shough | -0.025 | T.Shough | 5.828 | 5.833 |
| 2026_07_PIT_NO | PIT | NO | A.Rodgers | -0.022 | A.Rodgers | -0.022 | A.Rodgers | 6.521 | 6.450 |
| 2026_07_SF_ATL | ATL | SF | M.Penix | 0.031 | M.Penix | 0.031 | M.Penix | 6.664 | 6.683 |
| 2026_07_SF_ATL | SF | ATL | B.Purdy | 0.242 | B.Purdy | 0.242 | B.Purdy | 7.093 | 7.105 |
| 2026_07_TB_CAR | CAR | TB | B.Young | 0.028 | B.Young | 0.028 | B.Young | 6.661 | 6.781 |
| 2026_07_TB_CAR | TB | CAR | J.Daniels | -0.193 | J.Daniels | -0.193 | B.Mayfield | 5.990 | 5.840 |

"Offense" is the team throwing; the projection is the net yards per dropback that the defense allows. The two QB choices disagree for PHI (the schedule lists Hurts; T.McKee led PHI in passing last game) and NYG (McCarthy vs Winston). Those are exactly the cases where `c_qb_epa` depends on schedule information you wouldn't have a week ahead.

## 5. What was fixed, and how it was checked

- **Top-QB missingness leak.** In v1, the top QB's window row only existed in weeks he threw a pass or was the listed starter. When he sat out, all seven top-QB terms went missing, and missing rows averaged 5.91 NY/A against 6.49 otherwise. Now the window is computed for every week whether or not he plays. As a side effect, `c_qb_epa` now uses all of the QB's dropbacks, not only games where he was the listed starter.
- **Leak test** (`leak_test.py`, output in `leak_test_log.txt`). It scrambles every play outcome in 2021 Week 9, replaces that week's passer IDs and schedule starter IDs with fakes, rebuilds, and checks the 108 numeric features on that week's rows. The only ones that move are the `c_qb_*` schedule-starter group, which reads the starter field by design. The v1 code fails this test on all seven top-QB terms.
- **Same model path for history** (`week7/check_same.py`). The default rebuild and the Week 7 forecast build both reproduce all 123 columns of the frozen 2013–2025 table (`data/rows.parquet`) exactly. The forecast mode only adds rows; it does not change any historical value.
- **Earlier fixes kept:** injury team codes (OAK/SD/STL), injury rows edited after kickoff, `wind`/`temp` excluded.

## 6. Files and commands to reproduce Week 7

No RDS files are involved. Required inputs:

| file | what | where it comes from |
|---|---|---|
| `te_share/data/pbp/pbp_2012.parquet` … `pbp_2025.parquet` | play-by-play history (2012 is needed for 2013's prior season) | nflverse release `pbp/play_by_play_YYYY.parquet` |
| `te_share/data/aux/inj_2012.parquet` … `inj_2025.parquet` | injury reports; read by `build.py` but not used by any of the 14 or 17 terms | nflverse release `injuries/injuries_YYYY.parquet` |
| `week7/inputs_2026-10-23/pbp_2026.parquet` | 2026 play-by-play snapshot, sha256 `157a046f…8772` | downloaded 2026-10-23 20:46 ET |
| `week7/inputs_2026-10-23/games.csv` | schedule, lines, rest, starter IDs snapshot, sha256 `78517e64…d096` | nflverse `nfldata/data/games.csv`, same time |
| `week7/inputs_2026-10-23/inj_2026.parquet` | 2026 injury snapshot (not used by the terms), sha256 `0b64da77…f7fa3` | same time |

Outputs, all regenerated by the commands below:

| file | what |
|---|---|
| `week7/out/rows.parquet` | forecast-mode feature table: 2013–2025, 2026 Weeks 1–6, and the 28 Week 7 rows |
| `week7/week7_2026_terms_and_projections.csv` | every term value and both projections for Week 7 |
| `week7/week7_coefficients_14.csv`, `week7_coefficients_17.csv` | fitted coefficients |
| `week7/qb_variant_scores.csv` | the model scores in the table at the top |
| `pruned_terms_noqb.csv`, `pruned_terms.csv` | the 14 and 17 term lists |

From the `def_ypa_lift` folder:

```bash
week7/setup_inputs.sh
python3 build.py
S=week7/data FORECAST=2026,7 OUT=week7/out python3 build.py
python3 week7/check_same.py
python3 week7/project.py
```

The plain `python3 build.py` rebuilds the historical table `data/rows.parquet` from the 2012–2025 files. It is only needed for `check_same.py` and the evaluation scripts; the projection reads `week7/out/rows.parquet`.

`setup_inputs.sh` only creates symlinks and downloads nothing. The forecast build writes only to `week7/out`. The snapshot inputs are never rewritten.

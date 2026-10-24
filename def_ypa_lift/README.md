# Defensive YPA baseline: can pregame stats lift it?

Question: find pregame stats that, added to the 35-term `lm()` baseline in `defensive_ypa_old_baseline_lm.R`, raise the 2016-2025 mean season correlation a lot, raise 2023-2025 correlation, and lower MAE.

Short answer: no pregame stat does that. Removing the two leaky weather terms and cutting noisy ones helps a little; nothing added helps on the untouched recent seasons.

**For exact term definitions, source columns, the `c_qb_epa` question, and the Week 7 2026 values, see [SPEC.md](SPEC.md).**

> **Correction (2026-10-23).** The first version of this folder had a leak in the rebuilt top-QB terms. When a team's usual QB sat out a game, his stats were missing for that game, and the median fill told the model a backup was playing. It is fixed, a stronger leak test now catches it, and every number below is from the fixed table. The old "16-term" model was selected on the leaky table and has been replaced by the 14- and 17-term models below.

## How this was tested

- `build.py` rebuilds his 35 terms from nflverse play-by-play (2012-2025) and adds 63 pregame candidates. Target: net passing yards per dropback allowed (sacks count as attempts and their yards are subtracted).
- `evaluate.py` uses his protocol exactly: for season s and week w, refit OLS on all seasons before s plus weeks before w, fill missing cells with training medians, and score the raw prediction.
- `leak_test.py` scrambles every play outcome in one week and replaces that week's passer IDs and schedule starter IDs with fakes. It then rebuilds and checks the 108 features on that week's rows. The only ones that move are the `c_qb_*` group, which by design reads the schedule's listed starter (see SPEC.md section 2).
- Two Codex adversarial reviews (`review/report.txt`, `review2/report.txt`). The first found an injury-report team-code bug and post-game edits to injury rows. The second found documentation errors in SPEC.md and no remaining leak in the 14-term model. All are fixed.

The rebuild is close but not identical to his version: r 0.346 vs his 0.356 for 2016-2025, and 0.370 vs 0.401 for 2023-2025. His MAE (1.305) is lower than any version of this target produces (1.41), which suggests his target drops low-attempt games or trims outliers. Compare the changes below against the rebuilt baseline, not against his absolute numbers.

## 1. Two baseline terms are not pregame

`wind` and `temp` come from the nflverse schedule, which records game-day conditions after the fact. A forecast made during the week does not have them, and his own rule ("rain / wind does not work") excludes them.

| | r 2016-25 | r 2023-25 | MAE |
|---|---|---|---|
| his 35 terms (rebuilt) | 0.346 | 0.370 | 1.410 |
| same minus wind and temp (33 clean terms) | 0.340 | 0.367 | 1.414 |

## 2. Adding any one pregame stat moves almost nothing

Each of 63 candidates was added alone to the 33 clean terms (`results_single_adds.csv`). They cover defense and offense levels, opponent adjustment, sack, interception, CPOE, aDOT and YAC rates, two versions of the starting QB's record, injury report counts, referee crew, rest, byes, roof, surface, division games, and moneyline. Only 2 of 63 add more than +0.001, and both use the schedule's starter field:

| stat | change in r 2016-25 | change in r 2023-25 | change in MAE |
|---|---|---|---|
| `c_qb_epa`: listed starter's EPA per dropback | +0.0031 | +0.0025 | -0.0030 |
| `c_qb_ypa`: listed starter's NY/A | +0.0019 | +0.0030 | -0.0017 |
| `c_qb_epa_pre`: same, QB = last game's leading passer | -0.0002 | +0.0005 | +0.0001 |

The starter field is filled in with the actual starter after each game, so the historical values carry information a week-ahead forecast does not have. With the QB picked from pregame information, the gain disappears. The standard error of a 10-season mean correlation is about 0.013 anyway.

Forward selection on 2016-2022 picked 8 stats. They raised 2016-2025 a little (0.340 to 0.347) and lowered the untouched 2023-2025 (0.367 to 0.359): overfitting.

## 3. Removing terms helps more than adding them

`compact.py` drops terms one at a time, judged only on 2016-2022, with standard errors from resampling weeks within each season.

| model | r 2016-25 | r 2023-25 | MAE | change in r vs clean 33 (s.e.) |
|---|---|---|---|---|
| clean 33 | 0.340 | 0.367 | 1.414 | |
| **pruned to 14 terms, no QB-starter fields** | 0.351 | 0.369 | 1.408 | +0.011 (0.003) |
| pruned to 14 terms, pregame-QB block offered | 0.352 | 0.367 | 1.408 | +0.012 (0.004) |
| pruned to 17 terms, schedule-starter QB block offered | 0.355 | 0.369 | 1.405 | +0.015 (0.004) |

The pruned models gain mostly in the seasons used to pick the terms. On 2023-2025, which played no part in picking, every version sits within 0.002 of the clean 33. On 2026 Weeks 1-6 (186 offense-defense rows from 93 games, scored by `week7/project.py`), the 14-term and 17-term models tie at r 0.399 and 0.398.

## Why the ceiling is here

One dropback has a standard deviation of about 10 yards, and a team gets roughly 35 per game, so luck alone moves a single game's NY/A by about 1.7. The target's total SD is 1.93, which leaves a true-talent SD near 1.0 and caps the correlation any model can reach at about 0.5. The Vegas spread and total already carry most of the pregame information about the matchup.

## Recommendation

1. Drop `wind` and `temp`.
2. For a forecast that only uses pregame information, use the 14-term model (`pruned_terms_noqb.csv`). Leave out `c_qb_epa`: its gain depends on knowing the actual starter.
3. Don't expect another large lift from stats. The remaining room needs information the betting market doesn't already price.

## Files

`build.py` (features; `FORECAST=season,week` for forecast rows) · `evaluate.py` (baselines, single adds, forward selection) · `compact.py` (pruning, paired s.e.) · `leak_test.py` · `SPEC.md` (definitions, Week 7) · `week7/` (frozen 2026-10-23 inputs, setup script, Week 7 values and projections) · `review/`, `review2/` (Codex reviews)

From the `def_ypa_lift` folder: `python3 build.py`, then `evaluate.py`, then `compact.py`. These read `../te_share/data` by default, or `$S` if set.

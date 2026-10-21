# Defensive YPA baseline: can pregame stats lift it?

Question: find pregame stats that, added to the 35-term `lm()` baseline in `defensive_ypa_old_baseline_lm.R`, raise the 2016-2025 mean season correlation a lot, raise 2023-2025 correlation, and lower MAE.

Short answer: no pregame stat does that. The baseline is at the ceiling of what extra linear terms can add. The changes that do help are removing two leaky terms and cutting noisy ones, and even those gains are small.

## How this was tested

- `build.py` rebuilds his 35 terms from nflverse play-by-play (2012-2025) and adds 59 pregame candidates. Target: net passing yards per dropback allowed (sacks count as attempts and their yards are subtracted).
- `evaluate.py` uses his protocol exactly: for season s and week w, refit OLS on all seasons before s plus weeks before w, fill missing cells with training medians, and score the raw prediction.
- `leak_test.py` scrambles every play outcome in one week, rebuilds, and checks that none of the 104 features for that week change. None do.
- A Codex adversarial review found an injury-report team-code bug (OAK/SD/STL) and a few post-game edits to injury rows. Both are fixed, and the numbers below are from after the fixes.

The rebuild is close but not identical to his version: r 0.347 vs his 0.356 for 2016-2025, and 0.374 vs 0.401 for 2023-2025. His MAE (1.305) is lower than any version of this target produces (1.41). His target probably drops low-attempt games or trims outliers. Compare the changes below against the rebuilt baseline, not against his absolute numbers.

## 1. Two baseline terms are not pregame

`wind` and `temp` come from the nflverse schedule, which records game-day conditions after the fact. A forecast made during the week does not have them, and the rule he set ("rain / wind does not work") excludes them. Removing them:

| | r 2016-25 | r 2023-25 | MAE |
|---|---|---|---|
| his 35 terms (rebuilt) | 0.347 | 0.374 | 1.409 |
| same minus wind and temp (33 clean terms) | 0.341 | 0.371 | 1.413 |

So about 0.006 of his 2016-2025 correlation comes from data that is not available before the game. The clean 33 terms are the honest baseline.

## 2. Adding any one pregame stat moves almost nothing

Each of 59 candidates was added alone to the 33 clean terms. They cover defense and offense level stats (prior season, this season, and both pooled), opponent-adjusted yards allowed, sack, interception, CPOE, aDOT and YAC rates, the starting QB's own record, injury report counts (DB, pass rush, WR/TE, OL, QB), the referee crew's pass-penalty history, rest, byes, short weeks, roof, surface, division games, and moneyline. Only 2 of 59 add more than +0.001.

| stat | change in r 2016-25 | change in r 2023-25 | change in MAE |
|---|---|---|---|
| starting QB's EPA per dropback | +0.0019 | +0.0048 | -0.0015 |
| starting QB's sack rate | +0.0011 | +0.0004 | -0.0012 |
| starting QB is not the usual QB | +0.0010 | -0.0031 | -0.0009 |
| starting QB's NY/A | +0.0009 | +0.0060 | -0.0008 |
| every injury-report count | -0.0005 to 0.0000 | | |

The standard error of a 10-season mean correlation is about 0.013, so none of these clears the noise. The starting-QB stats are the only group pointing the right way in both windows. Caveat: they use the QB who actually started. That is usually announced before kickoff, but the data has no timestamp.

Forward selection on 2016-2022 picked 10 stats. They raised 2016-2025 a little (0.341 to 0.346) and lowered the untouched 2023-2025 (0.371 to 0.355). That is overfitting.

## 3. Removing terms helps more than adding them

`compact.py` drops terms one at a time, judged only on 2016-2022, and gives standard errors from resampling weeks within each season.

| model | r 2016-25 | r 2023-25 | MAE | change in r vs clean 33 (s.e.) |
|---|---|---|---|---|
| clean 33 | 0.341 | 0.371 | 1.413 | |
| pruned to 17 terms, no starter-QB fields | 0.349 | 0.370 | 1.409 | +0.007 (0.002) |
| pruned to 16 terms, with starting-QB EPA and new-QB flag | 0.354 | 0.374 | 1.406 | +0.013 (0.003) |
| 10 strong terms picked before seeing results* | 0.349 | 0.380 | 1.409 | +0.008 (0.004) |

\*Spread, total, defense and offense NY/A and EPA (prior season plus this season), the starting QB's EPA, NY/A and sack rate, and roof.

The pruned models gain mostly in the seasons used to pick the terms. On 2023-2025, which played no part in picking, they gain between -0.001 and +0.003. Pruning makes the model simpler and slightly lowers MAE (by about 0.004 to 0.007), but it does not lift 2023-2025.

## Why the ceiling is here

One dropback has a standard deviation of about 10 yards, and a team gets roughly 35 per game, so luck alone moves a single game's NY/A by about 1.7. The target's total SD is 1.93, which leaves a true-talent SD near 1.0 and caps the correlation any model can reach at about 0.5. The Vegas spread and total already carry most of the pregame information about the matchup. Situational splits (a quarter, a down, the score state) are mostly noise once those are in.

## Recommendation

1. Drop `wind` and `temp`. Report the clean baseline: about 0.341 / 0.371 in this rebuild.
2. If the starting QB counts as pregame for this project, add the starting QB's EPA per dropback and drop the 21 terms in `compact_log.txt`. That gives the best result here (0.354 / 0.374, MAE -0.007), but treat the 2016-2025 number as partly in-sample.
3. Don't expect another large lift from stats. The remaining room needs information the market does not already price, such as real-time injury news and closing lines.

## Files

`build.py` (features) → `data/rows.parquet` · `evaluate.py` (baselines, single adds, forward selection) → `results_baselines.csv`, `results_single_adds.csv`, `results_final.csv` · `compact.py` (pruning, paired s.e.) → `results_compact.csv`, `pruned_terms*.csv` · `leak_test.py` · `review/report.txt` (Codex review)

Run from the repo root with `python3 def_ypa_lift/build.py`, then `evaluate.py`, then `compact.py`. These read `te_share/data` by default or `$S` if set.

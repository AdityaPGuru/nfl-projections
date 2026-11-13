# Player-state dataset for defensive YPA

Built for the three jobs in `H1_LAMBDA4_YPA_HANDOFF.md`: (1) historical opponent-credit cleanup, (2) return restoration, (3) a timestamped pregame expected-starter panel that can say whether a QB transition helps, hurts, or does nothing. Target and rows match his: ordinary passing yards per attempt (non-penalty, `pass == 1`, `pass_attempt == 1`, no sacks), and the same **5,278 keyed rows** for 2016–2025.

**Result: the dataset works, but it does not qualify under his promotion rule.** The pregame version adds nothing. Even an oracle version that knows who actually started (same QB model) adds only +0.003 Pearson.

## Read first: the "clean" H1 λ=4 numbers are still leaked

His repaired control reports Pearson 0.511 (two-way replacement) and 0.544 (hybrid). On these exact 5,278 rows, the luck in a single game puts the highest Pearson a forecast can reasonably reach at **about 0.31–0.32** (`ceiling.py`, `out/ceiling_log.txt`):

- A game's YPA has an SD of 1.87. Luck alone (about 33 throws, each with its own spread) accounts for an SD of 1.77, which leaves an SD of 0.59 that isn't luck. That gives a ceiling of 0.318.
- Split each game's throws into two random halves: over 200 random splits they correlate at r 0.051, which scales to 0.311 for a full game (95% interval **0.197 to 0.388**, resampling games).

Both estimates count everything the throws in a game share as signal: the QB who actually played, the real weather, game script, and the order of plays. A pregame forecast knows less than that, so the estimates are generous. They are estimates, not a proof, but 0.511 and 0.544 sit far above the top of the interval.

`his_twoway_spec.py` rebuilds his component from its written spec (same season only, `week < target week`, one-week half-life on both sides, λ = 4 attempts):

| | Pearson 2016–25 | MAE | RMSE |
|---|---|---|---|
| H1 λ=4 component alone | **0.155** (he reports 0.511) | 1.445 | 1.845 |
| 14-term pregame base | 0.304 | 1.383 | 1.779 |
| base + H1 λ=4 component | 0.304 (+0.0002) | 1.382 | 1.779 |

Fixing `week < w` was necessary but not sufficient. His audits check the history used to fit each snapshot. They don't check how fitted values are joined back to target rows, whether the 44 base terms, the line panel, NFLELO, or the calibration step carry outcome information, or whether the target in the evaluation file is the game's own result. The test that settles it: scramble one week's play outcomes and passer IDs, rerun the whole R pipeline, and see whether that week's forecasts change (see `leak_test_state.py` here and `def_ypa_lift/leak_test.py`). **Until the control comes back inside the ceiling interval (below about 0.39), its promotion rule will reject every honest improvement**, because no real feature can close a gap of about 0.2.

## What was built

| file | job | contents |
|---|---|---|
| `out/ledger_completed_games.csv` | 1, 2 | per completed team-game: primary QB, previous game's QB, transition flag, and the missing share of the established receiving core (top 4 by targets), OL (top 5 by snaps), DB (top 5) and front (top 6), each core set from the team's previous 8 games |
| `out/expected_state_pregame.parquet` | 3 | per team-game: expected QB from pregame sources only, expected availability of each core, source and timestamp flags |
| `out/state_features.parquet`, `out/state_panel_5278.csv` | 1–3 | per target row: state-aware ratings and matchup expectation, keyed `game_id, posteam, defteam`, ready to merge into his evaluator |
| `out/oof_predictions_5278.csv` | | out-of-fold predictions for every arm |

**Ratings (jobs 1 and 2).** For each target week, using only games from earlier weeks (the previous season plus earlier weeks of this one), fit by alternating weighted ridge:

```text
passer-game YPA = league + qb[passer] + support[offense] + defense[defense] + b · (missing share of rec, OL, DB, front cores)
```

A defense that faced Baltimore's backup is judged against the backup's own rating, not normal Baltimore (job 1). The backup's games load onto his own effect rather than Baltimore's support level, so when Lamar returns the offense is rated at its established level (job 2). Depleted secondaries and fronts are handled the same way through the defensive core slopes, whose signs are learned, not assumed. Ridge penalties were chosen on 2013–2015 forward error only (λ_qb = 3000, λ_team = 1000 attempts; `out/tuning_*.csv`).

**Pregame expected QB (job 3).** Three rules were compared on 2013–2015 only: carry the previous starter forward (88.9% correct), replace him only when the final injury report lists him Out/Doubtful (90.6%, chosen), or use depth-chart QB1 (90.4%). The model then prices the matchup with the **expected** QB's prior opponent-adjusted rating against the established QB's (`st_qb_delta`), so a Brissett → Minshew change can come out positive, negative, or zero.

## Source certification

`sources_check.py` compares each source with the QB who actually started:

| source | seasons | timestamp | right on all games | right on games where the starter changed |
|---|---|---|---|---|
| carry forward last game's starter | all | always pregame | 0.871 | 0.000 |
| NFL weekly depth chart QB1 | 2012–2024 | none | 0.861 | 0.356 |
| ESPN depth chart, last snapshot before kickoff | 2025–2026 | yes (`dt`; median 10.8 h before kickoff) | 0.890 | 0.583 |

The weekly charts are not revised after the fact (a revised chart would be near 100% on starter changes), but they are stale and unstamped. Only 2025 onward is certifiably pregame. Injury rows are kept only if last modified before the team's kickoff; the 2025 injury file has no timestamp, so its 544 rows are flagged `inj_timestamp_verified = False`. No source here gives a certified expected-starter panel for 2016–2024. That would need archived snapshots (for example Wayback captures of team depth charts or final injury reports), and none are in nflverse.

## Evaluation (`evaluate_state.py`)

Same 5,278 rows, expanding week-ahead OLS (train on seasons before s plus weeks before w, training-median fill, raw prediction). Paired differences against control B, with standard errors from resampling weeks within each season:

| vs B (base + two-way) | ΔPearson (se) | ΔMAE (se) | ΔRMSE (se) |
|---|---|---|---|
| C pregame state, 2016–25 | −0.002 (0.003) | +0.001 (0.002) | +0.001 (0.002) |
| C pregame state, 2023–25 | −0.007 (0.004) | +0.005 (0.002) | +0.004 (0.002) |
| C pregame state, QB-transition games | −0.016 (0.011) | +0.011 (0.006) | +0.013 (0.006) |
| **D oracle (actual QB), 2016–25** | +0.003 (0.002) | −0.000 (0.001) | −0.002 (0.001) |
| D oracle, 2023–25 | +0.004 (0.004) | +0.002 (0.002) | −0.004 (0.002) |
| D oracle, QB-transition games | +0.010 (0.004) | −0.004 (0.002) | −0.005 (0.002) |

Levels: base 0.304, control B 0.306, C 0.303, oracle D 0.309 (`out/eval_state_metrics.csv`; also early-season, stable-QB, by season, pairwise ranking accuracy).

What this means:
- **Pregame player state fails the rule.** It loses on 2023–25 and does worst on QB-transition games, because the pregame expected QB is wrong in most of those games (36–58% right).
- **Even the oracle doesn't qualify.** In this QB model, knowing the true starter is worth about +0.003 overall and +0.010 on transition games, and MAE doesn't fall on 2023–25. The betting lines already price QB news, and transitions are 13% of games (692 of 5,278).
- So for the QB part, the value is in timing: a certified pre-kickoff expected-starter feed can at best recover the oracle row above. The oracle row is a bound for this QB specification only. Better receiver, OL or defensive-player models, interactions, or nonlinear fits are not bounded by it, but each would still have to clear the luck ceiling above.

**Reviews and leak test.** A Codex adversarial review (`review/report.txt`) confirmed the windows, 2013–15-only tuning, the 5,278 keys, the paired standard errors and the H1 rebuild. It found one injury row edited after kickoff on gameday (the filter compared dates, now kickoff timestamps) and two overstated claims (the ceiling was called a proof; the oracle was called a bound on all player data). All three are fixed above. Leak test (`leak_test_state.py`, `out/leak_test_log.txt`): scrambling 2021 Week 9's outcomes and faking that week's passers, receivers and snap-count players leaves all 14 pregame features and the expected/established QB IDs unchanged. The oracle columns, the outcome, and later weeks move, as they should.

## Run

From the repo root (needs `te_share/data` history and `def_ypa_lift/data/rows.parquet`):

```bash
python3 player_state/ceiling.py
python3 player_state/sources_check.py
python3 player_state/build_state.py
python3 player_state/evaluate_state.py
python3 player_state/his_twoway_spec.py
python3 player_state/leak_test_state.py
```

Raw inputs under `raw/`: nflverse depth charts 2012–2026 (downloaded 2026-11-12) and injury reports 2012–2025 plus the 2026-10-23 snapshot.

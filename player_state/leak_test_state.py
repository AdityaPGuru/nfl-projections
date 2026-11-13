"""Leak test for the player-state build: rebuild with 2021 week 9 play outcomes scrambled and that week's passers, receivers and
snap-count players replaced by fakes, then check every pregame feature on that week's rows. Pregame features must not move; the
ORACLE columns (actual QB) and the outcome must. Usage (repo root): python player_state/leak_test_state.py"""
import os, subprocess, sys, numpy as np, pandas as pd

H = os.path.dirname(os.path.abspath(__file__)); OUTL = os.path.join(H, "out_leaktest")
subprocess.run([sys.executable, "-W", "ignore", os.path.join(H, "build_state.py")], check=True, env={**os.environ, "LEAK_WEEK": "2021,9", "OUT": OUTL},
               stdout=subprocess.DEVNULL)
a = pd.read_parquet(f"{H}/out/state_features.parquet"); b = pd.read_parquet(f"{OUTL}/state_features.parquet")
k = ["game_id", "posteam", "defteam"]; a = a.set_index(k); b = b.set_index(k).loc[a.index]
sel = (a.season == 2021) & (a.week == 9)
PRE = ["tw_naive", "st_match", "st_qb_exp", "st_sup", "st_def", "st_core", "st_qb_est", "st_qb_delta", "st_qb_exp_logatt", "st_qb_changed",
       "exp_avail_rec", "exp_avail_ol", "exp_avail_db", "exp_avail_front"]
moved = [c for c in PRE if not np.allclose(a.loc[sel, c].astype(float), b.loc[sel, c].astype(float), equal_nan=True)]
ids = [c for c in ("qb_expected", "qb_established") if not a.loc[sel, c].astype(str).equals(b.loc[sel, c].astype(str))]
later = (a.season * 100 + a.week) > 202109
print(f"{len(PRE)} pregame features on {int(sel.sum())} rows of 2021 week 9; moved: {moved or 'none'}; expected/established QB ids moved: {ids or 'none'}")
print("outcome moved (should be yes):", not np.allclose(a.loc[sel, "ypa"], b.loc[sel, "ypa"]))
print("oracle QB columns moved (should be yes):", not np.allclose(a.loc[sel, "oracle_st_qb"], b.loc[sel, "oracle_st_qb"]))
print("later-week pregame features moved (should be yes):", not np.allclose(a.loc[later, "st_match"], b.loc[later, "st_match"]))

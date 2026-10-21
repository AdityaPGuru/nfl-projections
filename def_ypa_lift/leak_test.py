"""Leakage test: scramble every play outcome in one target week, rebuild all features, and confirm that no feature for that
week's rows changes. Anything that moves is using same-game information. Usage: python def_ypa_lift/leak_test.py"""
import os, numpy as np, pandas as pd
import build

HERE = os.path.dirname(os.path.abspath(__file__))
WEEKS = [(2021, 9)]
orig_load = build.load


def corrupted():
    p = orig_load(); rng = np.random.default_rng(0)
    m = np.zeros(len(p), bool)
    for s, w in WEEKS: m |= (p.season == s).values & (p.week == w).values
    for c in ["yards_gained", "air_yards", "yards_after_catch", "epa", "yac_epa", "success", "cpoe", "complete_pass", "sack", "qb_hit", "interception",
              "pass_touchdown", "first_down", "penalty_yards", "pass_oe", "xpass", "receiver_player_id"]:
        v = p.loc[m, c].to_numpy(); p.loc[m, c] = rng.permutation(v)
    return p


if __name__ == "__main__":
    clean = pd.read_parquet(f"{HERE}/data/rows.parquet")
    build.load = corrupted; build.OUT = os.path.join(HERE, "data_leaktest"); os.makedirs(build.OUT, exist_ok=True); build.main()
    bad = pd.read_parquet(f"{build.OUT}/rows.parquet")
    key = ["game_id", "posteam", "defteam"]
    a = clean.set_index(key); b = bad.set_index(key).loc[a.index]
    sel = np.zeros(len(a), bool)
    for s, w in WEEKS: sel |= (a.season.values == s) & (a.week.values == w)
    feats = [c for c in a.columns if c not in ("yds", "n", "ypa") and pd.api.types.is_numeric_dtype(a[c])]
    moved = [c for c in feats if not np.allclose(a.loc[sel, c].to_numpy(float), b.loc[sel, c].to_numpy(float), equal_nan=True)]
    later = (a.season.values * 100 + a.week.values) > 202109
    print(f"{len(feats)} features checked on {sel.sum()} rows in the scrambled weeks; moved: {moved or 'none'}")
    print("target moved (should be yes):", not np.allclose(a.loc[sel, 'ypa'], b.loc[sel, 'ypa']))
    print("later-week features moved (should be yes):", any(not np.allclose(a.loc[later, c].to_numpy(float), b.loc[later, c].to_numpy(float), equal_nan=True) for c in feats))

"""Rush YPC allowed: (1) leakage test, (2) how few training seasons are enough, (3) every stat in the model with its
week-ahead correlation, (4) which model is best on week-ahead correlation and on the tracked matchups."""
import os, numpy as np, pandas as pd, warnings
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "rush_model.py")).read().split("SETS = {")[0])
HIST = H; FULL = ["eb_def", "eb_off"] + vegas + dfe + ofe; rm = lambda a, c: float(np.sqrt(((np.asarray(a) - np.asarray(c)) ** 2).mean()))
_sd = sd_parts
def sd_parts(tr):                       # with a single training season the year-to-year carry-over cannot be estimated -> 0 (fold to league)
    s, t, r = _sd(tr); return s, t, (0.0 if not np.isfinite(r) else r)

# ---------------------------------------------------------------- 1. leakage
print("=" * 110 + "\n1. LEAKAGE TEST: corrupt every rushing outcome in 2024 week 10, rebuild all features, compare")
keepD, keepO = D.copy(), O.copy(); ALL_ok = ALL.copy(); m = (ALL.season == 2024) & (ALL.week == 10)
for c_ in ("rushing_yards", "epa", "first_down", "success"): ALL.loc[m, c_] = ALL.loc[m, c_] * 5 + 3
D2, _ = side(False); O2, _ = side(True); ALL = ALL_ok
for lab, a, c2, fl in (("defense side", keepD, D2, dfe), ("offense side", keepO.rename(columns=lambda x: x[2:] if x.startswith("o_") else x), O2, [f[2:] for f in ofe])):
    cols_ = [f for f in fl if f in a.columns and f in c2.columns] + ["s2d_num", "s2d_den", "prev_num", "prev_den"]
    key = ["game_id", "defteam"] if "defteam" in a.columns else ["game_id", "off_team"]; c2 = c2.rename(columns={"defteam": key[1]})
    for wk, want in ((10, "must be 0"), (11, "should be many")):
        j = a[(a.season == 2024) & (a.week == wk)][key + cols_].merge(c2[key + cols_], on=key, suffixes=("", "_c"))
        moved = sum(np.nanmax(np.abs(j[f].astype(float) - j[f + "_c"].astype(float))) > 1e-9 for f in cols_)
        print(f"   {lab}, week {wk}: {len(j)} rows, features that moved: {moved} of {len(cols_)}  ({want})")
print("   shrinkage constants and calibration are re-estimated inside each training set; Vegas lines are pre-game closing numbers (see note in write-up)")

# ---------------------------------------------------------------- fitters
def fitters(a, x, cols, w=None):
    w = np.ones(len(a)) if w is None else w; out = {}
    out["ridge"] = b.lin(RidgeCV(alphas=np.logspace(0, 5, 25))).fit(a[cols], a.y, ridgecv__sample_weight=w).predict(x[cols])
    g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=6, min_child_samples=120, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, reg_lambda=20, verbose=-1, random_state=0, n_jobs=4).fit(a[cols], a.y, sample_weight=w)
    out["lgbm"] = g.predict(x[cols]); out["blend"] = (out["ridge"] + out["lgbm"]) / 2
    return out, lambda z: (b.lin(RidgeCV(alphas=np.logspace(0, 5, 25))).fit(a[cols], a.y).predict(z[cols]) + g.predict(z[cols])) / 2

# ---------------------------------------------------------------- 2. minimum seasons
print("\n" + "=" * 110 + "\n2. HOW FEW SEASONS? Train on only the last N seasons before S, predict S. Same test seasons for every N: 2021-2025 (2,700 games)")
TESTS = [2021, 2022, 2023, 2024, 2025]; NS = [1, 2, 3, 4, 5, 6, 7, 8, 9]; rec = {n: [] for n in NS}; gap = {n: [] for n in NS}
for s in TESTS:
    for n in NS:
        tr, te = HIST[(HIST.season < s) & (HIST.season >= s - n)], HIST[HIST.season == s]; tre, _, _ = add_eb(tr, tr); tee, _, _ = add_eb(tr, te)
        out, insample = fitters(tre, tee, FULL); gap[n].append(pearsonr(tre.y, insample(tre))[0])
        rec[n].append(pd.DataFrame({"s": s, "w": te.week.values, "y": te.y.values, "p": out["blend"]}))
R = {n: pd.concat(v, ignore_index=True) for n, v in rec.items()}; bestn = min(NS, key=lambda n: rm(R[n].y, R[n].p)); blk = (R[1].s * 100 + R[1].w).values; ub = np.unique(blk); rng = np.random.default_rng(0)
rows = []
for n in NS:
    d = R[n]; e, e0 = (d.y - d.p) ** 2, (R[bestn].y - R[bestn].p) ** 2; bs = []
    for _ in range(600):
        idx = np.concatenate([np.where(blk == u)[0] for u in rng.choice(ub, len(ub))]); bs.append(np.sqrt(e.values[idx].mean()) - np.sqrt(e0.values[idx].mean()))
    wk = d.groupby(["s", "w"]).apply(lambda x: x.y.corr(x.p))
    rows.append({"training seasons": n, "training games": int(np.mean([len(HIST[(HIST.season < s) & (HIST.season >= s - n)]) for s in TESTS])), "in-sample r": np.mean(gap[n]), "forward r": d.y.corr(d.p),
                 "overfit gap (in - forward)": np.mean(gap[n]) - d.y.corr(d.p), "forward RMSE": rm(d.y, d.p), "RMSE vs best": rm(d.y, d.p) - rm(R[bestn].y, R[bestn].p), "noise (1 s.e.)": np.std(bs), "avg weekly r": wk.mean()})
T = pd.DataFrame(rows); print(T.round(4).to_string(index=False)); print(f"(RMSE of guessing the average: {rm(R[1].y, HIST.y.mean()):.4f})")
ok = T[T["RMSE vs best"] <= T["noise (1 s.e.)"].where(T["training seasons"] != bestn, 0) + 1e-12]; nmin = int(ok["training seasons"].min())
print(f"best N = {bestn}; smallest N whose forward RMSE is within one standard error of the best: N = {nmin} seasons (~{int(T.loc[T['training seasons'] == nmin, 'training games'].iloc[0]):,} games)")

# ---------------------------------------------------------------- 3. every stat
print("\n" + "=" * 110 + "\n3. EVERY STAT IN THE MODEL with its own week-ahead correlation (all seasons, 7,326 games). Columns = look-back window")
He, _, _ = add_eb(HIST, HIST); r = He[FULL].corrwith(He.y)
print("core:", ", ".join(f"{k} {r[k]:+.3f}" for k in ["eb_def", "eb_off"] + vegas + ["prev_rate", "prev_adj_rate", "prev_epa", "o_prev_rate", "o_prev_adj_rate", "o_prev_epa", "lg_prev", "week"] if k in r))
for lab, pre in (("DEFENSE (what this defense has allowed)", ""), ("OFFENSE (what the opposing run game has done)", "o_")):
    tab = {}
    for f in FULL:
        g_ = f[len(pre):] if pre and f.startswith(pre) else (f if not pre and not f.startswith("o_") else None)
        if g_ is None: continue
        for wdw in ("ewS", "ewL", "l4", "s2d"):
            if g_.startswith(wdw + "_"): tab.setdefault(g_[len(wdw) + 1:], {})[wdw] = r[f]
    t = pd.DataFrame(tab).T[["ewL", "ewS", "l4", "s2d"]]; t.columns = ["~14 games wtd", "~4 games wtd", "last 4", "season to date"]
    print(f"\n{lab}"); print(t.reindex(t.iloc[:, 0].abs().sort_values(ascending=False).index).round(3).to_string())
pd.DataFrame({"stat": r.index, "week_ahead_r": r.values}).sort_values("week_ahead_r", key=abs, ascending=False).to_csv(os.path.join(HERE, "results", "rush_model_all_stats.csv"), index=False)

# ---------------------------------------------------------------- 4. best model
print("\n" + "=" * 110 + "\n4. WHICH MODEL? Strict forward 2018-2025 (4,254 games, 141 weekly vectors); tracked matchups = the six 2025 games")
SETS = {"A defense shrunk level": ["eb_def"], "B + opposing run game": ["eb_def", "eb_off"], "C + Vegas/home/roof/wind": ["eb_def", "eb_off"] + vegas, "D + all 146 rushing stats": FULL}
acc = {}
for s in range(2018, 2026):
    tr, te = HIST[HIST.season < s], HIST[HIST.season == s]; tre, _, _ = add_eb(tr, tr); tee, _, _ = add_eb(tr, te)
    for sn, cols in SETS.items():
        out, _ = fitters(tre, tee, cols)
        if sn.startswith(("C", "D")):
            out["OLS (no penalty)"] = b.lin(LinearRegression()).fit(tre[cols], tre.y).predict(tee[cols]); out["KNN k=200"] = b.lin(KNeighborsRegressor(200)).fit(tre[cols], tre.y).predict(tee[cols])
        if sn.startswith("A"): out = {"shrunk level itself": tee.eb_def.values}
        for mn, p in out.items(): acc.setdefault((sn, mn), []).append(pd.DataFrame({"s": s, "w": te.week.values, "d": te.defteam.values, "y": te.y.values, "p": p}))
rows = []
for (sn, mn), v in acc.items():
    d = pd.concat(v); wk = d.groupby(["s", "w"]).apply(lambda x: x.y.corr(x.p)); t25 = d[d.s == 2025]; T6 = pd.concat([t25[(t25.d == a) & (t25.w == w_)] for a, w_ in dm.TRACKED])
    rows.append({"features": sn, "model": mn, "forward r": d.y.corr(d.p), "avg weekly r": wk.mean(), "weeks r>0": f"{(wk > 0).mean():.0%}", "avg per-game L2 (RMSE)": rm(d.y, d.p), "2025 r": t25.y.corr(t25.p), "tracked-6 r": T6.y.corr(T6.p)})
print(pd.DataFrame(rows).sort_values("avg weekly r", ascending=False).round(3).to_string(index=False))

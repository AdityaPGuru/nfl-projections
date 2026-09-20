"""Defensive pass YPA allowed only: (1) leakage test, (2) how few training seasons are enough, (3) every baseline stat with its
week-ahead correlation, (4) which model is best on week-ahead correlation and on the tracked matchups.
Strictly forward everywhere: train on seasons before S, predict S. Game-time weather is excluded (not known a week ahead)."""
import os, numpy as np, pandas as pd, warnings
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import bakeoff as b, def_model as dm
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
S = os.environ["S"]; HERE = os.path.dirname(os.path.abspath(__file__)); rm = lambda a, c: float(np.sqrt(((np.asarray(a) - np.asarray(c)) ** 2).mean()))
PLAYS = dm.plays()


def side(plays, swap):
    p = plays.copy()
    if swap: p[["posteam", "defteam"]] = p[["defteam", "posteam"]].values
    dm.plays = lambda: p
    g, feats, base = dm.build("pass"); g = g.sort_values(["defteam", "t"]).reset_index(drop=True)
    s = b.prior_sums(g, ["defteam", "season"], ["num", "den"], "s2d"); g["s2d_num"], g["s2d_den"] = s.s2d_num, s.s2d_den
    yr = g.groupby(["defteam", "season"], as_index=False)[["num", "den"]].sum().rename(columns={"num": "prev_num", "den": "prev_den"}); yr["season"] += 1
    return g.merge(yr, on=["defteam", "season"], how="left"), feats, base


def assemble(plays):
    D, dfe, base = side(plays, False); O, ofe, _ = side(plays, True)
    OC = list(dict.fromkeys(ofe + ["s2d_num", "s2d_den", "prev_num", "prev_den", "lg_prev", "num", "den", "y"]))
    O = O.rename(columns={c: "o_" + c for c in OC}).rename(columns={"defteam": "off_team"})
    M = D.merge(O[["game_id", "off_team"] + ["o_" + c for c in OC]], left_on=["game_id", "posteam"], right_on=["game_id", "off_team"], how="inner")
    ctx = pd.read_parquet(S + "/pbp_game_ctx.parquet"); M = M.merge(ctx, on="game_id", how="left"); sgn = np.where(M.posteam == M.home_team, 1, -1)
    M["off_spread"] = sgn * M.spread_line; M["off_implied"] = M.total_line / 2 + M.off_spread / 2; M["indoors"] = M.roof.isin(["dome", "closed"]).astype(int); M["off_home"] = (sgn == 1) * 1
    return M[M.season <= 2025].reset_index(drop=True), dfe, ["o_" + c for c in ofe if c not in ("lg_prev", "week")], base


def sd_parts(tr):
    u = tr.defteam + tr.season.astype(str); N = tr.den.groupby(u).sum(); m = tr.num.groupby(u).sum() / N
    sigma2 = (tr.den * (tr.y - u.map(m)) ** 2).sum() / max(len(tr) - len(N), 1); mu = tr.num.sum() / tr.den.sum()
    tau2 = max(np.average((m - mu) ** 2, weights=N) - sigma2 * np.average(1 / N, weights=N), 1e-9)
    sm = pd.DataFrame({"m": m, "n": N, "team": tr.defteam.groupby(u).first(), "season": tr.season.groupby(u).first()})
    nx = sm.assign(season=sm.season + 1).merge(sm, on=["team", "season"], suffixes=("_p", "")); rel = tau2 / (tau2 + sigma2 / sm.n.mean())
    rho = nx.m_p.corr(nx.m) / rel if len(nx) > 5 else np.nan       # one training season -> carry-over cannot be estimated -> fold to league
    return np.sqrt(sigma2), np.sqrt(tau2), float(np.clip(rho, 0, 1)) if np.isfinite(rho) else 0.0


def eb(x, tr):
    sg, ta, rho = sd_parts(tr); k = sg ** 2 / ta ** 2; lg = x.lg_prev
    return (x.s2d_num + k * (lg + rho * ((x.prev_num.fillna(0) + k * lg) / (x.prev_den.fillna(0) + k) - lg))) / (x.s2d_den + k), (sg, ta, k, rho)


offview = lambda x: pd.DataFrame({"defteam": x.off_team, "season": x.season, "num": x.o_num, "den": x.o_den, "y": x.o_y, "s2d_num": x.o_s2d_num, "s2d_den": x.o_s2d_den,
                                  "prev_num": x.o_prev_num, "prev_den": x.o_prev_den, "lg_prev": x.o_lg_prev}, index=x.index)


def add_eb(tr, x):
    x = x.copy(); x["eb_def"], pdv = eb(x, tr); x["eb_off"], pov = eb(offview(x), offview(tr)); return x, pdv, pov


def fitters(a, x, cols):
    rdg = b.lin(RidgeCV(alphas=np.logspace(0, 5, 25))).fit(a[cols], a.y)
    g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=6, min_child_samples=120, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, reg_lambda=20,
                          verbose=-1, random_state=0, n_jobs=4).fit(a[cols], a.y)
    f = lambda z: {"ridge": rdg.predict(z[cols]), "lgbm": g.predict(z[cols]), "blend": (rdg.predict(z[cols]) + g.predict(z[cols])) / 2}
    return f(x), f(a)["blend"]


HIST, dfe, ofe, BASE23 = assemble(PLAYS); vegas = ["off_spread", "off_implied", "total_line", "indoors", "off_home"]
newdef = [f for f in dfe if f not in ("lg_prev", "week")]
print(f"DEFENSIVE PASS YPA ALLOWED. {len(HIST):,} defense-games, 2012-2025 regular season; mean {HIST.y.mean():.2f}, sd {HIST.y.std():.2f}", flush=True)

# ---------------------------------------------------------------- 1. leakage
print("\n" + "=" * 110 + "\n1. LEAKAGE TEST: corrupt every passing outcome in 2024 week 10, rebuild every feature from scratch, compare")
bad = PLAYS.copy(); m = (bad.season == 2024) & (bad.week == 10)
for c_ in ("passing_yards", "air_yards", "yards_after_catch", "epa", "success", "first_down", "yards_gained"): bad.loc[m, c_] = bad.loc[m, c_] * 5 + 3
bad.loc[m, "complete_pass"] = 1.0; bad.loc[m, "cpoe"] = 40.0
H2, _, _, _ = assemble(bad); chk = newdef + ofe + ["s2d_num", "s2d_den", "prev_num", "prev_den", "o_s2d_num", "o_s2d_den", "o_prev_num", "o_prev_den", "lg_prev", "o_lg_prev"]
for wk, want in ((10, "must be 0"), (11, "should be many")):
    j = HIST[(HIST.season == 2024) & (HIST.week == wk)][["game_id", "defteam"] + chk].merge(H2[["game_id", "defteam"] + chk], on=["game_id", "defteam"], suffixes=("", "_c"))
    moved = sum(np.nanmax(np.abs(j[f].astype(float) - j[f + "_c"].astype(float))) > 1e-9 for f in chk); print(f"   week {wk}: {len(j)} defense-games, features that moved: {moved} of {len(chk)}  ({want})")
print("   your 23 baseline variables are rebuilt per (season, week) from plays strictly before that week (pbp_feats.py)")
print("   shrinkage constants, variable pruning and calibration are all re-estimated inside each training set; game-time wind/temperature are NOT used", flush=True)
dm.plays = lambda: PLAYS

# ---------------------------------------------------------------- 2. minimum seasons
FULL = ["eb_def", "eb_off"] + vegas + dfe + ofe + BASE23; SMALL = ["eb_def", "eb_off"] + vegas
TESTS = [2021, 2022, 2023, 2024, 2025]; NS = [1, 2, 3, 4, 5, 6, 7, 8, 9]; rng = np.random.default_rng(0)
print("\n" + "=" * 110 + f"\n2. HOW FEW SEASONS? Train on only the last N seasons before S, predict S. Same test seasons for every N: 2021-2025 ({int(HIST.season.isin(TESTS).sum()):,} games)")
for lab, cols in ((f"FULL MODEL ({len(FULL)} inputs: shrunk levels, Vegas, every defensive + offensive passing stat, your 23)", FULL), (f"SMALL MODEL ({len(SMALL)} inputs: defense + offense shrunk levels, Vegas, home, roof)", SMALL)):
    rec, ins = {n: [] for n in NS}, {n: [] for n in NS}
    for s in TESTS:
        for n in NS:
            tr, te = HIST[(HIST.season < s) & (HIST.season >= s - n)], HIST[HIST.season == s]; tre, _, _ = add_eb(tr, tr); tee, _, _ = add_eb(tr, te)
            out, fitted = fitters(tre, tee, cols); ins[n].append(pearsonr(tre.y, fitted)[0]); rec[n].append(pd.DataFrame({"s": s, "w": te.week.values, "y": te.y.values, "p": out["blend"]}))
    R = {n: pd.concat(v, ignore_index=True) for n, v in rec.items()}; bestn = min(NS, key=lambda n: rm(R[n].y, R[n].p)); blk = (R[1].s * 100 + R[1].w).values; ub = np.unique(blk); rows = []
    for n in NS:
        d = R[n]; e, e0 = (d.y - d.p) ** 2, (R[bestn].y - R[bestn].p) ** 2; bs = []
        for _ in range(600):
            idx = np.concatenate([np.where(blk == u)[0] for u in rng.choice(ub, len(ub))]); bs.append(np.sqrt(e.values[idx].mean()) - np.sqrt(e0.values[idx].mean()))
        wk = d.groupby(["s", "w"]).apply(lambda x: x.y.corr(x.p))
        rows.append({"training seasons": n, "training games": n * 544, "in-sample r": np.mean(ins[n]), "forward r": d.y.corr(d.p), "overfit gap": np.mean(ins[n]) - d.y.corr(d.p),
                     "forward RMSE": rm(d.y, d.p), "RMSE vs best": rm(d.y, d.p) - rm(R[bestn].y, R[bestn].p), "noise (1 s.e.)": np.std(bs), "avg weekly r": wk.mean()})
    T = pd.DataFrame(rows); print(f"\n{lab}"); print(T.round(4).to_string(index=False))
    ok = T[(T["training seasons"] == bestn) | (T["RMSE vs best"] <= T["noise (1 s.e.)"])]; nmin = int(ok["training seasons"].min())
    print(f"   best N = {bestn}. Smallest N whose forward RMSE is within one standard error of the best: N = {nmin} seasons (~{nmin * 544:,} games). RMSE of guessing the average: {rm(R[1].y, HIST[HIST.season < 2021].y.mean()):.4f}", flush=True)

# ---------------------------------------------------------------- 3. every baseline stat
print("\n" + "=" * 110 + "\n3. EVERY STAT IN THE BASELINE with its own week-ahead correlation to YPA allowed (all 7,326 games; n this large makes |r| < 0.03 indistinguishable from zero)")
He, pdv, pov = add_eb(HIST, HIST); fw = HIST[HIST.season >= 2018]
t = pd.DataFrame({"r all seasons": He[BASE23].corrwith(He.y), "r 2012-2018": He[He.season <= 2018][BASE23].corrwith(He[He.season <= 2018].y), "r 2019-2025": He[He.season >= 2019][BASE23].corrwith(He[He.season >= 2019].y)})
t["verdict"] = np.where((t.iloc[:, 1] > 0.03) & (t.iloc[:, 2] > 0.03), "holds in both halves", np.where(t.iloc[:, 0].abs() <= 0.03, "noise", "unstable")); print("\nYOUR 23 BASELINE DEFENSIVE VARIABLES")
print(t.sort_values("r all seasons", ascending=False).round(3).to_string())
print(f"\nSHRUNK LEVELS (k from standard deviations): defense luck sd/attempt {pdv[0]:.2f}, true sd {pdv[1]:.2f}, k = {pdv[2]:.0f} attempts, carry-over {pdv[3]:.2f} | offense luck sd {pov[0]:.2f}, true sd {pov[1]:.2f}, k = {pov[2]:.0f}, carry-over {pov[3]:.2f}")
core = He[["eb_def", "eb_off"] + vegas + ["prev_rate", "prev_adj_rate", "prev_epa", "lg_prev", "week"]].corrwith(He.y); print("core inputs:", ", ".join(f"{k} {v:+.3f}" for k, v in core.items()))
tab = {}
for f in newdef:
    for wdw in ("ewS", "ewL", "l4", "s2d"):
        if f.startswith(wdw + "_"): tab.setdefault(f[len(wdw) + 1:], {})[wdw] = He[f].corr(He.y)
g_ = pd.DataFrame(tab).T[["ewL", "ewS", "l4", "s2d"]]; g_.columns = ["~14 games wtd", "~4 games wtd", "last 4", "season to date"]
print("\nNEW DEFENSIVE STATS (what this defense has allowed), by look-back window"); print(g_.reindex(g_.iloc[:, 0].abs().sort_values(ascending=False).index).round(3).to_string(), flush=True)
pd.DataFrame({"stat": FULL, "week_ahead_r": [He[f].corr(He.y) for f in FULL]}).sort_values("week_ahead_r", key=abs, ascending=False).to_csv(os.path.join(HERE, "results", "pass_def_all_stats.csv"), index=False)

# ---------------------------------------------------------------- 4. best model
print("\n" + "=" * 110 + f"\n4. WHICH MODEL? Strict forward 2018-2025 ({len(fw):,} games); tracked matchups = the six 2025 games")
def pruned(tr): r = tr[BASE23].corrwith(tr.y); return list(r[r > 0.03].index)
SETS = {"A defense shrunk level": lambda tr: ["eb_def"], "B your 23 baseline defensive vars": lambda tr: BASE23, "C your baseline, pruned in training": pruned,
        "D all defensive stats + pruned baseline (defense-only)": lambda tr: ["eb_def"] + dfe + pruned(tr), "E defense + opposing offense shrunk levels": lambda tr: ["eb_def", "eb_off"],
        "F E + Vegas, home, roof": lambda tr: SMALL, "G F + all defensive stats + pruned baseline": lambda tr: SMALL + dfe + pruned(tr), "H G + all offensive stats (full)": lambda tr: SMALL + dfe + ofe + pruned(tr)}
acc = {}
for s in range(2018, 2026):
    tr, te = HIST[HIST.season < s], HIST[HIST.season == s]; tre, _, _ = add_eb(tr, tr); tee, _, _ = add_eb(tr, te)
    for sn, fn in SETS.items():
        cols = fn(tre)
        if sn.startswith("A"): out = {"shrunk level itself": tee.eb_def.values}
        else:
            out, _ = fitters(tre, tee, cols)
            if sn.startswith(("B", "F", "H")): out["OLS (no penalty)"] = b.lin(LinearRegression()).fit(tre[cols], tre.y).predict(tee[cols]); out["KNN k=200"] = b.lin(KNeighborsRegressor(200)).fit(tre[cols], tre.y).predict(tee[cols])
        for mn, p in out.items(): acc.setdefault((sn, mn), []).append(pd.DataFrame({"s": s, "w": te.week.values, "d": te.defteam.values, "o": te.posteam.values, "y": te.y.values, "p": p}))
rows, allp = [], {}
for (sn, mn), v in acc.items():
    d = pd.concat(v); allp[(sn, mn)] = d; wk = d.groupby(["s", "w"]).apply(lambda x: x.y.corr(x.p)); t25 = d[d.s == 2025]; T6 = pd.concat([t25[(t25.d == a) & (t25.w == w_)] for a, w_ in dm.TRACKED])
    rows.append({"features": sn, "model": mn, "forward r": d.y.corr(d.p), "avg weekly r": wk.mean(), "weeks r>0": f"{(wk > 0).mean():.0%}", "per-game L2 (RMSE)": rm(d.y, d.p), "2025 r": t25.y.corr(t25.p), "tracked-6 r": T6.y.corr(T6.p)})
RT = pd.DataFrame(rows).sort_values("avg weekly r", ascending=False); print(RT.round(3).to_string(index=False)); print(f"(per-game L2 of guessing the prior seasons' average: {rm(fw.y, 6.74):.3f}; {len(wk)} weekly vectors)")
RT.to_csv(os.path.join(HERE, "results", "pass_def_model_table.csv"), index=False)
top = RT.iloc[0]; d = allp[(top.features, top.model)]; t25 = d[d.s == 2025]; T6 = pd.concat([t25[(t25.d == a) & (t25.w == w_)] for a, w_ in dm.TRACKED])
print(f"\nbest on week-ahead correlation: {top.features} / {top.model}. Its tracked matchups:"); print(T6[["d", "w", "o", "p", "y"]].rename(columns={"d": "defense", "w": "week", "o": "vs", "p": "projected", "y": "actual"}).round(2).to_string(index=False))
print(f"chance check on the tracked-6 column: spread of r across random sets of six 2025 games for this same model = {np.std([t25.sample(6, random_state=i).pipe(lambda x: x.y.corr(x.p)) for i in range(2000)]):.2f} (so differences smaller than that between models mean nothing)")

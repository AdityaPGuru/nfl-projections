"""Pass YPA allowed (team matchup model): how few training seasons are enough, and which model is best week-ahead and on the tracked matchups."""
import os, numpy as np, pandas as pd, warnings
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import bakeoff as b, check45 as c45, def_model as dm
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250); S = os.environ["S"]; rm = lambda a, c: float(np.sqrt(((np.asarray(a) - np.asarray(c)) ** 2).mean()))
g = c45.team_games(); g = g.merge(pd.read_parquet(S + "/pbp_game_ctx.parquet"), on="game_id", how="left"); sgn = np.where(g.posteam == g.home_team, 1, -1)
g["off_spread"] = sgn * g.spread_line; g["off_implied"] = g.total_line / 2 + g.off_spread / 2; g["indoors"] = g.roof.isin(["dome", "closed"]).astype(int); g["wind_f"] = np.where(g.indoors == 1, 0, g.wind)
g = g[g.off_trail_ypa.notna() & (g.season <= 2025)].rename(columns={"ypa": "y"}).reset_index(drop=True); V = c45.OFF + c45.DEF
trail, veg = ["off_trail_ypa", "def_trail_ypa"], ["off_implied", "off_spread", "total_line", "indoors", "wind_f"]

def cols_for(sn, a):
    if sn.startswith("D"):
        r0 = a[V].corrwith(a.y); return trail + veg + list(r0[r0 > 0.03].index)     # pruning redone on the training set only
    return {"A": ["def_trail_ypa"], "B": trail, "C": trail + veg}[sn[0]]

def fitters(a, x, cols):
    out = {"ridge": b.lin(RidgeCV(alphas=np.logspace(-1, 4, 20))).fit(a[cols], a.y).predict(x[cols])}
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=6, min_child_samples=100, colsample_bytree=0.6, reg_lambda=10, verbose=-1, random_state=0, n_jobs=4).fit(a[cols], a.y)
    out["lgbm"] = m.predict(x[cols]); out["blend"] = (out["ridge"] + out["lgbm"]) / 2
    return out, lambda z: (b.lin(RidgeCV(alphas=np.logspace(-1, 4, 20))).fit(a[cols], a.y).predict(z[cols]) + m.predict(z[cols])) / 2

print("=" * 110 + "\nPASS YPA ALLOWED - HOW FEW SEASONS? Train on only the last N seasons before S, predict S; test seasons 2021-2025")
TESTS, NS = [2021, 2022, 2023, 2024, 2025], [1, 2, 3, 4, 5, 6, 7, 8, 9]; rec, gap = {n: [] for n in NS}, {n: [] for n in NS}
for s in TESTS:
    for n in NS:
        tr, te = g[(g.season < s) & (g.season >= s - n)], g[g.season == s]; out, ins = fitters(tr, te, cols_for("D", tr)); gap[n].append(pearsonr(tr.y, ins(tr))[0])
        rec[n].append(pd.DataFrame({"s": s, "w": te.week.values, "y": te.y.values, "p": out["blend"]}))
R = {n: pd.concat(v, ignore_index=True) for n, v in rec.items()}; bestn = min(NS, key=lambda n: rm(R[n].y, R[n].p)); blk = (R[1].s * 100 + R[1].w).values; ub = np.unique(blk); rng = np.random.default_rng(0); rows = []
for n in NS:
    d = R[n]; e, e0 = (d.y - d.p) ** 2, (R[bestn].y - R[bestn].p) ** 2; bs = []
    for _ in range(600):
        idx = np.concatenate([np.where(blk == u)[0] for u in rng.choice(ub, len(ub))]); bs.append(np.sqrt(e.values[idx].mean()) - np.sqrt(e0.values[idx].mean()))
    rows.append({"training seasons": n, "training games": int(np.mean([len(g[(g.season < s) & (g.season >= s - n)]) for s in TESTS])), "in-sample r": np.mean(gap[n]), "forward r": d.y.corr(d.p),
                 "overfit gap (in - forward)": np.mean(gap[n]) - d.y.corr(d.p), "forward RMSE": rm(d.y, d.p), "RMSE vs best": rm(d.y, d.p) - rm(R[bestn].y, R[bestn].p), "noise (1 s.e.)": np.std(bs),
                 "avg weekly r": d.groupby(["s", "w"]).apply(lambda x: x.y.corr(x.p)).mean()})
T = pd.DataFrame(rows); print(T.round(4).to_string(index=False)); print(f"(RMSE of guessing the average: {rm(R[1].y, g.y.mean()):.4f})")
ok = T[T["RMSE vs best"] <= T["noise (1 s.e.)"].where(T["training seasons"] != bestn, 0) + 1e-12]; print(f"best N = {bestn}; smallest N within one standard error of the best: N = {int(ok['training seasons'].min())} seasons")

print("\n" + "=" * 110 + "\nPASS YPA ALLOWED - WHICH MODEL? Strict forward 2018-2025; tracked matchups = the six 2025 games")
SETS = ["A defense trailing YPA allowed", "B + opposing offense trailing YPA", "C + Vegas/roof/wind", "D + pruned situational variables"]; acc = {}
for s in range(2018, 2026):
    tr, te = g[g.season < s], g[g.season == s]
    for sn in SETS:
        cols = cols_for(sn, tr); out, _ = fitters(tr, te, cols)
        if sn[0] in "CD": out["OLS (no penalty)"] = b.lin(LinearRegression()).fit(tr[cols], tr.y).predict(te[cols]); out["KNN k=200"] = b.lin(KNeighborsRegressor(200)).fit(tr[cols], tr.y).predict(te[cols])
        if sn[0] == "A": out = {"trailing number itself": te.def_trail_ypa.values}
        for mn, p in out.items(): acc.setdefault((sn, mn), []).append(pd.DataFrame({"s": s, "w": te.week.values, "d": te.defteam.values, "y": te.y.values, "p": p}))
rows = []
for (sn, mn), v in acc.items():
    d = pd.concat(v); wk = d.groupby(["s", "w"]).apply(lambda x: x.y.corr(x.p)); t25 = d[d.s == 2025]; T6 = pd.concat([t25[(t25.d == a) & (t25.w == w_)] for a, w_ in dm.TRACKED])
    rows.append({"features": sn, "model": mn, "forward r": d.y.corr(d.p), "avg weekly r": wk.mean(), "weeks r>0": f"{(wk > 0).mean():.0%}", "avg per-game L2 (RMSE)": rm(d.y, d.p), "2025 r": t25.y.corr(t25.p), "tracked-6 r": T6.y.corr(T6.p)})
print(pd.DataFrame(rows).sort_values("avg weekly r", ascending=False).round(3).to_string(index=False))

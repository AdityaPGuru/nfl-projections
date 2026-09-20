"""Recency half-life LEARNED from training data only (nested forward validation). No hand-picked weight.
For test season S: H*(S) = argmin over H of pooled forward RMSE on seasons 2015..S-1 (each predicted from seasons before it). Then apply H*(S) to S."""
import os, numpy as np, pandas as pd, warnings
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
import lightgbm as lgb
warnings.filterwarnings("ignore")
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rush_model.py")).read().split("SETS = {")[0])
HIST = H; cols = ["eb_def", "eb_off"] + vegas + dfe + ofe
HS = [np.inf, 40, 25, 16, 12, 9, 7, 5.5, 4.5, 3.5, 2.5, 1.5]; SEAS = list(range(2015, 2026))

def fit(a, x, w):
    r = b.lin(RidgeCV(alphas=np.logspace(0, 5, 25))).fit(a[cols], a.y, ridgecv__sample_weight=w).predict(x[cols])
    l = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=6, min_child_samples=120, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
                          reg_lambda=20, verbose=-1, random_state=0, n_jobs=4).fit(a[cols], a.y, sample_weight=w).predict(x[cols])
    return (r + l) / 2

wts = lambda seasons, ref, h: (lambda w: w * len(w) / w.sum())(np.ones(len(seasons)) if np.isinf(h) else 0.5 ** ((ref - 1 - seasons) / h))
SSE, N, PRED = {}, {}, {}
for s in SEAS:
    tr, te = HIST[HIST.season < s], HIST[HIST.season == s]; tre, _, _ = add_eb(tr, tr); tee, _, _ = add_eb(tr, te)
    for h in HS:
        p = fit(tre, tee, wts(tr.season.values, s, h)); PRED[(s, h)] = p; SSE[(s, h)] = float(((te.y.values - p) ** 2).sum()); N[s] = len(te)
    print("season", s, "done", flush=True)
curve = lambda upto: pd.Series({h: np.sqrt(sum(SSE[(s, h)] for s in SEAS if s < upto) / sum(N[s] for s in SEAS if s < upto)) for h in HS})
print(f"\nLEARNED HALF-LIFE, rush YPC matchup model. Candidates (seasons): {['equal' if np.isinf(h) else h for h in HS]}")
rows, out = [], []
for S_ in range(2018, 2026):
    c = curve(S_); hstar = c.idxmin(); te = HIST[HIST.season == S_]
    rows.append({"test season": S_, "trained/selected on": f"2012-{S_ - 1}", "learned half-life": "equal (no decay)" if np.isinf(hstar) else f"{hstar} seasons",
                 "RMSE learned": np.sqrt(SSE[(S_, hstar)] / N[S_]), "RMSE equal weights": np.sqrt(SSE[(S_, np.inf)] / N[S_]), "r learned": pearsonr(te.y, PRED[(S_, hstar)])[0], "r equal": pearsonr(te.y, PRED[(S_, np.inf)])[0]})
    out.append(pd.DataFrame({"season": S_, "week": te.week.values, "defteam": te.defteam.values, "posteam": te.posteam.values, "y": te.y.values, "p": PRED[(S_, hstar)], "p_eq": PRED[(S_, np.inf)]}))
print(pd.DataFrame(rows).round(4).to_string(index=False)); O = pd.concat(out)
rm = lambda a, c: float(np.sqrt(((a - c) ** 2).mean()))
print(f"\npooled 2018-2025 ({len(O):,} games): learned-weight r {O.y.corr(O.p):.4f}, RMSE {rm(O.y, O.p):.4f} | equal-weight r {O.y.corr(O.p_eq):.4f}, RMSE {rm(O.y, O.p_eq):.4f}")
c = curve(2026); print("\nforward-RMSE curve using everything through 2025 (this picks the 2026 value):"); print(pd.DataFrame({"half-life": ["equal" if np.isinf(h) else h for h in c.index], "RMSE": c.values.round(4)}).T.to_string(header=False))
hfin = c.idxmin(); print(f"==> half-life learned for 2026: {'equal weighting (no decay)' if np.isinf(hfin) else str(hfin) + ' seasons'}")
d = O[O.season == 2025]; wk = d.groupby("week").apply(lambda x: pd.Series({"r": x.y.corr(x.p), "rmse": rm(x.y, x.p), "soft5": x.nlargest(5, "p").y.mean(), "tough5": x.nsmallest(5, "p").y.mean()}))
print(f"\nVECTOR EVALUATION 2025 with the learned weight: season r {d.y.corr(d.p):.3f}, RMSE {rm(d.y, d.p):.3f} (guess-average {d.y.std(ddof=0):.3f})"); print(wk.round(2).T.to_string())
aw = O.groupby(["season", "week"]).apply(lambda x: x.y.corr(x.p)); print(f"avg weekly r 2025 {wk.r.mean():.3f} ({(wk.r > 0).sum()}/18 positive); all 8 seasons: {len(aw)} weekly vectors, avg r {aw.mean():.3f}, positive {(aw > 0).mean():.0%}")
live = M[~M.future].reset_index(drop=True); lt, _, _ = add_eb(live, live); F, _, _ = add_eb(live, M[M.future]); F["proj"] = fit(lt, F, wts(live.season.values, 2027, hfin))
sl, ic = np.polyfit(O.p, O.y, 1); F["proj"] = ic + sl * F.proj; print(f"\ncalibration from the forward predictions: actual = {ic:.3f} + {sl:.3f}*pred")
F.sort_values("proj")[["defteam", "posteam", "proj"]].to_csv(S + "/week2_rush_ypc_final.csv", index=False)
old = pd.read_csv(S + "/week2_rush_ypc.csv"); j = F.merge(old, left_on="defteam", right_on="Defense"); print(f"week 2 projections vs the earlier table: correlation {j.proj.corr(j['Proj YPC allowed']):.3f}, largest change {(j.proj - j['Proj YPC allowed']).abs().max():.2f} YPC")

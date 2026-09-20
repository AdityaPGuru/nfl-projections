"""How much of the 'fold to league average' is shrinkage I chose vs. what the data supports? Defense-only pass YPA allowed."""
import os, numpy as np, pandas as pd, warnings
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, RidgeCV
import lightgbm as lgb
import bakeoff as b, fantasy_bet as fb, def_model as dm
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
g, new, DEF = dm.build("pass"); h = g[g.season <= 2025].reset_index(drop=True); fold = fb.scattered_folds(h)
r0 = h[DEF].corrwith(h.y); cols = new + list(r0[r0 > 0.03].index)
models = {
 "CURRENT: ridge + regularized LightGBM blend": lambda a, x: dm.fitpred(a, x, cols)["blend"],
 "No penalty: plain least squares, all 122 stats": lambda a, x: b.lin(LinearRegression()).fit(a[cols], a.y).predict(x[cols]),
 "No penalty: LightGBM, deep trees, 1500 rounds, no regularization": lambda a, x: lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.05, num_leaves=63, min_child_samples=5, reg_lambda=0, verbose=-1, random_state=0).fit(a[cols], a.y).predict(x[cols]),
 "Zero shrinkage: defense's raw YPA allowed, last 4 games": lambda a, x: x.l4_rate.fillna(a.y.mean()).values,
 "Zero shrinkage: raw YPA allowed, season to date": lambda a, x: x.s2d_rate.fillna(x.prev_rate).fillna(a.y.mean()).values,
 "Zero shrinkage: raw YPA allowed, last ~14 games weighted": lambda a, x: x.ewL_rate.values}
rows, keep = [], {}
for lab, f in models.items():
    p = np.zeros(len(h))
    for k in range(fb.K): p[fold == k] = f(h[fold != k], h[fold == k])
    keep[lab] = p; sl, ic = np.polyfit(p, h.y, 1)
    rows.append({"projection method": lab, "sd of projections": p.std(), "lowest": p.min(), "highest": p.max(), "r": pearsonr(h.y, p)[0], "RMSE": dm.rmse(h.y, p),
                 "slope (1.0 = honest spread)": sl})
p = keep["CURRENT: ridge + regularized LightGBM blend"]
for m in (2, 4, h.y.std() / p.std()):
    s = p.mean() + (p - p.mean()) * m; sl, _ = np.polyfit(s, h.y, 1)
    rows.append({"projection method": f"CURRENT stretched x{m:.1f}", "sd of projections": s.std(), "lowest": s.min(), "highest": s.max(), "r": pearsonr(h.y, s)[0], "RMSE": dm.rmse(h.y, s), "slope (1.0 = honest spread)": sl})
print(f"DEFENSE-ONLY PASS YPA ALLOWED, all {len(h):,} defense-games 2012-2025, out-of-fold. Real games: sd {h.y.std():.2f}. RMSE of guessing the average: {h.y.std():.3f}\n")
print(pd.DataFrame(rows).round(3).to_string(index=False))
raw = keep["Zero shrinkage: defense's raw YPA allowed, last 4 games"]
d = pd.DataFrame({"raw": raw, "cur": p, "y": h.y}); d["bin"] = pd.cut(d.raw, [0, 5, 5.75, 6.5, 7.25, 8, 9, 20])
print("\nWHAT ACTUALLY HAPPENS NEXT WEEK, grouped by the unshrunk last-4-games number:")
print(d.groupby("bin", observed=True).agg(games=("y", "size"), unshrunk_says=("raw", "mean"), current_model_says=("cur", "mean"), actually_happened=("y", "mean")).round(2).to_string())
yr = h.groupby(["defteam", "season"]).apply(lambda x: x.num.sum() / x.den.sum())
print(f"\nFull-season YPA allowed by defense (the most 'true' a defense number gets): sd across teams {yr.std():.2f}, best {yr.min():.2f}, worst {yr.max():.2f}")
h1 = h[h.week <= 9].groupby(["defteam", "season"]).apply(lambda x: x.num.sum() / x.den.sum()); h2 = h[h.week >= 10].groupby(["defteam", "season"]).apply(lambda x: x.num.sum() / x.den.sum())
j = pd.concat([h1, h2], axis=1).dropna(); sl, ic = np.polyfit(j[0], j[1], 1)
print(f"First-half YPA allowed vs SAME defense's second half: r = {j[0].corr(j[1]):.2f}; a defense 1.00 above average in weeks 1-9 is only {sl:.2f} above average in weeks 10-18")

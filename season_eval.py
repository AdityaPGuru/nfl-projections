"""Full-season check of the team-level defensive YPA model on 2025, and a direct test of 'is it just guessing the average?'"""
import os, numpy as np, pandas as pd, warnings
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
import lightgbm as lgb
import bakeoff as b, fantasy_bet as fb, check45 as c45
warnings.filterwarnings("ignore"); S = os.environ["S"]; rmse = c45.rmse
g = c45.team_games(); ctx = pd.read_parquet(S + "/pbp_game_ctx.parquet")
g = g.merge(ctx, on="game_id", how="left"); sgn = np.where(g.posteam == g.home_team, 1, -1)
g["off_spread"] = sgn * g.spread_line; g["off_implied"] = g.total_line / 2 + g.off_spread / 2
g["indoors"] = g.roof.isin(["dome", "closed"]).astype(int); g["wind_f"] = np.where(g.indoors == 1, 0, g.wind)
core = ["off_trail_ypa", "def_trail_ypa", "off_implied", "off_spread", "total_line", "indoors", "wind_f"]; V = c45.OFF + c45.DEF
g = g[g.off_trail_ypa.notna() & (g.season <= 2025)].reset_index(drop=True)
def fit(a, x):
    r0 = a[V].corrwith(a.ypa); cols = core + list(r0[r0 > 0.03].index)
    r = b.lin(RidgeCV(alphas=np.logspace(-1, 4, 20))).fit(a[cols], a.ypa).predict(x[cols])
    l = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=6, min_child_samples=100, colsample_bytree=0.6, reg_lambda=10, verbose=-1, random_state=0).fit(a[cols], a.ypa).predict(x[cols])
    return (r + l) / 2
fold = fb.scattered_folds(g); oof = np.zeros(len(g))
for k in range(fb.K): oof[fold == k] = fit(g[fold != k], g[fold == k])
m25 = (g.season == 2025).values; tr = g[g.season < 2025]; te = g[m25].copy()
raw = fit(tr, te)
f2 = fb.scattered_folds(tr.reset_index(drop=True)); o2 = np.zeros(len(tr)); trr = tr.reset_index(drop=True)
for k in range(fb.K): o2[f2 == k] = fit(trr[f2 != k], trr[f2 == k])
sl, ic = np.polyfit(o2, trr.ypa, 1); te["fwd"] = ic + sl * raw; te["oof"] = oof[m25]
te["avg"] = tr[tr.season == 2024].ypa.mean(); te["trail"] = (te.off_trail_ypa + te.def_trail_ypa) / 2
y = te.ypa.values
print(f"2025 REGULAR SEASON, every game: {len(te)} team-games ({te.game_id.nunique()} games), actual YPA allowed mean {y.mean():.2f}, sd {y.std():.2f}")
print(f"forward test = trained on {len(tr):,} team-games from 2012-2024 only, calibrated on those seasons only (actual = {ic:.2f} + {sl:.2f}*pred)\n")
rows = []
for lab, c in (("Model, strict forward (no 2025 in training)", "fwd"), ("Model, scattered folds", "oof"), ("Plain trailing (off YPA + def YPA allowed)/2", "trail"), ("Guess the 2024 league average for everyone", "avg")):
    p = te[c].values; wk = te.groupby("week").apply(lambda d: pd.Series({"rmse": rmse(d.ypa, d[c]), "r": d.ypa.corr(d[c]) if d[c].std() > 0 else np.nan}))
    rows.append({"method": lab, "season r": pearsonr(y, p)[0] if p.std() > 0 else np.nan, "season Spearman": spearmanr(y, p)[0] if p.std() > 0 else np.nan, "season RMSE": rmse(y, p),
                 "avg weekly RMSE": wk.rmse.mean(), "avg weekly r": wk.r.mean(), "weeks r>0": f"{int((wk.r > 0).sum())}/{len(wk)}" if p.std() > 0 else "", "sd of predictions": p.std()})
print(pd.DataFrame(rows).round(3).to_string(index=False))
p = te.fwd.values
print(f"\nIS IT JUST THE AVERAGE?  predictions range {p.min():.2f} to {p.max():.2f} (sd {p.std():.2f}); actuals range {y.min():.2f} to {y.max():.2f} (sd {y.std():.2f})")
te["q"] = pd.qcut(te.fwd, 5, labels=["lowest 20% of projections", "20-40%", "40-60%", "60-80%", "highest 20% of projections"])
print(te.groupby("q", observed=True).agg(games=("ypa", "size"), avg_projection=("fwd", "mean"), avg_actual=("ypa", "mean"), share_over_league_avg=("ypa", lambda s: (s > y.mean()).mean())).round(2).to_string())
print("\nWHAT HAPPENS IF WE FORCE THE PROJECTIONS TO SPREAD OUT LIKE REAL GAMES (same ranking, so same correlation)")
out = []
for mult in (1.0, 1.5, 2.0, y.std() / p.std()):
    s = p.mean() + (p - p.mean()) * mult
    out.append({"spread multiplier": round(mult, 2), "sd of projections": s.std(), "min": s.min(), "max": s.max(), "r": pearsonr(y, s)[0], "RMSE": rmse(y, s)})
print(pd.DataFrame(out).round(3).to_string(index=False))
big = te[(te.ypa > 9.5) | (te.ypa < 4.5)]
print(f"\nextreme games (actual >9.5 or <4.5): {len(big)} of {len(te)}. avg projection for the >9.5 games {te[te.ypa > 9.5].fwd.mean():.2f}, for the <4.5 games {te[te.ypa < 4.5].fwd.mean():.2f}, everyone else {te[(te.ypa <= 9.5) & (te.ypa >= 4.5)].fwd.mean():.2f}")
te[["season", "week", "defteam", "posteam", "fwd", "ypa"]].rename(columns={"fwd": "proj_def_ypa", "ypa": "actual"}).to_csv(S + "/season2025_def_ypa.csv", index=False)

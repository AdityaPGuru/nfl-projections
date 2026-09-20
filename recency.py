"""Recency-weighted training for the rushing YPC matchup model. Weight = 0.5 ** (seasons_ago / H). H is picked by strictly
forward validation on a coarse grid, and only adopted if it beats equal weighting by more than the noise."""
import os, numpy as np, pandas as pd, warnings
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
import lightgbm as lgb
warnings.filterwarnings("ignore")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rush_model.py")).read().split("SETS = {")[0]; exec(src)          # builds M, H (history), add_eb, feature lists
HIST = H; cols = ["eb_def", "eb_off"] + vegas + dfe + ofe; GRID = [None, 12, 8, 5, 3, 2]; TESTS = list(range(2018, 2026))

def fit(a, x, w):
    r = b.lin(RidgeCV(alphas=np.logspace(0, 5, 25))).fit(a[cols], a.y, ridgecv__sample_weight=w).predict(x[cols])
    l = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=6, min_child_samples=120, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
                          reg_lambda=20, verbose=-1, random_state=0).fit(a[cols], a.y, sample_weight=w).predict(x[cols])
    return (r + l) / 2

pred = {h: [] for h in GRID}
for s in TESTS:
    tr, te = HIST[HIST.season < s], HIST[HIST.season == s]; tre, _, _ = add_eb(tr, tr); tee, _, _ = add_eb(tr, te)
    for h in GRID:
        w = np.ones(len(tr)) if h is None else 0.5 ** ((s - 1 - tr.season.values) / h)
        pred[h].append(pd.DataFrame({"season": s, "week": te.week.values, "defteam": te.defteam.values, "posteam": te.posteam.values, "y": te.y.values, "p": fit(tre, tee, w * len(w) / w.sum())}))
P = {h: pd.concat(v, ignore_index=True) for h, v in pred.items()}; base = P[None]; rows = []
rng = np.random.default_rng(0); blocks = base.season * 100 + base.week; ub = blocks.unique()
for h in GRID:
    d = P[h]; e2, b2 = (d.y - d.p) ** 2, (base.y - base.p) ** 2
    boot = []
    for _ in range(1000):
        pick = rng.choice(ub, len(ub)); idx = np.concatenate([np.where(blocks.values == u)[0] for u in pick]); boot.append(np.sqrt(e2.values[idx].mean()) - np.sqrt(b2.values[idx].mean()))
    per = d.assign(e=e2, b=b2).groupby("season").apply(lambda x: np.sqrt(x.e.mean()) < np.sqrt(x.b.mean()))
    eff = "all seasons equal" if h is None else f"{h} seasons"
    rows.append({"half-life": eff, "share of weight on last 3 seasons (2025 model)": (1.0 if h is None else 1) and round(float((0.5 ** (np.arange(13) / h))[:3].sum() / (0.5 ** (np.arange(13) / h)).sum()) if h else 3 / 13, 2),
                 "forward r": pearsonr(d.y, d.p)[0], "forward RMSE": np.sqrt(e2.mean()), "RMSE vs equal": np.sqrt(e2.mean()) - np.sqrt(b2.mean()), "noise (1 s.e.)": np.std(boot),
                 "seasons it beat equal": "-" if h is None else f"{int(per.sum())}/8"})
print(f"\nRECENCY WEIGHTING, RUSH YPC MATCHUP MODEL. Strictly forward: train on all seasons before S, test on S, for S = 2018..2025 ({len(base):,} test games)")
print(pd.DataFrame(rows).round(4).to_string(index=False)); print(f"(RMSE of guessing each season's prior mean: {np.sqrt(((base.y - HIST.y.mean()) ** 2).mean()):.4f})")
t = pd.DataFrame(rows).iloc[1:]; best = t.loc[t["forward RMSE"].idxmin()]
adopt = best["RMSE vs equal"] < -best["noise (1 s.e.)"]
print(f"\nbest half-life on the grid: {best['half-life']} (RMSE change {best['RMSE vs equal']:+.4f}, noise {best['noise (1 s.e.)']:.4f}) -> {'ADOPT' if adopt else 'NOT distinguishable from equal weighting; keeping a mild, safe setting'}")
lg = HIST.groupby("season").y.mean(); print("league designed-run YPC by season:", " ".join(f"{s}:{v:.2f}" for s, v in lg.items()))
hl = [h for h in GRID if h and f"{h} seasons" == best["half-life"]][0] if adopt else 8
d = P[hl][P[hl].season == 2025]; e = P[None][P[None].season == 2025]
print(f"\nVECTOR EVALUATION, 2025, half-life {hl} seasons (equal-weight model in brackets): season r {d.y.corr(d.p):.3f} [{e.y.corr(e.p):.3f}], RMSE {np.sqrt(((d.y - d.p) ** 2).mean()):.3f} [{np.sqrt(((e.y - e.p) ** 2).mean()):.3f}], guess-average {d.y.std(ddof=0):.3f}")
wk = d.groupby("week").apply(lambda x: pd.Series({"r": x.y.corr(x.p), "rmse": np.sqrt(((x.y - x.p) ** 2).mean()), "soft5": x.nlargest(5, "p").y.mean(), "tough5": x.nsmallest(5, "p").y.mean()}))
print(wk.round(2).T.to_string()); print(f"avg weekly r {wk.r.mean():.3f}, weeks r>0 {(wk.r > 0).sum()}/18; five softest projected defenses allowed {wk.soft5.mean():.2f}, five toughest {wk.tough5.mean():.2f}, soft > tough in {(wk.soft5 > wk.tough5).sum()}/18 weeks")
allw = P[hl].groupby(["season", "week"]).apply(lambda x: x.y.corr(x.p)); print(f"all 8 forward seasons: {len(allw)} weekly vectors, avg r {allw.mean():.3f}, median {allw.median():.3f}, positive in {(allw > 0).mean():.0%}")
P[hl].to_csv(S + "/rush_forward_recency.csv", index=False)
live = M[~M.future].reset_index(drop=True); lt, _, _ = add_eb(live, live); F, _, _ = add_eb(live, M[M.future]); w = 0.5 ** ((2026 - live.season.values) / hl)
F["proj"] = fit(lt, F, w * len(w) / w.sum()); old = pd.read_csv(S + "/week2_rush_ypc.csv").rename(columns={"Proj YPC allowed": "equal_wt"})
o = F[["defteam", "posteam", "proj"]].merge(old[["Defense", "equal_wt"]], left_on="defteam", right_on="Defense").sort_values("proj")
print(f"\n2026 WEEK 2 projected rush YPC allowed, recency-weighted (half-life {hl}); biggest moves vs equal weighting:")
o["move"] = o.proj - o.equal_wt; print(o.reindex(o.move.abs().sort_values(ascending=False).index).head(6)[["defteam", "posteam", "proj", "equal_wt", "move"]].round(2).to_string(index=False))
print(f"correlation between the two sets of week 2 projections: {o.proj.corr(o.equal_wt):.3f}"); o[["defteam", "posteam", "proj"]].to_csv(S + "/week2_rush_ypc_recency.csv", index=False)

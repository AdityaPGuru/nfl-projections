"""Week-ahead 'vector' check in the format of the user's R reference: train < 2025, predict every 2025 REG week."""
import numpy as np, pandas as pd
from scipy.stats import pearsonr
import bakeoff as b
w = b.load(); q, feats = b.build_qb(w)
tr, te = q[q.season < 2025], q[(q.season == 2025) & (q.season_type == "REG")]
# calibrator: leave-season-out OOF on the training seasons only, then actual ~ a + b*pred
name = "Blend: EB + Ridge + LightGBM (mean)"
oof = pd.Series(np.nan, index=tr.index)
for s in sorted(tr.season.unique()):
    p, _ = b.fit_predict(tr[tr.season != s], tr[tr.season == s], feats, True); oof[tr.season == s] = p[name]
sl, ic = np.polyfit(oof, tr.y, 1)
p, k = b.fit_predict(tr, te, feats, True)
te = te.assign(proj=ic + sl * p[name], eb=p["Empirical Bayes (shrunk career rate)"])
print(f"calibrator actual = {ic:.2f} + {sl:.2f}*pred | 2025 REG: {len(te)} QB-games | overall r={pearsonr(te.y, te.proj)[0]:.3f} rmse={np.sqrt(((te.y-te.proj)**2).mean()):.2f} slope={np.polyfit(te.proj, te.y, 1)[0]:.2f}")
wk = te.groupby("week").apply(lambda d: pd.Series({"games": len(d), "r": pearsonr(d.y, d.proj)[0], "mean_proj": d.proj.mean(), "mean_actual": d.y.mean()}))
print(wk.round(2).T.to_string())
print(f"weekly r: mean {wk.r.mean():.3f}, sd {wk.r.std():.3f}, weeks with r>0: {(wk.r>0).sum()}/{len(wk)}")
t = te[te.week == 2].sort_values("proj", ascending=False)
print("\n2025 WEEK 2, week-ahead projections"); print(t[["player_name", "team", "opp", "proj", "y", "n"]].rename(columns={"y": "actual_ypa", "n": "att", "proj": "proj_ypa"}).round(2).to_string(index=False))

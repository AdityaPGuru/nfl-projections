"""Vector evaluation with L2 distance. For each 2025 week: projection vector p, actual vector y, L2 = ||p - y|| = sqrt(sum (p_i - y_i)^2)."""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "recency_learned.py")).read().split("SSE, N, PRED")[0]; exec(src)
tr, te = HIST[HIST.season < 2025], HIST[HIST.season == 2025]; tre, _, _ = add_eb(tr, tr); tee, _, _ = add_eb(tr, te)
d = te[["week", "defteam", "posteam", "y"]].assign(p=fit(tre, tee, wts(tr.season.values, 2025, 12.0)), avg=tr[tr.season == 2024].y.mean(), eb=tee.eb_def.values)
L2 = lambda a, c: float(np.sqrt(((a - c) ** 2).sum()))
wk = d.groupby("week").apply(lambda x: pd.Series({"games": len(x), "L2 model": L2(x.p, x.y), "L2 guess-average": L2(x.avg, x.y), "L2 defense-only": L2(x.eb, x.y),
                                                  "RMSE model (L2/sqrt n)": L2(x.p, x.y) / np.sqrt(len(x)), "r": x.p.corr(x.y)}))
wk["model beats average"] = wk["L2 model"] < wk["L2 guess-average"]
print("RUSH YPC ALLOWED, 2025, strict forward (trained 2012-2024, learned 12-season half-life)\n"); print(wk.round(2).to_string())
print(f"\nSEASON VECTOR ({len(d)} games): L2 model {L2(d.p, d.y):.2f} | L2 guess-average {L2(d.avg, d.y):.2f} | L2 defense-only shrunk level {L2(d.eb, d.y):.2f}")
print(f"average weekly L2: model {wk['L2 model'].mean():.2f}, guess-average {wk['L2 guess-average'].mean():.2f}, defense-only {wk['L2 defense-only'].mean():.2f}; model beats average in {int(wk['model beats average'].sum())}/18 weeks")
print(f"L2 reduction vs guessing the average: {1 - L2(d.p, d.y) / L2(d.avg, d.y):.1%} (season), RMSE {L2(d.p, d.y) / np.sqrt(len(d)):.3f} vs {L2(d.avg, d.y) / np.sqrt(len(d)):.3f}")
d.to_csv(S + "/rush_2025_vectors.csv", index=False)

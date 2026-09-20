"""Per-game ranges and tail probabilities on top of the full stack. For season S the miss distribution (actual - projection) is
taken ONLY from the stack's forward results in seasons before S, split by whether the projection was low / middle / high.
Outputs: projected 5/10/50/90/95th percentile per game, P(blow-up) and P(shutdown), and checks that they are honest."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); d = pd.read_csv(os.path.join(HERE, "results", "full_stack_forward.csv")).rename(columns={"y": "actual", "final": "proj"})
QS = [.05, .10, .50, .90, .95]; out = []
for S_ in range(2022, 2026):
    past, x = d[d.season < S_].copy(), d[d.season == S_].copy(); cuts = past.proj.quantile([1 / 3, 2 / 3]).values
    past["g"], x["g"] = np.digitize(past.proj, cuts), np.digitize(x.proj, cuts); past["e"] = past.actual - past.proj; hi, lo = past.actual.quantile(.90), past.actual.quantile(.10)
    for g_ in (0, 1, 2):
        e = past[past.g == g_].e.values; m = x.g == g_
        for q in QS: x.loc[m, f"q{int(q * 100)}"] = x.loc[m, "proj"] + np.quantile(e, q)
        x.loc[m, "p_blowup"] = [(p + e >= hi).mean() for p in x.loc[m, "proj"]]; x.loc[m, "p_shutdown"] = [(p + e <= lo).mean() for p in x.loc[m, "proj"]]
    x["hi"], x["lo"] = hi, lo; out.append(x)
R = pd.concat(out); R["blow"], R["shut"] = (R.actual >= R.hi) * 1, (R.actual <= R.lo) * 1
print(f"PER-GAME RANGES, 2022-2025 ({len(R):,} games). Blow-up = actual in the top 10% of prior seasons (about {R.hi.mean():.1f}+), shutdown = bottom 10% (about {R.lo.mean():.1f} or less)\n")
print("how wide the outputs are now:")
print(pd.DataFrame({"point projection": R.proj.describe()[["min", "max"]], "projected 5th pct": R.q5.describe()[["min", "max"]], "projected 95th pct": R.q95.describe()[["min", "max"]]}).round(2).to_string())
print(f"typical game: projection {R.proj.median():.2f}, range {R.q5.median():.2f} to {R.q95.median():.2f} (5th-95th), {R.q10.median():.2f} to {R.q90.median():.2f} (10th-90th)")
print("\ncoverage check (share of actual results that fell below each projected percentile; should match the label):"); print("  ".join(f"q{int(q * 100)}: {(R.actual <= R[f'q{int(q * 100)}']).mean():.1%}" for q in QS))
print(f"inside the 10th-90th range: {((R.actual >= R.q10) & (R.actual <= R.q90)).mean():.1%} (target 80%); inside 5th-95th: {((R.actual >= R.q5) & (R.actual <= R.q95)).mean():.1%} (target 90%)")
for col, ev, lab in (("p_blowup", "blow", "BLOW-UP"), ("p_shutdown", "shut", "SHUTDOWN")):
    R["b"] = pd.qcut(R[col].rank(method="first"), 5, labels=["lowest 20%", "20-40%", "40-60%", "60-80%", "highest 20%"])
    t = R.groupby("b", observed=True).agg(games=(ev, "size"), avg_probability=(col, "mean"), actually_happened=(ev, "mean")); top = R[R[col] >= R[col].quantile(.95)]
    print(f"\n{lab} probability, games grouped by the model's probability:"); print(t.round(3).to_string()); print(f"   the 5% of games with the highest {lab.lower()} probability: predicted {top[col].mean():.1%}, happened {top[ev].mean():.1%} (base rate {R[ev].mean():.1%}); range of probabilities {R[col].min():.1%} to {R[col].max():.1%}")
t = R[R.season == 2025]; TR = [("LV", 12), ("DAL", 15), ("CHI", 2), ("NE", 2), ("SEA", 18), ("MIN", 13)]; T6 = pd.concat([t[(t.defteam == a) & (t.week == w)] for a, w in TR])
print("\n2025 tracked matchups with ranges:"); print(T6[["defteam", "week", "posteam", "actual", "proj", "q5", "q10", "q90", "q95", "p_blowup", "p_shutdown"]].round(2).to_string(index=False))
R.drop(columns=["b", "g"]).to_csv(os.path.join(HERE, "results", "stack_ranges_2022_2025.csv"), index=False)

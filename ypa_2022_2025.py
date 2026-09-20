"""Defensive pass YPA allowed, 2022-2025 focus: best stats in the feature libraries, the defensive baseline on the 2025 tracked
matchups, and how projections behave in the tails (top/bottom 5% and 10% of actual YPA). Strictly forward: season S is
predicted by a model trained on every season before S."""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "def_ypa_full_eval.py")).read().split("# ---------------------------------------------------------------- 1. leakage")[0])
SMALL = ["eb_def", "eb_off"] + vegas; YEARS = [2022, 2023, 2024, 2025]
def pruned(tr): r = tr[BASE23].corrwith(tr.y); return list(r[r > 0.03].index)

# ---------------------------------------------------------------- A. best stats, 2022-2025
He, _, _ = add_eb(HIST, HIST); W = He[He.season.isin(YEARS)]; OLD = He[He.season < 2022]
LIB = {"baseline 23": BASE23, "defense library": newdef + ["eb_def", "prev_rate", "prev_adj_rate", "prev_epa"], "offense library": ofe + ["eb_off"], "Vegas/venue": vegas}
rows = []
for lib, fs in LIB.items():
    for f in fs:
        per = [W[W.season == s][f].corr(W[W.season == s].y) for s in YEARS]
        rows.append({"stat": f, "library": lib, "r 2022-25": W[f].corr(W.y), "same sign in": f"{sum(np.sign(p) == np.sign(W[f].corr(W.y)) for p in per)}/4 seasons", "weakest season": min(per, key=abs) if all(np.sign(p) == np.sign(per[0]) for p in per) else 0.0,
                     "r 2012-21 (check)": OLD[f].corr(OLD.y)})
ST = pd.DataFrame(rows); ST["abs"] = ST["r 2022-25"].abs(); ST = ST.sort_values("abs", ascending=False).drop(columns="abs")
print(f"A. MOST EFFECTIVE STATS FOR YPA ALLOWED, 2022-2025 ({len(W):,} defense-games; {len(ST)} stats searched). With this many games |r| under ~0.04 is noise.")
print("\nTOP 25 OVERALL"); print(ST.head(25).round(3).to_string(index=False))
print("\nTOP 15 DEFENSE-ONLY STATS (baseline 23 + defense library)"); print(ST[ST.library.isin(["baseline 23", "defense library"])].head(15).round(3).to_string(index=False))
print("\nYOUR 23 BASELINE VARIABLES ON 2022-2025"); print(ST[ST.library == "baseline 23"].round(3).to_string(index=False))
ST.to_csv(os.path.join(HERE, "results", "ypa_stats_2022_2025.csv"), index=False)
robust = ST[(ST["same sign in"] == "4/4 seasons") & (np.sign(ST["r 2022-25"]) == np.sign(ST["r 2012-21 (check)"])) & (ST["r 2012-21 (check)"].abs() > 0.03)]
print(f"\nstats that are same-signed in all four seasons AND confirmed on 2012-2021: {len(robust)} of {len(ST)}")

# ---------------------------------------------------------------- B. forward projections 2022-2025
MODELS = {"DEFENSIVE BASELINE (your 23 vars)": lambda tr: BASE23, "Defense-only best (all defensive stats + pruned baseline)": lambda tr: ["eb_def"] + dfe + pruned(tr),
          "Best matchup model (def + off shrunk levels, Vegas, home, roof)": lambda tr: SMALL}
P = {}
for s in YEARS:
    tr, te = HIST[HIST.season < s], HIST[HIST.season == s]; tre, _, _ = add_eb(tr, tr); tee, _, _ = add_eb(tr, te)
    for mn, fn in MODELS.items():
        out, _ = fitters(tre, tee, fn(tre)); P.setdefault(mn, []).append(pd.DataFrame({"season": s, "week": te.week.values, "defense": te.defteam.values, "vs": te.posteam.values, "actual": te.y.values, "proj": out["ridge"]}))
P = {k: pd.concat(v, ignore_index=True) for k, v in P.items()}
print("\n" + "=" * 110 + f"\nB. FORWARD RESULTS 2022-2025 ({len(P[list(P)[0]]):,} games, ridge)")
rows = []
for mn, d in P.items():
    wk = d.groupby(["season", "week"]).apply(lambda x: x.actual.corr(x.proj)); rows.append({"model": mn, "r": d.actual.corr(d.proj), "avg weekly r": wk.mean(), "weeks r>0": f"{(wk > 0).mean():.0%}", "per-game L2 (RMSE)": rm(d.actual, d.proj),
                 "sd of projections": d.proj.std(), **{f"r {s}": d[d.season == s].actual.corr(d[d.season == s].proj) for s in YEARS}})
print(pd.DataFrame(rows).round(3).to_string(index=False)); a = P[list(P)[0]].actual; print(f"(actual YPA allowed: mean {a.mean():.2f}, sd {a.std():.2f}; per-game L2 of guessing the average {a.std():.3f})")

print("\n" + "=" * 110 + "\nC. 2025 TRACKED MATCHUPS")
T = None
for mn, d in P.items():
    t = d[d.season == 2025]; t6 = pd.concat([t[(t.defense == a_) & (t.week == w_)] for a_, w_ in dm.TRACKED])[["defense", "week", "vs", "actual", "proj"]].rename(columns={"proj": mn.split(" (")[0]})
    T = t6 if T is None else T.merge(t6.drop(columns="actual"), on=["defense", "week", "vs"])
print(T.round(2).to_string(index=False)); print("r on the six:", ", ".join(f"{c}: {T.actual.corr(T[c]):+.2f}" for c in T.columns[4:]))

# ---------------------------------------------------------------- D. tails
print("\n" + "=" * 110 + "\nD. TAILS, 2022-2025. Games are grouped by where their ACTUAL YPA allowed landed; 'projected' is the average projection those same games had beforehand")
for mn, d in P.items():
    q = d.actual.quantile([.05, .10, .90, .95]); rows = []
    for lab, m in ((f"bottom 5%  (actual <= {q[.05]:.2f})", d.actual <= q[.05]), (f"bottom 10% (actual <= {q[.10]:.2f})", d.actual <= q[.10]), ("middle 80%", (d.actual > q[.10]) & (d.actual < q[.90])),
                   (f"top 10%    (actual >= {q[.90]:.2f})", d.actual >= q[.90]), (f"top 5%     (actual >= {q[.95]:.2f})", d.actual >= q[.95])):
        x = d[m]; rows.append({"group": lab, "games": len(x), "avg actual": x.actual.mean(), "avg projected": x.proj.mean(), "projected minus all-game avg": x.proj.mean() - d.proj.mean(), "miss (actual - proj)": (x.actual - x.proj).mean(),
                               "share projected on the correct side of average": ((x.proj > d.proj.mean()) == (x.actual.mean() > d.actual.mean())).mean()})
    print(f"\n{mn}"); print(pd.DataFrame(rows).round(2).to_string(index=False))
    pq = d.proj.quantile([.05, .10, .90, .95]); print("   percentile values   actual:    " + "  ".join(f"p{int(k * 100)}={v:.2f}" for k, v in q.items())); print("                       projected: " + "  ".join(f"p{int(k * 100)}={v:.2f}" for k, v in pq.items()))
    rows = []
    for lab, m in (("projected bottom 5%", d.proj <= pq[.05]), ("projected bottom 10%", d.proj <= pq[.10]), ("projected top 10%", d.proj >= pq[.90]), ("projected top 5%", d.proj >= pq[.95])):
        x = d[m]; rows.append({"group": lab, "games": len(x), "avg projected": x.proj.mean(), "avg actual": x.actual.mean(), "share of these that landed in actual top 10%": (x.actual >= q[.90]).mean(), "landed in actual bottom 10%": (x.actual <= q[.10]).mean()})
    print("   the other direction - what happened in the games the model was most extreme about:"); print(pd.DataFrame(rows).round(2).to_string(index=False))
    d.to_csv(os.path.join(HERE, "results", f"ypa_forward_2022_2025_{mn.split()[0].lower()}.csv"), index=False)

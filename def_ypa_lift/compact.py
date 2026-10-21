"""Second pass: (1) does the starting-QB block help as a group, (2) does REMOVING his noisy terms help more than adding,
(3) a compact model of strong level stats, (4) paired standard errors so a gain can be told apart from noise.
Backward elimination is judged on 2016-2022 only; 2023-2025 is the untouched check. Usage: python def_ypa_lift/compact.py"""
import numpy as np, pandas as pd, warnings
import evaluate as ev
warnings.filterwarnings("ignore")
QB = ["c_qb_epa", "c_qb_ypa", "c_qb_sack", "c_qb_new"]
COMPACT = ["off_spread_line", "total_line", "c_def_ypa", "c_def_epa", "c_off_ypa", "c_off_epa", "c_qb_epa", "c_qb_ypa", "c_qb_sack", "indoors"]


def paired(pa, pb):
    """Gain of b over a in mean season r and MAE, with a standard error from resampling whole weeks (2016-2025)."""
    m = np.isin(ev.SEA, ev.SEASONS); keys = (ev.SEA * 100 + ev.WK)[m]; uk = np.unique(keys); rng = np.random.default_rng(0)
    ya, yb, y, s = pa[m], pb[m], ev.Y[m], ev.SEA[m]
    pos = {k: np.flatnonzero(keys == k) for k in uk}
    by_season = {q: uk[uk // 100 == q] for q in ev.SEASONS}          # resample weeks WITHIN each season
    def stat_fast(idx):
        sel = np.concatenate([pos[k] for k in idx])
        r = lambda p: np.mean([np.corrcoef(p[sel][s[sel] == q], y[sel][s[sel] == q])[0, 1] for q in ev.SEASONS])
        return r(yb) - r(ya), np.mean(np.abs(yb[sel] - y[sel])) - np.mean(np.abs(ya[sel] - y[sel]))
    draw = lambda: np.concatenate([rng.choice(v, len(v)) for v in by_season.values()])
    pt = stat_fast(uk); bs = np.array([stat_fast(draw()) for _ in range(300)])
    return pt[0], bs[:, 0].std(), pt[1], bs[:, 1].std()


def backward(start):
    cols = list(start); cur = ev.sel_obj(ev.forward(cols, range(2016, 2023))); log = []
    while len(cols) > 3:
        best = max(((ev.sel_obj(ev.forward([c for c in cols if c != d], range(2016, 2023))), d) for d in cols))
        if best[0] < cur: break
        cur = best[0]; cols.remove(best[1]); log.append(best[1]); print("drop", best[1], round(cur, 4), flush=True)
    return cols, log


if __name__ == "__main__":
    P = {}
    P["clean 33"] = ev.forward(ev.BASE33)
    P["his 35 incl wind/temp"] = ev.forward(ev.BASE35)
    P["clean 33 + starting-QB block"] = ev.forward(ev.BASE33 + QB)
    P["compact 10 (level stats + Vegas + starting QB)"] = ev.forward(COMPACT)
    kept0, dropped0 = backward(ev.BASE33)
    P[f"backward-pruned, no starter-QB fields ({len(kept0)} terms, chosen on 2016-22)"] = ev.forward(kept0)
    P["compact 7 (no starter-QB fields)"] = ev.forward([c for c in COMPACT if not c.startswith("c_qb")])
    kept, dropped = backward(ev.BASE33 + QB)
    P[f"backward-pruned ({len(kept)} terms, chosen on 2016-22)"] = ev.forward(kept)
    rows = {}
    for k, p in P.items():
        sc = ev.score(p); g = paired(P["clean 33"], p) if k != "clean 33" else (0, 0, 0, 0)
        rows[k] = {**{x: sc[x] for x in ["r_2016_2025", "r_2023_2025", "MAE", "RMSE"]}, "d_r vs clean33": g[0], "se": g[1], "d_MAE": g[2], "se_MAE": g[3]}
    t = pd.DataFrame(rows).T; print(t.round(4).to_string()); t.to_csv(f"{ev.HERE}/results_compact.csv")
    print("kept (no QB):", kept0); print("kept:", kept); print("dropped:", dropped)
    pd.Series(kept0).to_csv(f"{ev.HERE}/pruned_terms_noqb.csv", index=False, header=["stat"])
    pd.Series(kept).to_csv(f"{ev.HERE}/pruned_terms.csv", index=False, header=["stat"])

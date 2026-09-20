"""Shrinkage chosen from standard deviations (empirical Bayes), not by hand. Defense pass YPA allowed and rush YPC allowed.
   estimate = (yards_so_far + k * prior_mean) / (plays_so_far + k),  k = (luck sd per play)^2 / (true defense-to-defense sd)^2
   prior_mean = league + rho * (defense's own shrunk level last season - league), rho measured from year-to-year carry-over."""
import os, sys, numpy as np, pandas as pd, warnings
from scipy.stats import pearsonr
import bakeoff as b, fantasy_bet as fb, def_model as dm
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)

def sd_parts(tr):
    """sigma = luck sd of ONE play's yards around the defense's level; tau = sd of true defense levels (within a season)."""
    u = tr.defteam + tr.season.astype(str); N = tr.den.groupby(u).sum(); m = tr.num.groupby(u).sum() / N
    sigma2 = (tr.den * (tr.y - u.map(m)) ** 2).sum() / (len(tr) - len(N))
    mu = tr.num.sum() / tr.den.sum(); obs = np.average((m - mu) ** 2, weights=N)
    tau2 = max(obs - sigma2 * np.average(1 / N, weights=N), 1e-9)
    return np.sqrt(sigma2), np.sqrt(tau2), np.sqrt(obs)

def project(h, tr, k, rho):
    lg = h.lg_prev
    prev_hat = (h.prev_num.fillna(0) + k * lg) / (h.prev_den.fillna(0) + k)          # last season, itself shrunk with the same k
    prior = lg + rho * (prev_hat - lg)
    return (h.s2d_num + k * prior) / (h.s2d_den + k)

def main(kind):
    g, _, _ = dm.build(kind); h = g[g.season <= 2025].sort_values(["defteam", "t"]).reset_index(drop=True)
    s = b.prior_sums(h, ["defteam", "season"], ["num", "den"], "s2d"); h["s2d_num"], h["s2d_den"] = s.s2d_num, s.s2d_den
    yr = h.groupby(["defteam", "season"], as_index=False)[["num", "den"]].sum().rename(columns={"num": "prev_num", "den": "prev_den"}); yr["season"] += 1
    h = h.merge(yr, on=["defteam", "season"], how="left"); fold = fb.scattered_folds(h)
    unit = {"pass": "pass attempts", "rush": "carries"}[kind]; per_game = h.den.mean()
    print(f"\n{'=' * 108}\n{'PASS YPA ALLOWED' if kind == 'pass' else 'RUSH YPC ALLOWED'} - shrinkage from standard deviations. {len(h):,} defense-games, ~{per_game:.0f} {unit} a game")
    sig, tau, obs = sd_parts(h)
    # rho: how much of a defense's TRUE level carries to next season = corr(season means) / reliability of a season mean
    sm = h.groupby(["defteam", "season"]).apply(lambda x: pd.Series({"m": x.num.sum() / x.den.sum(), "n": x.den.sum()})).reset_index()
    nx = sm.assign(season=sm.season + 1).merge(sm, on=["defteam", "season"], suffixes=("_prev", "")); rel = tau ** 2 / (tau ** 2 + sig ** 2 / sm.n.mean())
    rho = float(np.clip(nx.m_prev.corr(nx.m) / rel, 0, 1)); k = sig ** 2 / tau ** 2
    print(f"  luck sd of one play                       sigma = {sig:.2f} yards")
    print(f"  sd of full-season defense numbers (seen)          {obs:.2f}   <- part real, part luck")
    print(f"  sd of TRUE defense levels (luck removed)    tau = {tau:.2f}")
    print(f"  => k = sigma^2 / tau^2 = {k:.0f} {unit}  (= {k / per_game:.1f} games of data before the defense's own number gets half the weight)")
    print(f"  year-to-year carry-over of true level       rho = {rho:.2f}  (season-mean correlation {nx.m_prev.corr(nx.m):.2f}, reliability of a season mean {rel:.2f})")
    print("  weight on the defense's OWN number after: " + ", ".join(f"{n} games {n * per_game / (n * per_game + k):.0%}" for n in (1, 4, 9, 17)))
    rows = []
    for lab, kk in [("0  (no shrinkage)", 1e-6)] + [(f"{v:.0f}", v) for v in (k / 8, k / 4, k / 2)] + [(f"{k:.0f}  <- FROM THE STANDARD DEVIATIONS", k)] + [(f"{v:.0f}", v) for v in (2 * k, 4 * k, 8 * k)] + [("infinite (always the prior)", 1e9)]:
        p = np.zeros(len(h))
        for f in range(fb.K):
            tr = h[fold != f]; sg, ta, _ = sd_parts(tr)
            k_use = kk if kk != k else sg ** 2 / ta ** 2                       # the SD-derived k is re-derived inside every fold
            p[fold == f] = project(h[fold == f], tr, k_use, rho)
        sl, _ = np.polyfit(p, h.y, 1)
        rows.append({"k": lab, "sd of projections": p.std(), "lowest": p.min(), "highest": p.max(), "r": pearsonr(h.y, p)[0], "RMSE": dm.rmse(h.y, p), "slope (1.0 = spread is honest)": sl})
        if "FROM" in lab: best = p
    print("\n" + pd.DataFrame(rows).round(3).to_string(index=False))
    p0 = np.zeros(len(h))
    for f in range(fb.K):
        tr = h[fold != f]; sg, ta, _ = sd_parts(tr); kf = sg ** 2 / ta ** 2; x = h[fold == f]; p0[fold == f] = (x.s2d_num + kf * x.lg_prev) / (x.s2d_den + kf)
    print(f"\n  same k but folding to LEAGUE AVERAGE instead of the defense's own last-season level: r {pearsonr(h.y, p0)[0]:.3f}, RMSE {dm.rmse(h.y, p0):.3f}, sd of projections {p0.std():.3f}")
    print(f"  folding to the defense's own prior (above):                                              r {pearsonr(h.y, best)[0]:.3f}, RMSE {dm.rmse(h.y, best):.3f}, sd of projections {best.std():.3f}")
    t = h.assign(p=best)[h.season == 2025]; print(f"  2025 only: r {pearsonr(t.y, t.p)[0]:.3f}, RMSE {dm.rmse(t.y, t.p):.3f} (guess-the-average {t.y.std():.3f}); projections {t.p.min():.2f} to {t.p.max():.2f}")
    w18 = t[t.week == 12].sort_values("p"); print("  2025 week 12, five toughest / five softest:", ", ".join(f"{a} {v:.2f}" for a, v in zip(w18.defteam.head(5), w18.p.head(5))), "|", ", ".join(f"{a} {v:.2f}" for a, v in zip(w18.defteam.tail(5), w18.p.tail(5))))

for kind in (sys.argv[1:] or ["pass", "rush"]): main(kind)

"""Fantasy points + betting props. Season-scattered grouped K-fold (every fold holds games from every season).

Population is chosen from PRE-GAME information only (recent attempts / snap share), never from the outcome.
"""
import sys, numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import lightgbm as lgb
import bakeoff as b

b.BASE += ["rush_touchdown", "fantasy_points_ppr", "fantasy_points_standard", "fantasy_points_half_ppr"]
K = 5
nz = lambda s: s.replace(0, np.nan)


def scattered_folds(df, k=K, seed=0):
    """Within each season, shuffle games and deal them round-robin into k folds. Grouped by game_id."""
    rng = np.random.default_rng(seed)
    fold = {}
    for s, g in df.groupby("season"):
        ids = g.game_id.unique(); rng.shuffle(ids)
        fold.update({gid: i % k for i, gid in enumerate(ids)})
    return df.game_id.map(fold).values


def build_qb(w, target):
    q = w[w.position == "QB"].copy()
    q["gp"] = 1
    q["fp"] = q.fantasy_points_ppr
    base = ["passing_yards", "pass_attempts", "complete_pass", "passing_air_yards", "pass_touchdown", "interception",
            "times_sacked", "rushing_yards", "rush_attempts", "rush_touchdown", "fp", "gp"]
    car = b.prior_sums(q, ["player_id"], base, "car")
    s2d = b.prior_sums(q, ["player_id", "season"], ["passing_yards", "pass_attempts", "fp", "gp"], "s2d")
    l4 = b.prior_sums(q, ["player_id"], ["passing_yards", "pass_attempts", "fp", "gp"], "l4", window=4)
    ew = b.prior_sums(q, ["player_id"], ["passing_yards", "pass_attempts", "complete_pass", "passing_air_yards",
                                         "pass_touchdown", "rushing_yards", "rush_attempts", "fp"], "ew", halflife=8)
    q = pd.concat([q, car, s2d, l4, ew], axis=1)
    a, g = nz(q.car_pass_attempts), nz(q.car_gp)
    F = {"car_att": q.car_pass_attempts, "car_gp": q.car_gp, "car_ypa": q.car_passing_yards / a,
         "car_cmp": q.car_complete_pass / a, "car_adot": q.car_passing_air_yards / a, "car_tdr": q.car_pass_touchdown / a,
         "car_intr": q.car_interception / a, "car_sackr": q.car_times_sacked / nz(q.car_pass_attempts + q.car_times_sacked),
         "car_pyds_pg": q.car_passing_yards / g, "car_ryds_pg": q.car_rushing_yards / g, "car_ratt_pg": q.car_rush_attempts / g,
         "car_rtd_pg": q.car_rush_touchdown / g, "car_fp_pg": q.car_fp / g,
         "s2d_ypa": q.s2d_passing_yards / nz(q.s2d_pass_attempts), "s2d_fp_pg": q.s2d_fp / nz(q.s2d_gp),
         "s2d_pyds_pg": q.s2d_passing_yards / nz(q.s2d_gp), "s2d_gp": q.s2d_gp,
         "l4_ypa": q.l4_passing_yards / nz(q.l4_pass_attempts), "l4_fp_pg": q.l4_fp / nz(q.l4_gp),
         "l4_pyds_pg": q.l4_passing_yards / nz(q.l4_gp),
         "ew_ypa": q.ew_passing_yards / nz(q.ew_pass_attempts), "ew_cmp": q.ew_complete_pass / nz(q.ew_pass_attempts),
         "ew_adot": q.ew_passing_air_yards / nz(q.ew_pass_attempts), "ew_att": q.ew_pass_attempts,
         "ew_pyds": q.ew_passing_yards, "ew_ptd": q.ew_pass_touchdown, "ew_ryds": q.ew_rushing_yards,
         "ew_ratt": q.ew_rush_attempts, "ew_fp": q.ew_fp}
    q = q.assign(**F)
    starters = (w.position == "QB") & (w.pass_attempts >= 15)
    for nm, num, den in [("dypa", "passing_yards", "pass_attempts"), ("dpyds", "passing_yards", "gp"), ("dfp", "fp", "gp")]:
        ww = w.assign(gp=1, fp=w.fantasy_points_ppr)
        d = b.defense_table(ww, num, den, starters).rename(columns=lambda c: c.replace("def_", nm + "_"))
        q = q.merge(d, on=["opp", "t"], how="left")
    feats = list(F) + [f"{nm}_{s}" for nm in ["dypa", "dpyds", "dfp"] for s in ["s2d_rate", "prev_rate"]] + \
            ["dypa_s2d_den", "is_home", "age", "years_exp", "draft_pick", "week", "season", "lg_prev"]
    q = q[q.ew_att >= 15]  # pre-game filter: has recently been throwing like a starter
    spec = {"fp": ("fp", "gp", "car_fp", "car_gp", "ew_fp"), "pass_yds": ("passing_yards", "gp", "car_passing_yards", "car_gp", "ew_pyds"),
            "ypa": ("passing_yards", "pass_attempts", "car_passing_yards", "car_pass_attempts", "ew_ypa")}[target]
    if target == "ypa":
        q = q[q.pass_attempts >= 10]
    return finish(q, feats, *spec)


def build_te(w, target):
    te, feats = b.build_te(w.assign(), "yds")  # only to reuse its team/defense joins on the full TE table below
    t = w[w.position == "TE"].copy()
    t["gp"], t["fp"] = 1, t.fantasy_points_ppr
    car = b.prior_sums(t, ["player_id"], ["fp", "receptions", "receiving_touchdown", "gp"], "c2")
    ew = b.prior_sums(t, ["player_id"], ["fp", "receptions", "receiving_touchdown"], "e2", halflife=6)
    l4 = b.prior_sums(t, ["player_id"], ["fp", "receptions", "gp"], "l2", window=4)
    t = pd.concat([t[["player_id", "game_id", "fp", "receptions"]], car, ew, l4], axis=1)
    t["car_fp_pg"], t["car_rec_pg"] = t.c2_fp / nz(t.c2_gp), t.c2_receptions / nz(t.c2_gp)
    t["car_td_pg"] = t.c2_receiving_touchdown / nz(t.c2_gp)
    t["l4_fp_pg"], t["l4_rec_pg"] = t.l2_fp / nz(t.l2_gp), t.l2_receptions / nz(t.l2_gp)
    t = t.rename(columns={"e2_fp": "ew_fp", "e2_receptions": "ew_rec", "e2_receiving_touchdown": "ew_td", "receptions": "rec_y"})
    extra = ["car_fp_pg", "car_rec_pg", "car_td_pg", "l4_fp_pg", "l4_rec_pg", "ew_fp", "ew_rec", "ew_td"]
    te = te.drop(columns=["y", "n", "eb_num", "eb_den", "lg_prev"]).merge(
        t[["player_id", "game_id", "fp", "rec_y", "c2_fp", "c2_receptions", "c2_gp"] + extra], on=["player_id", "game_id"])
    te["gp"] = 1
    spec = {"fp": ("fp", "gp", "c2_fp", "c2_gp", "ew_fp"), "rec_yds": ("receiving_yards", "gp", "car_receiving_yards", "c2_gp", "ew_yds"),
            "rec": ("rec_y", "gp", "c2_receptions", "c2_gp", "ew_rec")}[target]
    return finish(te, [f for f in feats if f != "lg_prev"] + extra + ["lg_prev"], *spec)


def finish(df, feats, num, den, eb_num, eb_den, line_col):
    df = df.copy()
    df["lg_prev"] = b.league_prev(df, num, den)
    df["y"] = df[num] / nz(df[den])
    df["n"] = df[den].astype(float)
    df["eb_num"], df["eb_den"] = df[eb_num], df[eb_den]
    df["line"] = (df[line_col] * 2).round() / 2 if df.y.std() > 3 else df[line_col].round(1)
    return df[df.y.notna() & df.line.notna()].reset_index(drop=True), feats


def regression_table(df, feats, label):
    fold = scattered_folds(df)
    oof = {}
    for k in range(K):
        tr, te = df[fold != k], df[fold == k]
        preds, _ = b.fit_predict(tr, te, feats, True)
        for m, p in preds.items():
            oof.setdefault(m, np.full(len(df), np.nan))[fold == k] = p
    trf, tef = df[df.season < 2023], df[df.season >= 2023]
    fwd, _ = b.fit_predict(trf, tef, feats, True)
    per = pd.DataFrame({"season": df.season, "fold": fold}).groupby("fold").season.agg(["size", "nunique"])
    print(f"\n{'=' * 96}\n{label}: {len(df):,} games, {df.player_id.nunique()} players, {df.season.min()}-{df.season.max()}; "
          f"mean {df.y.mean():.2f}, sd {df.y.std():.2f}")
    print(f"split: {K} folds scattered across seasons, grouped by game -> each fold trains on ~{int(len(df) * (K - 1) / K):,} games, "
          f"tests on ~{int(len(df) / K):,}; seasons present per fold: {per['nunique'].min()}-{per['nunique'].max()} of {df.season.nunique()}")
    rows = []
    for m, p in oof.items():
        s, f = b.score(df.y.values, p, df.n.values), b.score(tef.y.values, np.asarray(fwd[m]), tef.n.values)
        by_season = [pearsonr(df.y[df.season == yr], p[df.season == yr])[0] if np.std(p[df.season == yr]) > 1e-9 else np.nan
                     for yr in sorted(df.season.unique())]
        rows.append({"model": m, "OOF r": s["r"], "OOF Spearman": s["rho"], "OOF RMSE": s["rmse"],
                     "worst-season r": np.nanmin(by_season), "best-season r": np.nanmax(by_season), "Fwd 2023+ r (ref)": f["r"]})
    tab = pd.DataFrame(rows).sort_values("OOF r", ascending=False).reset_index(drop=True)
    tab.loc[tab.model.str.startswith("Logistic"), "OOF RMSE"] = np.nan
    print(tab.round(3).to_string(index=False))
    tab.to_csv(f"{b.S}/scattered_{label.split(':')[0].replace(' ', '_')}.csv", index=False)
    best = tab[~tab.model.str.startswith(("Logistic", "Baseline"))].iloc[0].model
    p, y = oof[best], df.y.values
    cal = np.zeros(len(df))
    for k in range(K):
        sl, ic = np.polyfit(p[fold != k], y[fold != k], 1); cal[fold == k] = ic + sl * p[fold == k]
    sl, ic = np.polyfit(p, y, 1); sl2, ic2 = np.polyfit(cal, y, 1)
    print(f"calibration of best ({best}): raw slope {sl:.3f} / intercept {ic:.3f} -> cross-fitted slope {sl2:.3f} / intercept {ic2:.3f}; "
          f"final map actual = {ic:.3f} + {sl:.3f}*pred")
    d = pd.DataFrame({"pred": cal, "y": y}); d["bin"] = pd.qcut(d.pred, 10, labels=False, duplicates="drop")
    r = d.groupby("bin").agg(pred=("pred", "mean"), actual=("y", "mean"))
    print("deciles  pred:  ", " ".join(f"{v:6.1f}" for v in r.pred)); print("deciles  actual:", " ".join(f"{v:6.1f}" for v in r.actual))
    res_season = pd.DataFrame({"season": df.season, "resid": y - cal}).groupby("season").resid.mean()
    print("mean residual by season (season bias check):", " ".join(f"{yr}:{v:+.2f}" for yr, v in res_season.items()))
    return fold


def over_under(df, feats, fold, label):
    """P(actual > line). Line = the player's own recent exponentially-weighted average (a stand-in, NOT a sportsbook line)."""
    d = df[df.y != df.line].reset_index(drop=True); fold = fold[(df.y != df.line).values]
    z = (d.y > d.line).astype(int).values
    X = d[feats + ["line"]].assign(eb=0.0)
    P = {"Logistic": np.zeros(len(d)), "LightGBM clf": np.zeros(len(d))}
    for k in range(K):
        a, t = fold != k, fold == k
        kk = b.eb_k(d[a]); X.loc[:, "eb"] = b.add_eb(d, kk)
        X["eb_minus_line"] = X.eb - X.line
        P["Logistic"][t] = b.lin(LogisticRegression(C=0.1, max_iter=3000)).fit(X[a], z[a]).predict_proba(X[t])[:, 1]
        P["LightGBM clf"][t] = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.02, num_leaves=8, min_child_samples=80, subsample=0.8,
                                                  subsample_freq=1, colsample_bytree=0.7, reg_lambda=10, verbose=-1, random_state=0
                                                  ).fit(X[a], z[a]).predict_proba(X[t])[:, 1]
    P["Blend"] = (P["Logistic"] + P["LightGBM clf"]) / 2
    base = brier_score_loss(z, np.full(len(z), z.mean()))
    print(f"\n--- OVER/UNDER {label}: {len(d):,} games, line = own recent avg (median line {d.line.median():.1f}), over rate {z.mean():.1%}")
    rows = [{"model": m, "AUC": roc_auc_score(z, p), "Brier": brier_score_loss(z, p), "Brier skill vs base rate": 1 - brier_score_loss(z, p) / base} for m, p in P.items()]
    t = pd.DataFrame(rows).sort_values("Brier"); print(t.round(4).to_string(index=False))
    best = t.iloc[0].model; p = P[best]; cal = np.zeros(len(d))
    for k in range(K):
        cal[fold == k] = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(p[fold != k], z[fold != k]).predict(p[fold == k])
    print(f"calibrated ({best}, cross-fitted isotonic): Brier {brier_score_loss(z, cal):.4f}")
    bins = pd.cut(cal, [0, .3, .4, .45, .5, .55, .6, .7, 1]); r = pd.DataFrame({"p": cal, "z": z}).groupby(bins, observed=True).agg(games=("z", "size"), pred=("p", "mean"), actual=("z", "mean"))
    print(r.round(3).to_string())
    for edge in (0.05, 0.10):
        pick = np.abs(cal - 0.5) >= edge; win = np.where(cal[pick] > 0.5, z[pick], 1 - z[pick])
        print(f"pick a side when calibrated P is >= {edge:.0%} from 50/50: {pick.sum():,} picks ({pick.mean():.0%} of games), hit rate {win.mean():.1%} (-110 breakeven 52.4%)")


if __name__ == "__main__":
    w = b.load()
    jobs = {"qb_fp": (build_qb, "fp", "QB fantasy points (PPR scoring)", False), "te_fp": (build_te, "fp", "TE fantasy points (PPR)", False),
            "qb_yds": (build_qb, "pass_yds", "QB passing yards: betting prop", True), "qb_ypa": (build_qb, "ypa", "QB yards per attempt: betting", True),
            "te_yds": (build_te, "rec_yds", "TE receiving yards: betting prop", True), "te_rec": (build_te, "rec", "TE receptions: betting prop", True)}
    for key in (sys.argv[1:] or list(jobs)):
        fn, tgt, label, bet = jobs[key]
        df, feats = fn(w, tgt)
        fold = regression_table(df, feats, label)
        if bet:
            over_under(df, feats, fold, label)

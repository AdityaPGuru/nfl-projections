"""Model bake-off: QB yards/attempt and TE stats, leak-free features, leave-season-out OOF.

Every feature uses only games strictly before the row's game. EB shrinkage strength is
re-estimated on the training seasons of each fold.
"""
import glob, os, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
warnings.filterwarnings("ignore")

S = os.environ["S"]
B = glob.glob(S + "/cache/datasets/philiphyde1/*/versions/*")[0]
BASE = ["player_id", "player_name", "position", "team", "season", "week", "season_type", "game_id",
        "birth_date", "draft_pick", "years_exp", "pass_attempts", "complete_pass", "passing_yards",
        "passing_air_yards", "pass_touchdown", "interception", "times_sacked", "qb_scramble",
        "rush_attempts", "rushing_yards", "targets", "receptions", "receiving_yards",
        "receiving_air_yards", "yards_after_catch", "receiving_touchdown", "offense_snaps",
        "team_offense_snaps"]


def load():
    w = pd.read_csv(B + "/weekly_player_stats_offense.csv", usecols=BASE)
    p = w.game_id.str.split("_", expand=True)
    reloc = {"OAK": "LV", "SD": "LAC", "STL": "LA"}  # game_id keeps the old city code
    w["away"], w["home"] = p[2].replace(reloc), p[3].replace(reloc)
    w["is_home"] = (w.team == w.home).astype(int)
    w["opp"] = np.where(w.team == w.home, w.away, w.home)
    bad = ~((w.team == w.home) | (w.team == w.away))
    print(f"rows whose team is not in game_id: {bad.mean():.3%} (dropped)")
    w = w[~bad].copy()
    w["t"] = w.season * 100 + w.week
    w["age"] = w.season + (w.week / 52) - pd.to_datetime(w.birth_date, errors="coerce").dt.year
    return w.sort_values(["player_id", "t"]).reset_index(drop=True)


def prior_sums(df, keys, cols, prefix, window=None, halflife=None):
    """Sum of `cols` over earlier rows within `keys` group (current row excluded)."""
    g = df.groupby(keys, sort=False)
    out = {}
    for c in cols:
        if window:
            v = g[c].transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
        elif halflife:
            v = g[c].transform(lambda s: s.shift(1).ewm(halflife=halflife, adjust=True).mean())
        else:
            v = g[c].cumsum() - df[c]
        out[f"{prefix}_{c}"] = v
    return pd.DataFrame(out, index=df.index)


def defense_table(w, num, den, mask):
    """Per defense-game: what it allowed (num/den), then point-in-time prior sums."""
    d = w[mask].groupby(["opp", "season", "t"], as_index=False)[[num, den]].sum().sort_values(["opp", "t"])
    s2d = prior_sums(d, ["opp", "season"], [num, den], "s2d")
    d["def_s2d_rate"] = s2d.iloc[:, 0] / s2d.iloc[:, 1].replace(0, np.nan)
    d["def_s2d_den"] = s2d.iloc[:, 1]
    yr = d.groupby(["opp", "season"], as_index=False)[[num, den]].sum()
    yr["def_prev_rate"] = yr[num] / yr[den]
    yr["season"] += 1
    d = d.merge(yr[["opp", "season", "def_prev_rate"]], on=["opp", "season"], how="left")
    return d[["opp", "t", "def_s2d_rate", "def_s2d_den", "def_prev_rate"]]


def league_prev(df, num, den):
    yr = df.groupby("season")[[num, den]].sum()
    s = (yr[num] / yr[den]).shift(1)
    return df.season.map(s).fillna(s.dropna().iloc[0])


def build_qb(w):
    q = w[(w.position == "QB")].copy()
    q["dropbacks"] = q.pass_attempts + q.times_sacked
    cols = ["passing_yards", "pass_attempts", "complete_pass", "passing_air_yards", "pass_touchdown",
            "interception", "times_sacked", "dropbacks", "rushing_yards", "rush_attempts"]
    car = prior_sums(q, ["player_id"], cols, "car")
    s2d = prior_sums(q, ["player_id", "season"], cols[:2], "s2d")
    l4 = prior_sums(q, ["player_id"], cols[:2], "l4", window=4)
    ew = prior_sums(q, ["player_id"], cols[:4], "ew", halflife=8)
    q = pd.concat([q, car, s2d, l4, ew], axis=1)
    nz = lambda s: s.replace(0, np.nan)
    a = nz(q.car_pass_attempts)
    q["car_att"] = q.car_pass_attempts
    q["car_ypa"] = q.car_passing_yards / a
    q["car_cmp"] = q.car_complete_pass / a
    q["car_adot"] = q.car_passing_air_yards / a
    q["car_tdr"] = q.car_pass_touchdown / a
    q["car_intr"] = q.car_interception / a
    q["car_sackr"] = q.car_times_sacked / nz(q.car_dropbacks)
    q["car_rush_pg"] = q.car_rush_attempts / nz(q.groupby("player_id").cumcount())
    q["s2d_att"] = q.s2d_pass_attempts
    q["s2d_ypa"] = q.s2d_passing_yards / nz(q.s2d_pass_attempts)
    q["l4_ypa"] = q.l4_passing_yards / nz(q.l4_pass_attempts)
    q["l4_att"] = q.l4_pass_attempts
    q["ew_ypa"] = q.ew_passing_yards / nz(q.ew_pass_attempts)
    q["ew_cmp"] = q.ew_complete_pass / nz(q.ew_pass_attempts)
    q["ew_adot"] = q.ew_passing_air_yards / nz(q.ew_pass_attempts)
    q["ew_att"] = q.ew_pass_attempts
    q["lg_prev"] = league_prev(q, "passing_yards", "pass_attempts")
    q = q.merge(defense_table(w, "passing_yards", "pass_attempts", w.pass_attempts > 0), on=["opp", "t"], how="left")
    q["y"] = q.passing_yards / nz(q.pass_attempts)
    q["n"] = q.pass_attempts
    q["eb_num"], q["eb_den"] = q.car_passing_yards, q.car_pass_attempts
    q = q[q.pass_attempts >= 10].reset_index(drop=True)
    feats = ["car_att", "car_ypa", "car_cmp", "car_adot", "car_tdr", "car_intr", "car_sackr", "car_rush_pg",
             "s2d_att", "s2d_ypa", "l4_ypa", "l4_att", "ew_ypa", "ew_cmp", "ew_adot", "ew_att", "lg_prev",
             "def_s2d_rate", "def_s2d_den", "def_prev_rate", "is_home", "age", "years_exp", "draft_pick",
             "week", "season"]
    return q, feats


def build_te(w, target):
    te = w[w.position == "TE"].copy()
    tm = w.groupby(["team", "t"], as_index=False).agg(tm_tgt=("targets", "sum"), tm_att=("pass_attempts", "sum"),
                                                       tm_yds=("passing_yards", "sum"))
    tm = tm.sort_values(["team", "t"])
    tm["season"] = tm.t // 100
    ps = prior_sums(tm, ["team"], ["tm_att", "tm_yds"], "ew", halflife=8)
    tm["tm_ew_att"] = ps.ew_tm_att
    tm["tm_ew_ypa"] = ps.ew_tm_yds / ps.ew_tm_att.replace(0, np.nan)
    te = te.merge(tm[["team", "t", "tm_tgt", "tm_ew_att", "tm_ew_ypa"]], on=["team", "t"], how="left")
    te = te.sort_values(["player_id", "t"]).reset_index(drop=True)
    te["tgt_share"] = te.targets / te.tm_tgt.replace(0, np.nan)
    te["snap_pct"] = te.offense_snaps / te.team_offense_snaps.replace(0, np.nan)
    te["gp"] = 1
    cols = ["receiving_yards", "targets", "receptions", "receiving_air_yards", "yards_after_catch",
            "receiving_touchdown", "gp"]
    car = prior_sums(te, ["player_id"], cols, "car")
    s2d = prior_sums(te, ["player_id", "season"], ["receiving_yards", "targets", "gp"], "s2d")
    l4 = prior_sums(te, ["player_id"], ["receiving_yards", "targets", "gp"], "l4", window=4)
    ew = prior_sums(te, ["player_id"], ["receiving_yards", "targets", "receiving_air_yards", "tgt_share", "snap_pct"], "ew", halflife=6)
    te = pd.concat([te, car, s2d, l4, ew], axis=1)
    nz = lambda s: s.replace(0, np.nan)
    te["car_tgt"] = te.car_targets
    te["car_gp"] = te.car_gp
    te["car_ypt"] = te.car_receiving_yards / nz(te.car_targets)
    te["car_catch"] = te.car_receptions / nz(te.car_targets)
    te["car_adot"] = te.car_receiving_air_yards / nz(te.car_targets)
    te["car_yac_r"] = te.car_yards_after_catch / nz(te.car_receptions)
    te["car_tgt_pg"] = te.car_targets / nz(te.car_gp)
    te["car_yds_pg"] = te.car_receiving_yards / nz(te.car_gp)
    te["s2d_ypt"] = te.s2d_receiving_yards / nz(te.s2d_targets)
    te["s2d_tgt_pg"] = te.s2d_targets / nz(te.s2d_gp)
    te["s2d_yds_pg"] = te.s2d_receiving_yards / nz(te.s2d_gp)
    te["l4_ypt"] = te.l4_receiving_yards / nz(te.l4_targets)
    te["l4_tgt_pg"] = te.l4_targets / nz(te.l4_gp)
    te["l4_yds_pg"] = te.l4_receiving_yards / nz(te.l4_gp)
    te["ew_ypt"] = te.ew_receiving_yards / nz(te.ew_targets)
    te["ew_tgt"], te["ew_yds"] = te.ew_targets, te.ew_receiving_yards
    te["ew_adot"] = te.ew_receiving_air_yards / nz(te.ew_targets)
    te = te.merge(defense_table(w, "receiving_yards", "targets", (w.position == "TE") & (w.targets > 0)), on=["opp", "t"], how="left")
    feats = ["car_tgt", "car_gp", "car_ypt", "car_catch", "car_adot", "car_yac_r", "car_tgt_pg", "car_yds_pg",
             "s2d_ypt", "s2d_tgt_pg", "s2d_yds_pg", "l4_ypt", "l4_tgt_pg", "l4_yds_pg", "ew_ypt", "ew_tgt",
             "ew_yds", "ew_adot", "ew_tgt_share", "ew_snap_pct", "tm_ew_att", "tm_ew_ypa", "def_s2d_rate",
             "def_s2d_den", "def_prev_rate", "is_home", "age", "years_exp", "draft_pick", "week", "season", "lg_prev"]
    if target == "ypt":
        te["lg_prev"] = league_prev(te, "receiving_yards", "targets")
        te["y"] = te.receiving_yards / nz(te.targets)
        te["n"] = te.targets
        te["eb_num"], te["eb_den"] = te.car_receiving_yards, te.car_targets
        te = te[te.targets >= 2]
    else:  # receiving yards per game
        te["lg_prev"] = league_prev(te, "receiving_yards", "gp")
        te["y"] = te.receiving_yards
        te["n"] = 1.0
        te["eb_num"], te["eb_den"] = te.car_receiving_yards, te.car_gp
        te = te[te.ew_snap_pct.fillna(0) > 0.25]  # a player we'd plausibly be asked about pre-game
    return te.reset_index(drop=True), feats


def eb_k(tr):
    """Method-of-moments k = sigma^2/tau^2 for y_i ~ N(theta_p, sigma^2/n_i), theta_p ~ N(mu, tau^2)."""
    g = tr.groupby("player_id")
    N = g.n.sum()
    m = (tr.y * tr.n).groupby(tr.player_id).sum() / N
    within = (tr.n * (tr.y - tr.player_id.map(m)) ** 2).sum()
    sigma2 = within / max(len(tr) - g.ngroups, 1)
    keep = N >= N.median()
    mu = np.average(m[keep], weights=N[keep])
    tau2 = np.average((m[keep] - mu) ** 2, weights=N[keep]) - sigma2 * np.average(1 / N[keep], weights=N[keep])
    tau2 = max(tau2, 1e-6)
    return sigma2 / tau2


def add_eb(df, k):
    den = df.eb_den.fillna(0)
    return (df.eb_num.fillna(0) + k * df.lg_prev) / (den + k)


def lin(m):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), m)


def models():
    return {
        "Linear regression (OLS)": lambda: lin(LinearRegression()),
        "Ridge regression": lambda: lin(RidgeCV(alphas=np.logspace(-1, 4, 20))),
        "Lasso regression": lambda: lin(LassoCV(cv=5, alphas=30)),
        "KNN (k=50, distance-wtd)": lambda: lin(KNeighborsRegressor(50, weights="distance")),
        "KNN (k=200)": lambda: lin(KNeighborsRegressor(200)),
        "SVR (RBF)": lambda: lin(SVR(C=1.0, epsilon=0.5)),
        "Random forest": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                               RandomForestRegressor(300, min_samples_leaf=40, max_features=0.4, n_jobs=-1, random_state=0)),
        "Neural net (MLP 32-16)": lambda: lin(MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1.0, early_stopping=True, max_iter=500, random_state=0)),
        "Gradient boosting (sklearn Hist)": lambda: HistGradientBoostingRegressor(learning_rate=0.03, max_iter=250, max_depth=3,
                                                                                  min_samples_leaf=60, l2_regularization=5.0, random_state=0),
        "Gradient boosting (LightGBM)": lambda: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=8, min_child_samples=80,
                                                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10,
                                                                  verbose=-1, random_state=0),
    }


def fit_predict(tr, te_, feats, with_eb):
    """Returns dict name -> test predictions. with_eb adds the EB estimate as a feature."""
    k = eb_k(tr)
    tr, te_ = tr.copy(), te_.copy()
    tr["eb"], te_["eb"] = add_eb(tr, k), add_eb(te_, k)
    out = {"Baseline: last season's league avg": te_.lg_prev.values,
           "Baseline: raw career rate (no shrinkage)": (te_.eb_num / te_.eb_den.replace(0, np.nan)).fillna(te_.lg_prev).values,
           "Empirical Bayes (shrunk career rate)": te_.eb.values}
    f_eb = feats + ["eb"]
    for name, mk in models().items():
        out[name] = mk().fit(tr[feats], tr.y).predict(te_[feats])
    for name in ["Ridge regression", "Gradient boosting (LightGBM)", "Gradient boosting (sklearn Hist)"]:
        out["EB + " + name] = models()[name]().fit(tr[f_eb], tr.y).predict(te_[f_eb])
    thr = tr.y.median()
    lg = lin(LogisticRegression(C=0.1, max_iter=2000)).fit(tr[f_eb], (tr.y > thr).astype(int))
    out["Logistic regression (P[above median])"] = lg.predict_proba(te_[f_eb])[:, 1]
    out["Blend: EB + Ridge + LightGBM (mean)"] = np.mean([out["Empirical Bayes (shrunk career rate)"], out["EB + Ridge regression"],
                                                          out["EB + Gradient boosting (LightGBM)"]], axis=0)
    return out, k


def score(y, p, n):
    ok = np.isfinite(p)
    if np.std(p[ok]) < 1e-9:
        return dict(r=np.nan, rho=np.nan, r_w=np.nan, rmse=np.sqrt(np.mean((y - p) ** 2)))
    yw, pw = y - np.average(y, weights=n), p - np.average(p, weights=n)
    r_w = (n * yw * pw).sum() / np.sqrt((n * yw ** 2).sum() * (n * pw ** 2).sum())
    return dict(r=pearsonr(y, p)[0], rho=spearmanr(y, p)[0], r_w=r_w, rmse=np.sqrt(np.mean((y - p) ** 2)))


def run(df, feats, label, fwd_from=2023):
    print(f"\n{'=' * 100}\n{label}: {len(df):,} games, {df.player_id.nunique()} players, seasons {df.season.min()}-{df.season.max()}, "
          f"target mean {df.y.mean():.2f} sd {df.y.std():.2f}")
    oof, ks = {}, []
    for s in sorted(df.season.unique()):
        tr, te_ = df[df.season != s], df[df.season == s]
        preds, k = fit_predict(tr, te_, feats, True)
        ks.append(k)
        for m, p in preds.items():
            oof.setdefault(m, pd.Series(np.nan, index=df.index))[te_.index] = p
    tr, te_ = df[df.season < fwd_from], df[df.season >= fwd_from]
    fwd, kf = fit_predict(tr, te_, feats, True)
    print(f"EB prior strength k (pseudo-{'attempts' if df.n.max() > 1 else 'games'}): mean {np.mean(ks):.0f} across folds, forward fit {kf:.0f}")
    rows = []
    for m in oof:
        a, b = score(df.y.values, oof[m].values, df.n.values), score(te_.y.values, np.asarray(fwd[m]), te_.n.values)
        rows.append({"model": m, "OOF r": a["r"], "OOF r (n-wtd)": a["r_w"], "OOF Spearman": a["rho"], "OOF RMSE": a["rmse"],
                     f"Fwd {fwd_from}+ r": b["r"], f"Fwd RMSE": b["rmse"]})
    tab = pd.DataFrame(rows).sort_values("OOF r", ascending=False).reset_index(drop=True)
    is_logit = tab.model.str.startswith("Logistic")
    tab.loc[is_logit, ["OOF RMSE", "Fwd RMSE"]] = np.nan
    print(tab.round(3).to_string(index=False))
    thr = df.y.median()
    print(f"Logistic AUC (OOF, above overall median): {roc_auc_score(df.y > thr, oof['Logistic regression (P[above median])']):.3f}")
    tab.to_csv(f"{S}/table_{label.split()[0]}_{label.split()[1].replace('/', '-')}.csv", index=False)
    best = tab[~is_logit & ~tab.model.str.startswith("Baseline")].iloc[0].model
    calibrate(df, oof[best], te_, np.asarray(fwd[best]), best)
    return tab, oof, best


def calib_stats(y, p):
    slope, icpt = np.polyfit(p, y, 1)
    return slope, icpt, np.sqrt(np.mean((y - p) ** 2)), np.mean(y - p)


def calibrate(df, oofp, fwd_df, fwdp, name):
    """Cross-fitted calibration of OOF predictions: calibrator for season s is fit on OOF preds of other seasons."""
    print(f"\n--- Calibration of best model: {name}")
    y, p = df.y.values, oofp.values
    lin_c, iso_c = np.zeros(len(df)), np.zeros(len(df))
    for s in sorted(df.season.unique()):
        a, b = (df.season != s).values, (df.season == s).values
        sl, ic = np.polyfit(p[a], y[a], 1)
        lin_c[b] = ic + sl * p[b]
        iso_c[b] = IsotonicRegression(out_of_bounds="clip").fit(p[a], y[a]).predict(p[b])
    print(f"{'':28s}{'slope':>8s}{'intercept':>11s}{'RMSE':>8s}{'bias':>8s}   (perfect: slope 1, intercept 0)")
    for lab, pp in [("raw OOF", p), ("linear-calibrated (x-fit)", lin_c), ("isotonic-calibrated (x-fit)", iso_c)]:
        print(f"{lab:28s}" + "".join(f"{v:>{w}.3f}" for v, w in zip(calib_stats(y, pp), [8, 11, 8, 8])))
    pre = df.season < fwd_df.season.min()
    sl, ic = np.polyfit(p[pre.values], y[pre.values], 1)
    print(f"forward {fwd_df.season.min()}+ raw            " + "".join(f"{v:>{w}.3f}" for v, w in zip(calib_stats(fwd_df.y.values, fwdp), [8, 11, 8, 8])))
    print(f"forward, calibrator from OOF<{fwd_df.season.min()}" + "".join(f"{v:>{w}.3f}" for v, w in zip(calib_stats(fwd_df.y.values, ic + sl * fwdp), [8, 11, 8, 8])))
    print(f"final linear calibrator on all OOF: actual = {np.polyfit(p, y, 1)[1]:.3f} + {np.polyfit(p, y, 1)[0]:.3f} * pred")
    d = pd.DataFrame({"pred": lin_c, "y": y, "n": df.n.values})
    d["bin"] = pd.qcut(d.pred, 10, labels=False, duplicates="drop")
    r = d.groupby("bin").agg(games=("y", "size"), mean_pred=("pred", "mean"), mean_actual=("y", "mean"))
    r["gap"] = r.mean_actual - r.mean_pred
    print("reliability by decile of calibrated prediction:"); print(r.round(2).to_string())


if __name__ == "__main__":
    w = load()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "qb"):
        q, f = build_qb(w); run(q, f, "QB YPA (>=10 att)")
    if which in ("all", "te"):
        t, f = build_te(w, "ypt"); run(t, f, "TE yards-per-target (>=2 tgt)")
        t, f = build_te(w, "yds"); run(t, f, "TE receiving-yards per game")

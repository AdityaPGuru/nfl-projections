"""TE receiving-yards share: forward-season evaluation and 2026 projections.

Every model for season S is fit on seasons 2013..S-1 only. 2026 games are never used to fit anything;
2026 season-to-date stats only enter as pregame features (the same way week-N features always work).
Tuning values (shrinkage k for offense and defense, ridge alpha, calibration) are learned on the training
seasons and printed.

Models compared:
  league    prior-season league share (no information about the teams)
  naive     ridge on raw (unshrunk) season-to-date and prior-season shares, same coefficients every week.
            This is the structure of the frozen baseline: fixed weights on season-to-date rates.
  eb        empirical-Bayes posterior shares (prior from last season + roster, weight n/(n+k) on this season)
  eb_full   eb + personnel (injury report, returning production, snap roles) + Vegas, ridge
  eb_gbm    same inputs, LightGBM
"""
import numpy as np, pandas as pd, os, sys, warnings
from sklearn.linear_model import RidgeCV, LinearRegression
import lightgbm as lgb
warnings.filterwarnings("ignore")

D = os.path.join(os.path.dirname(__file__), "data")
R = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(R, exist_ok=True)
t = pd.read_parquet(f"{D}/team_games.parquet")
t = t[t.season >= 2013].copy()

def share(a, b):
    return a / b.where(b > 0)

# raw rates
t["o_ytd"] = share(t.o_yds_TE_ytd, t.o_yds_ALL_ytd)
t["d_ytd"] = share(t.d_yds_TE_ytd, t.d_yds_ALL_ytd)
t["o_tgt_ytd"] = share(t.o_tgt_TE_ytd, t.o_tgt_ALL_ytd)
t["o_ay_ytd"] = share(t.o_ay_TE_ytd, t.o_ay_ALL_ytd)
t["lg_ytd"] = share(t.lg_ytd_te, t.lg_ytd_all)
t["d_n"] = t.groupby(["season", "defteam"]).week.rank() - 1  # defense games before W (approx; refined below)
# exact defense games-before-W from the defense's own played games
dg = t[t.played][["season", "week", "defteam"]].sort_values("week")
dg["dn"] = dg.groupby(["season", "defteam"]).cumcount() + 1
dg = dg.assign(week=dg.week + 1)
t = pd.merge_asof(t.sort_values("week"), dg.sort_values("week"), on="week", by=["season", "defteam"], direction="backward")
t["d_n"] = t.dn.fillna(0); t = t.drop(columns="dn")
t["o_n"] = t.o_n_ytd

# Everything share-like is expressed relative to last season's league level, and the target is too.
# Fitting a coefficient on the league level itself extrapolates when a season sets a record (2025 did).
L = t.lg_prev
t["y_rel"] = t.y - L
for c in ["o_ytd", "d_ytd", "o_tgt_ytd", "o_ay_ytd", "prev_off_share", "prev_off_tgt_share", "prev_def_share"]:
    t[c] = t[c] - L
# league reversion input: how far last season's league level sits above the long-run level of all earlier
# seasons (uses completed seasons only). Its coefficient is learned, so a record year like 2025 is pulled back
# only as much as past record years were.
_full = pd.read_parquet(f"{D}/team_games.parquet"); _full = _full[_full.played]
_lg_all = _full.groupby("season").apply(lambda x: x.yds_TE.sum() / x.yds_ALL.sum())
t["lg_gap"] = t.season.map({S: _lg_all[S - 1] - _lg_all[_lg_all.index < S].mean() for S in t.season.unique()})
# this season's league drift so far, as a pregame input (0 in week 1)
t["lg_drift"] = (t.lg_ytd - L).fillna(0)
t["lg_n"] = (t.week - 1).clip(lower=0)
# Roster/snap features drift across eras (2025 was a record TE season, 12-personnel keeps rising), so each is
# centered on the same week's league-wide mean, which is known before kickoff from rosters and snap logs.
for c in ["te_rel_prod", "te_prevypg_avail", "te1_pct_prev", "te_pct_sum_prev", "te1_pct_ytd", "te_pct_sum_ytd",
          "wr_prevypg_avail"]:
    t[c] = t[c] - t.groupby(["season", "week"])[c].transform("mean")

FEATS_PRIOR_O = ["prev_off_share", "prev_off_tgt_share", "te_rel_prod", "te_prevypg_avail", "te1_pct_prev", "te_pct_sum_prev"]
FEATS_PRIOR_D = ["prev_def_share"]


def fill(X, med):
    return X.fillna(med)


class EB:
    """prior = linear model on preseason/roster features; posterior = (k*prior + n*ytd) / (k + n).
    k is chosen on the training rows by minimizing squared error against the game outcome."""
    def __init__(self, feats, ytd, n):
        self.feats, self.ytd, self.n = feats, ytd, n

    def fit(self, df):
        self.med = df[self.feats].median()
        X = fill(df[self.feats], self.med)
        self.lm = LinearRegression().fit(X, df.y_rel)
        prior = self.lm.predict(X)
        y, n, ytd = df.y_rel.values, df[self.n].values, df[self.ytd].fillna(0).values
        ks = np.unique(np.round(np.geomspace(0.5, 200, 60), 2))
        err = [np.mean((y - self._post(prior, ytd, n, k)) ** 2) for k in ks]
        self.k = ks[int(np.argmin(err))]
        self.curve = pd.Series(err, index=ks)
        return self

    @staticmethod
    def _post(prior, ytd, n, k):
        return (k * prior + n * ytd) / (k + n)

    def predict(self, df):
        prior = self.lm.predict(fill(df[self.feats], self.med))
        return self._post(prior, df[self.ytd].fillna(0).values, df[self.n].values, self.k), prior


FULL = ["eb_o", "eb_d", "eb_od", "te1_out", "wr1_out", "te_lost_frac", "wr_lost_frac", "te_rel_prod",
        "o_tgt_ytd", "o_ay_ytd", "te1_pct_ytd", "te_pct_sum_ytd", "implied", "spread", "home", "lg_drift", "lg_gap"]
NAIVE = ["o_ytd", "d_ytd", "prev_off_share", "prev_def_share", "o_tgt_ytd", "o_ay_ytd", "lg_gap"]


def core(train, test):
    out = pd.DataFrame(index=test.index)
    out["league"] = test.lg_prev.values
    # naive: fixed weights on raw rates (missing -> training median, as in the baseline's rule)
    med = train[NAIVE].median()
    rn = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(fill(train[NAIVE], med), train.y_rel)
    out["naive"] = rn.predict(fill(test[NAIVE], med))
    # empirical Bayes components
    ebo = EB(FEATS_PRIOR_O, "o_ytd", "o_n").fit(train)
    ebd = EB(FEATS_PRIOR_D, "d_ytd", "d_n").fit(train)
    for df in (train, test):
        df["eb_o"], df["prior_o"] = ebo.predict(df)
        df["eb_d"], df["prior_d"] = ebd.predict(df)
        df["eb_od"] = df.eb_o * df.eb_d
    rb = LinearRegression().fit(train[["eb_o", "eb_d", "eb_od", "lg_gap"]], train.y_rel)
    out["eb"] = rb.predict(test[["eb_o", "eb_d", "eb_od", "lg_gap"]])
    medf = train[FULL].median()
    rf = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(fill(train[FULL], medf), train.y_rel)
    out["eb_full"] = rf.predict(fill(test[FULL], medf))
    gb = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=80,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
    gb.fit(train[FULL], train.y_rel)
    out["eb_gbm"] = gb.predict(test[FULL])
    out["eb_blend"] = 0.5 * out.eb_full + 0.5 * out.eb_gbm
    out["w"] = test.o_n.values / (test.o_n.values + ebo.k)
    info = dict(k_off=ebo.k, k_def=ebd.k, ridge_alpha=rf.alpha_, naive_alpha=rn.alpha_,
                coef=dict(zip(FULL, np.round(rf.coef_, 4))))
    return out, info


CAL = ["eb_full", "eb_blend"]


def cal_X(p, w):
    return np.column_stack([p, p * w, w])


def fit_predict(train, test, calibrate=True):
    """Relative-to-league predictions plus a week-aware calibration. The calibration is fit on
    leave-one-season-out predictions inside the training seasons, so it never sees the test season."""
    out, info = core(train.copy(), test.copy())
    if calibrate:
        oof = []
        for s in sorted(train.season.unique()):
            if s < train.season.min() + 2:
                continue  # need at least two earlier... any other seasons; keep the fold large enough
            o, _ = core(train[train.season != s].copy(), train[train.season == s].copy())
            oof.append(o.assign(y_rel=train.loc[train.season == s, "y_rel"].values))
        oof = pd.concat(oof)
        info["cal"] = {}
        for m in CAL:
            lr = LinearRegression().fit(cal_X(oof[m].values, oof.w.values), oof.y_rel)
            out[m + "_cal"] = lr.predict(cal_X(out[m].values, out.w.values))
            info["cal"][m] = np.round(np.r_[lr.intercept_, lr.coef_], 4).tolist()
    for m in [c for c in out.columns if c not in ("league", "w")]:
        out[m] = out[m] + test.lg_prev.values
    return out, info


def metrics(y, p, wk):
    q75 = np.quantile(y, 0.75)
    top = y >= q75
    rec = np.mean(p[top] >= np.quantile(p, 0.75)) if top.any() else np.nan
    hi = y >= np.quantile(y, 0.8)
    return pd.Series({"r": np.corrcoef(y, p)[0, 1], "mae": np.mean(np.abs(y - p)), "rmse": np.sqrt(np.mean((y - p) ** 2)),
                      "hi_rmse": np.sqrt(np.mean((y[hi] - p[hi]) ** 2)), "topq_recall": rec, "pred_sd": np.std(p), "n": len(y)})


MODELS = ["league", "naive", "eb", "eb_full", "eb_gbm", "eb_blend", "eb_full_cal", "eb_blend_cal"]

if __name__ == "__main__":
    played = t[t.played].copy()
    preds, infos = [], []
    for S in list(range(2016, 2027)):
        train = played[played.season < S].copy()
        test = played[played.season == S].copy()
        out, info = fit_predict(train, test)
        preds.append(pd.concat([test[["season", "week", "game_id", "posteam", "defteam", "y", "o_n"]], out], axis=1))
        infos.append(dict(season=S, **{k: v for k, v in info.items() if k not in ("coef", "cal")}, cal=info.get("cal")))
        if S == 2026:
            coef26 = info["coef"]
    P = pd.concat(preds)
    P.to_csv(f"{R}/forward_predictions.csv", index=False)
    print("Learned tuning values per forecast season (fit on earlier seasons only):")
    print(pd.DataFrame(infos).to_string(index=False))
    print("\n2026 ridge coefficients (eb_full):", coef26)

    def table(df):
        return pd.DataFrame({m: metrics(df.y.values, df[m].values, df.week.values) for m in MODELS}).T.round(4)

    hist = P[P.season < 2026]
    print("\n=== Forward seasons 2016-2025 (each trained on earlier seasons) ===")
    print(table(hist).to_string())
    print("\n=== 2026 weeks 1-5 (trained on 2013-2025 only) ===")
    print(table(P[P.season == 2026]).to_string())
    print("\n2026 r by week:")
    print(P[P.season == 2026].groupby("week").apply(lambda g: pd.Series({m: g.y.corr(g[m]) for m in MODELS})).round(3).to_string())
    print("\nForward r by season:")
    print(P.groupby("season").apply(lambda g: pd.Series({m: g.y.corr(g[m]) for m in MODELS})).round(3).to_string())

    # late-season shrinkage diagnostic: projection spread and calibration slope by week bucket
    P["wkb"] = pd.cut(P.week, [0, 4, 8, 12, 18], labels=["1-4", "5-8", "9-12", "13-18"])
    h = P[P.season < 2026]
    sd = h.groupby("wkb").apply(lambda g: pd.Series({m: g[m].std() for m in ["naive", "eb", "eb_full", "eb_blend", "eb_full_cal", "eb_blend_cal"]}))
    sl = h.groupby("wkb").apply(lambda g: pd.Series({m: np.polyfit(g[m], g.y, 1)[0] for m in ["naive", "eb", "eb_full", "eb_blend", "eb_full_cal", "eb_blend_cal"]}))
    rr = h.groupby("wkb").apply(lambda g: pd.Series({m: g.y.corr(g[m]) for m in ["naive", "eb", "eb_full", "eb_blend", "eb_full_cal", "eb_blend_cal"]}))
    print("\nProjection sd by week (2016-2025 forward):\n", sd.round(4).to_string())
    print("\nCalibration slope (actual on projected; 1.0 = right width, >1 = projections too narrow):\n", sl.round(3).to_string())
    print("\nr by week bucket:\n", rr.round(3).to_string())

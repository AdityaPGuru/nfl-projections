"""Full stack for defensive pass YPA allowed. Level 1 = every model family built in this project, each predicting season s from
seasons < s only. Level 2 = a meta-model trained on those forward level-1 predictions from seasons < S. Level 3 = calibration
fitted on the stack's own forward predictions from seasons < S. Nothing is hand-tuned on the test seasons."""
import os, numpy as np, pandas as pd, warnings
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "def_ypa_full_eval.py")).read().split("# ---------------------------------------------------------------- 1. leakage")[0])
import lightgbm as lgb
SMALL = ["eb_def", "eb_off"] + vegas; L1S = list(range(2015, 2026)); HIST["rel"] = HIST.y - HIST.lg_prev
def pruned(tr): r = tr[BASE23].corrwith(tr.y); return list(r[r > 0.03].index)
STYLE = ["rate", "net", "adj_rate", "epa", "succ", "fd", "expl", "td", "cmp", "cpoe", "adot", "deep_share", "deep_cmp", "deep_ypa", "short_ypa", "yac_c", "sack", "hit", "press", "int", "early_rate", "late_rate", "vol"]
VEC = ["ewL_" + s for s in STYLE] + ["o_ewL_" + s for s in STYLE]; KS = [50, 100, 200, 400, 800]

def comps(cols, first=2013):
    """Team-vector KNN: season s predicted from the K most similar matchups of seasons < s; K for s learned on seasons first..s-1."""
    pk = {k: pd.Series(np.nan, index=HIST.index) for k in KS}
    for s in range(first, 2026):
        lib, q = HIST[HIST.season < s], HIST[HIST.season == s]; mu, sd = lib[cols].mean(), lib[cols].std().replace(0, 1); w = np.abs(lib[cols].corrwith(lib.rel).fillna(0).values); w = w / w.mean()
        L, Q = ((lib[cols] - mu) / sd).fillna(0).values * w, ((q[cols] - mu) / sd).fillna(0).values * w
        order = np.argsort((Q ** 2).sum(1)[:, None] + (L ** 2).sum(1)[None, :] - 2 * Q @ L.T, axis=1)[:, :max(KS)]
        for k in KS: pk[k][q.index] = q.lg_prev.values + lib.rel.values[order[:, :min(k, len(lib))]].mean(1)
    out = pd.Series(np.nan, index=HIST.index)
    for s in L1S:
        past = HIST[(HIST.season >= first) & (HIST.season < s)]; kb = min(KS, key=lambda k: rm(past.y, pk[k][past.index])); out[HIST.season == s] = pk[kb][HIST.season == s]
    return out

lgbm = lambda: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=6, min_child_samples=120, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, reg_lambda=20, verbose=-1, random_state=0, n_jobs=4)
imp = lambda m: make_pipeline(SimpleImputer(strategy="median"), m)
CACHE = os.path.join(S, "full_stack_level1.parquet")
L1 = {}
def put(name, s, idx, p): L1.setdefault(name, pd.Series(np.nan, index=HIST.index))[idx] = p
for s in ([] if os.path.exists(CACHE) else L1S):
    tr, te = HIST[HIST.season < s], HIST[HIST.season == s]; tre, _, _ = add_eb(tr, tr); tee, _, _ = add_eb(tr, te); ix = te.index
    FULL = SMALL + dfe + ofe + pruned(tre); DEFO = ["eb_def"] + dfe + pruned(tre)
    put("EB defense level", s, ix, tee.eb_def.values); put("EB defense + offense levels", s, ix, (tee.eb_def + tee.eb_off - tee.lg_prev).values)
    put("Ridge, 7 inputs", s, ix, b.lin(RidgeCV(alphas=np.logspace(0, 5, 25))).fit(tre[SMALL], tre.y).predict(tee[SMALL]))
    put("OLS, 7 inputs", s, ix, b.lin(LinearRegression()).fit(tre[SMALL], tre.y).predict(tee[SMALL]))
    put("Ridge, full (230+ inputs)", s, ix, b.lin(RidgeCV(alphas=np.logspace(0, 5, 25))).fit(tre[FULL], tre.y).predict(tee[FULL]))
    put("Lasso, full", s, ix, b.lin(LassoCV(cv=5, alphas=30)).fit(tre[FULL], tre.y).predict(tee[FULL]))
    put("LightGBM, 7 inputs", s, ix, lgbm().fit(tre[SMALL], tre.y).predict(tee[SMALL])); put("LightGBM, full", s, ix, lgbm().fit(tre[FULL], tre.y).predict(tee[FULL]))
    put("Random forest, full", s, ix, imp(RandomForestRegressor(300, min_samples_leaf=40, max_features=0.3, n_jobs=-1, random_state=0)).fit(tre[FULL], tre.y).predict(tee[FULL]))
    put("SVR, 7 inputs", s, ix, b.lin(SVR(C=1.0, epsilon=0.5)).fit(tre[SMALL], tre.y).predict(tee[SMALL]))
    put("Neural net, 7 inputs", s, ix, b.lin(MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1.0, early_stopping=True, max_iter=500, random_state=0)).fit(tre[SMALL], tre.y).predict(tee[SMALL]))
    put("KNN on features (k=200)", s, ix, b.lin(KNeighborsRegressor(200)).fit(tre[SMALL], tre.y).predict(tee[SMALL]))
    put("Logistic P(above median)", s, ix, b.lin(LogisticRegression(C=0.1, max_iter=2000)).fit(tre[SMALL], (tre.y > tre.y.median()).astype(int)).predict_proba(tee[SMALL])[:, 1])
    put("Defense baseline: your 23, ridge", s, ix, b.lin(RidgeCV(alphas=np.logspace(0, 5, 25))).fit(tre[BASE23], tre.y).predict(tee[BASE23]))
    put("Defense baseline: your 23, OLS no penalty", s, ix, b.lin(LinearRegression()).fit(tre[BASE23], tre.y).predict(tee[BASE23]))
    put("Defense-only, all defensive stats", s, ix, b.lin(RidgeCV(alphas=np.logspace(0, 5, 25))).fit(tre[DEFO], tre.y).predict(tee[DEFO]))
    print("level 1 season", s, flush=True)
if os.path.exists(CACHE): Z = pd.read_parquet(CACHE)
else:
    L1["KNN team-vector comps"] = comps(VEC); L1["KNN team-vector comps + Vegas"] = comps(VEC + ["off_spread", "off_implied", "total_line", "off_home", "indoors"])
    Z = pd.DataFrame(L1); Z.to_parquet(CACHE)
Z.index = HIST.index; names = list(Z.columns); Y = HIST.y

# ---------------------------------------------------------------- level 2: meta-models, trained on forward level-1 predictions of seasons < S
META = {"Stack: simple average": None, "Stack: non-negative weights": "nnls", "Stack: ridge meta": "ridge", "Stack: LightGBM meta": "lgbm"}
M = {k: pd.Series(np.nan, index=HIST.index) for k in META}; W = {}
use = [n for n in names if n != "Logistic P(above median)"]
for S_ in range(2018, 2026):
    a, x = HIST[(HIST.season >= 2015) & (HIST.season < S_)].index, HIST[HIST.season == S_].index
    M["Stack: simple average"][x] = Z.loc[x, use].mean(1)
    A = np.column_stack([np.ones(len(a)), Z.loc[a, use].values - Y[a].mean()]); w, _ = nnls(np.column_stack([A[:, 0], A[:, 1:]]), Y[a].values - 0)   # intercept + non-negative weights
    M["Stack: non-negative weights"][x] = np.column_stack([np.ones(len(x)), Z.loc[x, use].values - Y[a].mean()]) @ w; W[S_] = pd.Series(w[1:], index=use)
    M["Stack: ridge meta"][x] = b.lin(RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(Z.loc[a, names], Y[a]).predict(Z.loc[x, names])
    M["Stack: LightGBM meta"][x] = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.02, num_leaves=4, min_child_samples=150, reg_lambda=20, verbose=-1, random_state=0).fit(Z.loc[a, names].values, Y[a]).predict(Z.loc[x, names].values)
# choose the meta-model for season S by its record on earlier seasons (needs at least one earlier stacked season)
sel = pd.Series(np.nan, index=HIST.index); chosen = {}
for S_ in range(2019, 2026):
    past = HIST[(HIST.season >= 2018) & (HIST.season < S_)].index; kb = min(META, key=lambda k: rm(Y[past], M[k][past])); chosen[S_] = kb; sel[HIST.season == S_] = M[kb][HIST.season == S_]
# ---------------------------------------------------------------- level 3: calibration fitted on the selected stack's forward predictions of seasons < S
cal_lin, cal_iso, maps = pd.Series(np.nan, index=HIST.index), pd.Series(np.nan, index=HIST.index), {}
for S_ in range(2020, 2026):
    past = HIST[(HIST.season >= 2019) & (HIST.season < S_)].index; x = HIST[HIST.season == S_].index; sl, ic = np.polyfit(sel[past], Y[past], 1); maps[S_] = (ic, sl)
    cal_lin[x] = ic + sl * sel[x]; cal_iso[x] = IsotonicRegression(out_of_bounds="clip").fit(sel[past], Y[past]).predict(sel[x])

def line(name, p, yrs):
    d = HIST[HIST.season.isin(yrs)].assign(p=p).dropna(subset=["p"]); wk = d.groupby(["season", "week"]).apply(lambda q: q.y.corr(q.p)); sl = np.polyfit(d.p, d.y, 1)[0]
    return {"model": name, "r": d.y.corr(d.p), "avg weekly r": wk.mean(), "weeks r>0": f"{(wk > 0).mean():.0%}", "per-game L2": rm(d.y, d.p), "sd of proj": d.p.std(), "p5": d.p.quantile(.05), "p95": d.p.quantile(.95),
            "min": d.p.min(), "max": d.p.max(), "slope (>1 = too narrow)": sl}
YR = range(2020, 2026)
print(f"\nFULL STACK, pass YPA allowed. All rows scored on the same forward seasons 2020-2025 ({int(HIST.season.isin(YR).sum()):,} games); guess-the-average L2 = {HIST[HIST.season.isin(YR)].y.std():.3f}\n")
rows = [line(n, Z[n], YR) for n in names if n != "Logistic P(above median)"] + [line(k, v, YR) for k, v in M.items()] + [line("STACK, meta chosen from its past record", sel, YR), line("STACK + linear calibration  <- final", cal_lin, YR), line("STACK + isotonic calibration", cal_iso, YR)]
T = pd.DataFrame(rows).sort_values("avg weekly r", ascending=False); print(T.round(3).to_string(index=False)); T.to_csv(os.path.join(HERE, "results", "full_stack_table.csv"), index=False)
d = HIST[HIST.season.isin(YR)]; print(f"logistic member on its own: AUC-style check, r of P(above median) with actual = {d.y.corr(Z['Logistic P(above median)'][d.index]):.3f}")
print("\nmeta-model chosen per season:", chosen); print("calibration map per season (actual = a + b*stack):", {k: (round(v[0], 2), round(v[1], 3)) for k, v in maps.items()})
print("\nnon-negative stack weights, average over 2018-2025 (zero = the stack found nothing new in that model):"); print(pd.DataFrame(W).mean(1).sort_values(ascending=False).round(3).to_string())
f = HIST[HIST.season.isin(YR)].assign(p=cal_lin); f["b"] = pd.qcut(f.p, 10, labels=False); print("\nfinal model, reliability by decile of projection (2020-2025):"); print(f.groupby("b").agg(games=("y", "size"), projected=("p", "mean"), actual=("y", "mean")).round(2).T.to_string())
t = f[f.season == 2025]; T6 = pd.concat([t[(t.defteam == a_) & (t.week == w_)] for a_, w_ in dm.TRACKED]); print("\n2025 tracked matchups, final model:"); print(T6[["defteam", "week", "posteam", "p", "y"]].rename(columns={"p": "projected", "y": "actual"}).round(2).to_string(index=False))
HIST.assign(stack=sel, final=cal_lin)[["season", "week", "defteam", "posteam", "y", "stack", "final"]].dropna().to_csv(os.path.join(HERE, "results", "full_stack_forward.csv"), index=False)

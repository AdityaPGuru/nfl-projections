"""Best-effort final model for defensive pass YPA allowed: every experiment as a level-1 member (17 cached models + new members that
know the opposing STARTING QB), ridge meta-model, calibration, then a head-to-head against baselines. Strictly forward throughout."""
import os, glob, numpy as np, pandas as pd, warnings
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.isotonic import IsotonicRegression
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "def_ypa_full_eval.py")).read().split("# ---------------------------------------------------------------- 1. leakage")[0])
import lightgbm as lgb
Z = pd.read_parquet(os.path.join(S, "full_stack_level1.parquet")); assert len(Z) == len(HIST); Z.index = HIST.index
SMALL = ["eb_def", "eb_off"] + vegas; rng = np.random.default_rng(0)
def pruned(tr): r = tr[BASE23].corrwith(tr.y); return list(r[r > 0.03].index)

# ---------------------------------------------------------------- opposing starting QB (known at kickoff): first passer of the game for that offense
pb = pd.concat([pd.read_parquet(f, columns=["game_id", "season", "week", "play_id", "posteam", "pass", "pass_attempt", "passing_yards", "epa", "cpoe", "passer_player_id", "passer_player_name"])
                for f in sorted(glob.glob(S + "/pbp/pbp_*.parquet"))]); pb = pb[(pb["pass"] == 1) & pb.passer_player_id.notna() & pb.posteam.notna()]
start = pb.sort_values("play_id").groupby(["game_id", "posteam"], as_index=False).first()[["game_id", "posteam", "passer_player_id", "passer_player_name"]].rename(columns={"passer_player_id": "qb", "passer_player_name": "qb_name"})
qg = pb.groupby(["passer_player_id", "game_id", "season", "week"], as_index=False).agg(yds=("passing_yards", "sum"), att=("pass_attempt", "sum"), epa=("epa", "sum"), db=("pass", "sum"), cpoe_s=("cpoe", "sum"), cpoe_n=("cpoe", "count")).rename(columns={"passer_player_id": "qb"})
qg["t"] = qg.season * 100 + qg.week; qg = qg.sort_values(["qb", "t"]).reset_index(drop=True); qg["gp"] = 1.0
car = b.prior_sums(qg, ["qb"], ["yds", "att", "epa", "db", "cpoe_s", "cpoe_n", "gp"], "c"); ew = b.prior_sums(qg, ["qb"], ["yds", "att", "epa", "db"], "e", halflife=8)
qg = pd.concat([qg, car, ew], axis=1); nz = lambda s_: s_.replace(0, np.nan)
qg["qb_car_att"], qg["qb_car_ypa"], qg["qb_ew_ypa"] = qg.c_att, qg.c_yds / nz(qg.c_att), qg.e_yds / nz(qg.e_att)
qg["qb_ew_epa"], qg["qb_car_cpoe"], qg["qb_starts"] = qg.e_epa / nz(qg.e_db), qg.c_cpoe_s / nz(qg.c_cpoe_n), qg.c_gp
HIST = HIST.merge(start, on=["game_id", "posteam"], how="left").merge(qg[["qb", "game_id", "c_yds", "c_att", "qb_car_att", "qb_car_ypa", "qb_ew_ypa", "qb_ew_epa", "qb_car_cpoe", "qb_starts"]], on=["qb", "game_id"], how="left")
HIST = HIST.sort_values(["posteam", "t"]); HIST["qb_change"] = (HIST.groupby("posteam").qb.shift(1) != HIST.qb).astype(int); HIST = HIST.sort_index(); assert len(HIST) == len(Z)
HIST["qb_log_att"] = np.log1p(HIST.qb_car_att.fillna(0)); QBF = ["qb_eb", "qb_ew_ypa", "qb_ew_epa", "qb_car_cpoe", "qb_log_att", "qb_starts", "qb_change", "qb_minus_team"]

def qb_eb(tr, x):
    """QB shrunk career YPA; k = (luck sd per attempt)^2 / (true QB-to-QB sd)^2, from the training seasons only."""
    d = tr[tr.c_att.notna()]; N = d.groupby("qb").den.sum(); m = d.groupby("qb").num.sum() / N; keep = N >= N.median()
    sig2 = (d.den * (d.y - d.qb.map(m)) ** 2).sum() / max(len(d) - len(N), 1); mu = np.average(m[keep], weights=N[keep]); tau2 = max(np.average((m[keep] - mu) ** 2, weights=N[keep]) - sig2 * np.average(1 / N[keep], weights=N[keep]), 1e-6)
    k = sig2 / tau2; rook = d[d.c_att.fillna(0) < 50]; prior = x.lg_prev + (rook.y.mean() - rook.lg_prev.mean())           # low-experience starters fold to what such starters have done, not to league average
    return (x.c_yds.fillna(0) + k * prior) / (x.c_att.fillna(0) + k), k

lgbm = lambda: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=6, min_child_samples=120, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, reg_lambda=20, verbose=-1, random_state=0, n_jobs=4)
NEW = {}; KQ = {}
def put(n, ix, p): NEW.setdefault(n, pd.Series(np.nan, index=HIST.index))[ix] = p
for s in range(2015, 2026):
    tr, te = HIST[HIST.season < s], HIST[HIST.season == s]; tre, _, _ = add_eb(tr, tr); tee, _, _ = add_eb(tr, te)
    tre["qb_eb"], KQ[s] = qb_eb(tre, tre); tee["qb_eb"], _ = qb_eb(tre, tee)
    for d_ in (tre, tee): d_["qb_minus_team"] = d_.qb_eb - d_.eb_off
    A, FULLQ = SMALL + QBF, SMALL + QBF + dfe + ofe + pruned(tre)
    put("QB shrunk level + defense level", te.index, (tee.qb_eb + tee.eb_def - tee.lg_prev).values)
    put("Ridge, 7 inputs + starting QB", te.index, b.lin(RidgeCV(alphas=np.logspace(0, 5, 25))).fit(tre[A], tre.y).predict(tee[A]))
    put("LightGBM, 7 inputs + starting QB", te.index, lgbm().fit(tre[A], tre.y).predict(tee[A]))
    put("Lasso, full + starting QB", te.index, b.lin(LassoCV(cv=5, alphas=30)).fit(tre[FULLQ], tre.y).predict(tee[FULLQ]))
    print("new level-1 members, season", s, flush=True)
ZN = pd.concat([Z, pd.DataFrame(NEW)], axis=1); Y = HIST.y; oldn, alln = list(Z.columns), list(ZN.columns)

def stack(cols):
    raw, cal = pd.Series(np.nan, index=HIST.index), pd.Series(np.nan, index=HIST.index)
    for S_ in range(2018, 2026):
        a, x = HIST[(HIST.season >= 2015) & (HIST.season < S_)].index, HIST[HIST.season == S_].index; raw[x] = b.lin(RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(ZN.loc[a, cols], Y[a]).predict(ZN.loc[x, cols])
    for S_ in range(2020, 2026):
        past, x = HIST[(HIST.season >= 2018) & (HIST.season < S_)].index, HIST[HIST.season == S_].index; sl, ic = np.polyfit(raw[past], Y[past], 1); cal[x] = ic + sl * raw[x]
    return raw, cal
old_raw, old_cal = stack(oldn); new_raw, new_cal = stack(alln)
F = HIST[HIST.season >= 2020].copy(); F["FINAL"] = new_cal[F.index]; prior_avg = {s: HIST[HIST.season < s].y.mean() for s in range(2020, 2026)}
CAND = {"BASELINE 1: guess the prior seasons' average": F.season.map(prior_avg), "BASELINE 2: defense's trailing YPA allowed (raw)": F.ewL_rate, "BASELINE 3: your 23 defensive variables, ridge": Z["Defense baseline: your 23, ridge"][F.index],
        "BASELINE 4: defense shrunk level (SD-derived k)": Z["EB defense level"][F.index], "Ridge, 7 inputs (previous best single model)": Z["Ridge, 7 inputs"][F.index], "Previous stack (17 models), calibrated": old_cal[F.index],
        "Ridge, 7 inputs + starting QB": ZN["Ridge, 7 inputs + starting QB"][F.index], "Lasso, full + starting QB": ZN["Lasso, full + starting QB"][F.index], "FINAL: stack of all 21 models, calibrated": F.FINAL}
blk = (F.season * 100 + F.week).values; ub = np.unique(blk); rows = []
for n, p in CAND.items():
    d = F.assign(p=p.values if hasattr(p, "values") else p); wk = d.groupby(["season", "week"]).apply(lambda q: pd.Series({"r": q.y.corr(q.p) if q.p.std() > 0 else np.nan, "l2": rm(q.y, q.p), "fin": rm(q.y, q.FINAL)}))
    e = ((d.y - d.p) ** 2 - (d.y - d.FINAL) ** 2).values; bs = [np.concatenate([e[blk == u] for u in rng.choice(ub, len(ub))]).mean() for _ in range(1000)]
    rows.append({"model": n, "r": d.y.corr(d.p) if d.p.std() > 0 else np.nan, "avg weekly r": wk.r.mean(), "per-game L2": rm(d.y, d.p), "sd of proj": d.p.std(), "proj range": f"{d.p.min():.2f}-{d.p.max():.2f}", "slope": np.polyfit(d.p, d.y, 1)[0] if d.p.std() > 0 else np.nan,
                 "weeks FINAL has lower L2": f"{(wk.fin < wk.l2).mean():.0%}", "FINAL's MSE edge": e.mean(), "edge / noise": e.mean() / np.std(bs) if np.std(bs) > 0 else np.nan})
print(f"\nFINAL MODEL vs BASELINES. Defensive pass YPA allowed, strictly forward 2020-2025, {len(F):,} games, {len(ub)} weekly vectors\n"); T = pd.DataFrame(rows); print(T.round(3).to_string(index=False)); T.to_csv(os.path.join(HERE, "results", "best_stack_vs_baselines.csv"), index=False)
print(f"\nQB shrinkage constant learned per season (attempts): {', '.join(f'{s}: {k:.0f}' for s, k in KQ.items() if s >= 2020)}")
ch = F[F.qb_change == 1]; st = F[F.qb_change == 0]; rk = F[F.qb_car_att.fillna(0) < 150]
for lab, d in (("games where the offense changed starting QB from its previous game", ch), ("starter with under 150 career attempts", rk), ("same starter as last game", st)):
    print(f"   {lab}: {len(d):,} games | actual {d.y.mean():.2f} | L2 ridge-7 {rm(d.y, Z['Ridge, 7 inputs'][d.index]):.3f} -> previous stack {rm(d.y, old_cal[d.index]):.3f} -> FINAL {rm(d.y, d.FINAL):.3f} | bias ridge-7 {(d.y - Z['Ridge, 7 inputs'][d.index]).mean():+.2f}, FINAL {(d.y - d.FINAL).mean():+.2f}")
w = b.lin(RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(ZN.loc[HIST.season.between(2015, 2025), alln], Y[HIST.season.between(2015, 2025)]); co = pd.Series(w[-1].coef_, index=alln).sort_values(key=abs, ascending=False)
print("\nmeta-model weights (standardised), top 8:", ", ".join(f"{k} {v:+.3f}" for k, v in co.head(8).items()))
F["b"] = pd.qcut(F.FINAL, 10, labels=False); print("\nFINAL, reliability by decile:"); print(F.groupby("b").agg(projected=("FINAL", "mean"), actual=("y", "mean")).round(2).T.to_string())
yr = F.groupby("season").apply(lambda d: pd.Series({"r FINAL": d.y.corr(d.FINAL), "L2 FINAL": rm(d.y, d.FINAL), "L2 your 23 baseline": rm(d.y, Z["Defense baseline: your 23, ridge"][d.index]), "L2 guess average": rm(d.y, prior_avg[d.name])})); print("\nby season:"); print(yr.round(3).to_string())
t = F[F.season == 2025]; T6 = pd.concat([t[(t.defteam == a_) & (t.week == w_)] for a_, w_ in dm.TRACKED]).assign(base23=lambda q: Z["Defense baseline: your 23, ridge"][q.index])
print("\n2025 tracked matchups:"); print(T6[["defteam", "week", "posteam", "qb_name", "y", "FINAL", "base23"]].rename(columns={"y": "actual", "base23": "your-23 baseline"}).round(2).to_string(index=False))
HIST.assign(final=new_cal, stack=new_raw)[["season", "week", "defteam", "posteam", "qb_name", "y", "stack", "final"]].dropna().to_csv(os.path.join(HERE, "results", "full_stack_forward.csv"), index=False)

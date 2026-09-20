"""1) Check the user's 22 offense / 23 defense variables and their train-on-one-week protocol (team level).
2) Do they (and cleaner play-by-play + Vegas features) improve the QB models?  3) RMSE floor + multi-game windows."""
import os, sys, glob, warnings, numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, RidgeCV
import lightgbm as lgb
import bakeoff as b, fantasy_bet as fb
warnings.filterwarnings("ignore")
S = os.environ["S"]
F45 = pd.read_parquet(S + "/pbp_team_week_45.parquet")
OFF = [c for c in F45 if c.startswith("off_")]; DEF = [c for c in F45 if c.startswith("def_")]
rmse = lambda y, p: float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def team_games():
    pb = pd.concat([pd.read_parquet(f, columns=["game_id", "season", "week", "season_type", "posteam", "defteam", "pass_attempt", "passing_yards", "penalty", "pass", "rush"])
                    for f in sorted(glob.glob(S + "/pbp/pbp_*.parquet"))])
    pb = pb[(pb.season_type == "REG") & pb.posteam.notna() & ((pb["pass"] == 1) | (pb.rush == 1)) & ((pb.penalty == 0) | pb.penalty.isna()) & (pb.pass_attempt == 1)]
    g = pb.groupby(["game_id", "season", "week", "posteam", "defteam"], as_index=False).agg(yds=("passing_yards", "sum"), att=("pass_attempt", "sum"))
    g["ypa"] = g.yds.fillna(0) / g.att
    g = g[g.season >= 2012]
    g = g.merge(F45[["team", "season", "week"] + OFF], left_on=["posteam", "season", "week"], right_on=["team", "season", "week"], how="left").drop(columns="team")
    g = g.merge(F45[["team", "season", "week"] + DEF], left_on=["defteam", "season", "week"], right_on=["team", "season", "week"], how="left").drop(columns="team")
    # trailing YPA baselines over the same window the 45 variables use (prior season + season to date)
    g = g.sort_values(["season", "week"]).reset_index(drop=True); g["t"] = g.season * 100 + g.week
    for key, nm in (("posteam", "off_trail_ypa"), ("defteam", "def_trail_ypa")):
        v = np.full(len(g), np.nan)
        for team, d in g.groupby(key):
            for i, (idx, r) in enumerate(d.iterrows()):
                p = d[(d.t < r.t) & (d.t >= (r.season - 1) * 100)]
                if p.att.sum() > 0: v[idx] = p.yds.sum() / p.att.sum()
        g[nm] = v
    return g


def part1():
    g = team_games()
    print(f"\n{'=' * 100}\nPART 1. TEAM LEVEL, your definition of YPA. {len(g):,} team-games, 2012-2025 regular season; mean {g.ypa.mean():.2f}, sd {g.ypa.std():.2f}")
    print("\n1a. Each variable alone vs the NEXT game's YPA (offense vars vs own YPA, defense vars vs YPA allowed). n~7,400 so |r| < 0.03 is indistinguishable from zero")
    rows = [{"variable": c, "r": g[c].corr(g.ypa)} for c in OFF + DEF]
    rows += [{"variable": "** plain trailing offense YPA **", "r": g.off_trail_ypa.corr(g.ypa)}, {"variable": "** plain trailing defense YPA allowed **", "r": g.def_trail_ypa.corr(g.ypa)}]
    t = pd.DataFrame(rows).assign(abs_r=lambda d: d.r.abs()).sort_values("abs_r", ascending=False).drop(columns="abs_r")
    print(t.round(3).to_string(index=False))
    n_sig = int((t[~t.variable.str.startswith("**")].r.abs() > 0.03).sum()); print(f"variables with |r| > 0.03: {n_sig} of {len(OFF) + len(DEF)}")

    print("\n1b. Your protocol replayed on every consecutive week pair 2012-2025: fit lm on week w (32 rows), predict week w+1")
    res = []
    gg = g.copy(); gg[OFF + DEF] = gg[OFF + DEF].fillna(0)
    for s in range(2012, 2026):
        for wk in range(1, 18):
            a, c = gg[(gg.season == s) & (gg.week == wk)], gg[(gg.season == s) & (gg.week == wk + 1)]
            if len(a) < 26 or len(c) < 20: continue
            po = LinearRegression().fit(a[OFF], a.ypa).predict(c[OFF]); pdv = LinearRegression().fit(a[DEF], a.ypa).predict(c[DEF])
            res.append(pd.DataFrame({"y": c.ypa.values, "off": po, "def": pdv, "mean": (po + pdv) / 2, "in_r2_off": LinearRegression().fit(a[OFF], a.ypa).score(a[OFF], a.ypa),
                                     "lg": a.ypa.mean(), "trail": ((c.off_trail_ypa + c.def_trail_ypa) / 2).fillna(a.ypa.mean()).values, "s": s, "w": wk + 1}))
    r = pd.concat(res)
    print(f"week pairs: {r.groupby(['s', 'w']).ngroups}, predictions: {len(r):,}; average IN-SAMPLE R^2 of the week-w offense fit: {r.in_r2_off.mean():.2f}")
    tab = [{"method": lab, "r with actual": pearsonr(r.y, r[c])[0], "RMSE": rmse(r.y, r[c]), "weeks with r>0": f"{int((r.groupby(['s', 'w']).apply(lambda d: d.y.corr(d[c])) > 0).mean() * 100)}%"}
           for lab, c in [("your offense lm (22 vars, 1 week)", "off"), ("your defense lm (23 vars, 1 week)", "def"), ("your matchup mean", "mean"),
                          ("plain trailing (off YPA + def YPA allowed)/2", "trail")]]
    tab.append({"method": "just predict last week's league average", "r with actual": np.nan, "RMSE": rmse(r.y, r.lg), "weeks with r>0": ""})
    print(pd.DataFrame(tab).round(3).to_string(index=False))

    print("\n1c. SAME 45 variables, trained on all seasons instead of one week (5 season-scattered folds, ridge / LightGBM)")
    g2 = g.dropna(subset=["off_trail_ypa"]).reset_index(drop=True); fold = fb.scattered_folds(g2)
    sets = {"your 45 variables": OFF + DEF, "2 plain trailing YPAs": ["off_trail_ypa", "def_trail_ypa"], "both": OFF + DEF + ["off_trail_ypa", "def_trail_ypa"]}
    out = []
    for nm, cols in sets.items():
        pr, pl = np.zeros(len(g2)), np.zeros(len(g2))
        for k in range(fb.K):
            a, c = fold != k, fold == k
            pr[c] = b.lin(RidgeCV(alphas=np.logspace(-1, 4, 20))).fit(g2.loc[a, cols], g2.ypa[a]).predict(g2.loc[c, cols])
            pl[c] = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.02, num_leaves=6, min_child_samples=100, colsample_bytree=0.6, reg_lambda=10, verbose=-1).fit(g2.loc[a, cols], g2.ypa[a]).predict(g2.loc[c, cols])
        out.append({"features": nm, "ridge r": pearsonr(g2.ypa, pr)[0], "ridge RMSE": rmse(g2.ypa, pr), "LightGBM r": pearsonr(g2.ypa, pl)[0], "LightGBM RMSE": rmse(g2.ypa, pl)})
    print(pd.DataFrame(out).round(3).to_string(index=False)); print(f"(RMSE of always predicting the mean: {g2.ypa.std():.3f})")


def add_pbp(df, extra_qb=None, extra_def=None, extra_ctx=None):
    """Clean, lagged play-by-play features + pre-game Vegas/venue context, merged onto a QB-game frame."""
    qb = pd.read_parquet(S + "/pbp_qb_game.parquet"); dfn = pd.read_parquet(S + "/pbp_def_game.parquet"); ctx = pd.read_parquet(S + "/pbp_game_ctx.parquet")
    if extra_qb is not None:
        qb, dfn = pd.concat([qb, extra_qb], ignore_index=True), pd.concat([dfn, extra_def], ignore_index=True)
        ctx = pd.concat([ctx[~ctx.game_id.isin(extra_ctx.game_id)], extra_ctx], ignore_index=True)
    p = qb.game_id.str.split("_", expand=True); qb["t"] = p[0].astype(int) * 100 + p[1].astype(int)
    qb = qb.sort_values(["player_id", "t"]).reset_index(drop=True)
    cols = ["db", "epa_sum", "cpoe_sum", "cpoe_n", "succ", "press"]
    car, ew = b.prior_sums(qb, ["player_id"], cols, "car"), b.prior_sums(qb, ["player_id"], cols, "ew", halflife=8)
    for pre, d in (("car", car), ("ew", ew)):
        qb[f"{pre}_epa_db"] = d[f"{pre}_epa_sum"] / fb.nz(d[f"{pre}_db"]); qb[f"{pre}_cpoe"] = d[f"{pre}_cpoe_sum"] / fb.nz(d[f"{pre}_cpoe_n"])
        qb[f"{pre}_succ"] = d[f"{pre}_succ"] / fb.nz(d[f"{pre}_db"]); qb[f"{pre}_press"] = d[f"{pre}_press"] / fb.nz(d[f"{pre}_db"])
    qf = [f"{pre}_{x}" for pre in ("car", "ew") for x in ("epa_db", "cpoe", "succ", "press")]
    df = df.merge(qb[["player_id", "game_id"] + qf], on=["player_id", "game_id"], how="left")
    dfn["t"] = dfn.season * 100 + dfn.week; names = []
    for nm, num, den in (("depa", "epa_sum", "db"), ("dcpoe", "cpoe_sum", "cpoe_n"), ("dpress", "press", "db")):
        d = b.defense_table(dfn, num, den, dfn.db > 0).rename(columns=lambda c: c.replace("def_", nm + "_")).drop(columns=f"{nm}_s2d_den")
        df = df.merge(d, on=["opp", "t"], how="left"); names += [f"{nm}_s2d_rate", f"{nm}_prev_rate"]
    df = df.merge(ctx, on="game_id", how="left")
    sgn = np.where(df.team == df.home_team, 1, -1)
    df["team_spread"] = sgn * df.spread_line; df["implied_total"] = df.total_line / 2 + df.team_spread / 2; df["opp_implied"] = df.total_line - df.implied_total
    df["indoors"] = df.roof.isin(["dome", "closed"]).astype(int); df["wind_f"] = np.where(df.indoors == 1, 0, df.wind)
    vegas = ["team_spread", "implied_total", "opp_implied", "total_line", "indoors", "wind_f", "temp"]
    df = df.merge(F45[["team", "season", "week"] + OFF], on=["team", "season", "week"], how="left")
    df = df.merge(F45[["team", "season", "week"] + DEF].rename(columns={"team": "opp"}), on=["opp", "season", "week"], how="left")
    return df, qf + names, vegas


def cv_models(df, feats, fold):
    pr, pl = np.zeros(len(df)), np.zeros(len(df))
    for k in range(fb.K):
        a, c = fold != k, fold == k
        tr, te = df[a].copy(), df[c].copy(); kk = b.eb_k(tr); tr["eb"], te["eb"] = b.add_eb(tr, kk), b.add_eb(te, kk); f = feats + ["eb"]
        pr[c] = b.lin(RidgeCV(alphas=np.logspace(-1, 4, 20))).fit(tr[f], tr.y).predict(te[f])
        pl[c] = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=8, min_child_samples=80, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.6, reg_lambda=10, verbose=-1, random_state=0).fit(tr[f], tr.y).predict(te[f])
    return {"EB+Ridge": pr, "EB+LightGBM": pl, "Blend": (pr + pl) / 2}


def part2():
    w = b.load(); keep = {}
    for tgt, label in (("ypa", "QB yards per attempt"), ("pass_yds", "QB passing yards"), ("fp", "QB fantasy points (PPR)")):
        df, base = fb.build_qb(w, tgt); df, pbpf, vegas = add_pbp(df); fold = fb.scattered_folds(df)
        sets = {"A. my previous features": base, "B. your 45 variables only": OFF + DEF, "C. A + your 45": base + OFF + DEF,
                "D. A + clean QB/defense EPA, CPOE, pressure": base + pbpf, "E. D + Vegas spread/total, roof, wind, temp": base + pbpf + vegas,
                "F. E + your 45": base + pbpf + vegas + OFF + DEF}
        print(f"\n{'=' * 100}\nPART 2. {label}: {len(df):,} QB-games, 5 season-scattered folds. RMSE of predicting the mean = {df.y.std():.3f}")
        rows = []
        for nm, f in sets.items():
            P = cv_models(df, f, fold); keep[(tgt, nm)] = (df, P, fold)
            rows.append({"feature set": nm, **{f"{m} r": pearsonr(df.y, p)[0] for m, p in P.items()}, **{f"{m} RMSE": rmse(df.y, p) for m, p in P.items()}})
        print(pd.DataFrame(rows).round(3).to_string(index=False))
    return keep


def part3(keep):
    df, P, fold = keep[("ypa", "E. D + Vegas spread/total, roof, wind, temp")]
    p, y = P["Blend"], df.y.values; cal = np.zeros(len(df))
    for k in range(fb.K):
        sl, ic = np.polyfit(p[fold != k], y[fold != k], 1); cal[fold == k] = ic + sl * p[fold == k]
    sl, ic = np.polyfit(p, y, 1); sl2, ic2 = np.polyfit(cal, y, 1)
    print(f"\n{'=' * 100}\nPART 3. QB YPA, best feature set, blend. Calibration (cross-fitted on out-of-fold predictions): raw slope {sl:.3f}, intercept {ic:.3f} -> "
          f"slope {sl2:.3f}, intercept {ic2:.3f}; map actual = {ic:.3f} + {sl:.3f}*pred; RMSE {rmse(y, p):.3f} -> {rmse(y, cal):.3f}")
    pb = pd.concat([pd.read_parquet(f, columns=["pass_attempt", "sack", "passing_yards", "season_type"]) for f in sorted(glob.glob(S + "/pbp/pbp_*.parquet"))])
    v = pb[(pb.pass_attempt == 1) & (pb.sack != 1)].passing_yards.fillna(0).var()
    floor = float(np.sqrt(np.mean(v / df.pass_attempts)))
    print(f"single pass attempt: sd of yards gained = {np.sqrt(v):.1f}. With {df.pass_attempts.median():.0f} attempts a game, pure luck alone gives a single-game YPA error of {floor:.2f}")
    print(f"  -> RMSE now {rmse(y, cal):.3f}; RMSE if we knew every QB's true skill exactly ~ {floor:.2f}; RMSE of guessing the average {df.y.std():.3f}")
    print(f"  -> share of the REMOVABLE error already removed: {(df.y.var() - rmse(y, cal) ** 2) / (df.y.var() - floor ** 2):.0%}")
    print("\nMulti-game windows (your original 'predict N games, then see what happened'): forecast made before the first game, scored on the N-game total")
    d = df.assign(pred=cal).sort_values(["player_id", "season", "t"]); out = []
    for N in (1, 2, 4, 8, 16):
        d["blk"] = d.groupby(["player_id", "season"]).cumcount() // N
        a = d.groupby(["player_id", "season", "blk"]).agg(n=("y", "size"), yds=("passing_yards", "sum"), att=("pass_attempts", "sum"), pred=("pred", "first"), s2d=("s2d_ypa", "first"))
        a = a[a.n == N]; a["act"] = a.yds / a.att
        out.append({"games in window": N, "windows": len(a), "r": pearsonr(a.act, a.pred)[0], "RMSE": rmse(a.act, a.pred), "RMSE guessing the average": a.act.std(),
                    "luck-only floor": float(np.sqrt(np.mean(v / a.att)))})
    print(pd.DataFrame(out).round(3).to_string(index=False))


if __name__ == "__main__":
    part1()
    part3(part2())

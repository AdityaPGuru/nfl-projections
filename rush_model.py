"""Rushing YPC allowed, matchup model: defense (SD-derived shrinkage, own prior) + opposing run game (same method) + Vegas.
Week-ahead, season-scattered folds, strict 2025 forward test, weekly vector evaluation, 2026 week 2 projections."""
import os, numpy as np, pandas as pd, warnings
from scipy.stats import pearsonr
import bakeoff as b, fantasy_bet as fb, def_model as dm, week2 as w2
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250); S = os.environ["S"]
slate = w2.espn_slate(); fut = slate[~slate.final]
BASE = dm.plays()
ph = pd.concat([fut.assign(posteam=fut.home, defteam=fut.away), fut.assign(posteam=fut.away, defteam=fut.home)])[["game_id", "posteam", "defteam"]]
ph = ph.assign(season=w2.SEASON, week=w2.WEEK, season_type="REG", rush=1.0, rush_attempt=1.0, qb_kneel=0.0, qb_scramble=0.0, rushing_yards=0.0, epa=0.0,
               success=0.0, first_down=0.0, penalty=0.0, neutral=False)   # placeholder so an unplayed game gets a feature row; its own stats never enter its features
ALL = pd.concat([BASE, ph], ignore_index=True); ALL["pass"] = ALL["pass"].fillna(0)
FUT_IDS = set(fut.game_id)


def side(swap):
    p = ALL.copy()
    if swap: p[["posteam", "defteam"]] = p[["defteam", "posteam"]].values
    dm.plays = lambda: p
    g, feats, _ = dm.build("rush"); g = g.sort_values(["defteam", "t"]).reset_index(drop=True)
    s = b.prior_sums(g, ["defteam", "season"], ["num", "den"], "s2d"); g["s2d_num"], g["s2d_den"] = s.s2d_num, s.s2d_den
    real = g[~g.game_id.isin(FUT_IDS)]
    yr = real.groupby(["defteam", "season"], as_index=False)[["num", "den"]].sum().rename(columns={"num": "prev_num", "den": "prev_den"}); yr["season"] += 1
    return g.merge(yr, on=["defteam", "season"], how="left"), feats


def sd_parts(tr):
    u = tr.defteam + tr.season.astype(str); N = tr.den.groupby(u).sum(); m = tr.num.groupby(u).sum() / N
    sigma2 = (tr.den * (tr.y - u.map(m)) ** 2).sum() / (len(tr) - len(N)); mu = tr.num.sum() / tr.den.sum()
    tau2 = max(np.average((m - mu) ** 2, weights=N) - sigma2 * np.average(1 / N, weights=N), 1e-9)
    sm = pd.DataFrame({"m": m, "n": N, "team": tr.defteam.groupby(u).first(), "season": tr.season.groupby(u).first()})
    nx = sm.assign(season=sm.season + 1).merge(sm, on=["team", "season"], suffixes=("_p", ""))
    rel = tau2 / (tau2 + sigma2 / sm.n.mean()); rho = float(np.clip(nx.m_p.corr(nx.m) / rel, 0, 1))
    return np.sqrt(sigma2), np.sqrt(tau2), rho


def eb(x, tr):
    sg, ta, rho = sd_parts(tr); k = sg ** 2 / ta ** 2; lg = x.lg_prev
    prior = lg + rho * ((x.prev_num.fillna(0) + k * lg) / (x.prev_den.fillna(0) + k) - lg)
    return (x.s2d_num + k * prior) / (x.s2d_den + k), (sg, ta, k, rho)


D, dfe = side(False); O, ofe = side(True)
OC = list(dict.fromkeys(ofe + ["s2d_num", "s2d_den", "prev_num", "prev_den", "lg_prev", "num", "den", "y"]))
O = O.rename(columns={c: "o_" + c for c in OC}).rename(columns={"defteam": "off_team"})
M = D.merge(O[["game_id", "off_team"] + ["o_" + c for c in OC]],
            left_on=["game_id", "posteam"], right_on=["game_id", "off_team"], how="inner")
ctx = pd.read_parquet(S + "/pbp_game_ctx.parquet"); ctx = pd.concat([ctx[~ctx.game_id.isin(FUT_IDS)], fut[ctx.columns]])
M = M.merge(ctx, on="game_id", how="left"); sgn = np.where(M.posteam == M.home_team, 1, -1)
M["off_spread"] = sgn * M.spread_line; M["off_implied"] = M.total_line / 2 + M.off_spread / 2
M["indoors"] = M.roof.isin(["dome", "closed"]).astype(int); M["wind_f"] = np.where(M.indoors == 1, 0, M.wind); M["off_home"] = (sgn == 1) * 1
M["future"] = M.game_id.isin(FUT_IDS)
ofe = ["o_" + c for c in ofe if c not in ("lg_prev", "week")]; vegas = ["off_spread", "off_implied", "total_line", "indoors", "wind_f", "off_home"]
H = M[~M.future & (M.season <= 2025)].reset_index(drop=True); fold = fb.scattered_folds(H)


def offview(x):  # the offense table expressed in the columns sd_parts/eb expect
    return pd.DataFrame({"defteam": x.off_team, "season": x.season, "num": x.o_num, "den": x.o_den, "y": x.o_y, "s2d_num": x.o_s2d_num, "s2d_den": x.o_s2d_den,
                         "prev_num": x.o_prev_num, "prev_den": x.o_prev_den, "lg_prev": x.o_lg_prev}, index=x.index)


def add_eb(tr, x):
    x = x.copy(); x["eb_def"], pd_ = eb(x, tr); x["eb_off"], po_ = eb(offview(x), offview(tr)); return x, pd_, po_


SETS = {"1. defense shrunk level only": ["eb_def"], "2. + opposing run game shrunk level": ["eb_def", "eb_off"],
        "3. + Vegas spread/total, home, roof, wind": ["eb_def", "eb_off"] + vegas,
        "4. + all defensive rushing stats": ["eb_def", "eb_off"] + vegas + dfe,
        "5. + all offensive rushing stats (full model)": ["eb_def", "eb_off"] + vegas + dfe + ofe}
print(f"\n{'=' * 112}\nRUSHING YPC ALLOWED, MATCHUP MODEL. {len(H):,} defense-games 2012-2025, designed runs, mean {H.y.mean():.2f}, sd {H.y.std():.2f}; 5 season-scattered folds")
_, pdv, pov = add_eb(H, H)
print(f"shrinkage from standard deviations - defense: luck sd/carry {pdv[0]:.2f}, true sd {pdv[1]:.2f}, k = {pdv[2]:.0f} carries, carry-over {pdv[3]:.2f} | "
      f"offense: luck sd/carry {pov[0]:.2f}, true sd {pov[1]:.2f}, k = {pov[2]:.0f} carries, carry-over {pov[3]:.2f}")
rows, keep = [], {}
tr25, te25 = H[H.season < 2025], H[H.season == 2025]; tr25e, _, _ = add_eb(tr25, tr25); te25e, _, _ = add_eb(tr25, te25)
for lab, cols in SETS.items():
    P = {m: np.zeros(len(H)) for m in ("ridge", "lgbm", "blend")}
    for k in range(fb.K):
        a, x = H[fold != k], H[fold == k]; ae, _, _ = add_eb(a, a); xe, _, _ = add_eb(a, x); out = dm.fitpred(ae, xe, cols)
        for m in P: P[m][fold == k] = out[m]
    best = min(P, key=lambda m: dm.rmse(H.y, P[m])); fw = dm.fitpred(tr25e, te25e, cols)[best]; sl, ic = np.polyfit(P[best], H.y, 1); keep[lab] = (P[best], fw, best, cols)
    wk = te25.assign(p=fw).groupby("week").apply(lambda d: d.y.corr(d.p))
    rows.append({"feature set": lab, "model": best, "OOF r": pearsonr(H.y, P[best])[0], "OOF RMSE": dm.rmse(H.y, P[best]), "sd of projections": P[best].std(), "slope": sl,
                 "2025 fwd r": pearsonr(te25.y, fw)[0], "2025 RMSE": dm.rmse(te25.y, fw), "2025 avg weekly r": wk.mean(), "weeks r>0": f"{(wk > 0).sum()}/18"})
print(pd.DataFrame(rows).round(3).to_string(index=False)); print(f"(RMSE of guessing the mean: all seasons {H.y.std():.3f}, 2025 {te25.y.std():.3f}; defense-only ML model earlier: r 0.119)")
lab = min(rows, key=lambda r: r["OOF RMSE"])["feature set"]; oof, fw, m, cols = keep[lab]; sl, ic = np.polyfit(oof, H.y, 1)
print(f"\nBEST: '{lab}' ({m}); out-of-fold calibration: actual = {ic:.3f} + {sl:.3f} * pred")
t = te25.assign(proj=ic + sl * fw); wk = t.groupby("week").apply(lambda d: pd.Series({"r": d.y.corr(d.proj), "rmse": dm.rmse(d.y, d.proj), "rmse_avg": d.y.std(ddof=0)}))
print("2025 WEEKLY VECTOR EVALUATION (that week's projections vs that week's actuals):"); print(wk.round(2).T.to_string())
q = pd.qcut(t.proj, 5, labels=["toughest 20%", "20-40%", "40-60%", "60-80%", "softest 20%"])
print("2025 by projection bucket:"); print(t.groupby(q, observed=True).agg(games=("y", "size"), projected=("proj", "mean"), actual=("y", "mean")).round(2).T.to_string())
T = pd.concat([t[(t.defteam == a) & (t.week == w)] for a, w in dm.TRACKED])
print("tracked matchups (spot check):"); print(T[["defteam", "week", "posteam", "proj", "y"]].rename(columns={"proj": "projected", "y": "actual"}).round(2).to_string(index=False))
live_tr = M[~M.future].reset_index(drop=True); lt, _, _ = add_eb(live_tr, live_tr); F, _, _ = add_eb(live_tr, M[M.future]); F["proj"] = ic + sl * dm.fitpred(lt, F, cols)[m]
print(f"\n{'=' * 112}\n2026 WEEK 2 - PROJECTED RUSHING YPC ALLOWED (designed runs; league average ~{H[H.season == 2025].y.mean():.2f})")
o = F.sort_values("proj")[["defteam", "posteam", "proj", "eb_def", "eb_off", "off_spread"]]; o.columns = ["Defense", "vs Offense", "Proj YPC allowed", "Defense shrunk level", "Opp run game shrunk level", "Opp spread (+ = favored)"]
print(o.round(2).to_string(index=False)); o.to_csv(S + "/week2_rush_ypc.csv", index=False); t[["week", "defteam", "posteam", "proj", "y"]].to_csv(S + "/rush_matchup_2025.csv", index=False)

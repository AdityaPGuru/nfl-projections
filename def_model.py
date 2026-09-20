"""Defense-only, week-ahead models: passing YPA allowed and rushing YPC allowed. Every feature is built from the defense's
own PRIOR games (opponent strength is used only to adjust those past results). Usage: python def_model.py pass|rush"""
import os, sys, glob, warnings, numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
import lightgbm as lgb
import bakeoff as b, fantasy_bet as fb
warnings.filterwarnings("ignore"); S = os.environ["S"]; pd.set_option("display.width", 250)
rmse = lambda y, p: float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))
TRACKED = [("LV", 12), ("DAL", 15), ("CHI", 2), ("NE", 2), ("SEA", 18), ("MIN", 13)]
COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "pass", "rush", "pass_attempt", "sack", "qb_hit", "qb_scramble", "complete_pass",
        "passing_yards", "air_yards", "yards_after_catch", "epa", "success", "cpoe", "interception", "pass_touchdown", "rush_touchdown", "down", "ydstogo", "wp",
        "half_seconds_remaining", "yards_gained", "rush_attempt", "rushing_yards", "qb_kneel", "first_down", "penalty", "run_location", "run_gap", "tackled_for_loss", "score_differential"]


def plays():
    p = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob(S + "/pbp/pbp_*.parquet"))], ignore_index=True)
    p = p[(p.season_type == "REG") & p.posteam.notna() & p.defteam.notna() & ((p["pass"] == 1) | (p.rush == 1)) & ((p.penalty == 0) | p.penalty.isna())].copy()
    for c in ("epa", "success", "air_yards", "yards_after_catch", "passing_yards", "rushing_yards", "yards_gained", "first_down"): p[c] = p[c].astype(float).fillna(0)
    p["neutral"] = (p.wp.fillna(.5).between(.2, .8)) & (p.half_seconds_remaining > 120)
    return p


def game_sums(p, kind):
    """One row per (defense, game) with raw sums. `den`/`num` define the target rate."""
    if kind == "pass":
        d = p[p.pass_attempt == 1].copy(); ns = d.sack != 1      # user's definition: attempts include sacks, yards = passing yards
        d["num"], d["den"] = d.passing_yards, 1.0
        d["att_ns"], d["comp"], d["air"], d["yac"] = ns * 1.0, d.complete_pass.fillna(0), d.air_yards * ns, d.yards_after_catch
        d["cpoe_sum"], d["cpoe_n"] = d.cpoe.fillna(0), d.cpoe.notna() * 1.0
        d["sacks"], d["hits"], d["press"] = (d.sack == 1) * 1.0, (d.qb_hit == 1) * 1.0, ((d.sack == 1) | (d.qb_hit == 1)) * 1.0
        d["ints"], d["tds"], d["net_yds"] = d.interception.fillna(0), d.pass_touchdown.fillna(0), d.yards_gained
        d["expl"], d["deep_att"] = (d.passing_yards >= 20) * 1.0, (ns & (d.air_yards >= 15)) * 1.0
        d["deep_comp"], d["deep_yds"] = d.deep_att * d.comp, d.deep_att * d.passing_yards
        d["short_att"] = (ns & (d.air_yards < 15)) * 1.0; d["short_yds"] = d.short_att * d.passing_yards
        extra = ["att_ns", "comp", "air", "yac", "cpoe_sum", "cpoe_n", "sacks", "hits", "press", "ints", "tds", "net_yds", "expl", "deep_att", "deep_comp", "deep_yds", "short_att", "short_yds"]
    else:
        d = p[(p.rush_attempt == 1) & (p.qb_kneel != 1) & (p.qb_scramble != 1)].copy()   # designed runs only
        d["num"], d["den"] = d.rushing_yards, 1.0
        d["stuff"], d["expl"], d["tds"] = (d.rushing_yards <= 0) * 1.0, (d.rushing_yards >= 10) * 1.0, d.rush_touchdown.fillna(0)
        d["capped"] = d.rushing_yards.clip(upper=15)                      # YPC with long runs capped: far more stable than raw YPC
        ins = d.run_location.eq("middle") | d.run_gap.eq("guard"); d["in_att"], d["out_att"] = ins * 1.0, (~ins) * 1.0
        d["in_yds"], d["out_yds"] = d.in_att * d.rushing_yards, d.out_att * d.rushing_yards
        sy = (d.ydstogo <= 2) & d.down.isin([3, 4]); d["sy_att"], d["sy_conv"] = sy * 1.0, sy * d.first_down
        extra = ["stuff", "expl", "tds", "capped", "in_att", "out_att", "in_yds", "out_yds", "sy_att", "sy_conv"]
    d["epa_s"], d["succ"], d["fd"] = d.epa, d.success, d.first_down
    for tag, m in (("neu", d.neutral), ("early", d.down.isin([1, 2])), ("late", d.down.isin([3, 4]))):
        d[f"{tag}_num"], d[f"{tag}_den"] = d.num * m, m * 1.0
    sums = ["num", "den", "epa_s", "succ", "fd", "neu_num", "neu_den", "early_num", "early_den", "late_num", "late_den"] + extra
    g = d.groupby(["game_id", "season", "week", "posteam", "defteam"], as_index=False)[sums].sum()
    g["t"] = g.season * 100 + g.week; g["gp"] = 1.0
    return g[g.season >= 2011].sort_values(["defteam", "t"]).reset_index(drop=True), sums


def build(kind):
    g, sums = game_sums(plays(), kind)
    # opponent adjustment: what did the OFFENSE usually do coming into that game? (lagged, exponentially weighted)
    o = g.sort_values(["posteam", "t"]).reset_index(drop=True)
    e = b.prior_sums(o, ["posteam"], ["num", "den", "epa_s"], "o", halflife=8)
    o["exp_rate"], o["exp_epa"] = e.o_num / e.o_den, e.o_epa_s / e.o_den
    g = g.merge(o[["game_id", "posteam", "exp_rate", "exp_epa"]], on=["game_id", "posteam"]).sort_values(["defteam", "t"]).reset_index(drop=True)
    lg = g.num.sum() / g.den.sum()
    g["over_exp"] = g.num - g.exp_rate.fillna(lg) * g.den                 # yards allowed above what that offense normally gains
    g["epa_over_exp"] = g.epa_s - g.exp_epa.fillna(0) * g.den
    sums2 = sums + ["over_exp", "epa_over_exp", "gp"]
    W = {"ewS": dict(halflife=3), "ewL": dict(halflife=10), "l4": dict(window=4)}
    parts = {k: b.prior_sums(g, ["defteam"], sums2, k, **kw) for k, kw in W.items()}
    parts["s2d"] = b.prior_sums(g, ["defteam", "season"], sums2, "s2d")
    F = {}
    def rates(pre, d):
        n = lambda c: d[f"{pre}_{c}"]; den = n("den").replace(0, np.nan)
        F[f"{pre}_rate"] = n("num") / den; F[f"{pre}_adj_rate"] = n("over_exp") / den; F[f"{pre}_epa"] = n("epa_s") / den
        F[f"{pre}_adj_epa"] = n("epa_over_exp") / den; F[f"{pre}_succ"] = n("succ") / den; F[f"{pre}_fd"] = n("fd") / den; F[f"{pre}_expl"] = n("expl") / den
        F[f"{pre}_td"] = n("tds") / den
        for tag in ("neu", "early", "late"): F[f"{pre}_{tag}_rate"] = n(f"{tag}_num") / n(f"{tag}_den").replace(0, np.nan)
        if kind == "pass":
            ns = n("att_ns").replace(0, np.nan)
            F[f"{pre}_cmp"] = n("comp") / ns; F[f"{pre}_adot"] = n("air") / ns; F[f"{pre}_yac_c"] = n("yac") / n("comp").replace(0, np.nan)
            F[f"{pre}_cpoe"] = n("cpoe_sum") / n("cpoe_n").replace(0, np.nan); F[f"{pre}_sack"] = n("sacks") / den; F[f"{pre}_hit"] = n("hits") / den
            F[f"{pre}_press"] = n("press") / den; F[f"{pre}_int"] = n("ints") / ns; F[f"{pre}_net"] = n("net_yds") / den
            F[f"{pre}_deep_share"] = n("deep_att") / ns; F[f"{pre}_deep_cmp"] = n("deep_comp") / n("deep_att").replace(0, np.nan)
            F[f"{pre}_deep_ypa"] = n("deep_yds") / n("deep_att").replace(0, np.nan); F[f"{pre}_short_ypa"] = n("short_yds") / n("short_att").replace(0, np.nan)
        else:
            F[f"{pre}_stuff"] = n("stuff") / den; F[f"{pre}_capped"] = n("capped") / den; F[f"{pre}_in_ypc"] = n("in_yds") / n("in_att").replace(0, np.nan)
            F[f"{pre}_out_ypc"] = n("out_yds") / n("out_att").replace(0, np.nan); F[f"{pre}_sy_conv"] = n("sy_conv") / n("sy_att").replace(0, np.nan)
        F[f"{pre}_vol"] = n("den")
    for pre, d in parts.items(): rates(pre, d)
    g = pd.concat([g, pd.DataFrame(F)], axis=1)
    yr = g.groupby(["defteam", "season"], as_index=False)[["num", "den", "over_exp", "epa_s"]].sum()
    yr["prev_rate"], yr["prev_adj_rate"], yr["prev_epa"] = yr.num / yr.den, yr.over_exp / yr.den, yr.epa_s / yr.den; yr["season"] += 1
    g = g.merge(yr[["defteam", "season", "prev_rate", "prev_adj_rate", "prev_epa"]], on=["defteam", "season"], how="left")
    g["lg_prev"] = b.league_prev(g, "num", "den"); g["y"] = g.num / g.den
    feats = list(F) + ["prev_rate", "prev_adj_rate", "prev_epa", "lg_prev", "week"]
    g = g[(g.season >= 2012) & g.ewL_rate.notna()].reset_index(drop=True)
    F45 = pd.read_parquet(S + "/pbp_team_week_45.parquet"); DEF = [c for c in F45 if c.startswith("def_")]
    g = g.merge(F45[["team", "season", "week"] + DEF].rename(columns={"team": "defteam"}), on=["defteam", "season", "week"], how="left")
    return g, feats, DEF


def fitpred(a, x, cols):
    r = b.lin(RidgeCV(alphas=np.logspace(0, 5, 25))).fit(a[cols], a.y).predict(x[cols])
    l = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=6, min_child_samples=120, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
                          reg_lambda=20, verbose=-1, random_state=0).fit(a[cols], a.y).predict(x[cols])
    return {"ridge": r, "lgbm": l, "blend": (r + l) / 2}


def main(kind):
    g, new, DEF = build(kind); name = {"pass": "PASSING YPA ALLOWED", "rush": "RUSHING YPC ALLOWED (designed runs)"}[kind]
    hist = g[g.season <= 2025].reset_index(drop=True); fold = fb.scattered_folds(hist)
    print(f"\n{'=' * 110}\nDEFENSE-ONLY, WEEK-AHEAD: {name}. {len(hist):,} defense-games 2012-2025; target mean {hist.y.mean():.2f}, sd {hist.y.std():.2f}. {len(new)} new defensive stats built.")
    u = hist[new].corrwith(hist.y); u = u.reindex(u.abs().sort_values(ascending=False).index)
    print("\nBEST SINGLE DEFENSIVE STATS, week-ahead, all seasons (prefix: ewS = recent-weighted ~4 games, ewL = ~14 games, l4 = last 4, s2d = season to date, prev = last season; adj = opponent-adjusted)")
    print(pd.DataFrame({"r": u.head(15)}).round(3).T.to_string())
    if kind == "pass": print("for comparison, your best defensive variables:", ", ".join(f"{v} {hist[v].corr(hist.y):+.3f}" for v in hist[DEF].corrwith(hist.y).sort_values(ascending=False).head(3).index))
    top = list(u.index[:12]); r0 = hist[DEF].corrwith(hist.y); pruned = list(r0[r0 > 0.03].index)
    sets = {"trailing rate allowed only (season-to-date + last season)": ["s2d_rate", "prev_rate", "lg_prev"],
            "your 23 defensive variables": DEF, "your defensive variables, pruned": pruned,
            "NEW defensive stats (all)": new, "NEW: top 12 by single-stat r": top, "NEW + your pruned defensive variables": new + pruned}
    if kind == "rush": sets = {k: v for k, v in sets.items() if "your" not in k}
    rows, keep = [], {}
    te25, tr25 = hist[hist.season == 2025], hist[hist.season < 2025]
    for lab, cols in sets.items():
        P = {m: np.zeros(len(hist)) for m in ("ridge", "lgbm", "blend")}
        for k in range(fb.K):
            out = fitpred(hist[fold != k], hist[fold == k], cols)
            for m in P: P[m][fold == k] = out[m]
        fw = fitpred(tr25, te25, cols); best = max(P, key=lambda m: pearsonr(hist.y, P[m])[0]); keep[lab] = (P[best], fw[best], best)
        wk = te25.assign(p=fw[best]).groupby("week").apply(lambda d: d.y.corr(d.p))
        rows.append({"feature set": lab, "n": len(cols), "best model": best, "all-seasons OOF r": pearsonr(hist.y, P[best])[0], "OOF RMSE": rmse(hist.y, P[best]),
                     "2025 forward r": pearsonr(te25.y, fw[best])[0], "2025 RMSE": rmse(te25.y, fw[best]), "2025 avg weekly r": wk.mean(), "weeks r>0": f"{(wk > 0).sum()}/{len(wk)}"})
    print("\n" + pd.DataFrame(rows).round(3).to_string(index=False)); print(f"(RMSE of guessing the mean: all seasons {hist.y.std():.3f}, 2025 {te25.y.std():.3f})")
    lab = max(rows, key=lambda r: r["all-seasons OOF r"])["feature set"]; oof, fw, m = keep[lab]; sl, ic = np.polyfit(oof, hist.y, 1)
    print(f"\nBEST: '{lab}' ({m}). Calibration on out-of-fold predictions: actual = {ic:.3f} + {sl:.3f}*pred")
    t = te25.assign(proj=ic + sl * fw); T = pd.concat([t[(t.defteam == a) & (t.week == w)] for a, w in TRACKED])
    print("tracked matchups (2025, strict forward projections - spot check only, 6 games cannot rank models):")
    print(T[["defteam", "week", "posteam", "proj", "y"]].rename(columns={"proj": "projected", "y": "actual"}).round(2).to_string(index=False)); print(f"r on these 6: {T.proj.corr(T.y):+.2f}")
    q = pd.qcut(t.proj, 5, labels=["toughest 20%", "20-40%", "40-60%", "60-80%", "softest 20%"])
    print("2025 by projection bucket:"); print(t.groupby(q, observed=True).agg(games=("y", "size"), projected=("proj", "mean"), actual=("y", "mean")).round(2).T.to_string())
    nxt = g[(g.season == 2026)]
    if len(nxt): print(f"(2026 rows available for live use: weeks {sorted(nxt.week.unique())})")
    t[["season", "week", "defteam", "posteam", "proj", "y"]].to_csv(f"{S}/def_only_{kind}_2025.csv", index=False)


if __name__ == "__main__":
    for k in (sys.argv[1:] or ["pass"]): main(k)

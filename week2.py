"""2026 Week 2 projections. (1) Team-level projected defensive YPA allowed, with the user's variables pruned.
(2) QB yards-per-attempt / passing yards / fantasy points. Schedule + lines from ESPN, play-by-play from nflverse."""
import os, json, glob, warnings, numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
import lightgbm as lgb
import bakeoff as b, fantasy_bet as fb, check45 as c45
warnings.filterwarnings("ignore")
S = os.environ["S"]; ESPN2NFLV = {"WSH": "WAS", "LAR": "LA"}
SEASON, WEEK = 2026, 2
pd.set_option("display.width", 250)


def espn_slate():
    rows = []
    for e in json.load(open(f"{S}/espn_wk{WEEK}.json"))["events"]:
        c = e["competitions"][0]; t = {x["homeAway"]: ESPN2NFLV.get(x["team"]["abbreviation"], x["team"]["abbreviation"]) for x in c["competitors"]}
        o = (c.get("odds") or [{}])[0]; det = (o.get("details") or "").split()
        fav, line = (ESPN2NFLV.get(det[0], det[0]), abs(float(det[1]))) if len(det) == 2 else (None, np.nan)
        rows.append({"game_id": f"{SEASON}_{WEEK:02d}_{t['away']}_{t['home']}", "away": t["away"], "home": t["home"], "home_team": t["home"],
                     "spread_line": line if fav == t["home"] else -line, "total_line": o.get("overUnder"),
                     "roof": "dome" if (c.get("venue") or {}).get("indoor") else "outdoors", "wind": np.nan, "temp": np.nan,
                     "final": c["status"]["type"]["name"] == "STATUS_FINAL"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- team-level defensive YPA
def team_model(slate):
    g = c45.team_games()
    fut = slate[~slate.final]
    F = pd.concat([fut.assign(posteam=fut.home, defteam=fut.away), fut.assign(posteam=fut.away, defteam=fut.home)])[["game_id", "posteam", "defteam"]]
    F = F.assign(season=SEASON, week=WEEK, t=SEASON * 100 + WEEK)
    F = F.merge(c45.F45[["team", "season", "week"] + c45.OFF], left_on=["posteam", "season", "week"], right_on=["team", "season", "week"], how="left").drop(columns="team")
    F = F.merge(c45.F45[["team", "season", "week"] + c45.DEF], left_on=["defteam", "season", "week"], right_on=["team", "season", "week"], how="left").drop(columns="team")
    for key, nm in (("posteam", "off_trail_ypa"), ("defteam", "def_trail_ypa")):
        p = g[g.t >= (SEASON - 1) * 100].groupby(key)[["yds", "att"]].sum(); F[nm] = F[key].map(p.yds / p.att)
    g = pd.concat([g.assign(future=False), F.assign(future=True)], ignore_index=True)
    ctx = pd.read_parquet(S + "/pbp_game_ctx.parquet"); ctx = pd.concat([ctx[~ctx.game_id.isin(fut.game_id)], fut[ctx.columns]])
    g = g.merge(ctx, on="game_id", how="left"); sgn = np.where(g.posteam == g.home_team, 1, -1)
    g["off_spread"] = sgn * g.spread_line; g["off_implied"] = g.total_line / 2 + g.off_spread / 2
    g["indoors"] = g.roof.isin(["dome", "closed"]).astype(int); g["wind_f"] = np.where(g.indoors == 1, 0, g.wind)
    core = ["off_trail_ypa", "def_trail_ypa", "off_implied", "off_spread", "total_line", "indoors", "wind_f"]
    tr = g[~g.future & g.off_trail_ypa.notna()].reset_index(drop=True); fold = fb.scattered_folds(tr)
    V = c45.OFF + c45.DEF

    def select(d, rule):
        r = d[V].corrwith(d.ypa)
        return list(r[r > 0.03].index) if rule == "pos" else list(r[r.abs() > 0.03].index) if rule == "abs" else V

    def fit(a, x, cols):
        r = b.lin(RidgeCV(alphas=np.logspace(-1, 4, 20))).fit(a[cols], a.ypa).predict(x[cols])
        l = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=6, min_child_samples=100, colsample_bytree=0.6, reg_lambda=10,
                              verbose=-1, random_state=0).fit(a[cols], a.ypa).predict(x[cols])
        return (r + l) / 2

    print(f"\n{'=' * 100}\nTEAM-LEVEL DEFENSIVE YPA MODEL. {len(tr):,} team-games 2012-{SEASON} wk1, 5 season-scattered folds; variable pruning is redone inside every fold")
    res, oof = [], {}
    for lab, rule in (("all 45 of your variables + core", "all"), ("pruned: drop noise AND negative (your rule) + core", "pos"),
                      ("pruned: drop noise only, keep negative + core", "abs"), ("core only (trailing YPAs + Vegas + roof/wind)", None)):
        p = np.zeros(len(tr)); kept = []
        for k in range(fb.K):
            a, x = tr[fold != k], tr[fold == k]; cols = core + (select(a, rule) if rule else []); kept.append(len(cols) - len(core)); p[fold == k] = fit(a, x, cols)
        oof[lab] = p; res.append({"feature set": lab, "your vars kept": f"{min(kept)}-{max(kept)}", "OOF r": pearsonr(tr.ypa, p)[0], "OOF RMSE": c45.rmse(tr.ypa, p)})
    print(pd.DataFrame(res).round(4).to_string(index=False)); print(f"(RMSE of guessing the mean: {tr.ypa.std():.3f})")
    r = tr[V].corrwith(tr.ypa); keep = list(r[r > 0.03].index)
    print(f"\nFINAL VARIABLE LIST (r > +0.03 with next-game YPA): kept {len(keep)} of 45")
    print("  KEPT offense:", ", ".join(f"{v}({r[v]:+.3f})" for v in keep if v.startswith("off_")))
    print("  KEPT defense:", ", ".join(f"{v}({r[v]:+.3f})" for v in keep if v.startswith("def_")))
    print("  DROPPED as noise:", ", ".join(f"{v}({r[v]:+.3f})" for v in V if abs(r[v]) <= 0.03))
    print("  DROPPED as negative:", ", ".join(f"{v}({r[v]:+.3f})" for v in V if r[v] < -0.03))
    best = "pruned: drop noise AND negative (your rule) + core"; p = oof[best]; sl, ic = np.polyfit(p, tr.ypa, 1)
    print(f"calibration on out-of-fold predictions: actual = {ic:.3f} + {sl:.3f} * pred")
    tr["proj"] = ic + sl * p
    fu = g[g.future].copy(); fu["proj"] = ic + sl * fit(tr, fu, core + keep)
    return tr, fu


def show_defense(tr, fu, tracked):
    print(f"\n{'=' * 100}\nPROJECTED DEFENSIVE YARDS PER ATTEMPT ALLOWED, 2026 WEEK 2 (lower = tougher pass defense this week). League average ~{tr.ypa.mean():.2f}")
    t = fu.sort_values("proj")[["defteam", "posteam", "proj", "def_trail_ypa", "off_trail_ypa", "off_implied"]]
    t.columns = ["Defense", "vs Offense", "Proj YPA allowed", "Def trailing YPA allowed", "Opp offense trailing YPA", "Opp implied points"]
    print(t.round(2).to_string(index=False))
    print("\nTRACKED MATCHUPS")
    rows = []
    for team, wk in tracked:
        for season in (2025, 2026):
            h = tr[(tr.defteam == team) & (tr.season == season) & (tr.week == wk)]; f = fu[(fu.defteam == team) & (fu.week == wk)] if season == SEASON else fu.iloc[0:0]
            for _, x in h.iterrows(): rows.append({"Defense": team, "Season": season, "Week": wk, "vs Offense": x.posteam, "Proj def YPA allowed": x.proj, "Actual YPA allowed": x.ypa, "note": "out-of-fold projection"})
            for _, x in f.iterrows(): rows.append({"Defense": team, "Season": season, "Week": wk, "vs Offense": x.posteam, "Proj def YPA allowed": x.proj, "Actual YPA allowed": np.nan, "note": "not played yet"})
    print(pd.DataFrame(rows).round(2).to_string(index=False))


# ---------------------------------------------------------------- QB level
def qb_rows_2026(w, slate):
    cols = ["game_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team", "passer_player_id", "passer_player_name", "rusher_player_id",
            "pass_attempt", "sack", "complete_pass", "passing_yards", "air_yards", "pass_touchdown", "interception", "qb_scramble", "rushing_yards", "rush_touchdown", "rush_attempt"]
    p = pd.read_parquet(f"{S}/pbp/pbp_{SEASON}.parquet", columns=cols)
    a = p[(p.pass_attempt == 1) & (p.sack != 1) & p.passer_player_id.notna()]
    q = a.groupby(["passer_player_id", "game_id"], as_index=False).agg(player_name=("passer_player_name", "first"), team=("posteam", "first"), season=("season", "first"), week=("week", "first"),
                                                                        season_type=("season_type", "first"), pass_attempts=("pass_attempt", "sum"), complete_pass=("complete_pass", "sum"),
                                                                        passing_yards=("passing_yards", "sum"), passing_air_yards=("air_yards", "sum"), pass_touchdown=("pass_touchdown", "sum"),
                                                                        interception=("interception", "sum")).rename(columns={"passer_player_id": "player_id"})
    sk = p[p.sack == 1].groupby(["passer_player_id", "game_id"]).size().rename("times_sacked")
    ru = p[p.rusher_player_id.notna() & ((p.rush_attempt == 1) | (p.qb_scramble == 1))].groupby(["rusher_player_id", "game_id"]).agg(
        rush_attempts=("rushing_yards", "size"), rushing_yards=("rushing_yards", "sum"), rush_touchdown=("rush_touchdown", "sum"), qb_scramble=("qb_scramble", "sum"))
    q = q.merge(sk, left_on=["player_id", "game_id"], right_index=True, how="left").merge(ru, left_on=["player_id", "game_id"], right_index=True, how="left").fillna(
        {"times_sacked": 0, "rush_attempts": 0, "rushing_yards": 0, "rush_touchdown": 0, "qb_scramble": 0})
    q["fantasy_points_ppr"] = q.passing_yards / 25 + 4 * q.pass_touchdown - 2 * q.interception + q.rushing_yards / 10 + 6 * q.rush_touchdown
    starters = q[q.week == 1].sort_values("pass_attempts").groupby("team").tail(1)
    played = set(q[q.week == WEEK].team)
    sl = pd.concat([slate.assign(team=slate.home), slate.assign(team=slate.away)])
    ph = starters[~starters.team.isin(played)][["player_id", "player_name", "team"]].merge(sl[["team", "game_id"]], on="team")
    ph = ph.assign(season=SEASON, week=WEEK, season_type="REG", pass_attempts=30.0, placeholder=True)  # stats of an unplayed game never enter its own features
    q = pd.concat([q.assign(placeholder=False), ph], ignore_index=True).fillna({c: 0 for c in ["complete_pass", "passing_yards", "passing_air_yards", "pass_touchdown", "interception",
                                                                                                  "times_sacked", "rush_attempts", "rushing_yards", "rush_touchdown", "qb_scramble", "fantasy_points_ppr"]})
    last = w[w.position == "QB"].sort_values("t").groupby("player_id").tail(1).set_index("player_id")
    q["position"] = "QB"
    for col in ("birth_date", "draft_pick"): q[col] = q.player_id.map(last[col])
    q["years_exp"] = q.player_id.map(last.years_exp + (SEASON - last.season)).fillna(0)
    q["display"] = q.player_id.map(last.player_name).fillna(q.player_name)
    g = q.game_id.str.split("_", expand=True); q["away"], q["home"] = g[2], g[3]
    q["is_home"] = (q.team == q.home).astype(int); q["opp"] = np.where(q.team == q.home, q.away, q.home); q["t"] = q.season * 100 + q.week
    q["age"] = q.season + q.week / 52 - pd.to_datetime(q.birth_date, errors="coerce").dt.year
    return q


def qb_model(slate):
    w = b.load(); q26 = qb_rows_2026(w, slate)
    w = pd.concat([w.assign(placeholder=False), q26], ignore_index=True).sort_values(["player_id", "t"]).reset_index(drop=True)
    ph = q26[q26.placeholder]
    extra_qb = ph[["player_id", "game_id"]].assign(db=0.0, epa_sum=0.0, cpoe_sum=0.0, cpoe_n=0.0, succ=0.0, press=0.0)
    extra_def = ph[["opp", "game_id", "season", "week"]].assign(db=1e-9, epa_sum=0.0, cpoe_sum=0.0, cpoe_n=0.0, succ=0.0, press=0.0)
    out = ph[["player_id", "display", "team", "opp", "game_id"]].copy(); cv = []
    for tgt, lab, use in (("ypa", "ypa", "Blend"), ("pass_yds", "pass_yds", "EB+LightGBM"), ("fp", "fantasy_pts", "EB+LightGBM")):
        df, base = fb.build_qb(w, tgt); df, pbpf, vegas = c45.add_pbp(df, extra_qb, extra_def, slate[~slate.final][["game_id", "home_team", "spread_line", "total_line", "roof", "wind", "temp"]])
        feats = base + pbpf + vegas
        tr, te = df[~df.placeholder.astype(bool)].reset_index(drop=True), df[df.placeholder.astype(bool)].reset_index(drop=True)
        fold = fb.scattered_folds(tr); P = c45.cv_models(tr, feats, fold)[use]; sl, ic = np.polyfit(P, tr.y, 1)
        cv.append({"target": lab, "model": use, "train games": len(tr), "OOF r": pearsonr(tr.y, P)[0], "OOF RMSE": c45.rmse(tr.y, P), "calibration": f"{ic:.2f} + {sl:.3f}*pred"})
        k = b.eb_k(tr); tr["eb"], te["eb"] = b.add_eb(tr, k), b.add_eb(te, k); f = feats + ["eb"]
        pl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=8, min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                               reg_lambda=10, verbose=-1, random_state=0).fit(tr[f], tr.y).predict(te[f])
        pr = b.lin(RidgeCV(alphas=np.logspace(-1, 4, 20))).fit(tr[f], tr.y).predict(te[f])
        te[lab] = ic + sl * ((pl + pr) / 2 if use == "Blend" else pl)
        out = out.merge(te[["player_id", lab] + (["implied_total", "car_att"] if tgt == "ypa" else [])], on="player_id", how="left")
    print(f"\n{'=' * 100}\nQB MODELS refit through 2026 week 1 (+ DET@BUF). Out-of-fold check:"); print(pd.DataFrame(cv).round(3).to_string(index=False))
    print(f"\n2026 WEEK 2 QB PROJECTIONS (assumes each team's week-1 starter plays; DET/BUF already played Thursday)")
    o = out.sort_values("fantasy_pts", ascending=False).rename(columns={"display": "QB", "implied_total": "team implied pts", "car_att": "career att in data"})
    print(o[["QB", "team", "opp", "ypa", "pass_yds", "fantasy_pts", "team implied pts", "career att in data"]].round(2).to_string(index=False))
    missing = set(pd.concat([slate[~slate.final].home, slate[~slate.final].away])) - set(out.dropna(subset=["ypa"]).team)
    if missing: print("no projection (QB has too little history for the pre-game filter):", sorted(missing))
    o.to_csv(S + "/week2_qb_projections.csv", index=False)


if __name__ == "__main__":
    slate = espn_slate()
    tr, fu = team_model(slate)
    show_defense(tr, fu, [("LV", 12), ("DAL", 15), ("CHI", 2), ("NE", 2), ("SEA", 18), ("MIN", 13)])
    fu.to_csv(S + "/week2_def_ypa.csv", index=False)
    qb_model(slate)

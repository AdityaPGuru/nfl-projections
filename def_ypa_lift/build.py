"""Rebuild the defensive YPA baseline (35 terms) from nflverse play-by-play and add a pool of pregame candidate stats.
One row per (game, offense, defense), regular season 2013-2025. Target: net passing yards per dropback allowed
(yards_gained on pass attempts incl. sacks, two-point tries excluded).

Every feature is built from games completed before the row's kickoff: "ytd" = the team's full prior season pooled with
its completed weeks of the current season (the definition in his R script). Schedule items (spread, total, roof, rest,
starting QB, referee) are published before kickoff. Game-day temp/wind are built ONLY to reproduce his baseline and are
flagged as leaks.
Usage: python def_ypa_lift/build.py   (reads te_share/data, writes def_ypa_lift/data/rows.parquet)"""
import os, glob, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DATA = os.environ.get("S", os.path.join(ROOT, "te_share", "data")); OUT = os.environ.get("OUT", os.path.join(HERE, "data")); os.makedirs(OUT, exist_ok=True)
# Forecast mode: FORECAST=2026,7 keeps only plays from weeks strictly before that week and appends that week's scheduled games as rows
# with no outcome. Historical rows come out identical to the default build (week7/check_same.py verifies this).
FORECAST = tuple(int(v) for v in os.environ["FORECAST"].split(",")) if os.environ.get("FORECAST") else None
COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "pass_attempt", "rush_attempt", "sack", "qb_hit", "qb_scramble",
        "complete_pass", "yards_gained", "air_yards", "yards_after_catch", "epa", "yac_epa", "success", "cpoe", "interception", "pass_touchdown", "down",
        "ydstogo", "qtr", "score_differential", "yardline_100", "two_point_attempt", "passer_player_id", "receiver_player_id", "xpass", "pass_oe", "wp",
        "half_seconds_remaining", "penalty", "penalty_team", "penalty_type", "penalty_yards", "first_down", "qb_dropback", "shotgun", "no_huddle", "play_type"]


def load():
    p = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob(f"{DATA}/pbp/pbp_*.parquet"))], ignore_index=True)
    p = p[(p.season_type == "REG") & p.posteam.notna() & p.defteam.notna()].copy()
    p["t"] = p.season * 100 + p.week
    p = p[p.t < FORECAST[0] * 100 + FORECAST[1]] if FORECAST else p[p.season <= 2025]
    return p


def window(g, key, cols):
    """g: one row per (key, season, week) with additive sums in `cols`. Returns, per row, prior season total + current-season
    total of EARLIER weeks (never the row's own week)."""
    g = g.sort_values([key, "season", "week"]).reset_index(drop=True)
    cur = g.groupby([key, "season"])[cols].cumsum() - g[cols]
    yr = g.groupby([key, "season"], as_index=False)[cols].sum(); yr["season"] += 1
    prev = g[[key, "season"]].merge(yr, on=[key, "season"], how="left")[cols].fillna(0)
    out = g[[key, "season", "week"]].copy()
    for c in cols: out[c] = cur[c].values + prev[c].values
    return out


def rate_feats(d, key, specs, sched):
    """specs: name -> (num series, den series) on play frame d. Ratio of windowed sums, joined to `sched` (key, season, week)."""
    cols = []; frame = d[[key, "season", "week"]].copy()
    for n, (num, den) in specs.items():
        frame[n + "__n"], frame[n + "__d"] = num.astype(float).fillna(0).values, den.astype(float).fillna(0).values; cols += [n + "__n", n + "__d"]
    g = frame.groupby([key, "season", "week"], as_index=False)[cols].sum()
    g = sched[[key, "season", "week"]].drop_duplicates().merge(g, how="left").fillna(0)
    w = window(g, key, cols)
    for n in specs: w[n] = np.where(w[n + "__d"] > 0, w[n + "__n"] / w[n + "__d"].replace(0, np.nan), np.nan)
    return w[[key, "season", "week"] + list(specs)]


def std_feats(d, key, specs, sched):
    """specs: name -> (value series, mask). Windowed standard deviation from pooled sums of x and x^2."""
    frame = d[[key, "season", "week"]].copy(); cols = []
    for n, (x, m) in specs.items():
        x = x.astype(float); m = (m & x.notna()).astype(float); x = x.fillna(0)
        frame[n + "_a"], frame[n + "_b"], frame[n + "_c"] = x * m, x * x * m, m; cols += [n + "_a", n + "_b", n + "_c"]
    g = frame.groupby([key, "season", "week"], as_index=False)[cols].sum()
    g = sched[[key, "season", "week"]].drop_duplicates().merge(g, how="left").fillna(0)
    w = window(g, key, cols)
    for n in specs:
        c = w[n + "_c"]; mu = w[n + "_a"] / c.replace(0, np.nan)
        w[n] = np.sqrt(np.clip(w[n + "_b"] / c.replace(0, np.nan) - mu ** 2, 0, None) * c / (c - 1).where(c > 1))
    return w[[key, "season", "week"] + list(specs)]


def median_feats(d, key, specs, sched):
    """Windowed median (prior season + earlier current-season weeks), computed by brute force."""
    rows = sched[[key, "season", "week"]].drop_duplicates().sort_values([key, "season", "week"])
    out = {n: [] for n in specs}
    for n, (x, m) in specs.items():
        v = d.loc[m & x.notna(), [key, "season", "week"]].assign(x=x[m & x.notna()].astype(float))
        by = {k: grp for k, grp in v.groupby(key)}
        for team, s, w in rows.itertuples(index=False):
            grp = by.get(team)
            if grp is None: out[n].append(np.nan); continue
            sel = grp.x[((grp.season == s - 1)) | ((grp.season == s) & (grp.week < w))]
            out[n].append(float(sel.median()) if len(sel) >= 5 else np.nan)
    r = rows.copy()
    for n in specs: r[n] = out[n]
    return r


def main():
    p = load()
    db = p[(p.pass_attempt == 1) & (p.two_point_attempt != 1)].copy()          # dropbacks: pass attempts + sacks
    db["sack"] = db.sack.fillna(0); db["comp"] = db.complete_pass.fillna(0); ns = db.qb_scramble != 1; c = db.comp == 1
    sdiff = db.score_differential                                               # offense's view: >0 offense leads = defense trails
    blow, neut = sdiff.abs() >= 17, sdiff.abs() <= 7
    d_lead, d_trail = sdiff < 0, sdiff > 0
    behind = db.air_yards <= 0; deep = db.air_yards >= 20; expl = db.yards_gained >= 20; one = pd.Series(1.0, index=db.index)

    # ---- target rows ----
    tgt = db.groupby(["game_id", "season", "week", "posteam", "defteam", "home_team"], as_index=False).agg(yds=("yards_gained", "sum"), n=("yards_gained", "size"))
    tgt["ypa"] = tgt.yds / tgt.n
    games = pd.read_csv(f"{DATA}/aux/games.csv"); games = games[games.game_type == "REG"]
    if FORECAST:   # one row per (game, offense) for the forecast week, outcome unknown
        fw = games[(games.season == FORECAST[0]) & (games.week == FORECAST[1])]
        fr = pd.concat([fw.assign(posteam=fw.home_team, defteam=fw.away_team), fw.assign(posteam=fw.away_team, defteam=fw.home_team)])
        tgt = pd.concat([tgt, fr[["game_id", "season", "week", "posteam", "defteam", "home_team"]]], ignore_index=True)
    tgt = tgt.merge(games[["game_id", "away_team", "spread_line", "total_line", "roof", "surface", "temp", "wind", "away_rest", "home_rest", "div_game",
                           "away_qb_id", "home_qb_id", "referee", "away_moneyline", "home_moneyline", "gameday", "gametime"]], on="game_id", how="left")
    home = tgt.posteam == tgt.home_team; sgn = np.where(home, 1, -1)
    tgt["off_spread_line"] = sgn * tgt.spread_line                              # nflverse spread_line > 0 = home favored
    tgt["off_home"] = home * 1; tgt["indoors"] = tgt.roof.isin(["dome", "closed"]) * 1; tgt["off_implied"] = tgt.total_line / 2 + tgt.off_spread_line / 2
    tgt["def_rest_days"] = np.where(home, tgt.away_rest, tgt.home_rest); tgt["off_rest_days"] = np.where(home, tgt.home_rest, tgt.away_rest)
    tgt["rest_diff"] = tgt.def_rest_days - tgt.off_rest_days
    tgt["opp_qb_id"] = np.where(home, tgt.home_qb_id, tgt.away_qb_id)
    tgt["grass"] = tgt.surface.fillna("").str.contains("grass") * 1
    tgt["div_game"] = tgt.div_game.fillna(0)
    sched_d, sched_o = tgt.rename(columns={}).copy(), tgt.copy()

    # ---- his 35 baseline terms (approximate rebuilds where the R definition is not shown) ----
    D = "defteam"; O = "posteam"
    F = rate_feats(db, D, {
        "def_comp_all_sr": (db.success * c, c), "def_neut_q2_air_yards": (db.air_yards * (neut & (db.qtr == 2)), neut & (db.qtr == 2) & db.air_yards.notna()),
        "def_lead_d2_cpoe": (db.cpoe.fillna(0) * (d_lead & (db.down == 2)), d_lead & (db.down == 2) & db.cpoe.notna()),
        "def_trail_q3_complete": (db.comp * (d_trail & (db.qtr == 3) & (db.sack == 0)), d_trail & (db.qtr == 3) & (db.sack == 0)),
        "def_trail_d1_yac": (db.yards_after_catch.fillna(0) * (d_trail & (db.down == 1) & c), d_trail & (db.down == 1) & c),
        "def_qb_hit_rate_q4": (db.qb_hit.fillna(0) * (db.qtr == 4), db.qtr == 4),
        "def_explosive_rate_behind_los": ((db.yards_gained >= 15) & behind, behind), "def_epa_neutral_behind_los": (db.epa * (neut & behind), neut & behind),
        "def_td_rate_behind_los": (db.pass_touchdown.fillna(0) * behind, behind), "def_yac_epa_per_dropback_ytd": (db.yac_epa.fillna(0) * c, one),
        "def_hit_red_zone_pass_rate_ytd": ((db.qb_hit == 1) & (db.yardline_100 <= 20), db.qb_hit == 1),
    }, tgt)
    gp = rate_feats(db.assign(g1=(~db.duplicated(["game_id", D])) * 1.0), D, {"def_pass_volume": (one, db.assign(g1=(~db.duplicated(["game_id", D])) * 1.0).g1)}, tgt)
    F = F.merge(gp)
    F = F.merge(std_feats(db, D, {"def_blowout_std_air_comp_mean": (db.air_yards, blow & c), "def_ns_behind_los_yac_std": (db.yards_after_catch, ns & behind & c),
                                  "def_blowout_air_std": (db.air_yards, blow & db.air_yards.notna())}, tgt))
    F = F.merge(median_feats(db, D, {"def_ns_comp_deep_air_yards_med": (db.air_yards, ns & deep & c), "def_comp_deep_yac_med": (db.yards_after_catch, deep & c)}, tgt))
    # minimum per-game YPA allowed when backed up (own 20 or worse), over the window's games
    bu = db[db.yardline_100 >= 80].groupby([D, "season", "week"]).yards_gained.mean().rename("v").reset_index()
    rows = tgt[[D, "season", "week"]].drop_duplicates(); vals = []
    byt = {k: v for k, v in bu.groupby(D)}
    for team, s, w in rows.itertuples(index=False):
        g = byt.get(team); sel = g.v[(g.season == s - 1) | ((g.season == s) & (g.week < w))] if g is not None else []
        vals.append(float(np.min(sel)) if len(sel) else np.nan)
    F = F.merge(rows.assign(def_backedup_ypa_min=vals))

    allp = p[(p.pass_attempt == 1) | (p.rush_attempt == 1)]; allp = allp[allp.two_point_attempt != 1]
    OF = rate_feats(db, O, {"off_explosive_pass_rate_ytd": (expl, one), "off_success_rate_ytd": (db.success, one),
                            "off_top_receiver_target_share_ytd": (one * 0, one)}, tgt)  # placeholder, replaced below
    OF = OF.drop(columns="off_top_receiver_target_share_ytd")
    OF = OF.merge(rate_feats(allp, O, {"off_pass_rate_ytd": (allp.pass_attempt == 1, pd.Series(1.0, index=allp.index)),
                                       "off_pass_oe_ytd": (allp.pass_oe.fillna(0), allp.pass_oe.notna()), "off_xpass_ytd": (allp.xpass.fillna(0), allp.xpass.notna())}, tgt))
    # top receiver target share (max single-receiver targets / team targets, per game, windowed)
    rg = db[db.receiver_player_id.notna()].groupby(["game_id", "season", "week", O, "receiver_player_id"]).size().rename("tg").reset_index()
    rg = rg.groupby(["game_id", "season", "week", O]).tg.agg(["max", "sum"]).reset_index()
    rg2 = rate_feats(rg.reset_index(drop=True), O, {"off_top_receiver_target_share_ytd": (rg["max"], rg["sum"])}, tgt)
    OF = OF.merge(rg2)

    # opponent explosive-pass rate faced so far this season: mean, over the defense's earlier games this season, of the offense's pregame rate
    t2 = tgt[["game_id", "season", "week", O, D]].merge(OF[[O, "season", "week", "off_explosive_pass_rate_ytd"]], how="left").sort_values([D, "season", "week"])
    t2["cs"] = t2.groupby([D, "season"]).off_explosive_pass_rate_ytd.cumsum() - t2.off_explosive_pass_rate_ytd.fillna(0)
    t2["cn"] = t2.groupby([D, "season"]).cumcount()
    t2["def_opp_explosive_pass_rate_ytd_faced_ytd"] = t2.cs / t2.cn.replace(0, np.nan)

    # primary QB = the offense's passer with most dropbacks in the window; his stats over his own window (any team)
    qb = db[db.passer_player_id.notna()].copy()
    QA_SPEC = lambda: {"q_succ": (qb.success, pd.Series(1.0, index=qb.index)), "q_expl": (qb.yards_gained >= 20, pd.Series(1.0, index=qb.index)),
                                             "q_yacpc": (qb.yards_after_catch.fillna(0) * qb.comp, qb.comp), "q_yacepa": (qb.yac_epa.fillna(0) * qb.comp, pd.Series(1.0, index=qb.index)),
                                             "q_cay": (qb.air_yards.fillna(0) * qb.comp, qb.comp), "q_cmp": (qb.comp * (qb.sack == 0), qb.sack == 0)}
    cnt = qb.groupby([O, "season", "week", "passer_player_id"]).size().rename("n").reset_index()
    rows = tgt[[O, "season", "week"]].drop_duplicates(); top = []
    byo = {k: v for k, v in cnt.groupby(O)}
    for team, s, w in rows.itertuples(index=False):
        g = byo.get(team); sel = g[(g.season == s - 1) | ((g.season == s) & (g.week < w))] if g is not None else None
        top.append(sel.groupby("passer_player_id").n.sum().idxmax() if sel is not None and len(sel) else None)
    rows = rows.assign(top_qb=top)
    # The QB's window row must exist whether or not he plays this week. (v1 built the grid only from weeks he played, so the
    # feature went missing exactly when the usual QB sat out -- a same-game leak through missingness.)
    PLAYED = qb[["passer_player_id", "season", "week"]].drop_duplicates()
    qa = rate_feats(qb, "passer_player_id", QA_SPEC(), pd.concat([PLAYED, rows[["top_qb", "season", "week"]].rename(columns={"top_qb": "passer_player_id"}).dropna()]))
    rows = rows.merge(qa.rename(columns={"passer_player_id": "top_qb"}), on=["top_qb", "season", "week"], how="left")
    rows = rows.rename(columns={"q_succ": "off_top_qb_success_rate_ytd", "q_expl": "off_top_qb_explosive_pass_rate_ytd", "q_yacpc": "off_top_qb_yac_per_completion_ytd",
                                "q_yacepa": "off_top_qb_yac_epa_per_dropback_ytd", "q_cay": "off_ngs_top_qb_avg_completed_air_yards_ytd", "q_cmp": "off_ngs_top_qb_completion_percentage_ytd"})

    X = tgt.merge(F, on=[D, "season", "week"], how="left").merge(OF, on=[O, "season", "week"], how="left").merge(rows, on=[O, "season", "week"], how="left")
    X = X.merge(t2[["game_id", D, "def_opp_explosive_pass_rate_ytd_faced_ytd"]], on=["game_id", D], how="left")
    X["sq__off_top_qb_yac_epa_per_dropback_ytd"] = X.off_top_qb_yac_epa_per_dropback_ytd ** 2
    X["ix__off_top_receiver_target_share_ytd__X__def_hit_red_zone_pass_rate_ytd"] = X.off_top_receiver_target_share_ytd * X.def_hit_red_zone_pass_rate_ytd

    # ================= candidate pool (all pregame) =================
    # C1. level stats as raw ytd ratios (defense allowed, offense gained) -- shrinkage is done by the regression itself
    dspec = {"c_def_ypa": (db.yards_gained, one), "c_def_epa": (db.epa, one), "c_def_sack": (db.sack, one), "c_def_int": (db.interception.fillna(0), one),
             "c_def_cpoe": (db.cpoe.fillna(0), db.cpoe.notna()), "c_def_adot": (db.air_yards.fillna(0) * (db.sack == 0), (db.sack == 0) & db.air_yards.notna()),
             "c_def_yac": (db.yards_after_catch.fillna(0) * c, c), "c_def_succ": (db.success, one), "c_def_expl": (expl, one),
             "c_def_hit": (db.qb_hit.fillna(0), one), "c_def_fd": (db.first_down.fillna(0), one),
             "c_def_ypa_neutral": (db.yards_gained * neut, neut)}
    ospec = {k.replace("c_def_", "c_off_"): v for k, v in dspec.items()}
    CD = rate_feats(db, D, dspec, tgt); CO = rate_feats(db, O, ospec, tgt)
    # C2. current-season-only versions (no prior season) with dropback counts, so the model can weight them
    def s2d(d, key, spec, name_suffix):
        frame = d[[key, "season", "week"]].copy(); cols = []
        for n, (num, den) in spec.items():
            frame[n + "__n"], frame[n + "__d"] = num.astype(float).fillna(0).values, den.astype(float).fillna(0).values; cols += [n + "__n", n + "__d"]
        g = frame.groupby([key, "season", "week"], as_index=False)[cols].sum()
        g = tgt[[key, "season", "week"]].drop_duplicates().merge(g, how="left").fillna(0).sort_values([key, "season", "week"])
        cur = g.groupby([key, "season"])[cols].cumsum() - g[cols]; out = g[[key, "season", "week"]].copy()
        for n in spec: out[n + name_suffix] = cur[n + "__n"] / cur[n + "__d"].replace(0, np.nan)
        return out
    CD = CD.merge(s2d(db, D, {"c_def_ypa": dspec["c_def_ypa"], "c_def_epa": dspec["c_def_epa"]}, "_s2d"))
    CO = CO.merge(s2d(db, O, {"c_off_ypa": ospec["c_off_ypa"], "c_off_epa": ospec["c_off_epa"]}, "_s2d"))
    # prior-season-only levels
    for key, pre, df_ in ((D, "c_def", CD), (O, "c_off", CO)):
        yr = db.groupby([key, "season"]).agg(a=("yards_gained", "sum"), b=("yards_gained", "size"), e=("epa", "sum")).reset_index(); yr["season"] += 1
        yr[pre + "_ypa_prev"] = yr.a / yr.b; yr[pre + "_epa_prev"] = yr.e / yr.b
        df_[[pre + "_ypa_prev", pre + "_epa_prev"]] = df_[[key, "season"]].merge(yr[[key, "season", pre + "_ypa_prev", pre + "_epa_prev"]], how="left")[[pre + "_ypa_prev", pre + "_epa_prev"]].values
    # C3. opponent-adjusted defense: yards allowed per dropback minus what each faced offense averaged coming in
    t3 = tgt[["game_id", "season", "week", O, D, "yds", "n"]].merge(CO[[O, "season", "week", "c_off_ypa"]], how="left")
    t3["exp"] = t3.c_off_ypa.fillna(t3.yds / t3.n) * t3.n; t3["over"] = t3.yds - t3.exp
    adj = rate_feats(t3, D, {"c_def_ypa_oppadj": (t3.over, t3.n)}, tgt)
    t4 = tgt[["game_id", "season", "week", O, D, "yds", "n"]].merge(CD[[D, "season", "week", "c_def_ypa"]], how="left")
    t4["exp"] = t4.c_def_ypa.fillna(t4.yds / t4.n) * t4.n; t4["over"] = t4.yds - t4.exp
    adjo = rate_feats(t4, O, {"c_off_ypa_oppadj": (t4.over, t4.n)}, tgt)
    # C4. the starting QB listed in the schedule (nflverse fills a projected starter days before the game and overwrites it with the
    # actual starter afterward) -- his ytd stats over all his dropbacks, and whether he differs from the team's usual QB
    qs = rate_feats(qb, "passer_player_id", {"c_qb_ypa": (qb.yards_gained, pd.Series(1.0, index=qb.index)), "c_qb_epa": (qb.epa, pd.Series(1.0, index=qb.index)),
                                             "c_qb_sack": (qb.sack, pd.Series(1.0, index=qb.index)), "c_qb_cpoe": (qb.cpoe.fillna(0), qb.cpoe.notna()),
                                             "c_qb_n": (pd.Series(1.0, index=qb.index), pd.Series(1.0, index=qb.index) * 0 + 1)},
                    pd.concat([PLAYED, tgt[["opp_qb_id", "season", "week"]].rename(columns={"opp_qb_id": "passer_player_id"}).dropna()]))
    qn = qb.groupby(["passer_player_id", "season", "week"]).size().rename("k").reset_index()
    qn = pd.concat([PLAYED, tgt[["opp_qb_id", "season", "week"]].rename(columns={"opp_qb_id": "passer_player_id"}).dropna()]).drop_duplicates().merge(qn, how="left").fillna(0)
    qn2 = window(qn, "passer_player_id", ["k"]).rename(columns={"k": "c_qb_dropbacks"})
    qs = qs.drop(columns="c_qb_n").merge(qn2, how="left").rename(columns={"passer_player_id": "opp_qb_id"})

    X = X.merge(CD, on=[D, "season", "week"], how="left").merge(CO, on=[O, "season", "week"], how="left").merge(adj, on=[D, "season", "week"], how="left")
    X = X.merge(adjo, on=[O, "season", "week"], how="left").merge(qs, on=["opp_qb_id", "season", "week"], how="left")
    X["c_qb_new"] = (X.opp_qb_id != X.top_qb) * 1
    # C4b. pregame-only QB choice: the offense's leading passer (most dropbacks) in its most recent completed game,
    # with that QB's stats over ALL his dropbacks in the window (prior season + earlier weeks, any team).
    lead = qb.groupby([O, "season", "week", "passer_player_id"]).size().rename("k").reset_index().sort_values(["k", "passer_player_id"]).drop_duplicates([O, "season", "week"], keep="last")  # ties -> fixed order
    lead["t"] = (lead.season * 100 + lead.week).astype("int64")
    tt = X[[O, "season", "week"]].drop_duplicates().assign(t=lambda z: (z.season * 100 + z.week).astype("int64")).sort_values("t")
    tt = pd.merge_asof(tt, lead.sort_values("t")[[O, "t", "passer_player_id"]], on="t", by=O, allow_exact_matches=False).rename(columns={"passer_player_id": "pre_qb_id"})
    X = X.merge(tt[[O, "season", "week", "pre_qb_id"]], on=[O, "season", "week"], how="left")
    one_q = pd.Series(1.0, index=qb.index)
    qp = rate_feats(qb, "passer_player_id", {"c_qb_epa_pre": (qb.epa, one_q), "c_qb_ypa_pre": (qb.yards_gained, one_q), "c_qb_sack_pre": (qb.sack, one_q)},
                    pd.concat([qb[["passer_player_id", "season", "week"]], X[["pre_qb_id", "season", "week"]].rename(columns={"pre_qb_id": "passer_player_id"}).dropna()]))
    X = X.merge(qp.rename(columns={"passer_player_id": "pre_qb_id"}), on=["pre_qb_id", "season", "week"], how="left")
    X["c_qb_new_pre"] = (X.pre_qb_id != X.top_qb) * 1
    X["c_qb_logdb"] = np.log1p(X.c_qb_dropbacks.fillna(0))

    # C5. injury report (final report, published before kickoff): starters-ish counts by position group listed Out/Doubtful
    inj = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DATA}/aux/inj_*.parquet"))], ignore_index=True)
    inj = inj[(inj.game_type == "REG") & inj.report_status.isin(["Out", "Doubtful"])].copy()
    inj["team"] = inj.team.replace({"OAK": "LV", "SD": "LAC", "STL": "LA"})          # injury files keep relocation-era codes; pbp does not
    # keep only report rows last touched on or before the team's gameday (a few were edited after the game)
    gd = pd.concat([games[["season", "week", "home_team", "gameday"]].rename(columns={"home_team": "team"}),
                    games[["season", "week", "away_team", "gameday"]].rename(columns={"away_team": "team"})])
    gd["team"] = gd.team.replace({"OAK": "LV", "SD": "LAC", "STL": "LA"})
    inj = inj.merge(gd, on=["team", "season", "week"], how="left")
    ts = pd.to_datetime(inj.get("date_modified"), utc=True, errors="coerce")
    mod = ts.dt.tz_convert("US/Eastern").dt.strftime("%Y-%m-%d")
    inj = inj[ts.isna() | (mod <= inj.gameday.astype(str))]      # 2025 file has no timestamp column: keep as is
    grp = {"DB": ["CB", "S", "FS", "SS", "DB"], "PR": ["DE", "DT", "OLB", "LB", "ILB", "MLB", "NT", "DL", "EDGE"], "WR": ["WR", "TE"], "OL": ["T", "G", "C", "OT", "OG", "OL"], "QB": ["QB"]}
    for gname, pos in grp.items():
        k = inj[inj.position.isin(pos)].groupby(["team", "season", "week"]).size().rename(f"inj_{gname}").reset_index()
        if gname in ("DB", "PR"):
            X = X.merge(k.rename(columns={"team": D, f"inj_{gname}": f"c_def_inj_{gname}"}), on=[D, "season", "week"], how="left")
        if gname in ("WR", "OL", "QB"):
            X = X.merge(k.rename(columns={"team": O, f"inj_{gname}": f"c_off_inj_{gname}"}), on=[O, "season", "week"], how="left")
    for col in [c for c in X.columns if "_inj_" in c]: X[col] = X[col].fillna(0)

    # C6. referee crew: defensive pass-penalty yards per game in the crew's earlier games (crews are assigned before the week)
    pen = p[(p.penalty == 1) & p.penalty_type.isin(["Defensive Pass Interference", "Defensive Holding", "Illegal Contact", "Roughing the Passer"])]
    pg = pen.groupby("game_id").penalty_yards.sum().rename("py").reset_index()
    rf = games[["game_id", "season", "week", "referee"]].merge(pg, how="left").fillna({"py": 0}).dropna(subset=["referee"])
    rf = rf.assign(gm=1.0).groupby(["referee", "season", "week"], as_index=False)[["py", "gm"]].sum()
    rw = window(rf, "referee", ["py", "gm"]); rw["c_ref_dpen"] = rw.py / rw.gm.replace(0, np.nan)
    X = X.merge(rw[["referee", "season", "week", "c_ref_dpen"]], on=["referee", "season", "week"], how="left")
    # C7. schedule context
    X["c_short_week"] = (X.def_rest_days <= 5) * 1; X["c_off_bye"] = (X.off_rest_days >= 12) * 1; X["c_def_bye"] = (X.def_rest_days >= 12) * 1
    ml = np.where(X.off_home == 1, X.home_moneyline, X.away_moneyline)
    X["c_off_ml_prob"] = np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))
    X["c_late_season"] = (X.week >= 13) * 1

    X = X[X.season >= 2013].reset_index(drop=True)
    X.to_parquet(f"{OUT}/rows.parquet"); print(X.shape, "->", f"{OUT}/rows.parquet")
    print(X[["ypa"]].describe().T)


if __name__ == "__main__":
    main()

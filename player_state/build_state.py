"""Player-state dataset for defensive YPA: completed-game state ledger, pregame expected state, and state-aware ratings.

Target everywhere: ordinary passing yards per attempt (non-penalty plays, pass == 1, pass_attempt == 1, no sacks, no two-point tries).
Rows: one per (game_id, posteam, defteam), regular season. 2016-2025 gives the same 5,278 keyed rows as the H1 panel.

Three jobs:
 1. Opponent-credit cleanup (history, postgame state allowed): ratings are fit on passer-level rows as
        ypa = league + qb[passer] + support[offense] + defense[defense] + b . (missing share of each core in that game)
    so a defense that faced a backup QB or a depleted receiving core / OL is judged against that state, not normal strength.
    Offense rows facing a depleted secondary / front are corrected the same way through the defensive core terms.
 2. Return restoration (history): a backup's games load onto his own qb[] effect, not support[offense], and the core slopes
    absorb depleted-core games, so when the established QB/core returns the offense's level is the established one.
 3. Pregame state (deployment): expected QB from a rule chosen on 2013-2015 only (previous starter, replaced when the final injury
    report lists him Out/Doubtful or the pregame depth chart moves him), expected core availability from the depth chart and the
    final injury report. The model then rates the matchup with the EXPECTED QB's prior rating, so a transition can come out better,
    worse, or equal; nothing assumes a direction.

Every rating used for a target week is fit only on games from weeks strictly before it (prior season + earlier current-season
weeks). Ridge penalties (attempt units) are chosen on 2013-2015 forward error only. Outputs in player_state/out/.
Usage (repo root): python player_state/build_state.py"""
import glob, os, itertools, numpy as np, pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); H = os.path.join(R, "player_state"); OUT = os.environ.get("OUT", os.path.join(H, "out"))
# LEAK_WEEK=2021,9 scrambles that week's play outcomes, passers, receivers and snap counts (used by leak_test_state.py)
LEAK = tuple(int(v) for v in os.environ["LEAK_WEEK"].split(",")) if os.environ.get("LEAK_WEEK") else None
os.makedirs(OUT, exist_ok=True)
PBP = sorted(glob.glob(f"{R}/te_share/data/pbp/pbp_20*.parquet"))
GAMES = f"{R}/def_ypa_lift/week7/inputs_2026-10-23/games.csv"
FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAC": "JAX"}
CORES = {"rec": 4, "ol": 5, "db": 5, "front": 6}
POS = {"ol": {"T", "G", "C", "OT", "OG", "OL", "LT", "RT", "LG", "RG"}, "db": {"CB", "S", "FS", "SS", "DB"},
       "front": {"DE", "DT", "NT", "LB", "OLB", "ILB", "MLB", "DL", "EDGE"}}
CORE_GAMES = 8                       # established core = most-used players over the team's previous 8 games (shown, not tuned)


# ----------------------------------------------------------------------------------------------------------------- inputs
def plays():
    c = ["game_id", "season", "week", "season_type", "posteam", "defteam", "pass", "pass_attempt", "sack", "penalty", "two_point_attempt",
         "passing_yards", "passer_player_id", "receiver_player_id"]
    p = pd.concat([pd.read_parquet(f, columns=c) for f in PBP if int(f[-12:-8]) <= 2025], ignore_index=True)
    p = p[(p.season_type == "REG") & p.posteam.notna() & p.defteam.notna()]
    t = p[(p["pass"] == 1) & (p.pass_attempt == 1) & (p.sack != 1) & ((p.penalty == 0) | p.penalty.isna()) & (p.two_point_attempt != 1)].copy()
    t["yds"] = t.passing_yards.fillna(0.0); t["t"] = t.season * 100 + t.week
    if LEAK:
        m = (t.season == LEAK[0]) & (t.week == LEAK[1]); rng = np.random.default_rng(0)
        t.loc[m, "yds"] = rng.permutation(t.loc[m, "yds"].to_numpy()) * 3 + 7
        t.loc[m, "passer_player_id"] = "FAKE_" + t.loc[m, "passer_player_id"].astype(str)
        t.loc[m, "receiver_player_id"] = "FAKE_" + t.loc[m, "receiver_player_id"].astype(str)
    return t


def schedule():
    g = pd.read_csv(GAMES); g = g[(g.game_type == "REG") & (g.season >= 2012)].copy()
    for c in ("home_team", "away_team"): g[c] = g[c].replace(FIX)
    g["kickoff"] = pd.to_datetime(g.gameday + " " + g.gametime).dt.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    return g


def injuries(g):
    fs = sorted(glob.glob(f"{H}/raw/inj/inj_20*.parquet"))
    i = pd.concat([pd.read_parquet(f).assign(src=os.path.basename(f)) for f in fs], ignore_index=True)
    i = i[(i.game_type == "REG") & i.report_status.isin(["Out", "Doubtful"])].copy()
    i["team"] = i.team.replace(FIX)
    ts = pd.to_datetime(i.get("date_modified"), utc=True, errors="coerce")
    i["inj_ts"] = ts
    # keep a report row only if it was last modified before the team's kickoff (rows without a timestamp are kept and
    # flagged unverified: the 2025 file has no date_modified column)
    gd = pd.concat([g[["season", "week", "home_team", "kickoff"]].rename(columns={"home_team": "team"}),
                    g[["season", "week", "away_team", "kickoff"]].rename(columns={"away_team": "team"})])
    i = i.merge(gd, on=["season", "week", "team"], how="left")
    i = i[i.inj_ts.isna() | (i.inj_ts < i.kickoff)]           # strictly before the team's kickoff (UTC timestamps)
    return i[["season", "week", "team", "gsis_id", "position", "report_status", "inj_ts"]]


def depth(g):
    """Pregame depth chart per team-week: rank of each player within position. 2012-2024 weekly NFL chart (no timestamp);
    2025 last ESPN snapshot strictly before kickoff (timestamped)."""
    w = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{H}/raw/depth_charts/dc_20*.parquet")) if int(f[-12:-8]) <= 2024])
    w = w[w.game_type == "REG"].copy(); w["team"] = w.club_code.replace(FIX); w["week"] = w.week.astype(int)
    w["rank"] = pd.to_numeric(w.depth_team, errors="coerce"); w["pos"] = w.position
    w = w[["season", "week", "team", "gsis_id", "pos", "rank"]].assign(dc_src="weekly_untimestamped", dc_ts=pd.NaT)
    s = pd.read_parquet(f"{H}/raw/depth_charts/dc_2025.parquet"); s["team"] = s.team.replace(FIX); s["ts"] = pd.to_datetime(s.dt, utc=True)
    rows = []
    g25 = g[g.season == 2025]
    for r in g25.itertuples(index=False):
        for team in (r.home_team, r.away_team):
            x = s[(s.team == team) & (s.ts < r.kickoff)]
            if not len(x): continue
            x = x[x.ts == x.ts.max()]
            rows.append(pd.DataFrame({"season": 2025, "week": r.week, "team": team, "gsis_id": x.gsis_id.values, "pos": x.pos_abb.values,
                                      "rank": x.pos_rank.values, "dc_src": "snapshot_pre_kickoff", "dc_ts": x.ts.values}))
    return pd.concat([w] + rows, ignore_index=True)


def snaps_map():
    pl = pd.read_parquet(f"{R}/te_share/data/aux/players.parquet", columns=["gsis_id", "pfr_id", "position"]).dropna(subset=["gsis_id"])
    s = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{R}/te_share/data/aux/snaps_20*.parquet")) if int(f[-12:-8]) <= 2025])
    s = s[s.game_type == "REG"].copy(); s["team"] = s.team.replace(FIX)
    s = s.merge(pl[["gsis_id", "pfr_id"]].dropna().drop_duplicates("pfr_id"), left_on="pfr_player_id", right_on="pfr_id", how="left")
    if LEAK:
        m = (s.season == LEAK[0]) & (s.week == LEAK[1]); s.loc[m, "gsis_id"] = "FAKE_" + s.loc[m, "gsis_id"].astype(str)
    return s[["game_id", "season", "week", "team", "gsis_id", "position", "offense_snaps", "defense_snaps"]]


# ------------------------------------------------------------------------------------------------- completed-game ledger
def ledger(t, s):
    """Per (game, team): actual primary QB, previous game's primary QB, and per-core postgame presence of the established core."""
    q = t[t.passer_player_id.notna()].groupby(["game_id", "season", "week", "posteam", "passer_player_id"]).size().rename("n").reset_index()
    q = q.sort_values(["n", "passer_player_id"]).drop_duplicates(["game_id", "posteam"], keep="last").rename(columns={"passer_player_id": "qb_actual"})
    q = q.sort_values(["posteam", "season", "week"]); q["qb_prev"] = q.groupby("posteam").qb_actual.shift()
    q["qb_transition"] = (q.qb_actual != q.qb_prev) & q.qb_prev.notna()
    # usage per player-game
    tg = t[t.receiver_player_id.notna()].groupby(["game_id", "season", "week", "posteam", "receiver_player_id"]).size().rename("use").reset_index()
    tg = tg.rename(columns={"posteam": "team", "receiver_player_id": "gsis_id"}).assign(core="rec")
    ol = s[s.position.isin(POS["ol"]) & (s.offense_snaps > 0)].assign(use=lambda z: z.offense_snaps, core="ol")
    db = s[s.position.isin(POS["db"]) & (s.defense_snaps > 0)].assign(use=lambda z: z.defense_snaps, core="db")
    fr = s[s.position.isin(POS["front"]) & (s.defense_snaps > 0)].assign(use=lambda z: z.defense_snaps, core="front")
    use = pd.concat([tg, ol[tg.columns], db[tg.columns], fr[tg.columns]], ignore_index=True).dropna(subset=["gsis_id"])
    played = s[(s.offense_snaps > 0) | (s.defense_snaps > 0)][["game_id", "team", "gsis_id"]].dropna().drop_duplicates()
    played_set = set(zip(played.game_id, played.team, played.gsis_id))
    # receivers can also count as present via a target in the game
    played_set |= set(zip(tg.game_id, tg.team, tg.gsis_id))
    return q, use, played_set


def established_cores(use, team_games):
    """For each team and each of its games, the established core from its previous CORE_GAMES games (strictly earlier)."""
    out = {}
    use = use.merge(team_games[["game_id", "team", "k"]], on=["game_id", "team"], how="inner")
    byteam = {tm: u for tm, u in use.groupby("team")}
    for tm, tgk in team_games.groupby("team"):
        u = byteam.get(tm)
        for gid, k in zip(tgk.game_id, tgk.k):
            if u is None: continue
            win = u[(u.k < k) & (u.k >= k - CORE_GAMES)]
            for core, n in CORES.items():
                w = win[win.core == core].groupby("gsis_id").use.sum().sort_values(ascending=False).head(n)
                if len(w): out[(gid, tm, core)] = (w.index.tolist(), (w / w.sum()).values if core == "rec" else np.full(len(w), 1 / len(w)))
    return out


# -------------------------------------------------------------------------------------------------- pregame expected state
def expected_state(tg, q, inj, dc, cores, rule=None):
    """Expected QB and expected core availability for every team-game, from pregame sources only."""
    injset = {}
    for r in inj.itertuples(index=False): injset.setdefault((r.season, r.week, r.team), set()).add(r.gsis_id)
    dcq = dc[dc.pos == "QB"].sort_values("rank")
    dcqb = {k: v.gsis_id.tolist() for k, v in dcq.groupby(["season", "week", "team"])}
    dcall = {k: set(v.gsis_id) for k, v in dc.groupby(["season", "week", "team"])}
    prev = q.set_index(["game_id", "posteam"]).qb_prev
    recs = []
    for r in tg.itertuples(index=False):
        key = (r.season, r.week, r.team); out = injset.get(key, set()); qbs = [x for x in dcqb.get(key, []) if x not in out]
        p = prev.get((r.game_id, r.team), None); p = None if (p is None or (isinstance(p, float) and np.isnan(p))) else p
        e1 = p if (p is not None and p not in out) else (qbs[0] if qbs else p)                   # R1: injury-driven change only
        e2 = qbs[0] if qbs else e1                                                              # R2: depth chart QB1 (healthy)
        rec = {"game_id": r.game_id, "team": r.team, "qb_prev": p, "qb_rule1": e1, "qb_rule2": e2, "prev_listed_out": p in out if p else False,
               "has_dc": key in dcall}
        for core in CORES:
            c = cores.get((r.game_id, r.team, core))
            if c is None: rec[f"exp_avail_{core}"] = np.nan; continue
            ids, w = c; on = dcall.get(key)
            avail = np.array([(x not in out) and (on is None or x in on) for x in ids], float)
            rec[f"exp_avail_{core}"] = float((avail * w).sum())
        recs.append(rec)
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------------------------------------------- ratings
def fit3(y, a, qi, oi, di, X, nq, no, nd, lq, lt, iters=25):
    """Weighted alternating ridge: y - lg = qb[qi] + sup[oi] + def[di] + X b. a = attempts."""
    lg = (y * a).sum() / a.sum(); r = y - lg
    qb, sup, de = np.zeros(nq), np.zeros(no), np.zeros(nd); b = np.zeros(X.shape[1]) if X is not None else None
    wq, wo, wd = np.bincount(qi, a, nq), np.bincount(oi, a, no), np.bincount(di, a, nd)
    for _ in range(iters):
        xb = X @ b if X is not None else 0.0
        qb = np.bincount(qi, a * (r - sup[oi] - de[di] - xb), nq) / (wq + lq) if lq is not None else qb
        sup = np.bincount(oi, a * (r - qb[qi] - de[di] - xb), no) / (wo + lt)
        de = np.bincount(di, a * (r - qb[qi] - sup[oi] - xb), nd) / (wd + lt)
        if X is not None:
            e = r - qb[qi] - sup[oi] - de[di]; Xa = X * a[:, None]
            b = np.linalg.solve(X.T @ Xa + 1e-6 * np.eye(X.shape[1]), Xa.T @ e)
    return lg, qb, sup, de, b


def snapshot_ratings(qrows, season, week, lq, lt, use_cov=True):
    win = qrows[(qrows.season == season - 1) | ((qrows.season == season) & (qrows.week < week))]
    if not len(win): return None
    qc, qn = pd.factorize(win.passer); oc, on_ = pd.factorize(win.posteam); dcd, dn = pd.factorize(win.defteam)
    X = win[["miss_rec", "miss_ol", "miss_db", "miss_front"]].fillna(0).to_numpy(float) if use_cov else None
    lg, qb, sup, de, b = fit3(win.ypa.to_numpy(float), win.att.to_numpy(float), qc, oc, dcd, X, len(qn), len(on_), len(dn), lq, lt)
    return {"lg": lg, "qb": dict(zip(qn, qb)), "sup": dict(zip(on_, sup)), "def": dict(zip(dn, de)), "b": b,
            "qb_att": win.groupby("passer").att.sum().to_dict()}


def snapshot_twoway(trows, season, week, lt):
    win = trows[(trows.season == season - 1) | ((trows.season == season) & (trows.week < week))]
    if not len(win): return None
    oc, on_ = pd.factorize(win.posteam); dcd, dn = pd.factorize(win.defteam); z = np.zeros(len(win), int)
    lg, _, off, de, _ = fit3(win.ypa.to_numpy(float), win.att.to_numpy(float), z, oc, dcd, None, 1, len(on_), len(dn), None, lt)
    return {"lg": lg, "off": dict(zip(on_, off)), "def": dict(zip(dn, de))}


# ------------------------------------------------------------------------------------------------------------------ main
def main():
    t = plays(); g = schedule(); s = snaps_map(); inj = injuries(g); dc = depth(g)
    trows = t.groupby(["game_id", "season", "week", "posteam", "defteam"]).agg(att=("yds", "size"), yds=("yds", "sum")).reset_index()
    trows["ypa"] = trows.yds / trows.att
    q, use, played = ledger(t, s)
    # team-game index (k = running game number per team) for core windows
    tgm = pd.concat([trows[["game_id", "season", "week", "posteam"]].rename(columns={"posteam": "team"}),
                     trows[["game_id", "season", "week", "defteam"]].rename(columns={"defteam": "team"})]).drop_duplicates(["game_id", "team"])
    tgm = tgm.sort_values(["team", "season", "week"]); tgm["k"] = tgm.groupby("team").cumcount()
    cores = established_cores(use, tgm)
    # postgame missing share per core per team-game (ledger, history only)
    miss = []
    for (gid, tm, core), (ids, w) in cores.items():
        pres = np.array([(gid, tm, x) in played for x in ids], float); miss.append((gid, tm, core, float(((1 - pres) * w).sum())))
    miss = pd.DataFrame(miss, columns=["game_id", "team", "core", "miss"]).pivot_table(index=["game_id", "team"], columns="core", values="miss").reset_index()
    miss.columns = ["game_id", "team"] + [f"miss_{c}" for c in miss.columns[2:]]
    # passer-level rating rows with the game's core state (offense cores from posteam, defense cores from defteam)
    qrows = t[t.passer_player_id.notna()].groupby(["game_id", "season", "week", "posteam", "defteam", "passer_player_id"]).agg(att=("yds", "size"), yds=("yds", "sum")).reset_index()
    qrows = qrows.rename(columns={"passer_player_id": "passer"}); qrows["ypa"] = qrows.yds / qrows.att
    mo = miss[["game_id", "team", "miss_rec", "miss_ol"]].rename(columns={"team": "posteam"}); md = miss[["game_id", "team", "miss_db", "miss_front"]].rename(columns={"team": "defteam"})
    qrows = qrows.merge(mo, on=["game_id", "posteam"], how="left").merge(md, on=["game_id", "defteam"], how="left")
    led = q.merge(miss.rename(columns={"team": "posteam"}), on=["game_id", "posteam"], how="left")
    led.to_parquet(f"{OUT}/ledger_completed_games.parquet")

    # pregame expected state for every team-game
    ex = expected_state(tgm, q, inj, dc, cores)
    ex = ex.merge(q[["game_id", "posteam", "qb_actual"]].rename(columns={"posteam": "team"}), on=["game_id", "team"], how="left")
    m = ex.game_id.str[:4].astype(int).between(2013, 2015) & ex.qb_prev.notna()
    acc = {r: (ex.loc[m, r] == ex.loc[m, "qb_actual"]).mean() for r in ("qb_prev", "qb_rule1", "qb_rule2")}
    best = max(acc, key=acc.get); print("expected-QB rule accuracy on 2013-2015:", {k: round(v, 3) for k, v in acc.items()}, "-> using", best)
    ex["qb_expected"] = ex[best]; ex["qb_expected_rule"] = best
    ex["dc_source"] = np.where(ex.game_id.str[:4] == "2025", "snapshot_pre_kickoff", "weekly_untimestamped")
    ex["inj_timestamp_verified"] = ~ex.game_id.str[:4].isin(["2025"])
    ex.to_parquet(f"{OUT}/expected_state_pregame.parquet")

    # choose ridge penalties on 2013-2015 forward error of the pregame matchup expectation
    target = trows.merge(ex[["game_id", "team", "qb_expected", "exp_avail_rec", "exp_avail_ol"]].rename(columns={"team": "posteam"}), on=["game_id", "posteam"], how="left")
    target = target.merge(ex[["game_id", "team", "exp_avail_db", "exp_avail_front"]].rename(columns={"team": "defteam"}), on=["game_id", "defteam"], how="left")
    weeks = sorted(set(zip(target.season, target.week)))
    GRID = [300, 1000, 3000, 10000, 30000]

    def predict_week(sn, rows, oracle=False):
        qb = rows.qb_expected if not oracle else rows.merge(q[["game_id", "posteam", "qb_actual"]], on=["game_id", "posteam"], how="left").qb_actual.values
        b = sn["b"] if sn["b"] is not None else np.zeros(4)
        cov = np.column_stack([1 - rows.exp_avail_rec.fillna(1), 1 - rows.exp_avail_ol.fillna(1), 1 - rows.exp_avail_db.fillna(1), 1 - rows.exp_avail_front.fillna(1)])
        qv = np.array([sn["qb"].get(x, 0.0) for x in qb]); sv = rows.posteam.map(sn["sup"]).fillna(0).to_numpy(); dv = rows.defteam.map(sn["def"]).fillna(0).to_numpy()
        return sn["lg"] + qv + sv + dv + cov @ b, qv, sv, dv, cov @ b

    tune = []
    for lq, lt in itertools.product(GRID, GRID):
        err, wsum = 0.0, 0.0
        for (sea, wk) in weeks:
            if not (2013 <= sea <= 2015): continue
            rows = target[(target.season == sea) & (target.week == wk)]
            sn = snapshot_ratings(qrows, sea, wk, lq, lt)
            if sn is None: continue
            p = predict_week(sn, rows)[0]; err += (rows.att * (rows.ypa - p) ** 2).sum(); wsum += rows.att.sum()
        tune.append((err / wsum, lq, lt)); print("tune", lq, lt, round(err / wsum, 4), flush=True)
    _, LQ, LT = min(tune)
    tune2 = []
    for lt in GRID:
        err, wsum = 0.0, 0.0
        for (sea, wk) in weeks:
            if not (2013 <= sea <= 2015): continue
            rows = target[(target.season == sea) & (target.week == wk)]; sn = snapshot_twoway(trows, sea, wk, lt)
            if sn is None: continue
            p = sn["lg"] + rows.posteam.map(sn["off"]).fillna(0) + rows.defteam.map(sn["def"]).fillna(0)
            err += (rows.att * (rows.ypa - p) ** 2).sum(); wsum += rows.att.sum()
        tune2.append((err / wsum, lt))
    _, LT2 = min(tune2)
    pd.DataFrame(tune, columns=["wmse_2013_15", "lambda_qb", "lambda_team"]).to_csv(f"{OUT}/tuning_state_model.csv", index=False)
    pd.DataFrame(tune2, columns=["wmse_2013_15", "lambda_team"]).to_csv(f"{OUT}/tuning_twoway.csv", index=False)
    print("chosen on 2013-2015: lambda_qb", LQ, "lambda_team", LT, "| two-way lambda", LT2)

    # features for every target row, each week from its own snapshot
    feats = []
    for (sea, wk) in weeks:
        if sea < 2013: continue
        rows = target[(target.season == sea) & (target.week == wk)].copy()
        sn = snapshot_ratings(qrows, sea, wk, LQ, LT); tw = snapshot_twoway(trows, sea, wk, LT2)
        if sn is None: continue
        p, qv, sv, dv, cb = predict_week(sn, rows)
        po, qo, _, _, _ = predict_week(sn, rows, oracle=True)
        est = q.merge(rows[["game_id", "posteam"]], on=["game_id", "posteam"])      # established QB = most prior attempts in window for team
        win = qrows[(qrows.season == sea - 1) | ((qrows.season == sea) & (qrows.week < wk))]
        top = win.groupby(["posteam", "passer"]).att.sum().reset_index().sort_values("att").drop_duplicates("posteam", keep="last").set_index("posteam").passer
        qest = rows.posteam.map(top)
        rows["st_qb_est"] = [sn["qb"].get(x, 0.0) for x in qest]
        rows["st_match"], rows["st_qb_exp"], rows["st_sup"], rows["st_def"], rows["st_core"] = p, qv, sv, dv, cb
        rows["st_qb_delta"] = rows.st_qb_exp - rows.st_qb_est
        rows["st_qb_exp_logatt"] = np.log1p([sn["qb_att"].get(x, 0.0) for x in rows.qb_expected])
        rows["st_qb_changed"] = (rows.qb_expected != qest).astype(int)
        rows["tw_naive"] = tw["lg"] + rows.posteam.map(tw["off"]).fillna(0) + rows.defteam.map(tw["def"]).fillna(0)
        rows["oracle_st_match"], rows["oracle_st_qb"] = po, qo
        rows["qb_established"] = qest.values
        feats.append(rows)
    F = pd.concat(feats, ignore_index=True)
    F = F.merge(q[["game_id", "posteam", "qb_actual", "qb_prev", "qb_transition"]], on=["game_id", "posteam"], how="left")
    F.to_parquet(f"{OUT}/state_features.parquet")
    print(F.shape, "->", f"{OUT}/state_features.parquet"); print("panel rows 2016-2025:", int(F.season.between(2016, 2025).sum()))


if __name__ == "__main__":
    main()

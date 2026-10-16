"""Build the team-game TE receiving-yards-share table and pregame features.

Target: TE receiving yards / all team receiving yards, per offense team-game (regular season).
Every feature for (season, week W) uses only games from earlier seasons or weeks < W of that season,
plus information published before kickoff (roster, Wednesday-Friday injury report, Vegas lines).
Writes data/team_games.parquet.
"""
import numpy as np, pandas as pd, glob, os

def I(df):
    """normalize season/week to int64 so merges line up across nflverse files"""
    for c in ("season", "week"):
        if c in df:
            df[c] = df[c].astype("int64")
    for c in ("team", "posteam", "defteam", "home_team", "away_team"):
        if c in df:
            df[c] = df[c].replace(TEAM_FIX)
    return df

TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "JAC": "JAX", "LAR": "LA"}

D = os.path.join(os.path.dirname(__file__), "data")
YEARS = range(2012, 2027)

# ---------- positions (season-level mode from weekly rosters, fallback to players table) ----------
ros = pd.concat([pd.read_parquet(f"{D}/aux/roster_weekly_{y}.parquet") for y in YEARS], ignore_index=True)
ros = I(ros)
ros = ros[ros.game_type.fillna("REG") == "REG"] if "game_type" in ros else ros
pos_season = (ros.dropna(subset=["gsis_id", "position"]).groupby(["season", "gsis_id", "position"]).size()
              .rename("n").reset_index().sort_values("n").drop_duplicates(["season", "gsis_id"], keep="last")
              .rename(columns={"position": "pos"})[["season", "gsis_id", "pos"]])
players = pd.read_parquet(f"{D}/aux/players.parquet")
pid_col = "gsis_id" if "gsis_id" in players else "player_id"
pos_any = players.set_index(pid_col)["position"].dropna().to_dict()

# ---------- play-by-play -> player-game receiving lines ----------
cols = ["season", "week", "season_type", "game_id", "posteam", "defteam", "home_team", "pass_attempt",
        "sack", "complete_pass", "two_point_attempt", "receiver_player_id", "receiving_yards", "air_yards"]
pbp = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{D}/pbp/pbp_*.parquet"))],
                ignore_index=True)
pbp = I(pbp)
pbp = pbp[(pbp.season_type == "REG") & (pbp.two_point_attempt != 1) & (pbp.pass_attempt == 1)
          & (pbp.sack != 1) & pbp.receiver_player_id.notna()].copy()
# position as listed on that week's roster (no look-ahead); season mode / players table only when the week is missing
pos_week = (ros.dropna(subset=["gsis_id", "position"]).drop_duplicates(["season", "week", "gsis_id"])
            [["season", "week", "gsis_id", "position"]].rename(columns={"position": "pos_wk"}))
pbp = pbp.merge(pos_week, left_on=["season", "week", "receiver_player_id"], right_on=["season", "week", "gsis_id"], how="left").drop(columns="gsis_id")
pbp = pbp.merge(pos_season, left_on=["season", "receiver_player_id"], right_on=["season", "gsis_id"], how="left")
pbp["pos"] = pbp.pos_wk.fillna(pbp.pos).fillna(pbp.receiver_player_id.map(pos_any)).fillna("UNK")
pbp["grp"] = np.select([pbp.pos == "TE", pbp.pos == "WR", pbp.pos.isin(["RB", "FB"])], ["TE", "WR", "RB"], "OTH")
pbp["ryds"] = pbp.receiving_yards.fillna(0) * (pbp.complete_pass == 1)
pbp["ay"] = pbp.air_yards.fillna(0)

pg = (pbp.groupby(["season", "week", "game_id", "posteam", "receiver_player_id", "grp"])
      .agg(yds=("ryds", "sum"), tgt=("ryds", "size"), ay=("ay", "sum")).reset_index())

# team-game totals by group
tg = pg.pivot_table(index=["season", "week", "game_id", "posteam"], columns="grp",
                    values=["yds", "tgt", "ay"], aggfunc="sum", fill_value=0)
tg.columns = [f"{a}_{b}" for a, b in tg.columns]
tg = tg.reset_index()
for v in ["yds", "tgt", "ay"]:
    tg[f"{v}_ALL"] = tg[[c for c in tg.columns if c.startswith(v + "_")]].sum(axis=1)
tg = tg[tg.yds_ALL > 0].copy()
tg["y"] = tg.yds_TE / tg.yds_ALL

meta = pbp.groupby(["game_id", "posteam"]).agg(defteam=("defteam", "first"), home_team=("home_team", "first")).reset_index()
tg = tg.merge(meta, on=["game_id", "posteam"])
tg["home"] = (tg.posteam == tg.home_team).astype(int)

# ---------- schedule rows (includes the not-yet-played week so it gets features too) ----------
games = pd.read_csv(f"{D}/aux/games.csv")
games = I(games)
games = games[(games.game_type == "REG") & games.season.isin(YEARS)]
sched = pd.concat([
    games.assign(posteam=games.home_team, defteam=games.away_team, home=1,
                 spread=games.spread_line, total=games.total_line),
    games.assign(posteam=games.away_team, defteam=games.home_team, home=0,
                 spread=-games.spread_line, total=games.total_line)])
sched = sched[["season", "week", "game_id", "posteam", "defteam", "home", "spread", "total", "result"]]
sched["implied"] = sched.total / 2 + sched.spread / 2  # spread_line is home-minus-away margin
# nflverse game ids use current abbreviations; pbp uses the same in recent releases
tg = sched.merge(tg.drop(columns=["defteam", "home", "home_team"]), on=["season", "week", "game_id", "posteam"], how="left")
tg["played"] = tg.y.notna()

# ---------- running season-to-date and prior-season aggregates ----------
tg = tg.sort_values(["posteam", "season", "week"]).reset_index(drop=True)

def ytd(df, key, cols):
    """season-to-date sums over weeks strictly before the row's week"""
    g = df.groupby(["season", key])
    out = {}
    for c in cols:
        s = df[c].fillna(0)
        out[c + "_ytd"] = g[c].transform(lambda x: x.fillna(0).cumsum().shift(1).fillna(0))
    out["n_ytd"] = g["played"].transform(lambda x: x.astype(int).cumsum().shift(1).fillna(0))
    return pd.DataFrame(out, index=df.index)

num = ["yds_TE", "yds_ALL", "tgt_TE", "tgt_ALL", "ay_TE", "ay_ALL", "yds_WR", "yds_RB"]
o = ytd(tg, "posteam", num)
tg = tg.join(o.add_prefix("o_"))

# defense side: same stats allowed, keyed by defteam
dd = tg[["season", "week", "defteam", "played"] + num].sort_values(["defteam", "season", "week"])
d = ytd(dd, "defteam", num).add_prefix("d_")
tg = tg.join(d)

# prior full season per team (offense and defense)
season_tot = (tg[tg.played].groupby(["season", "posteam"])[num].sum().reset_index())
season_tot["prev_off_share"] = season_tot.yds_TE / season_tot.yds_ALL
season_tot["prev_off_tgt_share"] = season_tot.tgt_TE / season_tot.tgt_ALL
season_tot["season"] += 1
tg = tg.merge(season_tot[["season", "posteam", "prev_off_share", "prev_off_tgt_share"]], on=["season", "posteam"], how="left")
dtot = tg[tg.played].groupby(["season", "defteam"])[num].sum().reset_index()
dtot["prev_def_share"] = dtot.yds_TE / dtot.yds_ALL
dtot["season"] += 1
tg = tg.merge(dtot[["season", "defteam", "prev_def_share"]], on=["season", "defteam"], how="left")

# league level of the previous season (the only "league" info allowed for a forecast)
lg = tg[tg.played].groupby("season").apply(lambda x: x.yds_TE.sum() / x.yds_ALL.sum()).rename("lg_prev").reset_index()
lg["season"] += 1
tg = tg.merge(lg, on="season", how="left")
# league level so far this season (weeks before W)
wk = tg[tg.played].groupby(["season", "week"])[["yds_TE", "yds_ALL"]].sum().reset_index().sort_values(["season", "week"])
wk["lg_ytd_te"] = wk.groupby("season").yds_TE.cumsum().shift(1)
wk["lg_ytd_all"] = wk.groupby("season").yds_ALL.cumsum().shift(1)
wk.loc[wk.groupby("season").cumcount() == 0, ["lg_ytd_te", "lg_ytd_all"]] = np.nan
full_weeks = pd.MultiIndex.from_product([sorted(tg.season.unique()), range(1, 19)], names=["season", "week"]).to_frame(index=False)
wk = full_weeks.merge(wk[["season", "week", "yds_TE", "yds_ALL"]], how="left").fillna(0).sort_values(["season", "week"])
wk["lg_ytd_te"] = wk.groupby("season").yds_TE.transform(lambda x: x.cumsum().shift(1))
wk["lg_ytd_all"] = wk.groupby("season").yds_ALL.transform(lambda x: x.cumsum().shift(1))
tg = tg.merge(wk[["season", "week", "lg_ytd_te", "lg_ytd_all"]], on=["season", "week"], how="left")

# ---------- personnel: returning TE production, injury-report availability, snap shares ----------
# player-season receiving yards and games (any team)
ps = pg.groupby(["season", "receiver_player_id", "grp"]).agg(yds=("yds", "sum"), g=("game_id", "nunique")).reset_index()
ps_prev = ps.assign(season=ps.season + 1).rename(columns={"yds": "prev_yds", "g": "prev_g"})
# player-season-to-date (before week W), per team
pg = pg.sort_values(["season", "receiver_player_id", "week"])
pg["p_ytd"] = pg.groupby(["season", "receiver_player_id"]).yds.transform(lambda x: x.cumsum().shift(1).fillna(0))

r = ros[["season", "week", "team", "gsis_id", "position", "status"]].dropna(subset=["gsis_id"])
r = r[r.status.isin(["ACT", "INA", "RES"]) & r.position.isin(["TE", "WR"])]
inj = pd.concat([pd.read_parquet(f"{D}/aux/inj_{y}.parquet") for y in YEARS], ignore_index=True)
inj = I(inj)
inj = inj[inj.game_type == "REG"][["season", "week", "team", "gsis_id", "report_status"]]
r = r.merge(inj, on=["season", "week", "team", "gsis_id"], how="left")
# pregame "unavailable": reserve list or Out/Doubtful on the final injury report
r["out"] = (r.status == "RES") | r.report_status.isin(["Out", "Doubtful"])
r = r.merge(ps_prev[["season", "receiver_player_id", "prev_yds", "prev_g"]].groupby(["season", "receiver_player_id"]).sum().reset_index(),
            left_on=["season", "gsis_id"], right_on=["season", "receiver_player_id"], how="left").drop(columns="receiver_player_id")
# season-to-date yards for each rostered player through week W-1
cum = pg.groupby(["season", "week", "receiver_player_id"]).yds.sum().reset_index()
cum = cum.sort_values(["season", "receiver_player_id", "week"])
r = r.sort_values(["season", "week"])
cum["cum"] = cum.groupby(["season", "receiver_player_id"]).yds.cumsum()
cum = cum.rename(columns={"receiver_player_id": "gsis_id"})
r = pd.merge_asof(r.sort_values("week"), cum[["season", "week", "gsis_id", "cum"]].assign(week=lambda x: x.week + 1).sort_values("week"),
                  on="week", by=["season", "gsis_id"], direction="backward")
r["cum"] = r.cum.fillna(0)
r[["prev_yds", "prev_g"]] = r[["prev_yds", "prev_g"]].fillna(0)
r["prev_ypg"] = np.where(r.prev_g > 0, r.prev_yds / r.prev_g.clip(lower=1), 0)
r["prod"] = r.prev_yds + r.cum  # last season + this season so far

def roster_agg(x):
    av = x[~x.out]
    te_all, te_av = x[x.position == "TE"], av[av.position == "TE"]
    wr_all, wr_av = x[x.position == "WR"], av[av.position == "WR"]
    return pd.Series({
        "te_prod_avail": te_av.nlargest(2, "prod")["prod"].sum(),
        "te_prod_all": te_all.nlargest(2, "prod")["prod"].sum(),
        "wr_prod_avail": wr_av.nlargest(3, "prod")["prod"].sum(),
        "wr_prod_all": wr_all.nlargest(3, "prod")["prod"].sum(),
        "te_prevypg_avail": te_av.nlargest(1, "prev_ypg").prev_ypg.sum(),
        "wr_prevypg_avail": wr_av.nlargest(2, "prev_ypg").prev_ypg.sum(),
        "te1_out": float(len(te_all) > 0 and te_all.nlargest(1, "prod").out.iat[0]),
        "wr1_out": float(len(wr_all) > 0 and wr_all.nlargest(1, "prod").out.iat[0]),
    })

rag = r.groupby(["season", "week", "team"]).apply(roster_agg).reset_index().rename(columns={"team": "posteam"})
tg = tg.merge(rag, on=["season", "week", "posteam"], how="left")
tg["te_rel_prod"] = tg.te_prod_avail / (tg.te_prod_avail + tg.wr_prod_avail + 1)
tg["te_lost_frac"] = 1 - tg.te_prod_avail / tg.te_prod_all.replace(0, np.nan)
tg["wr_lost_frac"] = 1 - tg.wr_prod_avail / tg.wr_prod_all.replace(0, np.nan)

# snap shares season-to-date (TE1 offense snap pct, and number of TEs used = 2-TE usage proxy)
sn = pd.concat([pd.read_parquet(f"{D}/aux/snaps_{y}.parquet") for y in YEARS], ignore_index=True)
sn = I(sn)
sn = sn[(sn.game_type == "REG") & (sn.position == "TE")]
tsn = sn.groupby(["season", "week", "team"]).offense_pct.agg(te_pct_sum="sum", te1_pct="max").reset_index()
tsn = tsn.sort_values(["team", "season", "week"])
for c in ["te_pct_sum", "te1_pct"]:
    tsn[c + "_ytd"] = tsn.groupby(["season", "team"])[c].transform(lambda x: x.expanding().mean())
tsn_prev = tsn.groupby(["season", "team"])[["te_pct_sum", "te1_pct"]].mean().reset_index()
tsn_prev["season"] += 1
tsn_prev = tsn_prev.rename(columns={"te_pct_sum": "te_pct_sum_prev", "te1_pct": "te1_pct_prev", "team": "posteam"})
# as-of join: running mean through the last game strictly before week W (works for unplayed weeks too)
snap_ytd = tsn[["season", "week", "team", "te_pct_sum_ytd", "te1_pct_ytd"]].rename(columns={"team": "posteam"})
snap_ytd = snap_ytd.assign(week=snap_ytd.week + 1).sort_values("week")
tg = pd.merge_asof(tg.sort_values("week"), snap_ytd, on="week", by=["season", "posteam"], direction="backward")
tg = tg.merge(tsn_prev, on=["season", "posteam"], how="left")

tg.to_parquet(f"{D}/team_games.parquet")
print(tg.shape, tg.played.sum())
print(tg[tg.played].groupby("season").apply(lambda x: pd.Series({
    "lg_share": x.yds_TE.sum() / x.yds_ALL.sum(), "mean_y": x.y.mean(), "sd_y": x.y.std(), "n": len(x)})).round(4))

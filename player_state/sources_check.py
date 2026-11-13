"""Which pregame starter sources can be trusted, and for which seasons?

For every regular-season team-game 2016-2025 the actual primary QB (most dropbacks in the game) is compared with:
  A. weekly NFL depth chart QB1 (2016-2024; no timestamp),
  B. last timestamped ESPN depth-chart snapshot strictly before kickoff (2025),
  C. "carry forward": the primary QB of the team's previous game (always pregame-valid).
Accuracy is reported for all games and for transition games (actual starter differs from last game's starter). A pregame source
should beat C on transitions but cannot be perfect; near-100% transition accuracy means the source was revised after the game.
Usage: python player_state/sources_check.py   (from the repo root)"""
import glob, os, numpy as np, pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); H = os.path.join(R, "player_state")
PBP = sorted(glob.glob(f"{R}/te_share/data/pbp/pbp_20*.parquet"))
GAMES = f"{R}/def_ypa_lift/week7/inputs_2026-10-23/games.csv"
FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAC": "JAX"}


def actual_starters():
    c = ["game_id", "season", "week", "season_type", "posteam", "pass_attempt", "passer_player_id", "two_point_attempt"]
    p = pd.concat([pd.read_parquet(f, columns=c) for f in PBP if 2015 <= int(f[-12:-8]) <= 2025])
    p = p[(p.season_type == "REG") & (p.pass_attempt == 1) & (p.two_point_attempt != 1) & p.passer_player_id.notna()]
    k = p.groupby(["game_id", "season", "week", "posteam", "passer_player_id"]).size().rename("n").reset_index()
    k = k.sort_values(["n", "passer_player_id"]).drop_duplicates(["game_id", "posteam"], keep="last").rename(columns={"passer_player_id": "actual"})
    k = k.sort_values(["posteam", "season", "week"]); k["prev"] = k.groupby("posteam").actual.shift()
    return k


def weekly_dc():
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{H}/raw/depth_charts/dc_20*.parquet")) if int(f[-12:-8]) <= 2024])
    q = d[(d.position == "QB") & (d.depth_team.astype(str) == "1") & (d.game_type == "REG")].copy()
    q["team"] = q.club_code.replace(FIX); q["week"] = q.week.astype(int)
    return q.drop_duplicates(["season", "week", "team"])[["season", "week", "team", "gsis_id"]].rename(columns={"gsis_id": "dc_qb1"})


def snapshot_dc(season, games):
    d = pd.read_parquet(f"{H}/raw/depth_charts/dc_{season}.parquet"); d = d[d.pos_abb == "QB"].copy()
    d["team"] = d.team.replace(FIX); d["ts"] = pd.to_datetime(d.dt, utc=True)
    snaps = d[d.pos_rank == 1].sort_values("ts")
    g = games[games.season == season].copy()
    out = []
    for r in g.itertuples(index=False):
        for team in (r.home_team, r.away_team):
            s = snaps[(snaps.team == team) & (snaps.ts < r.kickoff)]
            if len(s): out.append((r.game_id, team, s.gsis_id.iloc[-1], s.ts.iloc[-1], r.kickoff))
    return pd.DataFrame(out, columns=["game_id", "posteam", "snap_qb1", "snap_ts", "kickoff"])


if __name__ == "__main__":
    games = pd.read_csv(GAMES); games = games[games.game_type == "REG"].copy()
    games["kickoff"] = pd.to_datetime(games.gameday + " " + games.gametime).dt.tz_localize("US/Eastern").dt.tz_convert("UTC")
    for c in ("home_team", "away_team"): games[c] = games[c].replace(FIX)
    a = actual_starters(); a = a[a.season >= 2016]
    a = a.merge(weekly_dc().rename(columns={"team": "posteam"}), on=["season", "week", "posteam"], how="left")
    a = a.merge(snapshot_dc(2025, games), on=["game_id", "posteam"], how="left")
    a["trans"] = a.actual != a.prev
    rows = []
    for era, m, src in (("2016-2024 weekly depth chart", a.season <= 2024, "dc_qb1"), ("2025 last pre-kickoff snapshot", a.season == 2025, "snap_qb1")):
        x = a[m]
        for lab, sub in (("all games", x), ("transition games", x[x.trans])):
            rows.append({"era": era, "games": lab, "n": len(sub), "source coverage": sub[src].notna().mean(),
                         "source = actual": (sub[src] == sub.actual).mean(), "carry-forward = actual": (sub.prev == sub.actual).mean()})
    t = pd.DataFrame(rows); print(t.round(3).to_string(index=False)); t.to_csv(f"{H}/sources_check.csv", index=False)
    s25 = a[a.season == 2025].dropna(subset=["snap_ts"]); lag = (s25.kickoff - s25.snap_ts).dt.total_seconds() / 3600
    print("2025 snapshot age at kickoff (hours): median", round(lag.median(), 1), "90th pct", round(lag.quantile(.9), 1))
    a.to_parquet(f"{H}/starter_sources.parquet")

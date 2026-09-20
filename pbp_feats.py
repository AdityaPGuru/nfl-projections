"""Python port of the user's R feature engine (22 offense + 23 defense play-by-play variables), made point-in-time for
every (season, week) 2012-2025, plus clean QB/defense/game-context aggregates from the same play-by-play."""
import os, glob, numpy as np, pandas as pd

S = os.environ["S"]
COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team", "pass", "rush", "penalty",
        "qb_scramble", "pass_attempt", "complete_pass", "passing_yards", "cp", "cpoe", "air_yards", "epa", "first_down", "touchdown",
        "wp", "ydstogo", "yards_after_catch", "sack", "qb_hit", "down", "qtr", "score_differential", "yardline_100",
        "half_seconds_remaining", "passer_player_id", "id", "spread_line", "total_line", "roof", "wind", "temp", "success"]


def load():
    df = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob(S + "/pbp/pbp_*.parquet"))], ignore_index=True)
    df = df[df.posteam.notna() & df.defteam.notna() & ((df["pass"] == 1) | (df.rush == 1)) & ((df.penalty == 0) | df.penalty.isna())].copy()
    c = lambda col, v=0.0: df[col].astype(float).fillna(v)
    df["comp"], df["air"], df["epa_v"], df["cpoe_v"], df["cp_v"] = c("complete_pass"), c("air_yards"), c("epa"), c("cpoe"), c("cp")
    df["pyds"], df["yac"], df["ytg"], df["wp_v"] = c("passing_yards"), c("yards_after_catch"), c("ydstogo", 10), c("wp", 0.5)
    df["sr"] = (df.epa_v > 0).astype(float)
    df["fd"] = ((c("first_down") == 1) | (c("touchdown") == 1)).astype(float)
    df["dropback"] = ((df["pass"] == 1) | (df.qb_scramble == 1)).astype(float)
    df["pressure"] = ((df.dropback == 1) & ((df.sack == 1) | (df.qb_hit == 1) | (df.qb_scramble == 1))).astype(float)
    return df


def masks(d):
    P = d["pass"] == 1
    sd, hs = d.score_differential, d.half_seconds_remaining
    lead, trail, blow = sd > 0, sd < 0, sd.abs() > 16
    neut = (d.wp_v >= .15) & (d.wp_v <= .85) & (hs > 120)
    neut2 = (d.wp_v >= .20) & (d.wp_v <= .80) & (hs > 120)
    d1, d2, d3, d4 = (d.down == k for k in (1, 2, 3, 4))
    early, late = d.down.isin([1, 2]), d.down.isin([3, 4])
    q1, q2, q3, q4 = (d.qtr == k for k in (1, 2, 3, 4))
    C, short, med, long_ = d.comp == 1, d.ytg <= 4, (d.ytg >= 5) & (d.ytg <= 8), d.ytg >= 9
    q = lambda p: (lambda s: s.quantile(p))
    off = {  # name: (mask, column, aggregation)
        "off_late_downs_std_epa_std": (P & late, "epa_v", "std"), "off_d4_std_epa_neg_mean": (P & d4 & (d.epa_v < 0), "epa_v", "mean"),
        "off_d1_std_epa_pos_mean": (P & d1 & (d.epa_v > 0), "epa_v", "mean"), "off_blowout_std_air_comp_mean": (P & blow & C, "air", "mean"),
        "off_q3_pos_cpoe_std": (P & q3 & (d.cpoe_v > 0), "cpoe_v", "std"), "off_air_d4_short_comp_rate": (P & d4 & short, "comp", "mean"),
        "off_cpoe_all_past_sticks_mean": (P & (d.air >= d.ytg), "cpoe_v", "mean"), "off_neut_early_downs_adot_mean": (P & neut & early, "air", "mean"),
        "off_yac_median_neutral_q2": (P & neut & q2 & C, "yac", "median"), "off_cp_mean_neutral_q4": (P & neut & q4, "cp_v", "mean"),
        "off_cpoe_p90_all_short_distance": (P & short, "cpoe_v", q(.9)), "off_neut_adot_p75_d2_short": (P & neut & d2 & short, "air", q(.75)),
        "off_cp_p75_q1": (P & q1, "cp_v", q(.75)), "off_cpoe_mean_d3_short": (P & d3 & short, "cpoe_v", "mean"),
        "off_cpoe_p25_trail": (P & trail, "cpoe_v", q(.25)), "off_adot_p90_d1": (P & d1, "air", q(.9)),
        "off_d3_long_yac": (P & d3 & long_ & C, "yac", "mean"), "off_lead_late_air_yards": (P & lead & late, "air", "mean"),
        "off_comp_neut_d1_q1_epa_std": (P & C & neut & d1 & q1, "epa_v", "std"),
        "off_short_sticks_early_q1_cpoe_neg_mean": (P & (d.air < d.ytg) & early & q1 & (d.cpoe_v < 0), "cpoe_v", "mean"),
        "off_sr_trail_d1_q1_mean": (P & trail & d1 & q1, "sr", "mean"), "off_press_rate_d3": ((d.dropback == 1) & d3, "pressure", "mean")}
    de = {
        "def_blowout_std_air_comp_mean": (P & blow & C, "air", "mean"), "def_yac_median_neutral_q2": (neut2 & q2 & C, "yac", "median"),
        "def_comp_all_sr": (P & C, "sr", "mean"), "def_neut_q2_air_yards": (P & neut2 & q2, "air", "mean"),
        "def_lead_d2_cpoe": (P & lead & d2, "cpoe_v", "mean"), "def_trail_q3_complete": (P & trail & q3, "comp", "mean"),
        "def_trail_d1_yac": (P & trail & d1, "yac", "mean"), "def_ns_behind_los_yac_std": (P & neut & (d.air <= 0), "yac", "std"),
        "def_ns_comp_deep_air_yards_med": (P & neut & C & (d.air >= 15), "air", "median"), "def_ns_comp_q1_epa": (P & neut & C & q1, "epa_v", "mean"),
        "def_ns_d3_med_air_yards_med": (P & neut & d3 & med, "air", "median"), "def_ns_h2_yac_med": (P & neut & (d.qtr >= 3), "yac", "median"),
        "def_ns_d3_long_sr": (P & neut & d3 & long_, "sr", "mean"), "def_early_short_yac_med": (P & early & short, "yac", "median"),
        "def_comp_deep_yac_med": (P & C & (d.air >= 15), "yac", "median"), "def_backedup_ypa_min": (P & (d.yardline_100 >= 80), "pyds", "min"),
        "def_blowout_air_std": (P & blow, "air", "std"), "def_lead_comp_air_std": (P & lead & C, "air", "std"),
        "def_d3_med_inc_air_med": (P & d3 & med & ~C, "air", "median"), "def_pass_only_trail_d3_sr": (P & trail & d3, "sr", "mean"),
        "def_h1_cpoe_med": (P & (d.qtr <= 2), "cpoe_v", "median"), "def_pass_lead_d1_inc_fd_pct": (P & lead & d1 & ~C, "fd", "mean"),
        "def_ns_cpoe_min": (P & neut, "cpoe_v", "min")}
    return off, de


def team_week_features(df):
    """For each (season, week): features from REG plays of the prior season + current season before that week
    (the user's script used exactly that span: 2025 + 2026-to-date for a 2026 target)."""
    reg = df[df.season_type == "REG"].reset_index(drop=True)
    off, de = masks(reg)
    off = {k: (m.values, c, a) for k, (m, c, a) in off.items()}; de = {k: (m.values, c, a) for k, (m, c, a) in de.items()}
    tt = (reg.season * 100 + reg.week).values
    out = []
    for s in range(2012, 2027):
        for wk in range(1, 23):
            win = (tt >= (s - 1) * 100) & (tt < s * 100 + wk)
            row = {}
            for spec, key in ((off, "posteam"), (de, "defteam")):
                for name, (m, col, agg) in spec.items():
                    sub = reg.loc[win & m, [key, col]]
                    row[name] = sub.groupby(key)[col].agg(agg)
            f = pd.DataFrame(row); f.index.name = "team"
            out.append(f.reset_index().assign(season=s, week=wk))
        print("team-week features", s, flush=True)
    return pd.concat(out, ignore_index=True)


def clean_aggregates(df):
    P = df[(df["pass"] == 1)].copy()
    P["att"] = (P.pass_attempt == 1) & (P.sack != 1)
    P["cpoe_n"] = P.cpoe.notna().astype(float)
    P["succ"] = P.success.astype(float).fillna(0)
    agg = dict(db=("dropback", "sum"), epa_sum=("epa_v", "sum"), cpoe_sum=("cpoe_v", "sum"), cpoe_n=("cpoe_n", "sum"),
               succ=("succ", "sum"), press=("pressure", "sum"))
    qb = P[P["id"].notna()].groupby(["id", "game_id"], as_index=False).agg(**agg).rename(columns={"id": "player_id"})
    dfn = P.groupby(["defteam", "game_id", "season", "week"], as_index=False).agg(**agg).rename(columns={"defteam": "opp"})
    g = df.groupby("game_id", as_index=False).agg(home_team=("home_team", "first"), spread_line=("spread_line", "first"),
                                                  total_line=("total_line", "first"), roof=("roof", "first"),
                                                  wind=("wind", "first"), temp=("temp", "first"))
    return qb, dfn, g


if __name__ == "__main__":
    df = load()
    print("clean plays:", len(df), "seasons", df.season.min(), df.season.max())
    qb, dfn, g = clean_aggregates(df)
    qb.to_parquet(S + "/pbp_qb_game.parquet"); dfn.to_parquet(S + "/pbp_def_game.parquet"); g.to_parquet(S + "/pbp_game_ctx.parquet")
    team_week_features(df).to_parquet(S + "/pbp_team_week_45.parquet")
    print("done")

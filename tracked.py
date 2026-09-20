import os, numpy as np, pandas as pd, warnings
import check45 as c45
warnings.filterwarnings("ignore"); S = os.environ["S"]
g = c45.team_games(); ctx = pd.read_parquet(S + "/pbp_game_ctx.parquet"); g = g.merge(ctx, on="game_id", how="left")
sgn = np.where(g.posteam == g.home_team, 1, -1); g["off_spread"] = sgn * g.spread_line; g["off_implied"] = g.total_line / 2 + g.off_spread / 2
proj = pd.read_csv(S + "/season2025_def_ypa.csv").rename(columns={"proj_def_ypa": "MODEL_projection"})
g = g.merge(proj[["season", "week", "defteam", "posteam", "MODEL_projection"]], on=["season", "week", "defteam", "posteam"], how="left")
V = c45.OFF + c45.DEF + ["off_trail_ypa", "def_trail_ypa", "off_implied", "off_spread", "total_line", "MODEL_projection"]
tracked = [("LV", 12), ("DAL", 15), ("CHI", 2), ("NE", 2), ("SEA", 18), ("MIN", 13)]
s25 = g[(g.season == 2025) & g.off_trail_ypa.notna()].reset_index(drop=True)
T = pd.concat([s25[(s25.defteam == t) & (s25.week == w)] for t, w in tracked])
print(T[["defteam", "week", "posteam", "ypa", "MODEL_projection"]].round(2).to_string(index=False))
allyrs = g[g.off_trail_ypa.notna()]
r = pd.DataFrame({"r on the 6 tracked games": T[V].corrwith(T.ypa), "r on all 544 games of 2025": s25[V].corrwith(s25.ypa), "r on all seasons (7,300 games)": allyrs[[v for v in V if v != "MODEL_projection"]].corrwith(allyrs.ypa)})
r = r.reindex(r["r on the 6 tracked games"].abs().sort_values(ascending=False).index)
print("\nTOP 12 STATS ON THE 6 TRACKED GAMES"); print(r.head(12).round(3).to_string())
print("\nWHERE THE GENUINELY BEST STATS (by full-season r) RANK ON THE 6 GAMES"); rr = r.assign(rank_on_6=np.arange(1, len(r) + 1)).sort_values("r on all 544 games of 2025", ascending=False).head(8); print(rr.round(3).to_string())
print(f"\nstats with |r| > 0.8 on the 6 games: {(r.iloc[:, 0].abs() > 0.8).sum()} of {len(V)}; with |r| > 0.6: {(r.iloc[:, 0].abs() > 0.6).sum()}")
print(f"agreement between a stat's r on the 6 games and its r on the full season (correlation across the {len(V)} stats): {r.iloc[:, 0].corr(r.iloc[:, 1]):.2f}")
rng = np.random.default_rng(0); best, n8 = [], []
for _ in range(2000):
    d = s25.sample(6, random_state=int(rng.integers(1e9))); c = d[V].corrwith(d.ypa).abs(); best.append(c.max()); n8.append((c > 0.8).sum())
print(f"CHANCE CHECK, 2,000 random sets of 6 games from 2025: the best stat reaches |r| = {np.median(best):.2f} on a typical draw (95% of draws: {np.percentile(best, 2.5):.2f}-{np.percentile(best, 97.5):.2f}); typical number of stats above 0.8: {np.median(n8):.0f}")
top = r.index[0]; print(f"\n'{top}' scores {r.iloc[0, 0]:+.2f} on your 6 games but {r.iloc[0, 1]:+.3f} on the full 2025 season.")

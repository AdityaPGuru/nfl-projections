"""How high can a single-game Pearson go on the 5,278-row ordinary-YPA panel? Two estimates, both reproducible here.

1. Variance split: expected sampling variance of a game's mean = within-game variance of passing yards per attempt / attempts.
   Share of variance not explained by that = reliability; sqrt(reliability) = correlation an all-knowing game-level predictor reaches.
2. Split-half: assign each game's attempts at random to two halves, correlate the halves across games, Spearman-Brown to full length.
   Repeated over 200 random splits; a 95% interval comes from resampling games.

Both treat everything shared by the throws of a game (QB who played, real weather, game script, play sequencing) as signal, so they
are generous: a pregame forecast has less information than that. They are estimates with error, not a proof; the point is how far
outside the interval a reported number sits. Usage (repo root): python player_state/ceiling.py"""
import glob, os, numpy as np, pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
c = ["game_id", "season", "season_type", "posteam", "defteam", "pass", "pass_attempt", "penalty", "passing_yards", "sack", "two_point_attempt"]
p = pd.concat([pd.read_parquet(f, columns=c) for f in sorted(glob.glob(f"{R}/te_share/data/pbp/pbp_20*.parquet")) if 2016 <= int(f[-12:-8]) <= 2025])
d = p[(p.season_type == "REG") & (p["pass"] == 1) & (p.pass_attempt == 1) & (p.sack != 1) & ((p.penalty == 0) | p.penalty.isna()) & (p.two_point_attempt != 1)].copy()
d["y"] = d.passing_yards.fillna(0.0)
key = ["game_id", "posteam", "defteam"]
g = d.groupby(key).y.agg(["mean", "var", "size"]).reset_index()
noise = (g["var"] / g["size"]).mean(); tot = g["mean"].var()
print(f"rows {len(g)}  SD of game YPA {np.sqrt(tot):.3f}  luck SD {np.sqrt(noise):.3f}  non-luck SD {np.sqrt(max(tot - noise, 0)):.3f}")
print(f"variance-split ceiling r = {np.sqrt(max(tot - noise, 0) / tot):.3f}")

codes = d.groupby(key).ngroup().to_numpy(); y = d.y.to_numpy(); n = codes.max() + 1; rng = np.random.default_rng(0)
def split_r(sel=None):
    h = rng.integers(0, 2, len(y)); s0 = np.bincount(codes, y * (h == 0), n); c0 = np.bincount(codes, (h == 0) * 1.0, n)
    s1 = np.bincount(codes, y * (h == 1), n); c1 = np.bincount(codes, (h == 1) * 1.0, n); ok = (c0 > 0) & (c1 > 0)
    if sel is not None: ok = ok & sel
    return np.corrcoef(s0[ok] / c0[ok], s1[ok] / c1[ok])[0, 1]
rs = np.array([split_r() for _ in range(200)]); r = rs.mean(); rel = 2 * r / (1 + r)
boot = []
for _ in range(200):
    sel = np.zeros(n, bool); sel[np.unique(rng.integers(0, n, n))] = True; rb = split_r(sel); boot.append(np.sqrt(max(2 * rb / (1 + rb), 0)))
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"split-half r (mean of 200 splits) {r:.3f} -> reliability {rel:.3f} -> ceiling r {np.sqrt(rel):.3f}  (95% interval {lo:.3f} to {hi:.3f})")

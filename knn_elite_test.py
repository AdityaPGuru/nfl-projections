"""Does the team-vector KNN earn its keep specifically when the offense is elite? Pass YPA allowed, strictly forward 2018-2025.
'Elite' is defined before kickoff: offense's trailing YPA (or Vegas implied points) in the top 10% / 20% of all prior-season games."""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "def_ypa_full_eval.py")).read().split("# ---------------------------------------------------------------- 1. leakage")[0])
Z = pd.read_parquet(os.path.join(S, "full_stack_level1.parquet")); Z.index = HIST.index
HIST["ridge"], HIST["knn"], HIST["knn_v"] = Z["Ridge, 7 inputs"], Z["KNN team-vector comps"], Z["KNN team-vector comps + Vegas"]; HIST["rel"] = HIST.y - HIST.lg_prev
STYLE = ["rate", "net", "adj_rate", "epa", "succ", "fd", "expl", "td", "cmp", "cpoe", "adot", "deep_share", "deep_cmp", "deep_ypa", "short_ypa", "yac_c", "sack", "hit", "press", "int", "early_rate", "late_rate", "vol"]
DV, OV = ["ewL_" + s for s in STYLE], ["o_ewL_" + s for s in STYLE]; KS = [15, 30, 60, 120, 250]; rng = np.random.default_rng(0)

# elite-only comps: neighbours must themselves be elite offenses; match on the DEFENSE they faced (plus the offense vector). K learned on earlier seasons.
for tag, col, pct in (("trail", "o_ewL_rate", .90), ("vegas", "off_implied", .90), ("trail20", "o_ewL_rate", .80)):
    HIST["elite_" + tag] = False; pk = {k: pd.Series(np.nan, index=HIST.index) for k in KS}
    for s in range(2015, 2026):
        lib, q = HIST[HIST.season < s], HIST[HIST.season == s]; thr = lib[col].quantile(pct); HIST.loc[q.index, "elite_" + tag] = q[col] >= thr
        L_, Q_ = lib[lib[col] >= thr], q[q[col] >= thr]
        if len(Q_) == 0: continue
        cols = DV + OV; mu, sd = L_[cols].mean(), L_[cols].std().replace(0, 1); w = np.abs(L_[cols].corrwith(L_.rel).fillna(0).values); w = w / w.mean()
        A, B_ = ((L_[cols] - mu) / sd).fillna(0).values * w, ((Q_[cols] - mu) / sd).fillna(0).values * w
        order = np.argsort((B_ ** 2).sum(1)[:, None] + (A ** 2).sum(1)[None, :] - 2 * B_ @ A.T, axis=1)
        for k in KS: pk[k][Q_.index] = Q_.lg_prev.values + L_.rel.values[order[:, :min(k, len(L_))]].mean(1)
    HIST["knn_elite_" + tag] = np.nan
    for s in range(2018, 2026):
        past = HIST[(HIST.season >= 2015) & (HIST.season < s) & HIST["elite_" + tag]]; kb = min(KS, key=lambda k: rm(past.y, pk[k][past.index])); m = (HIST.season == s) & HIST["elite_" + tag]; HIST.loc[m, "knn_elite_" + tag] = pk[kb][m]

F = HIST[HIST.season >= 2018].copy()
def block(d, a, c):
    e = (d.y - d[a]) ** 2 - (d.y - d[c]) ** 2; ub = (d.season * 100 + d.week); u = ub.unique(); bs = []
    for _ in range(1000):
        pick = rng.choice(u, len(u)); bs.append(np.concatenate([e[ub == x].values for x in pick]).mean())
    return e.mean(), np.std(bs)
print("DOES KNN WORK WHEN THE OFFENSE IS ELITE? Forward 2018-2025. L2 = per-game RMSE (lower is better). 'MSE gain vs ridge' > 0 means KNN beat ridge; compare it with its noise.\n")
rows = []
for lab, m, ek in (("ALL GAMES", F.ridge.notna(), None), ("Elite offense: trailing YPA top 10%", F.elite_trail, "knn_elite_trail"), ("Elite offense: trailing YPA top 20%", F.elite_trail20, "knn_elite_trail20"),
                   ("Elite offense: Vegas implied points top 10%", F.elite_vegas, "knn_elite_vegas"), ("Everyone else (not top-20% trailing YPA)", ~F.elite_trail20, None)):
    d = F[m]; r = {"games": len(d), "avg actual YPA": d.y.mean(), "ridge proj": d.ridge.mean(), "L2 guess avg": d.y.std(), "L2 ridge": rm(d.y, d.ridge), "L2 KNN comps": rm(d.y, d.knn), "L2 KNN+Vegas": rm(d.y, d.knn_v)}
    if ek: r["L2 elite-only KNN"] = rm(d.y, d[ek])
    g, se = block(d, "ridge", ek or "knn_v"); r["MSE gain vs ridge (best KNN)"] = g; r["noise (1 s.e.)"] = se
    r.update({"r ridge": d.y.corr(d.ridge), "r KNN+Vegas": d.y.corr(d.knn_v), "r elite-only KNN": d.y.corr(d[ek]) if ek else np.nan})
    res = d.y - d.ridge; r["does KNN explain ridge's misses? r"] = res.corr(d[ek or "knn_v"] - d.ridge); rows.append({"group": lab, **r})
print(pd.DataFrame(rows).round(3).T.to_string())

# is there a style interaction at all? among elite offenses, does the KIND of defense change the outcome beyond how GOOD the defense is?
E = F[F.elite_trail20].copy(); E["res"] = E.y - E.ridge
print(f"\nSTYLE CHECK among the {len(E):,} elite-offense games: after ridge's projection, which defensive traits still line up with the miss? (|r| under ~{2 / np.sqrt(len(E)):.2f} is noise)")
s_ = E[DV].corrwith(E.res).sort_values(key=abs, ascending=False); print(s_.head(8).round(3).to_string())
A_ = F[~F.elite_trail20]; print("same check among everyone else:", ", ".join(f"{k} {v:+.3f}" for k, v in (A_[DV].corrwith(A_.y - A_.ridge)).sort_values(key=abs, ascending=False).head(4).items()))
top = E.assign(press_hi=E.ewL_press >= E.ewL_press.median(), good_def=E.eb_def <= E.eb_def.median() if "eb_def" in E else np.nan) if False else E.assign(press_hi=E.ewL_press >= E.ewL_press.median(), good_def=E.ewL_rate <= E.ewL_rate.median())
print("\nelite offenses by the defense they faced (actual YPA, ridge projection):"); print(top.groupby(["good_def", "press_hi"]).agg(games=("y", "size"), actual=("y", "mean"), ridge=("ridge", "mean"), knn_elite=("knn_elite_trail20", "mean")).round(2).rename(index={True: "yes", False: "no"}).to_string())

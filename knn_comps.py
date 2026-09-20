"""Team-vector KNN experiment for pass YPA allowed. Each offense and each defense is a vector of its pre-game passing profile
(~14 games, recency weighted). A game is predicted from the K most similar PAST matchups (same kind of offense vs same kind of
defense), using only earlier seasons. K and the distance weighting are learned from training seasons, never hand-picked."""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "def_ypa_full_eval.py")).read().split("# ---------------------------------------------------------------- 1. leakage")[0])
STYLE = ["rate", "net", "adj_rate", "epa", "succ", "fd", "expl", "td", "cmp", "cpoe", "adot", "deep_share", "deep_cmp", "deep_ypa", "short_ypa", "yac_c", "sack", "hit", "press", "int", "early_rate", "late_rate", "vol"]
DV, OV = ["ewL_" + s for s in STYLE], ["o_ewL_" + s for s in STYLE]; CTX = ["off_spread", "off_implied", "total_line", "off_home", "indoors"]
SMALL = ["eb_def", "eb_off"] + vegas; KS = [10, 25, 50, 100, 200, 400, 800]; SEAS = list(range(2015, 2026))
HIST["rel"] = HIST.y - HIST.lg_prev                                   # outcome relative to the league level known beforehand (era adjustment)

def knn_preds(cols, weighted):
    """For every season s >= 2015: predict its games from the K nearest games of seasons < s. Returns {K: series}."""
    out = {k: pd.Series(np.nan, index=HIST.index) for k in KS}; nbrs = {}
    for s in SEAS:
        lib, q = HIST[HIST.season < s], HIST[HIST.season == s]; mu, sd = lib[cols].mean(), lib[cols].std().replace(0, 1)
        L, Q = ((lib[cols] - mu) / sd).fillna(0).values, ((q[cols] - mu) / sd).fillna(0).values
        if weighted:                                                   # a stat counts in the distance in proportion to how much it tells you about YPA (learned on the library only)
            w = np.abs(lib[cols].corrwith(lib.rel).fillna(0).values); w = w / w.mean(); L, Q = L * w, Q * w
        d2 = (Q ** 2).sum(1)[:, None] + (L ** 2).sum(1)[None, :] - 2 * Q @ L.T; order = np.argsort(d2, axis=1)[:, :max(KS)]; rel = lib.rel.values
        for k in KS: out[k][q.index] = q.lg_prev.values + rel[order[:, :k]].mean(1)
        nbrs[s] = (lib.index.values, order[:, :10], q.index.values)
    return out, nbrs

def learned(pk):
    """Nested: for test season S use the K with the lowest pooled RMSE on seasons 2015..S-1 (each predicted from its own past)."""
    res, pick = pd.Series(np.nan, index=HIST.index), {}
    for s in range(2018, 2026):
        past = HIST[(HIST.season >= 2015) & (HIST.season < s)]; kbest = min(KS, key=lambda k: rm(past.y, pk[k][past.index])); pick[s] = kbest; res[HIST.season == s] = pk[kbest][HIST.season == s]
    return res, pick

VARIANTS = {"KNN team vectors, all stats equal": (DV + OV, False), "KNN team vectors, stats weighted by importance": (DV + OV, True),
            "KNN team vectors + Vegas context, weighted": (DV + OV + CTX, True), "KNN offense vector only, weighted": (OV, True), "KNN defense vector only, weighted": (DV, True)}
PRED, PICK, NB = {}, {}, {}
for lab, (cols, w) in VARIANTS.items():
    pk, nb = knn_preds(cols, w); PRED[lab], PICK[lab] = learned(pk); NB[lab] = nb; HIST["knn_" + str(len(PRED))] = pk[200]   # keep a fixed-K column only as a candidate stacking feature below
    if lab.startswith("KNN team vectors, stats weighted"): PKW = pk
# reference: the ridge model, and ridge with the KNN comp projection added as an extra input (the comp feature for every training row was itself made from earlier seasons only)
ridge, stack = pd.Series(np.nan, index=HIST.index), pd.Series(np.nan, index=HIST.index); compcol = PRED["KNN team vectors, stats weighted by importance"].copy()
for s in SEAS[:3]: compcol[HIST.season == s] = PKW[200][HIST.season == s]
HIST["comp"] = compcol
for s in range(2018, 2026):
    tr, te = HIST[HIST.season < s], HIST[HIST.season == s]; tre, _, _ = add_eb(tr, tr); tee, _, _ = add_eb(tr, te); ridge[te.index] = fitters(tre, tee, SMALL)[0]["ridge"]
    t2 = tre[tre.season >= 2015]; stack[te.index] = fitters(t2, tee, SMALL + ["comp"])[0]["ridge"]
PRED["Ridge, 7 inputs (current best)"] = ridge; PRED["Ridge + KNN comp projection as an 8th input"] = stack; PRED["50/50 blend: ridge and weighted KNN"] = (ridge + PRED["KNN team vectors, stats weighted by importance"]) / 2

rows = []
for lab, p in PRED.items():
    r_ = {}
    for nm, yrs in (("2018-25", range(2018, 2026)), ("2022-25", range(2022, 2026))):
        d = HIST[HIST.season.isin(yrs)].assign(p=p); wk = d.groupby(["season", "week"]).apply(lambda x: x.y.corr(x.p)); r_.update({f"r {nm}": d.y.corr(d.p), f"weekly r {nm}": wk.mean(), f"L2 {nm}": rm(d.y, d.p)})
    t = HIST[HIST.season == 2025].assign(p=p); T6 = pd.concat([t[(t.defteam == a) & (t.week == w_)] for a, w_ in dm.TRACKED]); rows.append({"method": lab, **r_, "weeks r>0": f"{(wk > 0).mean():.0%}", "sd of proj": d.p.std(), "tracked-6 r": T6.y.corr(T6.p)})
print(f"TEAM-VECTOR KNN vs THE CURRENT BEST. Pass YPA allowed, strictly forward. Vector = {len(STYLE)} passing-profile stats per side ({', '.join(STYLE)})\n")
print(pd.DataFrame(rows).sort_values("weekly r 2018-25", ascending=False).round(3).to_string(index=False)); print(f"(L2 of guessing the average: {HIST[HIST.season >= 2018].y.std():.3f})")
print("\nK learned from earlier seasons, by test season:"); print(pd.DataFrame(PICK).T.to_string())
cur = pd.Series({k: rm(HIST[HIST.season.between(2015, 2025)].y, PKW[k][HIST.season.between(2015, 2025)]) for k in KS}); print("forward L2 by K (weighted team vectors, 2015-2025):", "  ".join(f"K={k}: {v:.4f}" for k, v in cur.items()))

# ---------------------------------------------------------------- comps showcase
lib_i, order, q_i = NB["KNN team vectors, stats weighted by importance"][2025]; pos = {ix: n for n, ix in enumerate(q_i)}
def show(defteam, week):
    g = HIST[(HIST.season == 2025) & (HIST.defteam == defteam) & (HIST.week == week)]
    if g.empty: return
    ix = g.index[0]; nb = HIST.loc[lib_i[order[pos[ix]]]]; print(f"\n2025 week {week}: {g.posteam.iloc[0]} offense at/vs {defteam} defense. Actual YPA {g.y.iloc[0]:.2f}; KNN projection {PRED['KNN team vectors, stats weighted by importance'][ix]:.2f}; ridge {ridge[ix]:.2f}. Ten most similar past matchups:")
    print(nb.assign(matchup=nb.season.astype(str) + " wk" + nb.week.astype(str) + " " + nb.posteam + " off vs " + nb.defteam + " def")[["matchup", "y", "o_ewL_rate", "o_ewL_adot", "ewL_rate", "ewL_press"]].rename(
        columns={"y": "YPA that day", "o_ewL_rate": "off YPA coming in", "o_ewL_adot": "off depth of target", "ewL_rate": "def YPA allowed coming in", "ewL_press": "def pressure rate"}).round(2).to_string(index=False))
    print(f"   those ten averaged {nb.y.mean():.2f} (their seasons' league level {nb.lg_prev.mean():.2f})")
show("CHI", 2); show("SEA", 18)
last = HIST.sort_values("t").groupby(["posteam", "season"]).tail(1); mu, sd = last[OV].mean(), last[OV].std(); Z = ((last[OV] - mu) / sd).fillna(0)
def comps(team, season, n=6):
    a = Z[(last.posteam == team) & (last.season == season)]
    if a.empty: return
    d = np.sqrt(((Z[last.season < season] - a.values[0]) ** 2).sum(1)).sort_values().head(n); c = last.loc[d.index]
    full = HIST.groupby(["posteam", "season"]).apply(lambda x: x.o_num.sum() / x.o_den.sum()); me = last[(last.posteam == team) & (last.season == season)].iloc[0]
    print(f"\n{season} {team} passing offense (YPA {full[(team, season)]:.2f}, depth of target {me.o_ewL_adot:.1f}, sack rate {me.o_ewL_sack:.3f}) - most similar past offenses by style vector:")
    print(c.assign(team=c.season.astype(str) + " " + c.posteam, distance=d.values, season_ypa=[full[(t_, s_)] for t_, s_ in zip(c.posteam, c.season)])[["team", "distance", "season_ypa", "o_ewL_adot", "o_ewL_deep_share", "o_ewL_sack", "o_ewL_yac_c"]].rename(
        columns={"season_ypa": "season YPA", "o_ewL_adot": "depth of target", "o_ewL_deep_share": "deep share", "o_ewL_sack": "sack rate", "o_ewL_yac_c": "YAC per catch"}).round(3).to_string(index=False))
comps("KC", 2025); comps("KC", 2018); comps("DET", 2025)
pd.DataFrame({k: v for k, v in PRED.items()}).assign(season=HIST.season, week=HIST.week, defense=HIST.defteam, offense=HIST.posteam, actual=HIST.y).dropna().to_csv(os.path.join(HERE, "results", "knn_comps_forward.csv"), index=False)

"""Does the player-state dataset strengthen a two-way baseline? Same 5,278 keyed rows (2016-2025 REG, ordinary YPA), expanding
week-ahead OLS (train on seasons before s + weeks before w of s, training-median fill, raw prediction).

Arms (each adds to the previous control):
  A  base      : the 14 pregame terms of def_ypa_lift (pruned_terms_noqb.csv)
  B  control   : A + tw_naive  (team two-way rating, no player state; the analogue of the H1 hybrid's component)
  C  state     : B + st_match, st_qb_delta, st_qb_exp_logatt, st_qb_changed  (pregame expected state only)
  D  ORACLE    : B + oracle_st_match, oracle_st_qb  (uses the QB who actually played; NOT deployable, shown only as a ceiling)
Promotion rule (his): C must raise Pearson AND lower paired MAE AND RMSE vs B, hold up on 2023-2025, and on QB-transition games.
Usage (repo root): python player_state/evaluate_state.py"""
import os, numpy as np, pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); H = os.path.join(R, "player_state"); OUT = os.path.join(H, "out")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
F = pd.read_parquet(f"{OUT}/state_features.parquet")
B14 = pd.read_csv(f"{R}/def_ypa_lift/pruned_terms_noqb.csv").stat.tolist()
base = pd.read_parquet(f"{R}/def_ypa_lift/data/rows.parquet", columns=["game_id", "posteam", "defteam"] + B14)
X = F.merge(base, on=["game_id", "posteam", "defteam"], how="left").sort_values(["season", "week"]).reset_index(drop=True)
X = X[X.season >= 2013].reset_index(drop=True)
STATE = ["st_match", "st_qb_delta", "st_qb_exp_logatt", "st_qb_changed"]
ARMS = {"A base 14": B14, "B control: + two-way": B14 + ["tw_naive"], "C + player state (pregame)": B14 + ["tw_naive"] + STATE,
        "D ORACLE: + actual-QB state": B14 + ["tw_naive", "oracle_st_match", "oracle_st_qb"]}
Y, SEA, WK = X.ypa.to_numpy(), X.season.to_numpy(), X.week.to_numpy(); PANEL = (SEA >= 2016) & (SEA <= 2025)


def forward(cols):
    A = X[cols].to_numpy(float); pred = np.full(len(X), np.nan)
    for s in range(2016, 2026):
        for w in range(1, 19):
            te = (SEA == s) & (WK == w)
            if not te.any(): continue
            tr = (SEA < s) | ((SEA == s) & (WK < w))
            a = A[tr]; med = np.nanmedian(a, axis=0); med = np.where(np.isfinite(med), med, 0)
            a = np.where(np.isfinite(a), a, med); b = np.where(np.isfinite(A[te]), A[te], med)
            coef = np.linalg.lstsq(np.column_stack([np.ones(len(a)), a]), Y[tr], rcond=None)[0]
            pred[te] = np.column_stack([np.ones(len(b)), b]) @ coef
    return pred


def pairwise(p, m):
    """Share of same-week row pairs whose order the forecast gets right (ties ignored)."""
    ok = tot = 0
    for k in np.unique((SEA * 100 + WK)[m]):
        i = np.flatnonzero(m & ((SEA * 100 + WK) == k)); pp, yy = p[i], Y[i]
        d = np.sign(pp[:, None] - pp[None, :]) * np.sign(yy[:, None] - yy[None, :]); iu = np.triu_indices(len(i), 1)
        ok += (d[iu] > 0).sum(); tot += (d[iu] != 0).sum()
    return ok / tot


def metrics(p, m):
    return {"rows": int(m.sum()), "Pearson": np.corrcoef(p[m], Y[m])[0, 1], "MAE": np.mean(np.abs(p[m] - Y[m])), "RMSE": np.sqrt(np.mean((p[m] - Y[m]) ** 2)),
            "forecast SD": p[m].std(), "pairwise acc": pairwise(p, m)}


def paired(pa, pb, m, n=400):
    """b minus a on Pearson, MAE, RMSE; standard errors from resampling weeks within each season."""
    rng = np.random.default_rng(0); keys = SEA * 100 + WK; idx = {k: np.flatnonzero(m & (keys == k)) for k in np.unique(keys[m])}
    by = {}
    for k in idx: by.setdefault(k // 100, []).append(k)
    def stat(ks):
        i = np.concatenate([idx[k] for k in ks]); y = Y[i]
        return (np.corrcoef(pb[i], y)[0, 1] - np.corrcoef(pa[i], y)[0, 1], np.mean(np.abs(pb[i] - y)) - np.mean(np.abs(pa[i] - y)),
                np.sqrt(np.mean((pb[i] - y) ** 2)) - np.sqrt(np.mean((pa[i] - y) ** 2)))
    pt = stat(list(idx)); bs = np.array([stat(np.concatenate([rng.choice(v, len(v)) for v in by.values()])) for _ in range(n)])
    return pt, bs.std(axis=0)


if __name__ == "__main__":
    print("panel rows:", int(PANEL.sum()))
    P = {k: forward(v) for k, v in ARMS.items()}
    subsets = {"2016-2025 full": PANEL, "2016-2022 discovery": PANEL & (SEA <= 2022), "2023-2025 confirmation": PANEL & (SEA >= 2023),
               "weeks 1-4": PANEL & (WK <= 4), "QB-transition games (actual starter changed)": PANEL & X.qb_transition.fillna(False).to_numpy(bool),
               "expected QB differs from established (pregame flag)": PANEL & (X.st_qb_changed == 1).to_numpy(), "stable-QB games": PANEL & ~X.qb_transition.fillna(False).to_numpy(bool)}
    rows = []
    for sname, m in subsets.items():
        for arm, p in P.items(): rows.append({"subset": sname, "arm": arm, **metrics(p, m)})
    T = pd.DataFrame(rows); print(T.round(4).to_string(index=False)); T.to_csv(f"{OUT}/eval_state_metrics.csv", index=False)
    seas = pd.DataFrame([{"season": s, **{arm: np.corrcoef(p[PANEL & (SEA == s)], Y[PANEL & (SEA == s)])[0, 1] for arm, p in P.items()}} for s in range(2016, 2026)])
    print("\nPearson by season\n", seas.round(4).to_string(index=False)); seas.to_csv(f"{OUT}/eval_state_by_season.csv", index=False)
    pr = []
    for sname in ("2016-2025 full", "2023-2025 confirmation", "QB-transition games (actual starter changed)"):
        for b in ("C + player state (pregame)", "D ORACLE: + actual-QB state"):
            (dr, dm, de), (sr, sm, se) = paired(P["B control: + two-way"], P[b], subsets[sname])
            pr.append({"subset": sname, "challenger vs B": b, "dPearson": dr, "se": sr, "dMAE": dm, "se_MAE": sm, "dRMSE": de, "se_RMSE": se})
    pr = pd.DataFrame(pr); print("\nPaired vs control B\n", pr.round(4).to_string(index=False)); pr.to_csv(f"{OUT}/eval_state_paired.csv", index=False)
    X.assign(**{f"pred_{k.split()[0]}": v for k, v in P.items()})[["game_id", "season", "week", "posteam", "defteam", "ypa"] + [f"pred_{k.split()[0]}" for k in P]] \
        .loc[PANEL].to_csv(f"{OUT}/oof_predictions_5278.csv", index=False)

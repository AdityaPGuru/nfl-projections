"""Week 7 2026 values and projections for the 14-term (pregame only) and 17-term (schedule-starter QB) models, with their scores.
Reads week7/out/rows.parquet (built with FORECAST=2026,7: plays from 2026 weeks 1-6 plus earlier seasons only).
Writes week7/week7_2026_terms_and_projections.csv and week7/qb_variant_scores.csv. Usage: python def_ypa_lift/week7/project.py"""
import os, glob, numpy as np, pandas as pd

H = os.path.dirname(os.path.abspath(__file__)); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
X = pd.read_parquet(f"{H}/out/rows.parquet").sort_values(["season", "week"]).reset_index(drop=True)
TERMS14 = pd.read_csv(f"{H}/../pruned_terms_noqb.csv").stat.tolist()     # recommended: pregame-only, no QB-starter fields
TERMS17 = pd.read_csv(f"{H}/../pruned_terms.csv").stat.tolist()          # comparison: includes schedule-starter c_qb_epa, c_qb_new
Y, SEA, WK = X.ypa.to_numpy(), X.season.to_numpy(), X.week.to_numpy()


def fit_predict(cols, tr, te):
    A = X[cols].to_numpy(float); a = A[tr]; med = np.nanmedian(a, axis=0); med = np.where(np.isfinite(med), med, 0)
    a = np.where(np.isfinite(a), a, med); b = np.where(np.isfinite(A[te]), A[te], med)
    coef = np.linalg.lstsq(np.column_stack([np.ones(len(a)), a]), Y[tr], rcond=None)[0]
    return np.column_stack([np.ones(len(b)), b]) @ coef, coef


def forward(cols, seasons):
    pred = np.full(len(X), np.nan)
    for s in seasons:
        for w in range(1, 19):
            te = (SEA == s) & (WK == w) & np.isfinite(Y)
            if te.any(): pred[te] = fit_predict(cols, ((SEA < s) | ((SEA == s) & (WK < w))) & np.isfinite(Y), te)[0]
    return pred


def score(pred, seasons):
    ok = np.isfinite(Y) & np.isfinite(pred)
    r = [np.corrcoef(pred[(SEA == s) & ok], Y[(SEA == s) & ok])[0, 1] for s in seasons]; m = np.isin(SEA, seasons) & ok
    return {"mean season r": np.mean(r), "MAE": np.mean(np.abs(pred[m] - Y[m])), "n": int(m.sum())}


if __name__ == "__main__":
    rows = {}
    for label, cols in (("14 terms, pregame only (recommended)", TERMS14), ("17 terms, with schedule-starter QB", TERMS17)):
        p = forward(cols, range(2016, 2027))
        a, b, c = score(p, range(2016, 2026)), score(p, range(2023, 2026)), score(p, [2026])
        rows[label] = {"r 2016-25": a["mean season r"], "r 2023-25": b["mean season r"], "MAE 2016-25": a["MAE"], "r 2026 wk1-6": c["mean season r"], "MAE 2026 wk1-6": c["MAE"]}
    t = pd.DataFrame(rows).T; print(t.round(4).to_string()); t.to_csv(f"{H}/qb_variant_scores.csv")

    # Week 7: fit once on every completed row (2013 through 2026 week 6), predict the 28 offense-defense rows
    tr = np.isfinite(Y); te = (SEA == 2026) & (WK == 7)
    names = pd.concat([pd.read_parquet(f, columns=["passer_player_id", "passer_player_name"]) for f in sorted(glob.glob(f"{H}/data/pbp/pbp_202[4-6].parquet"))]).dropna()
    names = names.drop_duplicates("passer_player_id").set_index("passer_player_id").passer_player_name
    out = X.loc[te, ["game_id", "posteam", "defteam", "opp_qb_id", "pre_qb_id", "top_qb"]].copy()
    for c in ("opp_qb_id", "pre_qb_id", "top_qb"): out[c.replace("_id", "") + "_name"] = out[c].map(names)
    out["proj_ypa_14"], c14 = fit_predict(TERMS14, tr, te)
    out["proj_ypa_17"], c17 = fit_predict(TERMS17, tr, te)
    pd.DataFrame({"term": ["(intercept)"] + TERMS14, "coef": c14}).to_csv(f"{H}/week7_coefficients_14.csv", index=False)
    pd.DataFrame({"term": ["(intercept)"] + TERMS17, "coef": c17}).to_csv(f"{H}/week7_coefficients_17.csv", index=False)
    out = pd.concat([out, X.loc[te, sorted(set(TERMS17 + TERMS14 + ["c_qb_epa_pre", "c_qb_new_pre"]))]], axis=1).sort_values(["game_id", "posteam"])
    out.to_csv(f"{H}/week7_2026_terms_and_projections.csv", index=False)
    print(out[["game_id", "posteam", "defteam", "opp_qb_name", "pre_qb_name", "top_qb_name", "c_qb_epa", "c_qb_epa_pre", "c_qb_new", "proj_ypa_14", "proj_ypa_17"]].round(3).to_string(index=False))
    miss = out[TERMS17 + TERMS14].isna().sum(); print("missing cells (filled with training medians):", miss[miss > 0].to_dict() or "none")

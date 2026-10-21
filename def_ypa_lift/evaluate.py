"""His protocol, exactly: for target season s and week w, fit OLS on every row with season < s or (season == s and week < w),
fill missing cells with training medians, predict week w raw. Score: mean per-season correlation 2016-2025 and 2023-2025, MAE, RMSE.

Steps: (1) his 35-term baseline as rebuilt, (2) the same minus game-day wind/temp, (3) every candidate added alone,
(4) forward selection that only looks at 2016-2022, so 2023-2025 stays an untouched check.
Usage: python def_ypa_lift/evaluate.py"""
import os, sys, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
X = pd.read_parquet(f"{HERE}/data/rows.parquet")
BASE35 = ["def_blowout_std_air_comp_mean", "def_comp_all_sr", "def_neut_q2_air_yards", "def_lead_d2_cpoe", "def_trail_q3_complete", "def_trail_d1_yac",
          "def_ns_behind_los_yac_std", "def_ns_comp_deep_air_yards_med", "def_comp_deep_yac_med", "def_backedup_ypa_min", "def_blowout_air_std", "def_pass_volume",
          "def_qb_hit_rate_q4", "def_explosive_rate_behind_los", "def_epa_neutral_behind_los", "def_td_rate_behind_los", "def_yac_epa_per_dropback_ytd", "def_rest_days",
          "def_opp_explosive_pass_rate_ytd_faced_ytd", "off_explosive_pass_rate_ytd", "off_success_rate_ytd", "off_top_qb_success_rate_ytd",
          "off_top_qb_explosive_pass_rate_ytd", "off_top_qb_yac_per_completion_ytd", "sq__off_top_qb_yac_epa_per_dropback_ytd", "off_pass_rate_ytd", "off_pass_oe_ytd",
          "off_xpass_ytd", "off_ngs_top_qb_avg_completed_air_yards_ytd", "off_ngs_top_qb_completion_percentage_ytd", "off_spread_line", "total_line", "wind", "temp",
          "ix__off_top_receiver_target_share_ytd__X__def_hit_red_zone_pass_rate_ytd"]
LEAK = ["wind", "temp"]
BASE33 = [c for c in BASE35 if c not in LEAK]
CANDS = [c for c in X.columns if c.startswith("c_")] + ["off_implied", "off_home", "indoors", "grass", "div_game", "rest_diff", "off_rest_days"]
X = X.sort_values(["season", "week"]).reset_index(drop=True)
SEASONS = list(range(2016, 2026)); Y = X.ypa.values; SEA = X.season.values; WK = X.week.values


def forward(cols, seasons=SEASONS):
    A = X[cols].to_numpy(float); pred = np.full(len(X), np.nan)
    for s in seasons:
        for w in range(1, 19):
            te = (SEA == s) & (WK == w)
            if not te.any(): continue
            tr = (SEA < s) | ((SEA == s) & (WK < w))
            a = A[tr]; med = np.nanmedian(a, axis=0); med = np.where(np.isfinite(med), med, 0)
            a = np.where(np.isfinite(a), a, med); b = np.where(np.isfinite(A[te]), A[te], med)
            a1 = np.column_stack([np.ones(len(a)), a]); coef = np.linalg.lstsq(a1, Y[tr], rcond=None)[0]
            pred[te] = np.column_stack([np.ones(len(b)), b]) @ coef
    return pred


def score(pred, seasons=SEASONS):
    m = np.isin(SEA, seasons) & np.isfinite(pred)
    r = {s: np.corrcoef(pred[SEA == s], Y[SEA == s])[0, 1] for s in seasons}
    out = {"r_2016_2025": np.mean([r[s] for s in seasons if 2016 <= s <= 2025]), "r_2023_2025": np.mean([r[s] for s in seasons if s >= 2023]),
           "MAE": np.mean(np.abs(pred[m] - Y[m])), "RMSE": np.sqrt(np.mean((pred[m] - Y[m]) ** 2))}
    out.update({f"r{s}": r[s] for s in seasons})
    return out


def sel_obj(pred):
    """Selection objective uses 2016-2022 only: mean season r minus MAE (both on the same scale of a few hundredths)."""
    ss = range(2016, 2023); m = np.isin(SEA, ss)
    return np.mean([np.corrcoef(pred[SEA == s], Y[SEA == s])[0, 1] for s in ss]) - np.mean(np.abs(pred[m] - Y[m]))


if __name__ == "__main__":
    rows = {}
    rows["his 35 terms (rebuilt)"] = score(forward(BASE35))
    rows["his 35 minus wind/temp (33)"] = score(forward(BASE33))
    rows["defense-trailing NY/A only"] = score(forward(["c_def_ypa"]))
    tab = pd.DataFrame(rows).T; print(tab.round(4).to_string()); tab.to_csv(f"{HERE}/results_baselines.csv")

    p0 = forward(BASE33); b0 = score(p0)
    single = []
    for c in CANDS:
        sc = score(forward(BASE33 + [c])); single.append({"stat": c, "d_r_2016_2025": sc["r_2016_2025"] - b0["r_2016_2025"],
                                                          "d_r_2023_2025": sc["r_2023_2025"] - b0["r_2023_2025"], "d_MAE": sc["MAE"] - b0["MAE"]})
    single = pd.DataFrame(single).sort_values("d_r_2016_2025", ascending=False)
    print("\nEach candidate added alone to the 33 clean terms:\n", single.round(4).to_string(index=False)); single.to_csv(f"{HERE}/results_single_adds.csv", index=False)

    # forward selection judged on 2016-2022 only
    chosen, pool, cur = [], list(CANDS), sel_obj(p0); trace = []
    while pool:
        best = max(((sel_obj(forward(BASE33 + chosen + [c], range(2016, 2023))), c) for c in pool))
        if best[0] - cur < 0.001: break
        cur = best[0]; chosen.append(best[1]); pool.remove(best[1]); trace.append((best[1], round(cur, 4))); print("add", best[1], round(cur, 4), flush=True)
    final = score(forward(BASE33 + chosen))
    out = pd.DataFrame({"his 35 (with leaky wind/temp)": rows["his 35 terms (rebuilt)"], "clean 33": b0, f"clean 33 + {len(chosen)} added": final}).T
    print("\nadded:", chosen); print(out.round(4).to_string()); out.to_csv(f"{HERE}/results_final.csv")
    pd.Series(chosen).to_csv(f"{HERE}/added_stats.csv", index=False, header=["stat"])

"""Project TE receiving-yards share for an upcoming 2026 week with the model fit on 2013-2025 only.
usage: python3 predict_week.py 6
2026 weeks before the target week enter only as pregame features (season-to-date shares, roster, injury report).
"""
import sys, numpy as np, pandas as pd
from model import t, fit_predict, R

W = int(sys.argv[1]) if len(sys.argv) > 1 else 6
train = t[t.played & (t.season <= 2025)].copy()
test = t[(t.season == 2026) & (t.week == W)].copy()
out, info = fit_predict(train, test)
cols = ["posteam", "defteam", "home", "spread", "total", "o_n", "te1_out", "wr1_out"]
res = pd.concat([test[cols], out[["naive", "eb_full", "eb_full_cal"]]], axis=1)
res["off_ytd_share"] = (test.o_yds_TE_ytd / test.o_yds_ALL_ytd).round(3)
res["def_ytd_allowed"] = (test.d_yds_TE_ytd / test.d_yds_ALL_ytd).round(3)
res = res.rename(columns={"eb_full_cal": "projection", "naive": "fixed_weight_style"}).sort_values("projection", ascending=False)
res[["projection", "fixed_weight_style", "eb_full"]] = res[["projection", "fixed_weight_style", "eb_full"]].round(3)
res.to_csv(f"{R}/week{W}_2026_projections.csv", index=False)
print(f"2026 week {W}: target = TE receiving yards / team receiving yards; fit on 2013-2025 only")
print("k_off", info["k_off"], "k_def", info["k_def"], "calibration [a, b, b*w, w]", info["cal"]["eb_full"])
print(res[["posteam", "defteam", "projection", "fixed_weight_style", "off_ytd_share", "def_ytd_allowed", "te1_out", "wr1_out", "spread", "total"]].to_string(index=False))

"""His H1 lambda=4 two-way YPA component, rebuilt from its written spec on the same 5,278 rows, without the leak:
  metric = league + offense effect + defense effect, alternating weighted ridge, current-season games with week < target week only,
  game weight = attempts * 0.5 ** ((target_week - 1 - game_week) / 1)  (half-life 1 week, both sides), ridge lambda = 4 attempts.
Week 1 has no current-season games, so the component is missing and gets the training median (as in his evaluator).
Reports the component alone and added to the 14-term base, with the same week-ahead OLS. Usage: python player_state/his_twoway_spec.py"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_state as ev

t = ev.X[["game_id", "season", "week", "posteam", "defteam", "ypa"]].copy()
att = pd.read_parquet(os.path.join(ev.OUT, "state_features.parquet"), columns=["game_id", "posteam", "defteam", "att"])
t = t.merge(att, on=["game_id", "posteam", "defteam"])


def component(lam=4.0, hl=1.0, iters=50):
    out = pd.Series(np.nan, index=t.index)
    for (s, w), rows in t.groupby(["season", "week"]):
        h = t[(t.season == s) & (t.week < w)]
        if not len(h): continue
        a = h.att.to_numpy(float) * 0.5 ** ((w - 1 - h.week.to_numpy(float)) / hl); y = h.ypa.to_numpy(float)
        lg = (a * y).sum() / a.sum(); r = y - lg
        oc, on_ = pd.factorize(h.posteam); dc, dn = pd.factorize(h.defteam); off, de = np.zeros(len(on_)), np.zeros(len(dn))
        for _ in range(iters):
            off = np.bincount(oc, a * (r - de[dc]), len(on_)) / (np.bincount(oc, a, len(on_)) + lam)
            de = np.bincount(dc, a * (r - off[oc]), len(dn)) / (np.bincount(dc, a, len(dn)) + lam)
        out[rows.index] = lg + rows.posteam.map(dict(zip(on_, off))).fillna(0).values + rows.defteam.map(dict(zip(dn, de))).fillna(0).values
    return out


if __name__ == "__main__":
    ev.X["tw_h1_l4"] = component().reindex(ev.X.index).values
    arms = {"H1 lambda=4 component alone": ["tw_h1_l4"], "A base 14": ev.ARMS["A base 14"], "A + H1 lambda=4 component": ev.ARMS["A base 14"] + ["tw_h1_l4"],
            "B (window two-way)": ev.ARMS["B control: + two-way"]}
    rows = []
    for name, cols in arms.items():
        p = ev.forward(cols)
        for sub, m in (("2016-2025", ev.PANEL), ("2023-2025", ev.PANEL & (ev.SEA >= 2023))):
            rows.append({"model": name, "rows": sub, **ev.metrics(p, m)})
    T = pd.DataFrame(rows); print(T.round(4).to_string(index=False)); T.to_csv(os.path.join(ev.OUT, "his_twoway_spec_clean.csv"), index=False)

"""Confirms the Week 7 build did not change the model path: every column of the frozen data/rows.parquet (2013-2025, the table every
model was scored on; built by the default `python3 build.py`) is identical in the default rebuild and in the forecast build. Usage: python def_ypa_lift/week7/check_same.py"""
import os, numpy as np, pandas as pd

H = os.path.dirname(os.path.abspath(__file__))
frozen = pd.read_parquet(f"{H}/../data/rows.parquet")
key = ["game_id", "posteam", "defteam"]
for name in ("check_default", "out"):
    if not os.path.exists(f"{H}/{name}/rows.parquet"): print(f"{name}: not built, skipped"); continue
    new = pd.read_parquet(f"{H}/{name}/rows.parquet").set_index(key)
    a = frozen.set_index(key); b = new.loc[a.index, a.columns]
    bad = []
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]):
            if not np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True, rtol=0, atol=1e-12): bad.append(c)
        elif not a[c].astype(str).equals(b[c].astype(str)): bad.append(c)
    print(f"{name}: {len(a)} frozen rows, {len(a.columns)} columns compared, differing columns: {bad or 'none'}")

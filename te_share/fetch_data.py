"""Download the raw nflverse data into te_share/data/.

    python fetch_data.py              # past seasons once, current season fresh every run (use this weekly)
    python fetch_data.py --snapshot   # exact files behind the published results (GitHub release zip)

Works on Mac, Windows and Linux; needs only the Python standard library.
"""
import os, sys, urllib.request, zipfile, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
CURRENT = 2026
FIRST = 2012
NV = "https://github.com/nflverse/nflverse-data/releases/download"
SNAPSHOT = "https://github.com/AdityaPGuru/nfl-projections/releases/download/te-share-data-2026-10-15/te_share_data.zip"


def get(url, path, refresh=False):
    if os.path.exists(path) and not refresh:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    print("downloading", os.path.relpath(path, HERE))
    req = urllib.request.Request(url, headers={"User-Agent": "te-share-fetch"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(tmp, path)


def latest():
    for y in range(FIRST, CURRENT + 1):
        fresh = y == CURRENT
        get(f"{NV}/pbp/play_by_play_{y}.parquet", f"{D}/pbp/pbp_{y}.parquet", fresh)
        get(f"{NV}/weekly_rosters/roster_weekly_{y}.parquet", f"{D}/aux/roster_weekly_{y}.parquet", fresh)
        get(f"{NV}/snap_counts/snap_counts_{y}.parquet", f"{D}/aux/snaps_{y}.parquet", fresh)
        get(f"{NV}/injuries/injuries_{y}.parquet", f"{D}/aux/inj_{y}.parquet", fresh)
    get("https://github.com/nflverse/nfldata/raw/master/data/games.csv", f"{D}/aux/games.csv", True)
    get(f"{NV}/players/players.parquet", f"{D}/aux/players.parquet", True)


def snapshot():
    z = os.path.join(D, "te_share_data.zip")
    get(SNAPSHOT, z, True)
    with zipfile.ZipFile(z) as f:
        f.extractall(D)
    os.remove(z)


if __name__ == "__main__":
    snapshot() if "--snapshot" in sys.argv else latest()
    print("done ->", D)

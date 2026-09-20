"""Download everything the scripts need into the working directory $S (default ./data).
Kaggle player-game table, nflverse play-by-play 2011 onward, and the ESPN scoreboard for the weeks being projected."""
import os, sys, json, urllib.request
S = os.environ.get("S", "./data"); os.makedirs(S + "/pbp", exist_ok=True); os.environ["KAGGLEHUB_CACHE"] = S + "/cache"
import kagglehub
kagglehub.dataset_download("philiphyde1/nfl-stats-1999-2022")
last = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
for y in range(2011, last + 1):
    f = f"{S}/pbp/pbp_{y}.parquet"
    if not os.path.exists(f):
        urllib.request.urlretrieve(f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet", f); print("pbp", y)
for wk in (1, 2):
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={last}&seasontype=2&week={wk}"
    json.dump(json.load(urllib.request.urlopen(url)), open(f"{S}/espn_wk{wk}.json", "w"))
print("done ->", S)

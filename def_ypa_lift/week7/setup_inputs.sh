#!/usr/bin/env bash
# Builds def_ypa_lift/week7/data/ from the frozen 2026-10-23 snapshot plus the 2012-2025 history.
# History (pbp 2012-2025, injuries 2012-2025) comes from te_share/data, which `S=te_share/data python fetch_data.py` or
# te_share/build.py downloads. Nothing is downloaded or overwritten here; it only creates symlinks.
set -euo pipefail
W="$(cd "$(dirname "$0")" && pwd)"; R="$(cd "$W/../.." && pwd)"; H="${HIST:-$R/te_share/data}"; SNAP="$W/inputs_2026-10-23"
mkdir -p "$W/data/pbp" "$W/data/aux"
for y in $(seq 2012 2025); do
  ln -sfn "$H/pbp/pbp_$y.parquet" "$W/data/pbp/pbp_$y.parquet"
  ln -sfn "$H/aux/inj_$y.parquet" "$W/data/aux/inj_$y.parquet"
done
ln -sfn "$SNAP/pbp_2026.parquet" "$W/data/pbp/pbp_2026.parquet"
ln -sfn "$SNAP/inj_2026.parquet" "$W/data/aux/inj_2026.parquet"
ln -sfn "$SNAP/games.csv" "$W/data/aux/games.csv"
ls -L "$W/data/pbp" "$W/data/aux" > /dev/null && echo "inputs ready in $W/data"

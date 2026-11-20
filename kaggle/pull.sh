#!/usr/bin/env bash
# Download the run kernel's output and merge its results into ./results.
#
#   kaggle/pull.sh <kaggle-username>
set -euo pipefail
KUSER=${1:?usage: pull.sh <kaggle-username>}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT="$ROOT/build/kaggle_output"
rm -rf "$OUT" && mkdir -p "$OUT"
kaggle kernels status "$KUSER/crs-run"
kaggle kernels output "$KUSER/crs-run" -p "$OUT"
mkdir -p "$ROOT/results"
# kernel output = restored results + new chunks, so it supersedes the local copy
rsync -a "$OUT/results/" "$ROOT/results/"
[ -f "$OUT/status.json" ] && cp "$OUT/status.json" "$ROOT/results/last_kaggle_status.json"
echo "merged into $ROOT/results"

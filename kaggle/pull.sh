#!/usr/bin/env bash
# Download the run kernel's output and merge its results into ./results.
#
#   kaggle/pull.sh <kaggle-username> [notebook, default crs-run]
set -euo pipefail
KUSER=${1:?usage: pull.sh <kaggle-username> [notebook]}
KERNEL=${2:-crs-run}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT="$ROOT/build/kaggle_output-$KUSER-$KERNEL"
rm -rf "$OUT" && mkdir -p "$OUT"
kaggle kernels status "$KUSER/$KERNEL"
# only our outputs: an unfiltered pull fetches every runtime file and can hit a long 429
kaggle kernels output "$KUSER/$KERNEL" -p "$OUT" --file-pattern '^(results[^/]*/|logs/|status\.json)'
# kernel output = restored results + new chunks. Runs with a
# longer length limit come back in their own folder (e.g. results-32k/) and stay separate here too.
for d in "$OUT"/results*/; do
  [ -d "$d" ] || continue
  # merge, never overwrite: with parallel sessions this session's copy may be older than ours
  python3 -I "$ROOT/kaggle/merge_results.py" "$d" "$ROOT/$(basename "$d")"
done
[ -f "$OUT/status.json" ] && cp "$OUT/status.json" "$ROOT/results/last_kaggle_status-$KUSER-$KERNEL.json"
echo "merged into $ROOT/results"

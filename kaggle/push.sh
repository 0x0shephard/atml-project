#!/usr/bin/env bash
# Push the code, the results gathered so far and the run kernel to Kaggle.
#
#   kaggle/push.sh <kaggle-username> [smoke.yaml|priority.yaml] [GROUP ...]
#
# Needs the kaggle CLI (`uv tool install kaggle --with six`), signed in with `kaggle auth login`.
set -euo pipefail
KUSER=${1:?usage: push.sh <kaggle-username> [jobfile] [group ...]}
JOBS=${2:-priority.yaml}
shift $(( $# >= 2 ? 2 : 1 ))
GROUPS_PY=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@")

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD="$ROOT/build/kaggle"
rm -rf "$BUILD" && mkdir -p "$BUILD/code" "$BUILD/results" "$BUILD/kernel"

dataset() {  # dir slug title
  cat > "$1/dataset-metadata.json" <<JSON
{"title": "$3", "id": "$KUSER/$2", "licenses": [{"name": "CC0-1.0"}]}
JSON
  if kaggle datasets status "$KUSER/$2" >/dev/null 2>&1; then
    kaggle datasets version -p "$1" -m "update $(date -u +%FT%TZ)" --dir-mode zip -q
  else
    kaggle datasets create -p "$1" --dir-mode zip -q
  fi
}

cp -r "$ROOT/src" "$ROOT/configs" "$ROOT/data" "$ROOT/pyproject.toml" "$BUILD/code/"
find "$BUILD/code" -name __pycache__ -prune -exec rm -rf {} +
dataset "$BUILD/code" crs-code "crs-code"

mkdir -p "$BUILD/results/results"
[ -d "$ROOT/results" ] && cp -r "$ROOT/results/." "$BUILD/results/results/"
touch "$BUILD/results/results/KEEP" "$BUILD/results/crs-results.marker"
dataset "$BUILD/results" crs-results "crs-results"

sed -e "s/^JOB_FILE = .*/JOB_FILE = \"$JOBS\"  # set by push.sh/" \
    -e "s/^GROUPS: list = .*/GROUPS: list = $GROUPS_PY  # set by push.sh/" \
    "$ROOT/kaggle/run_kernel.py" > "$BUILD/kernel/run_kernel.py"
sed "s/{USER}/$KUSER/g" "$ROOT/kaggle/kernel-metadata.json.tmpl" > "$BUILD/kernel/kernel-metadata.json"

echo "waiting 60 s for the datasets to finish processing..."
sleep 60
kaggle kernels push -p "$BUILD/kernel"
echo "pushed (GPU T4 x2). Follow with:"
echo "  kaggle kernels status $KUSER/crs-run"
echo "  kaggle kernels logs -f $KUSER/crs-run"

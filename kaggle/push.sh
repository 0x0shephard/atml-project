#!/usr/bin/env bash
# Push the code, the results gathered so far and the run kernel to Kaggle.
#
#   [HOURS=10.5] [KERNEL=crs-run] kaggle/push.sh <kaggle-username> [smoke.yaml|priority.yaml] [GROUP ...]
#
# KERNEL names the notebook. Kaggle runs one session per notebook, so a second session at the same
# time on the same account needs another name (e.g. KERNEL=crs-run-b). Give parallel sessions
# different job files or groups: two sessions must never write the same run folder.
#
# Needs the kaggle CLI (`uv tool install kaggle --with six`), signed in with `kaggle auth login`.
set -euo pipefail
KUSER=${1:?usage: push.sh <kaggle-username> [jobfile] [group ...]}
JOBS=${2:-priority.yaml}
shift $(( $# >= 2 ? 2 : 1 ))
GROUPS_PY=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@")

ROOT=$(cd "$(dirname "$0")/.." && pwd)
KERNEL=${KERNEL:-crs-run}
BUILD="$ROOT/build/kaggle-$KUSER-$KERNEL"  # per notebook, so two pushes can overlap
rm -rf "$BUILD" && mkdir -p "$BUILD/code" "$BUILD/results" "$BUILD/kernel"

dsver() {  # "<status> <version>" of a dataset; retries through API hiccups, "unknown 0" if it gives up
  local out
  for _ in 1 2 3 4 5; do
    if out=$(kaggle datasets status "$1" --format 'json(status,current_version_number)' 2>/dev/null); then
      python3 - "$out" <<'PY' && return 0
import json, sys
d = json.loads(sys.argv[1])
v = d.get("current_version_number", d.get("currentVersionNumber", 0)) or 0
print(str(d.get("status", "")).lower().split(".")[-1], int(v))
PY
    fi
    sleep 15
  done
  echo "unknown 0"
}

wait_ready() {  # slug old_version: wait until a version newer than old_version is ready (up to 15 min)
  local st v
  for _ in $(seq 1 60); do
    read -r st v <<<"$(dsver "$KUSER/$1")"
    if [ "$st" = "ready" ] && [ "${v:-0}" -gt "$2" ]; then echo "$1: version $v is ready"; return 0; fi
    sleep 15
  done
  echo "warning: could not confirm a new $1 version is ready; the kernel stops itself if it finds old code" >&2
}

DSV=0  # version a dataset had before this upload (set by dataset)
dataset() {  # dir slug title
  cat > "$1/dataset-metadata.json" <<JSON
{"title": "$3", "id": "$KUSER/$2", "licenses": [{"name": "CC0-1.0"}]}
JSON
  local st
  read -r st DSV <<<"$(dsver "$KUSER/$2")"
  if [ "$st" != "unknown" ]; then
    kaggle datasets version -p "$1" -m "update $(date -u +%FT%TZ)" --dir-mode zip -q
  else
    DSV=0
    kaggle datasets create -p "$1" --dir-mode zip -q
  fi
}

cp -r "$ROOT/src" "$ROOT/configs" "$ROOT/data" "$ROOT/pyproject.toml" "$BUILD/code/"
find "$BUILD/code" -name __pycache__ -prune -exec rm -rf {} +
# stamp this upload; the kernel refuses to run code with another stamp. On 2026-11-23 a kernel
# started before Kaggle had processed the new crs-code version, ran the previous one and failed
CODE_VERSION="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)-$(date -u +%Y%m%dT%H%M%SZ)"
echo "$CODE_VERSION" > "$BUILD/code/CODE_VERSION"
dataset "$BUILD/code" crs-code "crs-code"
CODE_OLD=$DSV

mkdir -p "$BUILD/results/results"
[ -d "$ROOT/results" ] && cp -r "$ROOT/results/." "$BUILD/results/results/"
# runs with a longer length limit live in their own folders (results-32k-ext/ ...): ship them too,
# so an extend job resumes instead of starting over
for d in "$ROOT"/results-*/; do
  [ -d "$d" ] && cp -r "$d" "$BUILD/results/$(basename "$d")"
done
touch "$BUILD/results/results/KEEP" "$BUILD/results/crs-results.marker"
dataset "$BUILD/results" crs-results "crs-results"
RESULTS_OLD=$DSV

HOURS=${HOURS:-10.5}  # generation time budget for this session (Kaggle stops it at 12 h)
sed -e "s/^JOB_FILE = .*/JOB_FILE = \"$JOBS\"  # set by push.sh/" \
    -e "s/^TIME_BUDGET_H = [0-9.]*/TIME_BUDGET_H = $HOURS/" \
    -e "s/^GROUPS: list = .*/GROUPS: list = $GROUPS_PY  # set by push.sh/" \
    -e "s/^CODE_VERSION = .*/CODE_VERSION = \"$CODE_VERSION\"  # set by push.sh/" \
    "$ROOT/kaggle/run_kernel.py" > "$BUILD/kernel/run_kernel.py"
sed -e "s/{USER}/$KUSER/g" -e "s/{KERNEL}/$KERNEL/g" "$ROOT/kaggle/kernel-metadata.json.tmpl" > "$BUILD/kernel/kernel-metadata.json"

echo "waiting for Kaggle to finish processing the datasets..."
wait_ready crs-code "$CODE_OLD"
wait_ready crs-results "$RESULTS_OLD"
kaggle kernels push -p "$BUILD/kernel"
echo "pushed (GPU T4 x2). Follow with:"
echo "  kaggle kernels status $KUSER/$KERNEL"
echo "  kaggle kernels logs -f $KUSER/$KERNEL"
echo "  kaggle/pull.sh $KUSER $KERNEL"

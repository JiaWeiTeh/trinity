#!/usr/bin/env bash
# helix.sh -- up / submit / reduce / down for TRINITY batches on HPC helix.
#
# Design, in one line each:
#   up      rsync the repo (code only, no outputs) to helix
#   submit  ONE SLURM job array; one task = one config = one process  (PLAN.md C-3, free)
#   reduce  run the committed harness reducers ON HELIX; only CSVs are ever produced locally
#   down    rsync back data-new/*.csv + the manifest. Raw dictionary.jsonl STAYS on helix.
#
# Why reduce-on-cluster is not optional: a single run's dictionary.jsonl is ~9 MB, so a
# 1000-task array is ~9 GB of raw output against a few hundred kB of reduced CSV. Pulling
# raw down does not scale and is not needed -- and keeping it on helix means a reducer bug
# (cf. PLAN.md B11.0 S1, the thin-shell layer volume) costs a re-reduce, not a re-run.
#
# ponytail: rsync + sbatch already do 90% of this. No framework, no state file, no daemon.
set -euo pipefail

: "${HELIX_HOST:?set HELIX_HOST, e.g. export HELIX_HOST=user@helix.uni-heidelberg.de}"
: "${HELIX_ROOT:=~/trinity}"          # remote checkout
: "${ARM:?set ARM, e.g. export ARM=b13_grid}"
LOCAL_ROOT="$(git rev-parse --show-toplevel)"
SHA="$(git rev-parse --short HEAD)$(git status --porcelain | grep -q . && echo '+dirty' || true)"
REMOTE_OUT="${HELIX_ROOT}/outputs/${ARM}_${SHA}"      # C-7: dir embeds arm + SHA
DATA_NEW="${LOCAL_ROOT}/docs/dev/phii-identity/data-new"

case "${1:-}" in

up)
    case "$SHA" in *+dirty)
        echo "REFUSING: working tree is dirty. C-6 stamps would read '${SHA}' and C-1 needs a" >&2
        echo "pinned baseline. Commit (or stash) first." >&2; exit 1;; esac
    rsync -az --delete \
        --include='.git/' --exclude='outputs/' --exclude='scratch/' --exclude='.venv/' \
        --exclude='__pycache__/' --exclude='*.pyc' \
        "${LOCAL_ROOT}/" "${HELIX_HOST}:${HELIX_ROOT}/"
    echo "up: ${SHA} -> ${HELIX_HOST}:${HELIX_ROOT}"
    ;;

submit)
    # $2 = manifest of configs, one per line, e.g. docs/dev/phii-identity/harness/b13_grid.txt
    CFG="${2:?usage: helix.sh submit <config-list-file>}"
    N=$(grep -cve '^\s*$' -e '^\s*#' "${LOCAL_ROOT}/${CFG}")
    rsync -az "${LOCAL_ROOT}/${CFG}" "${HELIX_HOST}:${HELIX_ROOT}/${CFG}"
    ssh "${HELIX_HOST}" "mkdir -p '${REMOTE_OUT}' && cd '${HELIX_ROOT}' && \
        sbatch --array=0-$((N-1))%${THROTTLE:-200} \
               --export=ALL,TRINITY_CFG='${CFG}',TRINITY_OUT='${REMOTE_OUT}',TRINITY_SHA='${SHA}' \
               docs/dev/phii-identity/harness/helix_array.sbatch"
    echo "submit: ${N} tasks, arm ${ARM}, code ${SHA} -> ${REMOTE_OUT}"
    ;;

reduce)
    # Runs on helix. Add reducers here as they are needed -- each writes ONE stamped CSV.
    ssh "${HELIX_HOST}" "cd '${HELIX_ROOT}' && mkdir -p docs/dev/phii-identity/data-new && \
        python3 docs/dev/phii-identity/harness/make_manifest.py '${REMOTE_OUT}' \
            --arm '${ARM}' --sha '${SHA}' \
            --out docs/dev/phii-identity/data-new/${ARM}_manifest.csv && \
        python3 docs/dev/phii-identity/harness/alphap_screen.py '${REMOTE_OUT}'/*/ \
            --out docs/dev/phii-identity/data-new/${ARM}_alphap.csv"
    echo "reduce: done on helix (raw output left in place for re-reduce)"
    ;;

down)
    mkdir -p "${DATA_NEW}"
    rsync -az "${HELIX_HOST}:${HELIX_ROOT}/docs/dev/phii-identity/data-new/${ARM}_*.csv" "${DATA_NEW}/"
    echo "down: ${DATA_NEW}"
    ls -la "${DATA_NEW}"/${ARM}_*.csv
    echo
    echo "Check the manifest BEFORE using any of this:"
    echo "  a short CSV with a full manifest means tasks failed, not that the effect is small."
    ;;

*)  sed -n '2,20p' "$0"; exit 1;;
esac

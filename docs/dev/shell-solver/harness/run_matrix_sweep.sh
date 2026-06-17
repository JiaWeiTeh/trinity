#!/usr/bin/env bash
# Resumable matrix sweep over the analysis configs.
#
# Runs each config's ~10-min matrix capture (capture_replay_variants.py in
# PER_PHASE_N matrix mode), writing docs/dev/shell-solver/data/replay_variants_
# matrix_<tag>.csv. It SKIPS any config whose CSV already exists, so it is safe
# to re-run after a container reset / reclaim: only the missing configs run again
# (the committed+pushed CSVs are restored by the fresh clone and skipped).
#
# Resilience workflow:
#   1. Run this script (foreground or background).
#   2. As each config's CSV appears, commit AND PUSH it (only pushed commits
#      survive a full reclaim).
#   3. If the container resets, just re-run this script -- done configs are
#      skipped, the sweep resumes from where it stopped.
#
# Usage:
#   bash docs/dev/shell-solver/harness/run_matrix_sweep.sh
set -u
cd "$(git rev-parse --show-toplevel)"
DATA=docs/dev/shell-solver/data
H=docs/dev/shell-solver/harness/capture_replay_variants.py

run() {            # run <tag> <param-arg>
  local tag="$1" arg="$2"
  local csv="$DATA/replay_variants_matrix_${tag}.csv"
  if [ -f "$csv" ]; then echo ">>> SKIP $tag (csv exists)"; return; fi
  echo ">>> START $tag ($(date +%H:%M:%S))"
  PER_PHASE_N=15 MATRIX_MAX_S=600 timeout 640 \
    python "$H" "$arg" > "/tmp/matrix_${tag}.out" 2>&1
  echo ">>> DONE $tag rc=$? : $(grep -E 'captures per phase' "/tmp/matrix_${tag}.out" | tail -1)"
}

# tag (== harness-derived CSV tag) -> param arg
run sfe0.3             0.3
run sfe0.6             0.6
run steep              docs/dev/transition/harness/steep.param
run dense_flat         docs/dev/transition/harness/dense_flat.param
run mock_hybr          docs/dev/transition/harness/mock_hybr.param
run probe_typical_hybr docs/dev/archive/betadelta/diagnostics/probe_typical_hybr.param
echo ">>> MATRIX SWEEP COMPLETE"

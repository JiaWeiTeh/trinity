#!/usr/bin/env bash
# Resumable matrix sweep over the analysis configs.
#
# Diagnostic principle: what matters is the number of solves sampled *in* a
# phase, NOT the wall time spent reaching it. So each run targets a fixed sample
# count per phase (N_IMPLICIT=100 by default) and MATRIX_MAX_S is only a safety
# cap. A run stops as soon as every phase target is met (or the phase is
# exhausted / the cap is hit).
#
# Reset-safety: a config is SKIPPED if its CSV already holds >= N_IMPLICIT
# implicit-phase captures, so re-running after a container reclaim re-does only
# the unfinished configs (commit+push each CSV as it lands).
#
# Usage:  bash docs/dev/shell-solver/harness/run_matrix_sweep.sh
set -u
cd "$(git rev-parse --show-toplevel)"
DATA=docs/dev/shell-solver/data
H=docs/dev/shell-solver/harness/capture_replay_variants.py
IMPL_TARGET=${N_IMPLICIT:-100}

implicit_caps() {   # distinct implicit-phase captures already in a CSV
  local csv=$1
  [ -f "$csv" ] || { echo 0; return; }
  awk -F, 'NR>1 && $2=="implicit"{seen[$1]=1} END{print length(seen)+0}' "$csv"
}

run() {             # run <tag> <param-arg> <max_s> <n_transition>
  local tag=$1 arg=$2 maxs=$3 ntrans=${4:-0}
  local csv=$DATA/replay_variants_matrix_${tag}.csv
  local have; have=$(implicit_caps "$csv")
  if [ "$have" -ge "$IMPL_TARGET" ]; then
    echo ">>> SKIP $tag (implicit=$have >= $IMPL_TARGET)"; return
  fi
  echo ">>> START $tag (impl=$IMPL_TARGET ntrans=$ntrans maxs=${maxs}s $(date +%H:%M:%S))"
  N_ENERGY=${N_ENERGY:-20} N_IMPLICIT=$IMPL_TARGET N_TRANSITION=$ntrans N_MOMENTUM=0 \
    MATRIX_MAX_S=$maxs timeout $((maxs + 40)) \
    python "$H" "$arg" > "/tmp/matrix_${tag}.out" 2>&1
  echo ">>> DONE $tag rc=$? : $(grep -E 'captures per phase' "/tmp/matrix_${tag}.out" | tail -1)"
}

# Bumps: 100 implicit samples, no transition seek, 25-min safety cap.
run sfe0.6             0.6                                                    1500 0
run steep              docs/dev/transition/harness/steep.param               1500 0
run dense_flat         docs/dev/transition/harness/dense_flat.param          1500 0
run mock_hybr          docs/dev/transition/harness/mock_hybr.param           1500 0
run probe_typical_hybr docs/dev/archive/betadelta/diagnostics/probe_typical_hybr.param 1500 0
# Deep run on the current-default config: 100 implicit AND push into transition.
run sfe0.3             0.3                                                    2700 60
echo ">>> MATRIX SWEEP COMPLETE"

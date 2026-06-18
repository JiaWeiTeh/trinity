#!/usr/bin/env bash
# Resumable F1-resample P0 sweep over the 6 study configs.
#
# Drives capture_replay_bubble.py (built by the parallel task) BY PATH -- it runs
# one live TRINITY sim per config and, on each gated bubble call, replays the
# baseline (60k dense resample) + every coarse variant, timing + comparing each,
# writing one row per captured-call x variant to
# docs/dev/performance/data/bubble_resample_<config>.csv.
#
# Diagnostic principle (mirrors run_matrix_sweep.sh): what matters is the number
# of bubble solves sampled *in* a phase, NOT the wall time spent reaching it. Each
# run targets N_ENERGY=20 energy + N_IMPLICIT=100 implicit captures per config;
# MATRIX_MAX_S is only a per-config safety cap.
#
# Reset-safety: a config is SKIPPED if its CSV already holds >= N_IMPLICIT
# implicit-phase captures (distinct call_index), so re-running after a container
# reclaim re-does only the unfinished configs. Idempotent / re-runnable; commit
# each CSV as it lands.
#
# Usage:  bash docs/dev/performance/harness/run_p0_sweep.sh
set -u
cd "$(git rev-parse --show-toplevel)"
DATA=docs/dev/performance/data
H=docs/dev/performance/harness/capture_replay_bubble.py
mkdir -p "$DATA"

N_ENERGY=${N_ENERGY:-20}
IMPL_TARGET=${N_IMPLICIT:-100}

# Each gated call replays 6 variants; at TIMING_REPS=3 that is ~18 solves/call,
# which blows the wall budget before reaching 100 implicit. Accuracy (rel_*) is
# rep-independent (deterministic) and the speedup is a MEAN over ~100 calls, so
# 1 timing rep is statistically sufficient and ~3x cheaper. Override if desired.
export TIMING_REPS=${TIMING_REPS:-1}

# A temp sfe0.6 param (mCloud 1e5 / sfe 0.6); cleaned up on exit.
SFE06_PARAM=""
cleanup() { [ -n "$SFE06_PARAM" ] && rm -f "$SFE06_PARAM"; }
trap cleanup EXIT

make_sfe06_param() {
  SFE06_PARAM=$(mktemp /tmp/p0_sfe06_XXXX.param)
  printf 'mCloud    1e5\nsfe    0.6\n' > "$SFE06_PARAM"
  echo "$SFE06_PARAM"
}

implicit_caps() {   # distinct implicit-phase captures (call_index) already in a CSV
  local csv=$1
  [ -f "$csv" ] || { echo 0; return; }
  # schema: config,phase,call_index,...  -> gate on $2==implicit, key on $3
  awk -F, 'NR>1 && $2=="implicit"{seen[$3]=1} END{print length(seen)+0}' "$csv"
}

run() {             # run <tag> <param-arg> <max_s>
  local tag=$1 arg=$2 maxs=$3
  local csv=$DATA/bubble_resample_${tag}.csv
  local have; have=$(implicit_caps "$csv")
  if [ "$have" -ge "$IMPL_TARGET" ]; then
    echo ">>> SKIP $tag (implicit=$have >= $IMPL_TARGET)"; return
  fi
  echo ">>> START $tag (energy=$N_ENERGY impl=$IMPL_TARGET maxs=${maxs}s $(date +%H:%M:%S))"
  N_ENERGY=$N_ENERGY N_IMPLICIT=$IMPL_TARGET N_TRANSITION=0 N_MOMENTUM=0 \
    MATRIX_MAX_S=$maxs timeout $((maxs + 40)) \
    python "$H" "$arg" > "/tmp/p0_${tag}.out" 2>&1
  local rc=$?
  echo ">>> DONE $tag rc=$rc (implicit now=$(implicit_caps "$csv")) : $(tail -1 "/tmp/p0_${tag}.out")"
}

# Order: cheapest-to-reach-implicit first; degenerate sfe configs last (biggest cap).
# Caps sized for ~6 solves/gated-call (TIMING_REPS=1) x ~120 gated calls:
# mock ~720s, realistic ~1100s (~1.5s/solve), degenerate ~4300s (~6s/solve).
# Caps carry headroom so a config can actually REACH 100 implicit in one pass.
run mock_hybr           docs/dev/transition/harness/mock_hybr.param                    1200
run probe_typical_hybr  docs/dev/archive/betadelta/diagnostics/probe_typical_hybr.param 2400
run steep               docs/dev/transition/harness/steep.param                        2400
run dense_flat          docs/dev/transition/harness/dense_flat.param                   2400
run sfe0.3              param/simple_cluster.param                                     6000
run sfe0.6              "$(make_sfe06_param)"                                           6000

echo ">>> P0 SWEEP COMPLETE"
echo ">>> aggregate with: python docs/dev/performance/harness/aggregate_p0.py"

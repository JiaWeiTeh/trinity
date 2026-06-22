#!/usr/bin/env bash
# H5 clamp-width sweep: LEGACY solver x {W1,W2,W3} box widths over 6 configs.
# = 18 cells, plus a W0 consistency re-run for one config (small_dense) to verify
# the harness reproduces the committed c0_*_legacy.csv crossing.
#
# Each cell = one sim/process (trinity leaks module-global state in-process), so we
# run them via xargs -P 4 (box has 4 cores; runs are single-core-bound). Per-config
# stop_t is sized ~2-4x the committed W0 crossing time (or a few Myr for the
# slow/non-crossing configs) so we can see whether the ratio recovers vs crosses.
#
#   usage: bash h5_run_matrix.sh
#
# Writes h5_sweep.csv (summary, one row per cell, appended) + per-cell
# data/h5_traj_<cfg>_<W>.csv trajectories. W0 (committed c0_*_legacy.csv) and hybr
# (committed c0_*_h0.csv) rows are folded in later by h5_analyze.py.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFGDIR="$HERE/../../cleanroom/configs"
ROWS="$HERE/data/rows"   # one CSV row-file per cell (avoids the concurrent-append race)
TIMEOUT=1800   # 30 min/cell hard cap
export HERE CFGDIR ROWS TIMEOUT

rm -rf "$ROWS"
mkdir -p "$HERE/data" "$ROWS" /tmp/h5

# job lines: "config width stop_t"  (stop_t ~2-4x committed W0 cross_t; a few Myr
# for the slow/non-crossing configs). W0 is NOT re-run here — it is the committed
# c0_*_legacy.csv, and the harness->committed consistency is verified separately by
# the W0 small_dense smoke (h5_run_variant.py --width W0, recorded in the writeup).
JOBS=$(cat <<'EOF'
small_dense_highsfe W1 0.1
small_dense_highsfe W2 0.1
small_dense_highsfe W3 0.1
simple_cluster W1 0.6
simple_cluster W2 0.6
simple_cluster W3 0.6
pl2_steep W1 0.6
pl2_steep W2 0.6
pl2_steep W3 0.6
midrange_pl0 W1 2.5
midrange_pl0 W2 2.5
midrange_pl0 W3 2.5
be_sphere W1 3.5
be_sphere W2 3.5
be_sphere W3 3.5
large_diffuse_lowsfe W1 3.5
large_diffuse_lowsfe W2 3.5
large_diffuse_lowsfe W3 3.5
EOF
)

# -P 3 (not 4): the box has 4 cores and may be shared with another agent's heavy
# jobs; over-subscribing starves trinity's first-implicit-segment betadelta solve
# (25-75 bubble-structure solves) and inflates per-cell runtime.
echo "$JOBS" | xargs -P 3 -L 1 bash -c '
  cfg="$1"; w="$2"; st="$3"
  echo "[start] $cfg $w stop_t=$st $(date +%T)"
  timeout "$TIMEOUT" python "$HERE/h5_run_variant.py" \
    --width "$w" --param "$CFGDIR/${cfg}.param" --stop_t "$st" \
    --csv "$ROWS/${cfg}_${w}.csv" --traj "$HERE/data/h5_traj_${cfg}_${w}.csv" \
    --run-dir "/tmp/h5/${cfg}_${w}" \
    > "/tmp/h5/${cfg}_${w}.log" 2>&1 \
    && echo "[done]  $cfg $w $(date +%T)" \
    || echo "[FAIL/timeout] $cfg $w rc=$? $(date +%T)"
' _

echo "ALL DONE $(date +%T)"

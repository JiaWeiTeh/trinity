#!/usr/bin/env bash
# Drive the END-TO-END science gate matrix SEQUENTIALLY (one run at a time, so
# wall-clock timing is uncontended). 2 configs x {baseline, phiguard, clip, cgs}.
# Each run is bounded by stop_t so it stops NATURALLY at the same simulated time
# (identical-length trajectories) within the per-run timeout.
#
# Usage:  bash docs/dev/shell-solver/harness/run_endtoend_matrix.sh
# Writes per-run logs to /tmp/eteo_<config>_<idea>.log; the metrics JSON is the
# final ENDTOEND_METRICS line of each. ~30-40 min total.
set -u
cd /home/user/trinity

STOP_T=0.0015
PER_RUN_TIMEOUT=520

declare -A PARAM
PARAM[simple_cluster]="param/simple_cluster.param"
PARAM[probe_typical_hybr]="/tmp/probe_typical_hybr.param"

for cfg in simple_cluster probe_typical_hybr; do
  for idea in baseline phiguard clip cgs; do
    log="/tmp/eteo_${cfg}_${idea}.log"
    echo "=== RUN ${cfg} / ${idea} -> ${log} ==="
    timeout "${PER_RUN_TIMEOUT}" python docs/dev/shell-solver/harness/run_endtoend.py \
        "${PARAM[$cfg]}" "${idea}" "${STOP_T}" > "${log}" 2>&1
    echo "  exit=$? metrics: $(grep ENDTOEND_METRICS "${log}" | tail -1)"
  done
done
echo "=== MATRIX DONE ==="

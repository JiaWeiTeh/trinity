#!/usr/bin/env bash
# H3 Eb-floor experiment — run the full (config, variant) matrix, ONE sim per
# process (trinity leaks module-level globals in-process), OMP_NUM_THREADS=1,
# each cell bounded by a wall-clock `timeout` AND a small --stop_t.
#
# Outcomes recorded per cell in h3_eval.csv: crashed / completed (clean
# end_reason) / timeout (the runner won't append a row on timeout, so the
# absence of a row == timeout; we also echo a TIMEOUT marker to the log).
#
# Usage: bash docs/dev/transition/pt4/h3_run_matrix.sh
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
PT4=docs/dev/transition/pt4
CLEAN=docs/dev/transition/cleanroom/configs
FLC=docs/dev/failed-large-clouds/harness/params
EVAL=$PT4/h3_eval.csv
LOG=$PT4/h3_run_matrix.log
FLOOR=1e-3
export OMP_NUM_THREADS=1

rm -f "$EVAL"
: > "$LOG"

# config : param-path : stop_t : timeout(s) : class
# collapse configs reach the decision at t~0.003 Myr -> small stop_t, fits in timeout.
# stall/healthy configs: short stop_t (0.05) just to confirm the floor is a no-op.
CELLS=(
  "simple_cluster:$CLEAN/simple_cluster.param:0.05:360:stall"
  "large_diffuse_lowsfe:$CLEAN/large_diffuse_lowsfe.param:0.05:360:stall"
  "small_dense_highsfe:$CLEAN/small_dense_highsfe.param:0.05:360:stall"
  "midrange_pl0:$CLEAN/midrange_pl0.param:0.05:360:stall"
  "pl2_steep:$CLEAN/pl2_steep.param:0.05:360:stall"
  "be_sphere:$CLEAN/be_sphere.param:0.05:360:stall"
  "fail_repro:$FLC/fail_repro.param:0.01:300:collapse"
  "fail_helix:$FLC/fail_helix.param:0.01:300:collapse"
  "mass_5e8:$FLC/mass_5e8.param:0.01:300:collapse"
  "mass_1e9:$FLC/mass_1e9.param:0.01:300:collapse"
  "small_1e5:$FLC/small_1e5.param:0.05:360:healthy"
  "small_1e6:$FLC/small_1e6.param:0.05:360:healthy"
  "small_1e7:$FLC/small_1e7.param:0.05:360:healthy"
)

for variant in V0 EBFLOOR; do
  for cell in "${CELLS[@]}"; do
    IFS=':' read -r cfg param stopt tmo klass <<< "$cell"
    out=/tmp/h3/${cfg}_${variant}
    traj=$PT4/traj/h3_traj_${cfg}_${variant}.csv
    echo "=== [$variant $cfg] ($klass) stop_t=$stopt timeout=${tmo}s ===" | tee -a "$LOG"
    timeout "$tmo" python "$PT4/h3_run_variant.py" \
      --variant "$variant" --param "$param" --stop_t "$stopt" \
      --floor "$FLOOR" --csv "$EVAL" --traj "$traj" --out "$out" \
      >> "$LOG" 2>&1
    rc=$?
    if [ "$rc" -eq 124 ]; then
      echo "TIMEOUT [$variant $cfg] after ${tmo}s -- salvaging partial jsonl" | tee -a "$LOG"
      python "$PT4/h3_salvage_timeout.py" --config "$cfg" --variant "$variant" \
        --out "$out" --csv "$EVAL" --traj "$traj" --floor "$FLOOR" \
        --timeout_s "$tmo" >> "$LOG" 2>&1
    fi
  done
done
echo "DONE matrix" | tee -a "$LOG"

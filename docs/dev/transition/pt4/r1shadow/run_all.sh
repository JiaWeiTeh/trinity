#!/usr/bin/env bash
# Run the R1 shadow (V0 = plain shadow-instrumented production, NO monkeypatch)
# on all 8 configs, one sim per process, OMP_NUM_THREADS=1, timeout-bounded,
# up to 4 in parallel across the 4 cores. Each run writes dictionary.jsonl +
# shadow_R1_1b.csv to runs/<config>/, and a <config>_status.txt with runtime+status.
#
# Production untouched: --variant V0 in h3_run_variant.py applies NO patch.
#
# Usage: bash docs/dev/transition/pt4/r1shadow/run_all.sh
set -u
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"   # repo root (worktree)
cd "$ROOT"
DRV="docs/dev/transition/pt4/h3_run_variant.py"
OUTBASE="docs/dev/transition/pt4/r1shadow/runs"
CLEAN="docs/dev/transition/cleanroom/configs"
HEAVY="docs/dev/failed-large-clouds/harness/params"
TIMEOUT=900
mkdir -p "$OUTBASE"

# config:param:stop_t  (stop_t = ~1.3-1.5x known blowout epoch)
JOBS=(
  "simple_cluster:$CLEAN/simple_cluster.param:0.2"
  "small_dense_highsfe:$CLEAN/small_dense_highsfe.param:0.05"
  "midrange_pl0:$CLEAN/midrange_pl0.param:0.6"
  "pl2_steep:$CLEAN/pl2_steep.param:1.2"
  "be_sphere:$CLEAN/be_sphere.param:1.2"
  "large_diffuse_lowsfe:$CLEAN/large_diffuse_lowsfe.param:4.0"
  "fail_repro:$HEAVY/fail_repro.param:0.02"
  "fail_helix:$HEAVY/fail_helix.param:0.02"
)

run_one() {
  local cfg="$1" param="$2" stop_t="$3"
  local out="$OUTBASE/$cfg"
  mkdir -p "$out"
  local t0 rc status
  t0=$(date +%s)
  OMP_NUM_THREADS=1 timeout "$TIMEOUT" python "$DRV" \
    --variant V0 --param "$param" --stop_t "$stop_t" \
    --out "$out" --csv "$OUTBASE/${cfg}_row.csv" \
    > "$OUTBASE/${cfg}.log" 2>&1
  rc=$?
  local t1; t1=$(date +%s)
  if [ "$rc" -eq 124 ]; then status="timeout"; else status="completed"; fi
  {
    echo "config=$cfg"
    echo "param=$param"
    echo "stop_t=$stop_t"
    echo "timeout_s=$TIMEOUT"
    echo "exit_code=$rc"
    echo "status=$status"
    echo "runtime_s=$((t1 - t0))"
  } > "$OUTBASE/${cfg}_status.txt"
  echo "[done] $cfg rc=$rc status=$status runtime=$((t1-t0))s"
}
export -f run_one
export OUTBASE DRV TIMEOUT

# Skip small_dense_highsfe if already complete (smoke run).
PENDING=()
for j in "${JOBS[@]}"; do
  cfg="${j%%:*}"
  if [ "$cfg" = "small_dense_highsfe" ] && [ -f "$OUTBASE/$cfg/shadow_R1_1b.csv" ]; then
    # backfill its status if missing
    if [ ! -f "$OUTBASE/${cfg}_status.txt" ]; then
      run_one "$cfg" "$CLEAN/small_dense_highsfe.param" 0.05 &
    fi
    continue
  fi
  PENDING+=("$j")
done

# Launch up to 4 at once.
running=0
for j in "${PENDING[@]}"; do
  IFS=: read -r cfg param stop_t <<< "$j"
  run_one "$cfg" "$param" "$stop_t" &
  running=$((running + 1))
  if [ "$running" -ge 4 ]; then
    wait -n 2>/dev/null || wait
    running=$((running - 1))
  fi
done
wait
echo "ALL RUNS COMPLETE"

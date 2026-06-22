#!/usr/bin/env bash
# C0.2 substrate-certification batch: run every config (hybr) to a SHORT stop_t,
# 3 concurrent (box has 4 cores; runs are single-core-bound, ~35 s/implicit segment).
# Each config writes its own CSV+log, so partial progress survives interruption.
#   usage: bash run_c0_batch.sh [stop_t]      (default 0.05 Myr)
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ST="${1:-0.05}"
SUF="${2:-st${ST/./p}}"   # optional 2nd arg overrides the output suffix (e.g. h0)
CR="$(git rev-parse --show-toplevel)/docs/dev/transition/cleanroom"
export ST SUF CR
ls "$CR"/configs/*.param | xargs -P 3 -I{} bash -c '
  f="$1"; name=$(basename "$f" .param)
  echo "[start] $name $(date +%T)"
  python "$CR/c0_consistency.py" "$f" --stop-t "$ST" \
    --out "$CR/data/c0_${name}_${SUF}.csv" > "$CR/data/c0_${name}_${SUF}.log" 2>&1 \
    && echo "[done]  $name $(date +%T)" || echo "[FAIL]  $name $(date +%T)"
' _ {}
echo "ALL DONE $(date +%T)"

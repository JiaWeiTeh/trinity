#!/usr/bin/env bash
set -euo pipefail

NDENS="1e2"
SFES=("001" "010" "020" "030" "050" "080")
MASSES=("1e5" "1e7" "1e8")

PARAM_DIR="param"   # change to "param" if your folder is named that

run_batch () {
  local sfe="$1"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"

  echo "===== $(date) | START batch: sfe=${sfe}, n=${NDENS} ====="

  local -a pids=()

  for m in "${MASSES[@]}"; do
    local param="${PARAM_DIR}/${m}_sfe${sfe}_n${NDENS}.param"
    local log="txt/${m}_sfe${sfe}_n${NDENS}.txt"

    if [[ ! -f "$param" ]]; then
      echo "Missing param file: $param" >&2
      continue
    fi

    echo "Launching: $param -> $log"
    nohup python3 run.py "$param" > "$log" 2>&1 &
    local pid="$!"
    pids+=("$pid")
    echo "  PID: $pid"
  done

  if [[ "${#pids[@]}" -eq 0 ]]; then
    echo "No jobs launched for sfe=${sfe} (missing params?). Skipping."
    return 0
  fi

  echo "Waiting for ${#pids[@]} job(s) to finish for sfe=${sfe}..."
  for pid in "${pids[@]}"; do
    if wait "$pid"; then
      echo "  PID $pid finished OK"
    else
      rc=$?
      echo "  PID $pid finished with exit code $rc" >&2
    fi
  done

  echo "===== $(date) | END batch: sfe=${sfe}, n=${NDENS} ====="
}

for sfe in "${SFES[@]}"; do
  run_batch "$sfe"
done

echo "All SFE batches completed. $(date)"

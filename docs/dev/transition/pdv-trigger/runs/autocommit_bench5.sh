#!/bin/bash
# In-container bench5 heartbeat committer (Phase 5; the RECORDING half of the no-HPC run).
#
# Every ~2 min: merge the current container's completed arms into the committed bench5 summary +
# trajectory dir (via checkpoint_bench5.py) and, if anything grew, commit + push. A restart then
# loses at most ~2 min. Lives in the repo (not /tmp) so watchdogs can relaunch it after a restart.
# It is the SOLE git committer for the run (watchdogs only relaunch it) — no concurrent-commit race.
#
#   bash docs/dev/transition/pdv-trigger/runs/autocommit_bench5.sh <out_dir>
set -u
OUT="${1:?usage: autocommit_bench5.sh <out_dir>}"
cd "$(dirname "$0")" && cd "$(git rev-parse --show-toplevel)" || exit 1
SUMMARY=docs/dev/transition/pdv-trigger/runs/data/bench5_summary.csv
TRAJ=docs/dev/transition/pdv-trigger/runs/data/bench5_traj
CK=docs/dev/transition/pdv-trigger/runs/checkpoint_bench5.py
BRANCH=feature/pdv-trigger-pt4b
for _ in $(seq 1 300); do   # 300 * 120s = 10h ceiling; watchdogs relaunch if it ever exits
  sleep 120
  out=$(python "$CK" --out "$OUT" --summary "$SUMMARY" --traj-dir "$TRAJ" 2>&1 | tail -1)
  git add "$SUMMARY" "$TRAJ" 2>/dev/null
  if ! git diff --cached --quiet -- "$SUMMARY" "$TRAJ" 2>/dev/null; then
    git commit -m "pdv-trigger: bench5 checkpoint auto (${out})" >/dev/null 2>&1
    for a in 1 2 3 4; do git push origin "$BRANCH" >/dev/null 2>&1 && break; sleep $((2**a)); done
    echo "[autocommit] $(date -u +%H:%M:%S) pushed: ${out}"
  else
    echo "[autocommit] $(date -u +%H:%M:%S) no change: ${out}"
  fi
done

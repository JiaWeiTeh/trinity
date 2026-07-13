#!/bin/bash
# In-container Cf-scan heartbeat committer (the RECORDING half of the no-HPC run).
# Adapted from docs/dev/transition/pdv-trigger/runs/autocommit_bench5.sh.
#
# Every ~2 min: merge the current container's completed arms into the committed summary +
# trajectory dir (via harvest_cf_scan.py, which is an idempotent merge) and, if anything grew,
# commit + push. A restart then loses at most ~2 min of finished arms. It is the SOLE git
# committer while it runs — no concurrent-commit race.
#
# OPTIONAL: arm it only if the step-2 timing probe projects the campaign past ~1.5 h of wall
# clock (see docs/dev/rosette-cf/README.md §3); a campaign that fits one container window can
# just harvest + commit once at the end.
#
#   bash docs/dev/rosette-cf/harness/autocommit_cf_scan.sh <out_dir_glob_base>
#   e.g. bash docs/dev/rosette-cf/harness/autocommit_cf_scan.sh "$WS/outputs/rosette_cf_PISM1e5"
set -u
OUT="${1:?usage: autocommit_cf_scan.sh <base_output_dir>}"
cd "$(dirname "$0")" && cd "$(git rev-parse --show-toplevel)" || exit 1
SUMMARY=docs/dev/rosette-cf/data/cf_scan_PISM1e5_summary.csv
TRAJ=docs/dev/rosette-cf/data/cf_scan_PISM1e5_traj
HARVEST=docs/dev/rosette-cf/harness/harvest_cf_scan.py
BRANCH=$(git rev-parse --abbrev-ref HEAD)
for _ in $(seq 1 300); do   # 300 * 120s = 10h ceiling; relaunch if it ever exits
  sleep 120
  out=$(python "$HARVEST" "$OUT"/* --csv "$SUMMARY" --traj-dir "$TRAJ" 2>&1 | tail -1)
  git add "$SUMMARY" "$TRAJ" 2>/dev/null
  if ! git diff --cached --quiet -- "$SUMMARY" "$TRAJ" 2>/dev/null; then
    git commit -m "rosette-cf: scan checkpoint auto (${out})" >/dev/null 2>&1
    for a in 1 2 3 4; do git push -u origin "$BRANCH" >/dev/null 2>&1 && break; sleep $((2**a)); done
    echo "[autocommit] $(date -u +%H:%M:%S) pushed: ${out}"
  else
    echo "[autocommit] $(date -u +%H:%M:%S) no change: ${out}"
  fi
done

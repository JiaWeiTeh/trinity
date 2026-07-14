#!/bin/bash
# In-container Cf-scan heartbeat committer (the RECORDING half of the no-HPC run).
# Adapted from docs/dev/transition/pdv-trigger/runs/autocommit_bench5.sh.
#
# Every ~2 min: merge the current container's finished arms into the committed summary + trajectory
# dir + gzipped raw dicts (via harvest_cf_scan.py, an idempotent merge) and, if anything grew,
# commit + push. A restart then loses at most ~2 min of finished arms. It is the SOLE git committer
# while it runs — no concurrent-commit race.
#
# LOAD-BEARING here (not optional): the raw dictionary.jsonl per arm is the Rosette deliverable and
# the container is ephemeral (a restart already cost us a probe once), so the dicts must be pushed
# incrementally as arms finish — the campaign spans restarts.
#
#   bash docs/dev/rosette-cf/harness/autocommit_cf_scan.sh <base_output_dir>
#   e.g. bash docs/dev/rosette-cf/harness/autocommit_cf_scan.sh outputs/rosette_cf_survey_PISM1e5_fmix
set -u
OUT="${1:?usage: autocommit_cf_scan.sh <base_output_dir>}"
cd "$(dirname "$0")" && cd "$(git rev-parse --show-toplevel)" || exit 1
# Pin the committer identity here so every heartbeat commit is attributed correctly even after a
# container reclaim wipes the session's git config (maintainer's own address, already public in the
# branch history).
git config user.email jiaweiteh.astro@gmail.com
git config user.name "Jia Wei Teh"
DATA=docs/dev/rosette-cf/data
SUMMARY=$DATA/cf_scan_PISM1e5_summary.csv
TRAJ=$DATA/cf_scan_PISM1e5_traj
DICTS=$DATA/cf_scan_PISM1e5_dicts
HARVEST=docs/dev/rosette-cf/harness/harvest_cf_scan.py
BRANCH=$(git rev-parse --abbrev-ref HEAD)
for _ in $(seq 1 300); do   # 300 * 120s = 10h ceiling; relaunch if it ever exits
  sleep 120
  out=$(python "$HARVEST" "$OUT"/* --csv "$SUMMARY" --traj-dir "$TRAJ" --dicts-dir "$DICTS" 2>&1 | tail -1)
  # Stage the whole data dir: the dicts subdir does not exist until the first arm finishes, and a
  # missing pathspec makes `git add a b c` abort atomically (staging nothing). Adding the parent
  # dir is tolerant of that and picks up summary + traj + dicts together.
  git add "$DATA" 2>/dev/null
  if ! git diff --cached --quiet -- "$DATA" 2>/dev/null; then
    git commit -m "rosette-cf: scan checkpoint auto (${out})" >/dev/null 2>&1
    for a in 1 2 3 4; do git push -u origin "$BRANCH" >/dev/null 2>&1 && break; sleep $((2**a)); done
    echo "[autocommit] $(date -u +%H:%M:%S) pushed: ${out}"
  else
    echo "[autocommit] $(date -u +%H:%M:%S) no change: ${out}"
  fi
done

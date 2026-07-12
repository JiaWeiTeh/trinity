#!/bin/bash
# In-container theta5s heartbeat committer (the RECORDING half of the no-HPC fallback).
#
# Every ~2 min it merges the current container's completed arms into the committed
# runs/data/theta5s_summary.csv (via checkpoint_theta5s.py) and, if that grew, commits + pushes.
# So a container restart loses at most ~2 min of results — everything else is already in git.
#
# It lives in the REPO (not /tmp) on purpose: a hard restart wipes /tmp but not the git tree, so the
# watchdogs (send_later heartbeat + hourly cron) can always relaunch it after a restart. It is the
# SOLE git committer for the run — the watchdogs only relaunch it, they don't commit, so there is no
# concurrent-commit race.
#
#   bash docs/dev/transition/pdv-trigger/runs/autocommit_theta5s.sh <out_dir>
#   (out_dir = the runner's --out, e.g. $SCRATCHPAD/t5s_out)
set -u
OUT="${1:?usage: autocommit_theta5s.sh <out_dir>}"
cd "$(dirname "$0")" && cd "$(git rev-parse --show-toplevel)" || exit 1
SUMMARY=docs/dev/transition/pdv-trigger/runs/data/theta5s_summary.csv
CK=docs/dev/transition/pdv-trigger/runs/checkpoint_theta5s.py
BRANCH=pdv-trigger-pt4
for _ in $(seq 1 300); do   # 300 * 120s = 10h ceiling; watchdogs relaunch if it ever exits
  sleep 120
  out=$(python "$CK" --out "$OUT" --summary "$SUMMARY" 2>&1 | tail -1)
  if ! git diff --quiet "$SUMMARY" 2>/dev/null; then
    git add "$SUMMARY"
    git commit -m "pdv-trigger: theta5s checkpoint auto (${out})" >/dev/null 2>&1
    for a in 1 2 3 4; do git push origin "$BRANCH" >/dev/null 2>&1 && break; sleep $((2**a)); done
    echo "[autocommit] $(date -u +%H:%M:%S) pushed: ${out}"
  else
    echo "[autocommit] $(date -u +%H:%M:%S) no change: ${out}"
  fi
done

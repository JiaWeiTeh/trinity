#!/usr/bin/env bash
# phii-identity arm runner (Batch 14 K5 / Batch 18 K10 / Batch 21 K10-O1) — executes ON the cluster, from the trinity
# repo checkout ($CREPO). Normally driven by ../sync.sh; two steps because
# compute nodes cannot write /home (see paper/II-survey/sync.sh) while
# `git worktree add` writes metadata into $CREPO/.git:
#
#   bash run_arms.sh prep <SWEEP_DIR> <ARM>   # LOGIN node: worktree + patch
#   bash run_arms.sh run  <SWEEP_DIR> <ARM>   # compute job: the actual runs
#   ARM ∈ baseline | k5a_swap | k5a_driving | k10 | k10_o1
#
# Contamination rules honoured:
#   C-1/C-7  every arm runs in a detached worktree at the PINNED BASE_SHA, so
#            the sweep is immune to later pushes; SWEEP_DIR carries the UTC
#            stamp minted at submit time, so outputs are never reused across
#            versions. Worktrees live on /gpfs. The exact applied code delta
#            is recorded as <SWEEP>/<ARM>_applied.diff.
#   C-2..C-6 inherited from run_batch.py (one process per run, materialised
#            params with recorded overrides, provenance stamps).
#
# Ladder (pre-registered as G14.4, reused by G18.3/G21): SC (simple_cluster), F1LO/F1HI,
# B3M, B3MW01 — separate processes; matched-t comparison + fate table are the
# reduce step (harness/compare_trajectories.py).
set -euo pipefail

STEP=${1:?usage: run_arms.sh prep|run SWEEP_DIR ARM}
SWEEP=${2:?usage: run_arms.sh prep|run SWEEP_DIR ARM}
ARM=${3:?usage: run_arms.sh prep|run SWEEP_DIR ARM}
BASE_SHA=${BASE_SHA:-cce8c924}            # pinned base for ALL arms in this bundle
# $4/$5 win when non-empty; env is the fallback for login-node use. The compute job
# runs with --export=NONE, so it MUST get these positionally or it silently
# defaults -- which is exactly what happened to sweep 20260829_143834Z.
CONFIGS=${4:-}
CONFIGS=${CONFIGS:-${CONFIGS_ENV:-SC,F1LO,F1HI,B3M,B3MW01}}
STOP_T=${5:-}
STOP_T=${STOP_T:-1.5}

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../../.." && pwd)
WT=$SWEEP/wt_$ARM

case "$STEP" in
  prep)
    mkdir -p "$SWEEP"
    if [ -d "$WT" ]; then echo "prep: $WT already exists — keeping it (C-7)"; exit 0; fi
    git -C "$REPO" worktree add --detach "$WT" "$BASE_SHA"
    case "$ARM" in
      baseline)     ;;
      k5a_swap)     git -C "$WT" apply "$HERE/k5a_swap.patch" ;;
      k5a_driving)  git -C "$WT" apply "$HERE/k5a_driving.patch" ;;
      k10)          echo "REFUSED: k10 (Batch 18) is SUPERSEDED and HELD -- use k10_o1." >&2
                    echo "Override deliberately with ARM=k10_force if you really need it." >&2; exit 1 ;;
      k10_force)    git -C "$WT" apply "$HERE/k10_arm.patch" ;;
      k10_o1)       git -C "$WT" apply "$HERE/k10_o1_arm.patch" ;;
      *) echo "unknown arm: $ARM (baseline|k5a_swap|k5a_driving|k10_o1; k10 is held)"; exit 1 ;;
    esac
    git -C "$WT" diff > "$SWEEP/${ARM}_applied.diff"   # exact code delta, recorded
    echo "prep: $ARM worktree ready at $WT (base $BASE_SHA)"
    ;;
  run)
    [ -d "$WT" ] || { echo "run: no worktree at $WT — run 'prep' on the login node first"; exit 1; }
    python "$WT/docs/dev/phii-identity/harness/run_batch.py" \
      --arm "phii_$ARM" --configs "$CONFIGS" --stop-t "$STOP_T" \
      --root "$SWEEP/$ARM"
    ;;
  *) echo "unknown step: $STEP (prep|run)"; exit 1 ;;
esac

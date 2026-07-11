#!/usr/bin/env bash
# Laptop-side driver for the 📏 theta5s f_A-validation matrix (81 runs) on bwForCluster Helix.
# Same shape as ./sync_theta5k.sh — this workstream is TRACKED (under docs/dev), so code travels
# by `git pull`; `up` just pulls the latest committed code (which must include the Phase-2 f_A
# wiring — cooling_boost_fA in the registry).
#
#   ./sync_theta5s.sh up       # git pull the latest committed code on the cluster
#   ./sync_theta5s.sh submit   # git pull + make the logs dir + sbatch the 81-run array
#   ./sync_theta5s.sh watch    # your queue + tail the newest array task log
#   ./sync_theta5s.sh run      # reduce (harvest theta_max) ON HPC  -> theta5s_summary.csv
#   ./sync_theta5s.sh down     # rsync theta5s_summary.csv -> runs/data/
#
# NOTE: no calibration step here. make_theta5_calibration.py understands the MULTIPLIER knob
# (mult<X>) only — it maps every fa<X> arm to None, so it does NOT apply to this f_A matrix.
# `run` therefore stops at the harvest; analyse the fA arms with data/make_theta5s_analysis.py
# once theta5s_summary.csv is down.
#
# The summary CSV goes to $WS (gpfs), NOT the cluster repo, so `git pull` never conflicts.
# Override the ssh host with HELIX=myalias ./sync_theta5s.sh ...
set -euo pipefail

HOST=${HELIX:-helix}                                        # ssh host / alias
REPO=/home/hd/hd_hd/hd_cq295/trinity                        # trinity repo on Helix (/home, tracked)
WS=/gpfs/bwfor/work/ws/hd_cq295-trinity                     # writable workspace (/gpfs)
RUNS=$REPO/docs/dev/transition/pdv-trigger/runs
SBATCH=$RUNS/run_theta5s.sbatch
OUT=$WS/outputs/theta5s                                     # 81 run dirs (dictionary.jsonl live here)
LOGS=$WS/jobs_theta5s/logs                                  # --output dir (must exist BEFORE sbatch)
SUMMARY=$WS/outputs/theta5s_summary.csv                     # harvest writes here (gpfs -> repo stays clean)
ENV_SETUP=${ENV_SETUP:-"module load devel/miniforge && conda activate trinity"}  # your `condatrinity`

# this repo on the laptop (where `down` drops the committed CSV)
LAPTOP_DATA=/Users/jwt/unsync/Code/Trinity/docs/dev/transition/pdv-trigger/runs/data

case "${1:-}" in
  up)      echo ">> git pull the latest committed code on $HOST (commit + push locally first)"
           ssh "$HOST" "bash -lc 'cd $REPO && git pull --ff-only'" ;;

  submit)  echo ">> on $HOST: git pull -> mkdir logs -> sbatch the 81-run theta5s array"
           ssh "$HOST" "bash -lc 'cd $REPO && git pull && mkdir -p $LOGS && sbatch --array=${ARRAY:-1-81} $SBATCH'" ;;

  watch)   echo ">> queue + newest theta5s task log on $HOST (Ctrl-C to stop)"
           ssh -t "$HOST" "squeue --me -o '%.10i %.20j %.2t %.10M %.6D %R' 2>/dev/null; \
             f=\$(ls -t $LOGS/*.out 2>/dev/null | head -1); \
             if [ -n \"\$f\" ]; then echo \"== \$f ==\"; tail -f \"\$f\"; \
             else echo 'no logs yet in jobs_theta5s/logs — submit first / still queued (squeue).'; fi" ;;

  run)     echo ">> reduce (harvest theta_max) on $HOST -> $SUMMARY  (no calibration: f_A matrix)"
           ssh -t "$HOST" "bash -lc 'cd $REPO && $ENV_SETUP && \
             python $RUNS/harvest_theta_max.py $OUT/* --csv $SUMMARY'" ;;

  down)    echo ">> rsync theta5s_summary.csv <- $HOST -> runs/data/"
           mkdir -p "$LAPTOP_DATA"
           rsync -av "$HOST:$SUMMARY" "$LAPTOP_DATA/theta5s_summary.csv" 2>/dev/null \
             || echo ">> no theta5s_summary.csv yet — run './sync_theta5s.sh run' first"
           echo ">> committed deliverable now in runs/data/ — commit it from the laptop." ;;

  *)       echo "usage: $0 up|submit|watch|run|down   (HELIX=alias  ARRAY=1-81%16  ENV_SETUP=...)"; exit 1 ;;
esac

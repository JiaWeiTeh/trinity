#!/usr/bin/env bash
# Laptop-side driver for the 📏 theta5b referee matrix (43 runs) on bwForCluster Helix.
# Same shape as ./sync_theta5.sh — this workstream is TRACKED (under docs/dev), so code
# travels by `git pull` on the cluster; `up` just pulls the latest committed code.
#
#   ./sync_theta5b.sh up       # git pull the latest committed code on the cluster
#   ./sync_theta5b.sh submit   # git pull + make the logs dir + sbatch the 43-run array
#   ./sync_theta5b.sh watch    # your queue + tail the newest array task log
#   ./sync_theta5b.sh run      # reduce (harvest theta_max) + calibration scorecard, ON HPC
#   ./sync_theta5b.sh down     # rsync theta5b_summary.csv + theta5b_calibration.csv -> runs/data/
#
# Generated CSVs go to $WS/outputs/theta5b_reduce/ (gpfs), NOT the cluster repo, so
# `git pull` never conflicts and theta5's own reduce is never clobbered. Their committed
# copies land in runs/data/ on the LAPTOP after `down`.
# Override the ssh host with HELIX=myalias ./sync_theta5b.sh ...
set -euo pipefail

HOST=${HELIX:-helix}                                        # ssh host / alias
REPO=/home/hd/hd_hd/hd_cq295/trinity                        # trinity repo on Helix (/home, tracked)
WS=/gpfs/bwfor/work/ws/hd_cq295-trinity                     # writable workspace (/gpfs)
RUNS=$REPO/docs/dev/transition/pdv-trigger/runs
SBATCH=$RUNS/run_theta5b.sbatch
OUT=$WS/outputs/theta5b                                     # 43 run dirs (dictionary.jsonl live here)
LOGS=$WS/jobs_theta5b/logs                                  # --output dir (must exist BEFORE sbatch)
RED=$WS/outputs/theta5b_reduce                              # isolated reduce dir (avoids theta5 clash)
SUMMARY=$RED/theta5b_summary.csv                            # harvest writes here
CALIB=$RED/theta5_calibration.csv                           # calibration writes next to its input (fixed name)
ENV_SETUP=${ENV_SETUP:-"module load devel/miniforge && conda activate trinity"}  # your `condatrinity`

# this repo on the laptop (where `down` drops the committed CSVs)
LAPTOP_DATA=/Users/jwt/unsync/Code/Trinity/docs/dev/transition/pdv-trigger/runs/data

case "${1:-}" in
  up)      echo ">> git pull the latest committed code on $HOST (commit + push locally first)"
           ssh "$HOST" "bash -lc 'cd $REPO && git pull --ff-only'" ;;

  submit)  echo ">> on $HOST: git pull -> mkdir logs -> sbatch the 43-run theta5b array"
           ssh "$HOST" "bash -lc 'cd $REPO && git pull && mkdir -p $LOGS && sbatch --array=${ARRAY:-1-43} $SBATCH'" ;;

  watch)   echo ">> queue + newest theta5b task log on $HOST (Ctrl-C to stop)"
           ssh -t "$HOST" "squeue --me -o '%.10i %.20j %.2t %.10M %.6D %R' 2>/dev/null; \
             f=\$(ls -t $LOGS/*.out 2>/dev/null | head -1); \
             if [ -n \"\$f\" ]; then echo \"== \$f ==\"; tail -f \"\$f\"; \
             else echo 'no logs yet in jobs_theta5b/logs — submit first / still queued (squeue).'; fi" ;;

  run)     echo ">> reduce (harvest theta_max) + calibration scorecard on $HOST -> $RED"
           ssh -t "$HOST" "bash -lc 'cd $REPO && $ENV_SETUP && mkdir -p $RED && \
             python $RUNS/harvest_theta_max.py $OUT/* --csv $SUMMARY && \
             python $RUNS/make_theta5_calibration.py --csv $SUMMARY'" ;;

  down)    echo ">> rsync theta5b_summary.csv + theta5b_calibration.csv <- $HOST -> runs/data/"
           mkdir -p "$LAPTOP_DATA"
           rsync -av "$HOST:$SUMMARY" "$LAPTOP_DATA/theta5b_summary.csv" 2>/dev/null \
             || echo ">> no theta5b_summary.csv yet — run './sync_theta5b.sh run' first"
           rsync -av "$HOST:$CALIB" "$LAPTOP_DATA/theta5b_calibration.csv" 2>/dev/null || true
           echo ">> committed deliverables now in runs/data/ — commit them from the laptop." ;;

  *)       echo "usage: $0 up|submit|watch|run|down   (HELIX=alias  ARRAY=1-43%16  ENV_SETUP=...)"; exit 1 ;;
esac

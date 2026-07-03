#!/usr/bin/env bash
# Laptop-side driver for the 📏 theta5k kappa-validation matrix (56 runs) on bwForCluster Helix.
# Same shape as ./sync_theta5.sh / ./sync_theta5b.sh — this workstream is TRACKED (under
# docs/dev), so code travels by `git pull`; `up` just pulls the latest committed code.
#
#   ./sync_theta5k.sh up       # git pull the latest committed code on the cluster
#   ./sync_theta5k.sh submit   # git pull + make the logs dir + sbatch the 56-run array
#   ./sync_theta5k.sh watch    # your queue + tail the newest array task log
#   ./sync_theta5k.sh run      # reduce (harvest theta_max) ON HPC  -> theta5k_summary.csv
#   ./sync_theta5k.sh down     # rsync theta5k_summary.csv -> runs/data/
#
# NOTE: no calibration step. make_theta5_calibration.py understands the MULTIPLIER knob
# (mult<X>) only — it maps every kappa<X> arm to None, so it does NOT apply to this kappa
# matrix. `run` therefore stops at the harvest; analyse the kappa arms with the kappa
# tooling in ../data/ (e.g. make_kappa_*.py) once theta5k_summary.csv is down.
#
# The summary CSV goes to $WS (gpfs), NOT the cluster repo, so `git pull` never conflicts.
# Override the ssh host with HELIX=myalias ./sync_theta5k.sh ...
set -euo pipefail

HOST=${HELIX:-helix}                                        # ssh host / alias
REPO=/home/hd/hd_hd/hd_cq295/trinity                        # trinity repo on Helix (/home, tracked)
WS=/gpfs/bwfor/work/ws/hd_cq295-trinity                     # writable workspace (/gpfs)
RUNS=$REPO/docs/dev/transition/pdv-trigger/runs
SBATCH=$RUNS/run_theta5k.sbatch
OUT=$WS/outputs/theta5k                                     # 56 run dirs (dictionary.jsonl live here)
LOGS=$WS/jobs_theta5k/logs                                  # --output dir (must exist BEFORE sbatch)
SUMMARY=$WS/outputs/theta5k_summary.csv                     # harvest writes here (gpfs -> repo stays clean)
ENV_SETUP=${ENV_SETUP:-"module load devel/miniforge && conda activate trinity"}  # your `condatrinity`

# this repo on the laptop (where `down` drops the committed CSV)
LAPTOP_DATA=/Users/jwt/unsync/Code/Trinity/docs/dev/transition/pdv-trigger/runs/data

case "${1:-}" in
  up)      echo ">> git pull the latest committed code on $HOST (commit + push locally first)"
           ssh "$HOST" "bash -lc 'cd $REPO && git pull --ff-only'" ;;

  submit)  echo ">> on $HOST: git pull -> mkdir logs -> sbatch the 56-run theta5k array"
           ssh "$HOST" "bash -lc 'cd $REPO && git pull && mkdir -p $LOGS && sbatch --array=${ARRAY:-1-56} $SBATCH'" ;;

  watch)   echo ">> queue + newest theta5k task log on $HOST (Ctrl-C to stop)"
           ssh -t "$HOST" "squeue --me -o '%.10i %.20j %.2t %.10M %.6D %R' 2>/dev/null; \
             f=\$(ls -t $LOGS/*.out 2>/dev/null | head -1); \
             if [ -n \"\$f\" ]; then echo \"== \$f ==\"; tail -f \"\$f\"; \
             else echo 'no logs yet in jobs_theta5k/logs — submit first / still queued (squeue).'; fi" ;;

  run)     echo ">> reduce (harvest theta_max) on $HOST -> $SUMMARY  (no calibration: kappa matrix)"
           ssh -t "$HOST" "bash -lc 'cd $REPO && $ENV_SETUP && \
             python $RUNS/harvest_theta_max.py $OUT/* --csv $SUMMARY'" ;;

  down)    echo ">> rsync theta5k_summary.csv <- $HOST -> runs/data/"
           mkdir -p "$LAPTOP_DATA"
           rsync -av "$HOST:$SUMMARY" "$LAPTOP_DATA/theta5k_summary.csv" 2>/dev/null \
             || echo ">> no theta5k_summary.csv yet — run './sync_theta5k.sh run' first"
           echo ">> committed deliverable now in runs/data/ — commit it from the laptop." ;;

  *)       echo "usage: $0 up|submit|watch|run|down   (HELIX=alias  ARRAY=1-56%16  ENV_SETUP=...)"; exit 1 ;;
esac

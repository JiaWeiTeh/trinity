#!/usr/bin/env bash
# Laptop-side driver for the 📏 theta5n matrix (15 runs: 9th std config normal_n1e3,
# BOTH knobs) on bwForCluster Helix. Same shape as ./sync_theta5.sh / ._b / ._k —
# TRACKED workstream, so code travels by `git pull`; `up` just pulls it on Helix.
#
#   ./sync_theta5n.sh up       # git pull the latest committed code on the cluster
#   ./sync_theta5n.sh submit   # git pull + make the logs dir + sbatch the 15-run array
#   ./sync_theta5n.sh watch    # your queue + tail the newest array task log
#   ./sync_theta5n.sh run      # reduce (harvest theta_max) ON HPC  -> theta5n_summary.csv
#   ./sync_theta5n.sh down     # rsync theta5n_summary.csv -> runs/data/
#
# NOTE: no calibration step. theta5n mixes mult<X> and kappa<X> arms for a SINGLE
# config; make_theta5_calibration.py drops kappa arms and fits the multiplier law
# ACROSS configs, so it isn't meaningful here. `run` stops at the harvest; compare
# the two knobs straight from theta5n_summary.csv.
#
# The summary CSV goes to $WS (gpfs), NOT the cluster repo, so `git pull` never conflicts.
# Override the ssh host with HELIX=myalias ./sync_theta5n.sh ...
set -euo pipefail

HOST=${HELIX:-helix}                                        # ssh host / alias
REPO=/home/hd/hd_hd/hd_cq295/trinity                        # trinity repo on Helix (/home, tracked)
WS=/gpfs/bwfor/work/ws/hd_cq295-trinity                     # writable workspace (/gpfs)
RUNS=$REPO/docs/dev/transition/pdv-trigger/runs
SBATCH=$RUNS/run_theta5n.sbatch
OUT=$WS/outputs/theta5n                                     # 15 run dirs (dictionary.jsonl live here)
LOGS=$WS/jobs_theta5n/logs                                  # --output dir (must exist BEFORE sbatch)
SUMMARY=$WS/outputs/theta5n_summary.csv                     # harvest writes here (gpfs -> repo stays clean)
ENV_SETUP=${ENV_SETUP:-"module load devel/miniforge && conda activate trinity"}  # your `condatrinity`

# this repo on the laptop (where `down` drops the committed CSV)
LAPTOP_DATA=/Users/jwt/unsync/Code/Trinity/docs/dev/transition/pdv-trigger/runs/data

case "${1:-}" in
  up)      echo ">> git pull the latest committed code on $HOST (commit + push locally first)"
           ssh "$HOST" "bash -lc 'cd $REPO && git pull --ff-only'" ;;

  submit)  echo ">> on $HOST: git pull -> mkdir logs -> sbatch the 15-run theta5n array"
           ssh "$HOST" "bash -lc 'cd $REPO && git pull && mkdir -p $LOGS && sbatch --array=${ARRAY:-1-15} $SBATCH'" ;;

  watch)   echo ">> queue + newest theta5n task log on $HOST (Ctrl-C to stop)"
           ssh -t "$HOST" "squeue --me -o '%.10i %.20j %.2t %.10M %.6D %R' 2>/dev/null; \
             f=\$(ls -t $LOGS/*.out 2>/dev/null | head -1); \
             if [ -n \"\$f\" ]; then echo \"== \$f ==\"; tail -f \"\$f\"; \
             else echo 'no logs yet in jobs_theta5n/logs — submit first / still queued (squeue).'; fi" ;;

  run)     echo ">> reduce (harvest theta_max) on $HOST -> $SUMMARY  (no calibration: single config, mixed knobs)"
           ssh -t "$HOST" "bash -lc 'cd $REPO && $ENV_SETUP && \
             python $RUNS/harvest_theta_max.py $OUT/* --csv $SUMMARY'" ;;

  down)    echo ">> rsync theta5n_summary.csv <- $HOST -> runs/data/"
           mkdir -p "$LAPTOP_DATA"
           rsync -av "$HOST:$SUMMARY" "$LAPTOP_DATA/theta5n_summary.csv" 2>/dev/null \
             || echo ">> no theta5n_summary.csv yet — run './sync_theta5n.sh run' first"
           echo ">> committed deliverable now in runs/data/ — commit it from the laptop." ;;

  *)       echo "usage: $0 up|submit|watch|run|down   (HELIX=alias  ARRAY=1-15%16  ENV_SETUP=...)"; exit 1 ;;
esac

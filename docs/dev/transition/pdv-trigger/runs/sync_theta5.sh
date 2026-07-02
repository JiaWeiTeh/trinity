#!/usr/bin/env bash
# Laptop-side driver for the 📏 theta5 standard-protocol matrix on bwForCluster Helix.
# Sibling of ./sync.sh (the f_kappa sweep) and paper/II-survey/sync.sh. This workstream
# is TRACKED (under docs/dev), so code travels by `git pull` on the cluster — there is no
# rsync `up`: commit + push locally, then `up` pulls it on Helix.
#
#   ./sync_theta5.sh up       # git pull the latest committed code on the cluster
#   ./sync_theta5.sh submit   # git pull + make the logs dir + sbatch the 32-run array
#   ./sync_theta5.sh watch    # your queue + tail the newest array task log
#   ./sync_theta5.sh run      # reduce (harvest theta_max) + calibration scorecard, ON HPC
#   ./sync_theta5.sh down     # rsync theta5_summary.csv + theta5_calibration.csv -> runs/data/
#
# All generated CSVs are written under $WS (gpfs), NOT the cluster repo, so `git pull`
# never conflicts. Their committed copies land in runs/data/ on the LAPTOP after `down`.
# Override the ssh host with HELIX=myalias ./sync_theta5.sh ...
set -euo pipefail

HOST=${HELIX:-helix}                                        # ssh host / alias
REPO=/home/hd/hd_hd/hd_cq295/trinity                        # trinity repo on Helix (/home, tracked)
WS=/gpfs/bwfor/work/ws/hd_cq295-trinity                     # writable workspace (/gpfs)
RUNS=$REPO/docs/dev/transition/pdv-trigger/runs
SBATCH=$RUNS/run_theta5.sbatch
OUT=$WS/outputs/theta5                                      # 32 run dirs (dictionary.jsonl live here)
LOGS=$WS/jobs_theta5/logs                                   # --output dir (must exist BEFORE sbatch)
SUMMARY=$WS/outputs/theta5_summary.csv                      # harvest writes here (gpfs -> repo stays clean)
CALIB=$WS/outputs/theta5_calibration.csv                    # calibration writes next to the summary
ENV_SETUP=${ENV_SETUP:-"module load devel/miniforge && conda activate trinity"}  # your `condatrinity`

# this repo on the laptop (where `down` drops the committed CSVs)
LAPTOP_DATA=/Users/jwt/unsync/Code/Trinity/docs/dev/transition/pdv-trigger/runs/data

case "${1:-}" in
  up)      echo ">> git pull the latest committed code on $HOST (commit + push locally first)"
           ssh "$HOST" "bash -lc 'cd $REPO && git pull --ff-only'" ;;

  submit)  echo ">> on $HOST: git pull -> mkdir logs -> sbatch the 32-run theta5 array"
           ssh "$HOST" "bash -lc 'cd $REPO && git pull && mkdir -p $LOGS && sbatch --array=${ARRAY:-1-32} $SBATCH'" ;;

  watch)   echo ">> queue + newest theta5 task log on $HOST (Ctrl-C to stop)"
           ssh -t "$HOST" "squeue --me -o '%.10i %.20j %.2t %.10M %.6D %R' 2>/dev/null; \
             f=\$(ls -t $LOGS/*.out 2>/dev/null | head -1); \
             if [ -n \"\$f\" ]; then echo \"== \$f ==\"; tail -f \"\$f\"; \
             else echo 'no logs yet in jobs_theta5/logs — submit first / still queued (squeue).'; fi" ;;

  run)     echo ">> reduce (harvest theta_max) + calibration scorecard on $HOST -> $WS/outputs/"
           ssh -t "$HOST" "bash -lc 'cd $REPO && $ENV_SETUP && \
             python $RUNS/harvest_theta_max.py $OUT/* --csv $SUMMARY && \
             python $RUNS/make_theta5_calibration.py --csv $SUMMARY'" ;;

  down)    echo ">> rsync theta5_summary.csv + theta5_calibration.csv <- $HOST -> runs/data/"
           mkdir -p "$LAPTOP_DATA"
           rsync -av "$HOST:$SUMMARY" "$LAPTOP_DATA/" 2>/dev/null \
             || echo ">> no theta5_summary.csv yet — run './sync_theta5.sh run' first"
           rsync -av "$HOST:$CALIB" "$LAPTOP_DATA/" 2>/dev/null || true
           echo ">> committed deliverables now in runs/data/ — commit them from the laptop." ;;

  *)       echo "usage: $0 up|submit|watch|run|down   (HELIX=alias  ARRAY=1-32%16  ENV_SETUP=...)"; exit 1 ;;
esac

#!/usr/bin/env bash
# Laptop-side driver for the bench campaigns on bwForCluster Helix — one script, two campaigns:
#   bench5  = the 60-arm Phase-5 L21b matrix, HPC CONFIRMATION of the in-container §15h result
#             (summary lands as bench5_summary_hpc.csv — the in-container bench5_summary.csv is
#             kept; diff them with data/compare_bench5_hpc.py)
#   bench6  = the 60-arm Phase-6 DECISION matrix (f_A dose extension + f_mix head-to-head)
# Same shape as ./sync_theta5s.sh (tracked workstream: code travels by `git pull`).
#
#   ./sync_bench.sh bench5|bench6 up       # git pull the latest committed code on the cluster
#   ./sync_bench.sh bench5|bench6 submit   # git pull + mkdir logs + sbatch the 60-run array
#   ./sync_bench.sh bench5|bench6 watch    # your queue + tail the newest array task log
#   ./sync_bench.sh bench5|bench6 run      # harvest ON HPC -> summary csv + traj dir (gpfs)
#   ./sync_bench.sh bench5|bench6 down     # rsync summary + traj back into runs/data/
#
# The harvest (`run`) uses harvest_bench5.py for BOTH campaigns (it is campaign-agnostic):
# fire-map summary + per-arm theta(t) trajectory CSVs — the trajectories are what the Theta_cum
# L21b metric needs (make_bench5_analysis.py / make_bench6_analysis.py read them from runs/data/).
# Override the ssh host with HELIX=myalias ./sync_bench.sh ...
set -euo pipefail

CAMPAIGN=${1:-}
CMD=${2:-}
case "$CAMPAIGN" in
  bench5) SUMMARY_NAME=bench5_summary_hpc.csv; TRAJ_NAME=bench5_traj_hpc ;;
  bench6) SUMMARY_NAME=bench6_summary.csv;     TRAJ_NAME=bench6_traj ;;
  *) echo "usage: $0 bench5|bench6 up|submit|watch|run|down   (HELIX=alias  ARRAY=1-60%16)"; exit 1 ;;
esac

HOST=${HELIX:-helix}                                        # ssh host / alias
REPO=/home/hd/hd_hd/hd_cq295/trinity                        # trinity repo on Helix (/home, tracked)
WS=/gpfs/bwfor/work/ws/hd_cq295-trinity                     # writable workspace (/gpfs)
RUNS=$REPO/docs/dev/transition/pdv-trigger/runs
SBATCH=$RUNS/run_$CAMPAIGN.sbatch
OUT=$WS/outputs/$CAMPAIGN                                   # 60 run dirs (dictionary.jsonl live here)
LOGS=$WS/jobs_$CAMPAIGN/logs                                # --output dir (must exist BEFORE sbatch)
SUMMARY=$WS/outputs/$SUMMARY_NAME                           # harvest writes here (gpfs, repo stays clean)
TRAJ=$WS/outputs/$TRAJ_NAME
ENV_SETUP=${ENV_SETUP:-"module load devel/miniforge && conda activate trinity"}

# this repo on the laptop (where `down` drops the committed CSVs)
LAPTOP_DATA=/Users/jwt/unsync/Code/Trinity/docs/dev/transition/pdv-trigger/runs/data

case "$CMD" in
  up)      echo ">> git pull the latest committed code on $HOST (commit + push locally first)"
           ssh "$HOST" "bash -lc 'cd $REPO && git pull --ff-only'" ;;

  submit)  echo ">> on $HOST: git pull -> mkdir logs -> sbatch the 60-run $CAMPAIGN array"
           ssh "$HOST" "bash -lc 'cd $REPO && git pull && mkdir -p $LOGS && sbatch --array=${ARRAY:-1-60} $SBATCH'" ;;

  watch)   echo ">> queue + newest $CAMPAIGN task log on $HOST (Ctrl-C to stop)"
           ssh -t "$HOST" "squeue --me -o '%.10i %.20j %.2t %.10M %.6D %R' 2>/dev/null; \
             f=\$(ls -t $LOGS/*.out 2>/dev/null | head -1); \
             if [ -n \"\$f\" ]; then echo \"== \$f ==\"; tail -f \"\$f\"; \
             else echo 'no logs yet in jobs_$CAMPAIGN/logs — submit first / still queued (squeue).'; fi" ;;

  run)     echo ">> harvest on $HOST -> $SUMMARY + $TRAJ/"
           ssh -t "$HOST" "bash -lc 'cd $REPO && $ENV_SETUP && \
             python $RUNS/harvest_bench5.py $OUT/* --csv $SUMMARY --traj-dir $TRAJ'" ;;

  down)    echo ">> rsync $SUMMARY_NAME + $TRAJ_NAME/ <- $HOST -> runs/data/"
           mkdir -p "$LAPTOP_DATA/$TRAJ_NAME"
           rsync -av "$HOST:$SUMMARY" "$LAPTOP_DATA/$SUMMARY_NAME" 2>/dev/null \
             || echo ">> no $SUMMARY_NAME yet — run './sync_bench.sh $CAMPAIGN run' first"
           rsync -av "$HOST:$TRAJ/" "$LAPTOP_DATA/$TRAJ_NAME/" 2>/dev/null || true
           echo ">> committed deliverables now in runs/data/ — commit them from the laptop." ;;

  *)       echo "usage: $0 bench5|bench6 up|submit|watch|run|down   (HELIX=alias  ARRAY=1-60%16)"; exit 1 ;;
esac

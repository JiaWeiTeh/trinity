#!/usr/bin/env bash
# Laptop-side driver for the bench campaigns on bwForCluster Helix — one script, three campaigns:
#   bench5  = the 60-arm Phase-5 L21b matrix, HPC CONFIRMATION of the in-container §15h result
#             (summary lands as bench5_summary_hpc.csv — the in-container bench5_summary.csv is
#             kept; diff them with data/compare_bench5_hpc.py)
#   bench6  = the 60-arm Phase-6 DECISION matrix (f_A dose extension + f_mix head-to-head)
#   bench7  = the f_kappa re-open campaign, KAPPA_REOPEN_PLAN.md K1-K4 (~102 arms). K1/K1b/K2/K3/K4
#             all live in ONE params dir as one array — a K-phase is just a filename prefix, so
#             there is one submit, one reduce and one download for the whole campaign.
# Same shape as ./sync_theta5s.sh (tracked workstream: code travels by `git pull`).
#
#   ./sync_bench.sh <campaign> up       # git pull the latest committed code on the cluster
#   ./sync_bench.sh <campaign> submit   # git pull + mkdir logs + sbatch the array (AUTO-SIZED)
#   ./sync_bench.sh <campaign> watch    # your queue + tail the newest array task log
#   ./sync_bench.sh <campaign> reduce   # multi-GB jsonl -> small CSVs, ON HPC   (alias: `run`)
#   ./sync_bench.sh <campaign> down     # rsync ONLY the reduced CSVs into runs/data/
#
# WHY reduce-then-down. The raw dictionary.jsonl files are multi-GB and stay on gpfs; only the
# reduced CSVs ever travel. `reduce` runs harvest_bench5.py ON the cluster (it is campaign-agnostic):
# fire-map summary + per-arm theta(t) trajectory CSVs (<=4000 rows, log-t downsampled, endpoints
# kept) — the trajectories are what the Theta_cum L21b metric needs (make_bench*_analysis.py read
# them from runs/data/).
#
# ⚠️ THE REDUCE IS ONE-SHOT. gpfs workspaces get cleaned and the raw arms do NOT come back — the
# theta5s lesson (harvest_bench5.py docstring: its raw arms were lost to a /tmp wipe and dMdt had to
# be salvaged in a scramble). Anything the reduce does not capture is gone. So a campaign whose
# analysis needs more than the six default trajectory columns MUST declare them in EXTRA below,
# BEFORE the first reduce. bench7 declares Pb + bubble_dMdt (KAPPA_REOPEN_PLAN P2 and the K0.Q1b
# back-reaction both read them) and the L2/L3 split.
#
# `reduce` also writes <campaign>_hashes.csv: sha256 of each reduced trajectory CSV. That is what
# makes K3's determinism claim checkable (two runs of one param must hash identically) without ever
# shipping a raw dictionary down. Hash the REDUCED csv, not the jsonl — the physics columns only.
# Override the ssh host with HELIX=myalias ./sync_bench.sh ...
set -euo pipefail

CAMPAIGN=${1:-}
CMD=${2:-}
EXTRA=""
case "$CAMPAIGN" in
  bench5) SUMMARY_NAME=bench5_summary_hpc.csv; TRAJ_NAME=bench5_traj_hpc ;;
  bench6) SUMMARY_NAME=bench6_summary.csv;     TRAJ_NAME=bench6_traj ;;
  bench7) SUMMARY_NAME=bench7_summary.csv;     TRAJ_NAME=bench7_traj
          EXTRA="--extra-cols Pb,bubble_dMdt,bubble_L2Conduction,bubble_L3Intermediate" ;;
  *) echo "usage: $0 bench5|bench6|bench7 up|submit|watch|reduce|down   (HELIX=alias  ARRAY=1-60%16)"; exit 1 ;;
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
HASHES_NAME=${CAMPAIGN}_hashes.csv                          # sha256 per reduced traj csv (K3)
HASHES=$WS/outputs/$HASHES_NAME
PARAMS=$RUNS/params/$CAMPAIGN
ENV_SETUP=${ENV_SETUP:-"module load devel/miniforge && conda activate trinity"}

# this repo on the laptop (where `down` drops the committed CSVs)
LAPTOP_DATA=/Users/jwt/unsync/Code/Trinity/docs/dev/transition/pdv-trigger/runs/data

case "$CMD" in
  up)      echo ">> git pull the latest committed code on $HOST (commit + push locally first)"
           ssh "$HOST" "bash -lc 'cd $REPO && git pull --ff-only'" ;;

  submit)  echo ">> on $HOST: git pull -> mkdir logs -> sbatch the $CAMPAIGN array"
           # Array is AUTO-SIZED from the committed params unless ARRAY is set, so a grid change
           # needs no edit here (bench7's size is not fixed until the §6.0 ruling lands).
           ssh "$HOST" "bash -lc 'cd $REPO && git pull && mkdir -p $LOGS && \
             n=\$(ls $PARAMS/*.param 2>/dev/null | wc -l); \
             if [ \"\$n\" -eq 0 ]; then echo \"ERROR: no params in $PARAMS — generate + commit them first\" >&2; exit 1; fi; \
             a=\"${ARRAY:-}\"; [ -n \"\$a\" ] || a=\"1-\$n\"; \
             echo \">> \$n params, submitting --array=\$a\"; sbatch --array=\$a $SBATCH'" ;;

  watch)   echo ">> queue + newest $CAMPAIGN task log on $HOST (Ctrl-C to stop)"
           ssh -t "$HOST" "squeue --me -o '%.10i %.20j %.2t %.10M %.6D %R' 2>/dev/null; \
             f=\$(ls -t $LOGS/*.out 2>/dev/null | head -1); \
             if [ -n \"\$f\" ]; then echo \"== \$f ==\"; tail -f \"\$f\"; \
             else echo 'no logs yet in jobs_$CAMPAIGN/logs — submit first / still queued (squeue).'; fi" ;;

  reduce|run)                       # `run` kept as an alias — sync.sh calls this step `reduce`
           echo ">> reduce the multi-GB jsonl -> $SUMMARY + $TRAJ/ + $HASHES  (ON $HOST)"
           echo ">> extra trajectory columns: ${EXTRA:-<none — the six defaults>}"
           ssh -t "$HOST" "bash -lc 'cd $REPO && $ENV_SETUP && \
             python $RUNS/harvest_bench5.py $OUT/* --csv $SUMMARY --traj-dir $TRAJ $EXTRA && \
             { echo run_name,sha256,bytes; \
               for f in $TRAJ/*.csv; do \
                 echo \"\$(basename \"\$f\" .csv),\$(sha256sum \"\$f\" | cut -d\" \" -f1),\$(wc -c <\"\$f\")\"; \
               done; } > $HASHES && echo \">> hashed \$(( \$(wc -l <$HASHES) - 1 )) reduced trajectories\"'" ;;

  down)    echo ">> rsync ONLY the reduced CSVs <- $HOST -> runs/data/  (raw jsonl stays on gpfs)"
           mkdir -p "$LAPTOP_DATA/$TRAJ_NAME"
           rsync -av "$HOST:$SUMMARY" "$LAPTOP_DATA/$SUMMARY_NAME" 2>/dev/null \
             || echo ">> no $SUMMARY_NAME yet — run './sync_bench.sh $CAMPAIGN reduce' first"
           rsync -av "$HOST:$HASHES" "$LAPTOP_DATA/$HASHES_NAME" 2>/dev/null || true
           rsync -av "$HOST:$TRAJ/" "$LAPTOP_DATA/$TRAJ_NAME/" 2>/dev/null || true
           echo ">> committed deliverables now in runs/data/ — commit them from the laptop." ;;

  *)       echo "usage: $0 bench5|bench6|bench7 up|submit|watch|reduce|down   (HELIX=alias  ARRAY=1-60%16)"; exit 1 ;;
esac

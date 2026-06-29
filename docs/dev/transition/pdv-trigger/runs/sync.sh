#!/usr/bin/env bash
# Laptop-side driver for the f_kappa(n_H) calibration sweep on bwForCluster Helix.
# Mirrors paper/shellSSC6/sync.sh, but this workstream is TRACKED (under docs/dev),
# so the code travels by `git pull` on the cluster — no rsync `up` step needed.
#
#   ./sync.sh submit    # git pull + emit the 819-combo bundle (if absent) + sbatch the array
#   ./sync.sh watch     # show your queue + tail the newest array task log
#   ./sync.sh collect   # run.py --collect-report -> sweep_report.{txt,json} in the output dir
#   ./sync.sh reduce    # reduce_fkappa_sweep.py: the multi-GB jsonl -> one small summary.csv (ON HPC)
#   ./sync.sh down      # rsync summary.csv (+ sweep_report) back to data/; then plot on the laptop
# Plot (laptop, no cluster):  python docs/dev/transition/pdv-trigger/data/make_fkappa_nH_sweep.py
#   (reads data/summary.csv; writes fkappa_nH_sweep.csv + .png — the committed deliverables)
#
# Override the ssh host with HELIX=myalias ./sync.sh ...
# Re-run from scratch: ssh in and `rm -rf $WS/jobs_fkappa` first (emit refuses to clobber a live bundle).
# Cap concurrency differently: ARRAY=1-819%32 ./sync.sh submit   (passed to sbatch --array).
set -euo pipefail

HOST=${HELIX:-helix}                                          # ssh host / alias
REPO=/home/hd/hd_hd/hd_cq295/trinity                          # trinity repo on Helix (/home, tracked)
WS=/gpfs/bwfor/work/ws/hd_cq295-trinity                       # writable workspace (/gpfs)
JOBS=$WS/jobs_fkappa                                          # emit bundle: params/, runs.tsv, logs/
SWEEP_OUT=$WS/outputs/sweep_fkappa_nH                         # run outputs (relative path2output, resolved from $WS)
PARAM=$REPO/docs/dev/transition/pdv-trigger/runs/params/sweep_fkappa_nH.param
SBATCH=$REPO/docs/dev/transition/pdv-trigger/runs/run_fkappa.sbatch
REDUCE=$REPO/docs/dev/transition/pdv-trigger/data/reduce_fkappa_sweep.py
ENV_SETUP=${ENV_SETUP:-"module load devel/miniforge && conda activate trinity"}  # your `condatrinity`

# this repo on the laptop (where `down` drops the committed artifacts)
LAPTOP_PDV=/Users/jwt/unsync/Code/Trinity/docs/dev/transition/pdv-trigger

# NOTE on quoting: remote commands run via `bash -lc '...'` (login shell so module/conda exist).
# Keep them free of single quotes and of locally-deferred $(...); the fixed $REPO/$WS/... paths are
# meant to expand on the laptop. Emit only if the bundle is absent (run.py won't clobber a live array).
case "${1:-}" in
  submit)  echo ">> on $HOST: git pull -> emit (if $JOBS absent) -> sbatch the array"
           ssh "$HOST" "bash -lc 'cd $REPO && git pull && $ENV_SETUP && \
             { [ -f $JOBS/manifest.json ] || { cd $WS && python $REPO/run.py $PARAM --emit-jobs $JOBS ; } ; } && \
             sbatch ${ARRAY:+--array=$ARRAY} $SBATCH'" ;;

  watch)   echo ">> tailing newest array task log on $HOST (Ctrl-C to stop)"
           ssh -t "$HOST" "squeue --me -o '%.10i %.20j %.2t %.10M %.6D %R' 2>/dev/null; \
             f=\$(ls -t $JOBS/logs/*.out 2>/dev/null | head -1); \
             if [ -n \"\$f\" ]; then echo \"== \$f ==\"; tail -f \"\$f\"; \
             else echo 'no logs yet in jobs_fkappa/logs — submit first / still queued (squeue).'; fi" ;;

  collect) echo ">> collect per-task exit codes -> sweep_report.{txt,json} on $HOST"
           ssh "$HOST" "bash -lc 'cd $REPO && $ENV_SETUP && python $REPO/run.py --collect-report $JOBS'" ;;

  reduce)  echo ">> reduce the sweep jsonl -> summary.csv on $HOST  (workers=${WORKERS:-8}, stdlib-only)"
           ssh "$HOST" "bash -lc 'cd $REPO && $ENV_SETUP && python $REDUCE $SWEEP_OUT --workers ${WORKERS:-8}'" ;;

  down)    echo ">> rsync summary.csv (+ sweep_report) <- $HOST  -> data/  (then plot locally)"
           rsync -av "$HOST:$SWEEP_OUT/summary.csv" "$LAPTOP_PDV/data/" 2>/dev/null \
             || echo ">> no summary.csv yet — run './sync.sh reduce' first"
           rsync -av "$HOST:$SWEEP_OUT/sweep_report.txt" "$LAPTOP_PDV/data/" 2>/dev/null || true
           echo ">> now plot:  python $LAPTOP_PDV/data/make_fkappa_nH_sweep.py" ;;

  *)       echo "usage: $0 submit|watch|collect|reduce|down   (HELIX=alias  ARRAY=1-819%32  WORKERS=8  ENV_SETUP=...)"; exit 1 ;;
esac

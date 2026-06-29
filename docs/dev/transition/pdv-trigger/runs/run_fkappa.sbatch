#!/bin/bash -l
# f_kappa(n_H) calibration sweep — 819-combo SLURM array on bwForCluster Helix.
# Pre-patched for Helix (account/partition/--export=NONE/conda) so it runs straight
# from `git pull` — no per-cluster `patch_sbatch` step. Mirrors paper/shellSSC6/run.sbatch
# (sim batch) but as an --emit-jobs array because 819 runs outgrow one node.
#
#   driver:  ./docs/dev/transition/pdv-trigger/runs/sync.sh submit   (emits + sbatch)
#   by hand on Helix:
#     cd "$WS" && python "$REPO/run.py" \
#         "$REPO/docs/dev/transition/pdv-trigger/runs/params/sweep_fkappa_nH.param" \
#         --emit-jobs "$WS/jobs_fkappa"            # writes params/, runs.tsv, logs/
#     sbatch "$REPO/docs/dev/transition/pdv-trigger/runs/run_fkappa.sbatch"
#   watch:   ./sync.sh watch   (or: tail -f $WS/jobs_fkappa/logs/<jobid>_*.out)
#
#SBATCH --job-name=trinity-fkappa
#SBATCH --partition=cpu-single             # Helix single-node partition (matches your other jobs)
#SBATCH --account=bw22J006
#SBATCH --export=NONE                       # clean env (login shell below re-loads modules)
#SBATCH --array=1-819                       # 819 combos, one task per run (fair-share throttles; no %cap by default).
                                            # Cap concurrency only if you want to: sbatch --array=1-819%32 ... (or ARRAY=1-819%32 via sync.sh)
#SBATCH --cpus-per-task=1                   # one sim per task (parallelism = many independent tasks, not threads)
#SBATCH --mem=2G                            # one bubble integration is light on memory (matches your Paper II sweep)
#SBATCH --time=02:00:00                     # each run is ~1h max; 2h ceiling (matches your Paper II sweep)
#SBATCH --output=/gpfs/bwfor/work/ws/hd_cq295-trinity/jobs_fkappa/logs/%A_%a.out

# Body mirrors your Paper II sweep sbatch (the emit-jobs _SBATCH_TEMPLATE + patch_sbatch): NO `set -e` —
# capture the sim's own exit code into the .exit_code/.duration sentinels that run.py --collect-report reads.

module load devel/miniforge                                   # your `condatrinity`
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate trinity

REPO=/home/hd/hd_hd/hd_cq295/trinity                          # tracked code (reached via git pull)
WS=/gpfs/bwfor/work/ws/hd_cq295-trinity                       # writable workspace (/home is read-only at runtime)
JOBS="$WS/jobs_fkappa"                                        # emit bundle: params/, runs.tsv, logs/ (on /gpfs)

# single-thread the math libs: parallelism comes from running many array tasks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

# Optional OFFSET (sbatch --export=ALL,OFFSET=N) shifts the runs.tsv line so a grid larger than a
# MaxSubmitJobs cap can be submitted in chunks. Defaults to 0.
N=$(( SLURM_ARRAY_TASK_ID + ${OFFSET:-0} ))
LINE=$(sed -n "${N}p" "$JOBS/runs.tsv")
if [ -z "$LINE" ]; then
    echo "ERROR: no line $N in $JOBS/runs.tsv (TASK=$SLURM_ARRAY_TASK_ID OFFSET=${OFFSET:-0}) — emit first? (sync.sh submit)" >&2
    exit 1
fi
PARAM=$(printf '%s' "$LINE" | cut -f1)
OUTDIR=$(printf '%s' "$LINE" | cut -f2)
mkdir -p "$OUTDIR"
SECONDS=0
python "$REPO/run.py" "$PARAM"
code=$?
echo "$code" > "$OUTDIR/.exit_code"     # run.py --collect-report aggregates these
echo "$SECONDS" > "$OUTDIR/.duration"
exit $code

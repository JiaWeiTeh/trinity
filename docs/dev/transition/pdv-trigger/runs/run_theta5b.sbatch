#!/bin/bash -l
# 📏 Standard-protocol run matrix — fine f-bracket + long diffuse arm = 43 runs (referee matrix) on Helix.
# Same cluster conventions as run_fkappa.sbatch (account/partition/--export=NONE/conda).
#
#   emit params (once, committed):  python docs/dev/transition/pdv-trigger/runs/make_theta5b_params.py
#   submit:  sbatch docs/dev/transition/pdv-trigger/runs/run_theta5b.sbatch
#   harvest: python docs/dev/transition/pdv-trigger/runs/harvest_theta_max.py "$WS"/outputs/theta5b/* \
#              --csv docs/dev/transition/pdv-trigger/runs/data/theta5b_summary.csv
#
#SBATCH --job-name=trinity-theta5b
#SBATCH --partition=cpu-single
#SBATCH --account=bw22J006
#SBATCH --export=NONE
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=1-43
#SBATCH --time=3:00:00                     # stop_t=8 diffuse arms are the long pole (theta5 5-Myr arms fit in 1:30)
#SBATCH --output=/gpfs/bwfor/work/ws/hd_cq295-trinity/jobs_theta5b/logs/%A_%a.out

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate trinity

REPO=/home/hd/hd_hd/hd_cq295/trinity
WS=/gpfs/bwfor/work/ws/hd_cq295-trinity
PARAMS="$REPO/docs/dev/transition/pdv-trigger/runs/params/theta5b"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

mkdir -p "$WS/jobs_theta5b/logs"
PARAM=$(ls "$PARAMS"/*.param | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "$PARAM" ]; then
    echo "ERROR: no param #${SLURM_ARRAY_TASK_ID} in $PARAMS — run make_theta5b_params.py first?" >&2
    exit 1
fi
# path2output in each .param is relative -> run from the writable workspace
cd "$WS"
SECONDS=0
python "$REPO/run.py" "$PARAM"
code=$?
OUTDIR="$WS/outputs/theta5b/$(basename "$PARAM" .param)"
mkdir -p "$OUTDIR"
echo "$code" > "$OUTDIR/.exit_code"
echo "$SECONDS" > "$OUTDIR/.duration"
exit $code

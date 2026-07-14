#!/bin/bash -l
# 📏 bench6 — the Phase-6 DECISION matrix on Helix: f_A dose extension (bench1/2 × fa 24–128,
# bench3 × fa 24/32) + the f_mix head-to-head (all 5 benches × fm 2/3/4/8), prod+diag arms.
# 60 params, committed by make_bench6_params.py. Purpose + decision rubric: see that generator's
# docstring and PLAN.md (Phase-6 inputs). Same cluster conventions as run_theta5s.sbatch.
#
#   emit params (once, committed):  python docs/dev/transition/pdv-trigger/runs/make_bench6_params.py
#   submit:  sbatch docs/dev/transition/pdv-trigger/runs/run_bench6.sbatch
#            (or ./sync_bench.sh bench6 submit from the laptop)
#   harvest: python docs/dev/transition/pdv-trigger/runs/harvest_bench5.py "$WS"/outputs/bench6/* \
#              --csv "$WS"/outputs/bench6_summary.csv --traj-dir "$WS"/outputs/bench6_traj
#
# --time=1:30:00 (maintainer ruling 2026-07-13; evidence data/bench5_durations.csv — longest
# compliant bench5 arm 64 min under 3-worker contention). The only UNMEASURED-cost arms are the
# diffuse fa64/fa128 diag ones (blowout comes later as the boost strengthens; Eb back-reaction
# shrinks dt). If one wall-kills (exit 124 with t still advancing in dictionary.jsonl), resubmit
# just those ids with a CLI override — it beats the in-file directive:
#     sbatch --time=3:00:00 --array=<id1>,<id2> docs/dev/transition/pdv-trigger/runs/run_bench6.sbatch
# and re-harvest; quote no theta from a wall-killed arm. A FROZEN arm (t stalled at a fixed step,
# cf. bench5_fa16_diag §15h) is a stiffness signature, not a walltime problem — record, don't chase.
#
#SBATCH --job-name=trinity-bench6
#SBATCH --partition=cpu-single
#SBATCH --account=bw22J006
#SBATCH --export=NONE
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=1-60
#SBATCH --time=1:30:00
#SBATCH --output=/gpfs/bwfor/work/ws/hd_cq295-trinity/jobs_bench6/logs/%A_%a.out

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate trinity

REPO=/home/hd/hd_hd/hd_cq295/trinity
WS=/gpfs/bwfor/work/ws/hd_cq295-trinity
PARAMS="$REPO/docs/dev/transition/pdv-trigger/runs/params/bench6"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

mkdir -p "$WS/jobs_bench6/logs"
PARAM=$(ls "$PARAMS"/*.param | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "$PARAM" ]; then
    echo "ERROR: no param #${SLURM_ARRAY_TASK_ID} in $PARAMS — run make_bench6_params.py first?" >&2
    exit 1
fi
# path2output in each .param is relative -> run from the writable workspace
cd "$WS"
SECONDS=0
python "$REPO/run.py" "$PARAM"
code=$?
OUTDIR="$WS/outputs/bench6/$(basename "$PARAM" .param)"
mkdir -p "$OUTDIR"
echo "$code" > "$OUTDIR/.exit_code"
echo "$SECONDS" > "$OUTDIR/.duration"
exit $code

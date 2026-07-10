#!/bin/bash -l
# 📏 Standard-protocol run matrix — the f_A (cooling_boost_fA) all-9-config validation = 81 runs on Helix.
# Same cluster conventions as run_theta5k.sbatch (account/partition/--export=NONE/conda).
# SINGLE-KNOB by construction: every arm has cooling_boost_mode=none and cooling_boost_kappa=1
# (schema defaults); only cooling_boost_fA varies (1 = __none baseline, else __fa<v>).
#
#   emit params (once, committed):  python docs/dev/transition/pdv-trigger/runs/make_theta5s_params.py
#   submit:  sbatch docs/dev/transition/pdv-trigger/runs/run_theta5s.sbatch
#   harvest: python docs/dev/transition/pdv-trigger/runs/harvest_theta_max.py "$WS"/outputs/theta5s/* \
#              --csv docs/dev/transition/pdv-trigger/runs/data/theta5s_summary.csv
#
# NB requires the branch with cooling_boost_fA wired (SOURCE_TERM_DESIGN Phase 2) AND the
# no_physical_root=>momentum handoff (KAPPA_FREEZE_MECHANISM fix #1). --time=1:30:00 (maintainer
# ruling 2026-07-06: 90 min is plenty per arm on Helix): f_A is in-ODE (early-segment cost,
# FINDINGS §8d) and Eb-back-reacting (dt shrink, R4); fA=24-32 exceeds any boost run live; the diffuse
# arms (large_diffuse, small_1e6) are the long pole. Post-harvest, every arm must show t_final>=5
# or a physics termination; re-run any wall-killed / nonzero-exit arm at a longer limit BEFORE
# quoting any theta (protocol rule 2). Report "N/81 compliant" like theta5's 32/32.
#
#SBATCH --job-name=trinity-theta5s
#SBATCH --partition=cpu-single
#SBATCH --account=bw22J006
#SBATCH --export=NONE
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=1-81
#SBATCH --time=1:30:00
#SBATCH --output=/gpfs/bwfor/work/ws/hd_cq295-trinity/jobs_theta5s/logs/%A_%a.out

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate trinity

REPO=/home/hd/hd_hd/hd_cq295/trinity
WS=/gpfs/bwfor/work/ws/hd_cq295-trinity
PARAMS="$REPO/docs/dev/transition/pdv-trigger/runs/params/theta5s"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

mkdir -p "$WS/jobs_theta5s/logs"
PARAM=$(ls "$PARAMS"/*.param | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "$PARAM" ]; then
    echo "ERROR: no param #${SLURM_ARRAY_TASK_ID} in $PARAMS — run make_theta5s_params.py first?" >&2
    exit 1
fi
# path2output in each .param is relative -> run from the writable workspace
cd "$WS"
SECONDS=0
python "$REPO/run.py" "$PARAM"
code=$?
OUTDIR="$WS/outputs/theta5s/$(basename "$PARAM" .param)"
mkdir -p "$OUTDIR"
echo "$code" > "$OUTDIR/.exit_code"
echo "$SECONDS" > "$OUTDIR/.duration"
exit $code

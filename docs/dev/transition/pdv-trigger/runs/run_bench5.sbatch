#!/bin/bash -l
# 📏 bench5 HPC CONFIRMATION — the 60-arm Phase-5 L21b matrix (FINDINGS §15h) re-run on Helix.
# The in-container result (60/60, 59 compliant; fire map 1→4→12→>16→>16; Θ_cum calibration) is
# PROVISIONAL; this array makes it authoritative and measures in-container-vs-HPC fidelity
# (compare with data/compare_bench5_hpc.py). Same cluster conventions as run_theta5s.sbatch.
# Params are committed (runs/params/bench5/, emitted by make_bench5_params.py — do NOT re-emit).
#
#   submit:  sbatch docs/dev/transition/pdv-trigger/runs/run_bench5.sbatch
#            (or ./sync_bench.sh bench5 submit from the laptop)
#   harvest: python docs/dev/transition/pdv-trigger/runs/harvest_bench5.py "$WS"/outputs/bench5/* \
#              --csv "$WS"/outputs/bench5_summary_hpc.csv --traj-dir "$WS"/outputs/bench5_traj_hpc
#
# --time=1:30:00 (maintainer ruling 2026-07-13, same as theta5s): MEASURED evidence in
# data/bench5_durations.csv — the longest COMPLIANT in-container arm took 3856 s (64 min) WITH
# 3-worker contention, so 90 min on a dedicated Helix core is ample. The one wall-killer
# (bench5_m5e5_r2p5__fa16_diag) froze at t=0.037 — stiffness, not walltime; if it wall-kills here
# too, record it as the known non-critical freeze (§15h), do not chase with longer limits.
# Post-harvest compliance: every arm t_final>=5 or a physics termination; report "N/60 compliant".
#
#SBATCH --job-name=trinity-bench5
#SBATCH --partition=cpu-single
#SBATCH --account=bw22J006
#SBATCH --export=NONE
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=1-60
#SBATCH --time=1:30:00
#SBATCH --output=/gpfs/bwfor/work/ws/hd_cq295-trinity/jobs_bench5/logs/%A_%a.out

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate trinity

REPO=/home/hd/hd_hd/hd_cq295/trinity
WS=/gpfs/bwfor/work/ws/hd_cq295-trinity
PARAMS="$REPO/docs/dev/transition/pdv-trigger/runs/params/bench5"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

mkdir -p "$WS/jobs_bench5/logs"
PARAM=$(ls "$PARAMS"/*.param | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "$PARAM" ]; then
    echo "ERROR: no param #${SLURM_ARRAY_TASK_ID} in $PARAMS" >&2
    exit 1
fi
# path2output in each .param is relative -> run from the writable workspace
cd "$WS"
SECONDS=0
python "$REPO/run.py" "$PARAM"
code=$?
OUTDIR="$WS/outputs/bench5/$(basename "$PARAM" .param)"
mkdir -p "$OUTDIR"
echo "$code" > "$OUTDIR/.exit_code"
echo "$SECONDS" > "$OUTDIR/.duration"
exit $code

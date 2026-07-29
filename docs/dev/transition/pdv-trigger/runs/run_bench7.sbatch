#!/bin/bash -l
# 📏 bench7 — the f_kappa RE-OPEN campaign on Helix (KAPPA_REOPEN_PLAN.md K1–K4, ~102 arms).
# All five K-phases share this ONE array: a phase is just a filename prefix in params/bench7/, so
# there is one submit, one reduce and one download for the whole campaign.
#   K1  bench1/2/3 × f_kappa grid × prod/diag — the missing third leg of the band-entry head-to-head
#   K1b bench4/bench5 × f_kappa — dense fire-map completeness
#   K2  the 6 band configs × f_kappa {5,7,9} — the condensation-squeeze fine grid
#   K3  the 5 fate-flip arms, each run twice — determinism (P4)
#   K4  bench1/bench2 × f_mix {12,16} × prod/diag — the f_mix band-entry closure (P5)
# Params are committed by make_kappa_reopen_params.py; purpose, predictions P1–P5 and gates G0–G5
# live in KAPPA_REOPEN_PLAN.md. Same cluster conventions as run_bench6.sbatch.
#
#   emit params (once, committed):  python docs/dev/transition/pdv-trigger/runs/make_kappa_reopen_params.py
#   submit:  ./sync_bench.sh bench7 submit      # AUTO-SIZES --array from the committed param count
#            (or: sbatch --array=1-<N> docs/dev/transition/pdv-trigger/runs/run_bench7.sbatch)
#   reduce:  ./sync_bench.sh bench7 reduce      # ON HPC; declares the extra columns — see below
#   down:    ./sync_bench.sh bench7 down        # ships ONLY the reduced CSVs
#
# ⚠️ THE REDUCE IS ONE-SHOT — sync_bench.sh passes
#     --extra-cols Pb,bubble_dMdt,bubble_L2Conduction,bubble_L3Intermediate
# because P2 and the K0.Q1b back-reaction both read Pb and dMdt, and the six default trajectory
# columns do not carry them. gpfs gets cleaned; whatever the reduce misses is gone (the theta5s
# lesson). Do not hand-run the harvest without those columns.
#
# --time=1:30:00 (maintainer ruling 2026-07-13; evidence data/bench5_durations.csv — longest
# compliant bench5 arm 64 min under 3-worker contention). The UNMEASURED-cost arms here are the
# diffuse HIGH-DOSE f_kappa diag ones (bench1 at f_kappa ≥ 24): f_kappa enters the structure ODE, so
# they are the analogue of bench5_fa16_diag's stiffness corner. If one wall-kills (exit 124 with t
# still advancing in dictionary.jsonl), resubmit just those ids with a CLI override — it beats the
# in-file directive:
#     sbatch --time=3:00:00 --array=<id1>,<id2> docs/dev/transition/pdv-trigger/runs/run_bench7.sbatch
# and re-reduce; quote no theta from a wall-killed arm, and record it as a G3 non-compliance rather
# than dropping it silently. A FROZEN arm (t stalled at a fixed step, cf. bench5_fa16_diag §15h) is a
# stiffness signature, not a walltime problem — record, don't chase.
#
#SBATCH --job-name=trinity-bench7
#SBATCH --partition=cpu-single
#SBATCH --account=bw22J006
#SBATCH --export=NONE
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=1-102
#SBATCH --time=1:30:00
#SBATCH --output=/gpfs/bwfor/work/ws/hd_cq295-trinity/jobs_bench7/logs/%A_%a.out

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate trinity

REPO=/home/hd/hd_hd/hd_cq295/trinity
WS=/gpfs/bwfor/work/ws/hd_cq295-trinity
PARAMS="$REPO/docs/dev/transition/pdv-trigger/runs/params/bench7"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

mkdir -p "$WS/jobs_bench7/logs"
PARAM=$(ls "$PARAMS"/*.param | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "$PARAM" ]; then
    echo "ERROR: no param #${SLURM_ARRAY_TASK_ID} in $PARAMS — run make_kappa_reopen_params.py first?" >&2
    exit 1
fi
# path2output in each .param is relative -> run from the writable workspace
cd "$WS"
SECONDS=0
python "$REPO/run.py" "$PARAM"
code=$?
OUTDIR="$WS/outputs/bench7/$(basename "$PARAM" .param)"
mkdir -p "$OUTDIR"
echo "$code" > "$OUTDIR/.exit_code"
echo "$SECONDS" > "$OUTDIR/.duration"
exit $code

#!/bin/bash -l
# phii-identity arm runner as a SLURM job.
#   sbatch b14.sbatch <CREPO> <SWEEP> <ARM>        (normally: ./sync.sh submit)
#
# Everything comes in as a POSITIONAL ARGUMENT, never by environment. That is
# deliberate and is borrowed from paper/II-survey/reduce.sbatch, which records the
# cost of learning it the other way: `#SBATCH --export=NONE` gives the job a clean
# environment, so `--export=SWEEP=...` sets a variable in the SUBMITTING shell that
# the job never sees, and the job then runs against whatever the default was. An
# argument cannot be eaten.
#SBATCH --job-name=phii-arm
#SBATCH --partition=cpu-single
#SBATCH --account=bw22J006
#SBATCH --export=NONE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
# NOTE: --output is set on the sbatch command line by sync.sh, so the log lands
# beside the sweep. It is evaluated by the submitting shell, not the job.
set -euo pipefail

CREPO=${1:?usage: sbatch b14.sbatch CREPO SWEEP ARM}
SWEEP=${2:?usage: sbatch b14.sbatch CREPO SWEEP ARM}
ARM=${3:?usage: sbatch b14.sbatch CREPO SWEEP ARM [CONFIGS] [STOP_T]}
# Optional, and POSITIONAL for the same reason as the three above: an exported
# CONFIGS silently vanished under --export=NONE and the job ran the default ladder.
CONFIGS_IN=${4:-}
STOP_T_IN=${5:-}

module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate trinity

# ---- BANNER GUARD -----------------------------------------------------------
# Print what this job believes it is doing, as its FIRST output, and abort on
# anything obviously wrong. The failure this prevents is the expensive one: a job
# that runs happily against the wrong sweep or a worktree that was never prepped.
echo "=============================================================="
echo " CREPO  : $CREPO"
echo " SWEEP  : $SWEEP"
echo " ARM    : $ARM"
echo " CONFIGS: ${CONFIGS_IN:-<default: SC,F1LO,F1HI,B3M,B3MW01>}"
echo " STOP_T : ${STOP_T_IN:-<default: 1.5>}"
echo " WT     : $SWEEP/wt_$ARM"
echo " python : $(command -v python)  ($(python -V 2>&1))"
echo "=============================================================="
[ -d "$CREPO" ]           || { echo "ABORT: CREPO is not a directory"; exit 1; }
[ -d "$SWEEP" ]           || { echo "ABORT: SWEEP is not a directory -- was 'prep' run?"; exit 1; }
[ -d "$SWEEP/wt_$ARM" ]   || { echo "ABORT: no worktree for arm '$ARM' -- run 'sync.sh submit' (it preps on the login node)"; exit 1; }
python -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 9) else 1)' \
  || { echo "ABORT: python < 3.9; run_batch.py uses Path.is_relative_to and will crash after the runs"; exit 1; }

cd "$CREPO"
bash docs/dev/phii-identity/hpc/b14/run_arms.sh run "$SWEEP" "$ARM" "$CONFIGS_IN" "$STOP_T_IN"

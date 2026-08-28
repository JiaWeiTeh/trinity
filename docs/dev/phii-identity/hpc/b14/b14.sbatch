#!/usr/bin/env bash
#SBATCH --job-name=phii_b14
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# One job per arm; sync.sh submit passes SWEEP/ARM/CREPO_DIR via --export and
# points --output at $SWEEP/logs/. Runs are sequential inside the job (5
# configs × ~10-25 min); 8 h is a generous ceiling, not an estimate.
set -euo pipefail
module load devel/miniforge
conda activate trinity
cd "$CREPO_DIR"
bash docs/dev/phii-identity/hpc/b14/run_arms.sh run "$SWEEP" "$ARM"

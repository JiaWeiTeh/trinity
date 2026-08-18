#!/usr/bin/env bash
#SBATCH --job-name=trinity
#SBATCH --output=%A_%a.slurmout
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#
# One array task = one config = one Python process. That is exactly PLAN.md C-3
# ("one run per process, always -- trinity leaks module-level global state in-process"),
# satisfied by the scheduler rather than by a --workers flag.
#
# Env supplied by helix.sh submit: TRINITY_CFG, TRINITY_OUT, TRINITY_SHA
set -euo pipefail

CFG_LINE=$(grep -ve '^\s*$' -e '^\s*#' "${TRINITY_CFG}" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
[ -n "${CFG_LINE}" ] || { echo "no config for task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2; }

# Line format:  <name> <path/to/file.param> [extra run.py args...]
NAME=$(awk '{print $1}' <<<"${CFG_LINE}")
PARAM=$(awk '{print $2}' <<<"${CFG_LINE}")
EXTRA=$(cut -d' ' -f3- <<<"${CFG_LINE}")

RUNDIR="${TRINITY_OUT}/${NAME}"
mkdir -p "${RUNDIR}"

# Record what produced this BEFORE running, so a killed task still says what it was.
cat > "${RUNDIR}/.provenance" <<EOF
arm=$(basename "${TRINITY_OUT}")
name=${NAME}
param=${PARAM}
extra=${EXTRA}
code=${TRINITY_SHA}
slurm_job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
node=$(hostname)
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

START=$(date +%s)
set +e
srun python3 run.py "${PARAM}" --output-dir "${RUNDIR}" ${EXTRA}
RC=$?
set -e

echo "${RC}"                     > "${RUNDIR}/.exit_code"
echo "$(($(date +%s) - START))"  > "${RUNDIR}/.duration"
exit 0   # never fail the array on a single bad config -- the manifest records it instead

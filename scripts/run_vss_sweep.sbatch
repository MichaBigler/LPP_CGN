#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM array submission for VSS sweep runs.
#
# Each array task runs exactly one case (one row of the expanded VSS config),
# whose procedures may further expand into sub-runs internally (e.g. with
# procedure=bounds → 4 sub-runs per task).
#
# Submit via `bash scripts/submit_vss_sweep.sh` (recommended): the wrapper
# regenerates the expanded config and sets the --array=0-(N-1)%K bound for
# you. Manual submission also possible:
#
#   sbatch --array=0-127%40 scripts/run_vss_sweep.sbatch
#
# Adapt before first use:
#   MAIL_USER       — your e-mail for SLURM notifications
#   GUROBI_MODULE   — verify the version with `module avail Gurobi`
#   VENV_PATH       — path to a venv with pandas/numpy/scipy
#   EXPANDED_CFG    — path to expanded config (relative to repo root)
# ---------------------------------------------------------------------------

#SBATCH --job-name=lpp_vss
#SBATCH --qos=6hours
#SBATCH --time=05:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --output=logs/lpp_vss_%A_%a.out
#SBATCH --error=logs/lpp_vss_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=micha.bigler@unibas.ch

set -euo pipefail

GUROBI_MODULE="Gurobi/12.0.3-GCCcore-14.2.0"
VENV_PATH="${HOME}/envs/lpp"
EXPANDED_CFG="Data/config_vss_expanded.csv"

mkdir -p logs

module purge
module load "${GUROBI_MODULE}"

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

# Gurobi should use exactly the cores SLURM allocated to us.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "Working directory: $(pwd)"
date

# One row per task; --workers 1 so the bounds procedures inside the row
# run sequentially and don't oversubscribe Gurobi threads.
python run.py \
    --config "${EXPANDED_CFG}" \
    --row "${SLURM_ARRAY_TASK_ID}" \
    --stamp-suffix "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --workers 1

date
echo "Task ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} finished"

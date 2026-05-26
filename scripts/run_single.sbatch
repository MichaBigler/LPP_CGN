#!/bin/bash
# ---------------------------------------------------------------------------
# Generic single-run submission. Pass the config via env var:
#
#   CONFIG=Data/config_mumford0_test.csv sbatch scripts/run_single.sbatch
#
# Designed for sanity tests and one-off solves outside the VSS sweep machinery.
# ---------------------------------------------------------------------------

#SBATCH --job-name=lpp_single
#SBATCH --qos=6hours
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --output=logs/lpp_single_%j.out
#SBATCH --error=logs/lpp_single_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=micha.bigler@unibas.ch

set -euo pipefail

GUROBI_MODULE="Gurobi/12.0.3-GCCcore-14.2.0"
VENV_PATH="${HOME}/envs/lpp"
CONFIG="${CONFIG:-Data/config.csv}"

mkdir -p logs

module purge
module load "${GUROBI_MODULE}"

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "Config: ${CONFIG}"
date

python run.py --config "${CONFIG}" --workers 1

date
echo "Job ${SLURM_JOB_ID} finished"

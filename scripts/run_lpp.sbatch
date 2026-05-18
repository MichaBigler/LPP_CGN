#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM submission script for the LPP_CGN optimisation pipeline.
#
# Adapt before first use on a new account:
#   MAIL_USER       — your e-mail for SLURM notifications
#   GUROBI_MODULE   — `module avail Gurobi` and pick the version
#   VENV_PATH       — path to a venv with pandas/numpy/scipy installed
#                     (created with `python -m venv ... --system-site-packages`
#                      so that `gurobipy` from the module remains visible)
#
# Workflow assumption:
#   - `Data/config.csv` contains rows you want to evaluate
#   - The `threads` column in each row matches --cpus-per-task below
#   - `--workers 1` keeps Gurobi from oversubscribing the SLURM cgroup
#     when `procedure=bounds` expands one row into four sequential solves
# ---------------------------------------------------------------------------

#SBATCH --job-name=lpp_cgn
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --output=logs/lpp_%j.out
#SBATCH --error=logs/lpp_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=CHANGEME@unibas.ch

set -euo pipefail

GUROBI_MODULE="Gurobi/12.0.3-GCCcore-14.2.0"
VENV_PATH="${HOME}/envs/lpp"

mkdir -p logs

module purge
module load "${GUROBI_MODULE}"

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

# Gurobi should use exactly the cores SLURM allocated to us.
# `solve_*.py` reads the `threads` column from config.csv; this env var is
# informational only and serves as a sanity check in the log.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "Working directory: $(pwd)"
date

# IMPORTANT: keep --workers 1 unless every config row uses few enough Gurobi
# threads to fit ${SLURM_CPUS_PER_TASK}/workers without oversubscription.
python run.py --workers 1

date
echo "Job ${SLURM_JOB_ID} finished"

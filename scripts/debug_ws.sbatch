#!/bin/bash
# ---------------------------------------------------------------------------
# One-off debug submission for the WS > RP investigation.
#
# Runs a single row from an expanded config with LPP_DEBUG_WS=1 enabled so
# both `ws` and `integrated` print per-(sub-)scenario decompositions that can
# be diffed offline.
#
# sbatch does NOT inherit the caller's env vars by default — pass them via
# `--export=ALL,...`:
#
#   sbatch --export=ALL,\
# CONFIG=Data/config_vss_mumford0_expanded.csv,\
# ROW=2,\
# SUBDIR=debug_ws_811 \
#       scripts/debug_ws.sbatch
#
# After completion: grep -E '^\[(WS|RP)-DBG\]' logs/debug_ws_<JOBID>.out
# ---------------------------------------------------------------------------

#SBATCH --job-name=lpp_debug_ws
#SBATCH --qos=6hours
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --output=logs/debug_ws_%j.out
#SBATCH --error=logs/debug_ws_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=micha.bigler@unibas.ch

set -euo pipefail

GUROBI_MODULE="Gurobi/12.0.3-GCCcore-14.2.0"
VENV_PATH="${HOME}/envs/lpp"
CONFIG="${CONFIG:?CONFIG env var is required (path to expanded config CSV)}"
ROW="${ROW:?ROW env var is required (0-indexed config row)}"
SUBDIR="${SUBDIR:-debug_ws_${SLURM_JOB_ID}}"

mkdir -p logs

module purge
module load "${GUROBI_MODULE}"

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export LPP_DEBUG_WS=1

echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "Config: ${CONFIG}"
echo "Row:    ${ROW}"
echo "Subdir: ${SUBDIR}"
echo "LPP_DEBUG_WS=${LPP_DEBUG_WS}"
date

python run.py \
    --config "${CONFIG}" \
    --row "${ROW}" \
    --results-subdir "${SUBDIR}" \
    --workers 1

date
echo "Job ${SLURM_JOB_ID} finished"

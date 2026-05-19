#!/bin/bash
# ---------------------------------------------------------------------------
# Submit a VSS sweep:
#   1. expand `Data/config_vss.csv` → `Data/config_vss_expanded.csv` and
#      materialise the per-case scenario_infra/scenario_prob files
#   2. count expanded rows
#   3. submit `scripts/run_vss_sweep.sbatch` as a SLURM array job
#
# Defaults:
#   parallelism (concurrent tasks) = 40 — adapt via PARALLELISM=N before call
#
# Usage:
#   bash scripts/submit_vss_sweep.sh
#   PARALLELISM=20 bash scripts/submit_vss_sweep.sh
#   IN_CSV=Data/config_vss_custom.csv bash scripts/submit_vss_sweep.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

IN_CSV="${IN_CSV:-Data/config_vss.csv}"
OUT_CSV="${OUT_CSV:-Data/config_vss_expanded.csv}"
PARALLELISM="${PARALLELISM:-40}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-scripts/run_vss_sweep.sbatch}"
# Extra sbatch flags appended on submit, e.g. for memory / QOS overrides:
#   SBATCH_EXTRA="--mem=256G --qos=1day --time=23:00:00" bash scripts/submit_vss_sweep.sh
SBATCH_EXTRA="${SBATCH_EXTRA:-}"

if [[ ! -f "${IN_CSV}" ]]; then
    echo "[ERR] Input config not found: ${IN_CSV}" >&2
    exit 1
fi

echo "[1/3] Expanding ${IN_CSV} → ${OUT_CSV}"
python scripts/generate_vss_cases.py \
    --in "${IN_CSV}" \
    --out "${OUT_CSV}" \
    --data-root .

# Count data rows (header excluded). `grep -c .` counts non-empty lines, so
# trailing blank lines, missing-newline-at-eof, or CRLF artefacts all behave
# correctly — more robust than `wc -l - 1` which depends on the writer.
DATA_LINES=$(grep -c . "${OUT_CSV}" || true)
N=$((DATA_LINES - 1))
if [[ ${N} -le 0 ]]; then
    echo "[ERR] Expanded config has no rows" >&2
    exit 1
fi
echo "[2/3] Expanded to ${N} cases"

LAST=$((N - 1))
ARRAY_SPEC="0-${LAST}%${PARALLELISM}"
echo "[3/3] Submitting array ${ARRAY_SPEC} via ${SBATCH_SCRIPT}"
if [[ -n "${SBATCH_EXTRA}" ]]; then
    echo "     extra sbatch flags: ${SBATCH_EXTRA}"
fi
# shellcheck disable=SC2086  # intentional word-splitting of SBATCH_EXTRA
SUBMIT_OUT=$(sbatch ${SBATCH_EXTRA} --array="${ARRAY_SPEC}" "${SBATCH_SCRIPT}")
echo "${SUBMIT_OUT}"

# Persist the array job ID + expected task count so `aggregate_vss_map.py`
# can auto-detect them later without the user having to copy-paste numbers.
JOB_ID=$(echo "${SUBMIT_OUT}" | grep -oE '[0-9]+' | tail -1)
if [[ -n "${JOB_ID}" ]]; then
    cat > "${REPO_ROOT}/.last_vss_job" <<META
job_id=${JOB_ID}
tasks=${N}
expanded_config=${OUT_CSV}
submitted_at=$(date -Iseconds)
META
    echo "Saved job metadata to .last_vss_job"
fi

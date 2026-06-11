#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the full *matched* experiment program (SF + Mumford0, rho=0.10,
# equal operational parameters). One SLURM array per config.
#
# Long walltime + 1day QOS because rho=0.10 is SF's congestion-onset regime
# (high-k cases are hard). time_limit=12000 s/procedure is set in the configs.
#
# Job IDs + config paths are appended to .matched_jobs for aggregation.
#
# Usage:
#   bash scripts/submit_all_matched.sh
#   ONLY="main_sf pfail_sf" bash scripts/submit_all_matched.sh   # subset
#   PARALLELISM=60 bash scripts/submit_all_matched.sh
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PARALLELISM="${PARALLELISM:-40}"
ONLY="${ONLY:-}"
JOB_LOG="${REPO_ROOT}/.matched_jobs"
SBATCH_EXTRA="${SBATCH_EXTRA:---qos=1day --time=23:00:00}"

SLUGS=(
    main_sf bypass_sf overdemand_sf pfail_sf replcost_sf
    main_mumford0 bypass_mumford0 overdemand_mumford0 pfail_mumford0 replcost_mumford0
)

> "${JOB_LOG}.new"
for slug in "${SLUGS[@]}"; do
    if [[ -n "${ONLY}" ]]; then
        skip=1; for s in ${ONLY}; do [[ "${s}" == "${slug}" ]] && skip=0; done
        [[ ${skip} -eq 1 ]] && continue
    fi
    IN="Data/config_matched_${slug}.csv"
    OUT="Data/config_matched_${slug}_expanded.csv"
    if [[ ! -f "${IN}" ]]; then
        echo "[SKIP] ${slug}: ${IN} missing" >&2; continue
    fi
    echo "============================================================"
    echo "Matched sweep: ${slug}"
    echo "============================================================"
    rm -f "${REPO_ROOT}/.last_vss_job"
    IN_CSV="${IN}" OUT_CSV="${OUT}" PARALLELISM="${PARALLELISM}" \
        SBATCH_EXTRA="${SBATCH_EXTRA}" \
        bash scripts/submit_vss_sweep.sh
    if [[ -f "${REPO_ROOT}/.last_vss_job" ]]; then
        { echo "[${slug}]"; cat "${REPO_ROOT}/.last_vss_job"; echo; } >> "${JOB_LOG}.new"
    else
        echo "[WARN] ${slug}: no .last_vss_job produced" >&2
        { echo "[${slug}]"; echo "status=FAILED_TO_SUBMIT"; echo; } >> "${JOB_LOG}.new"
    fi
done
mv "${JOB_LOG}.new" "${JOB_LOG}"
echo
echo "All matched submissions in ${JOB_LOG}"

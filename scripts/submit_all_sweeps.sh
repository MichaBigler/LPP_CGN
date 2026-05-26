#!/bin/bash
# ---------------------------------------------------------------------------
# Submit ALL 12 VSS sweeps (SiouxFalls + Mumford0, 6 dimensions each) as
# independent SLURM array jobs.
#
# Used after the WS deterministic-settings fix to reproduce all bounds data
# with a reliable EVPI = RP - WS column.
#
# Each sweep:
#   - expands its config into per-case scenario files via generate_vss_cases.py
#   - submits a SLURM array (one task per case)
#
# Job IDs and expanded config paths are appended to .all_sweeps_jobs for the
# follow-up aggregation step.
#
# Usage:
#   bash scripts/submit_all_sweeps.sh
#   PARALLELISM=20 bash scripts/submit_all_sweeps.sh    # fewer concurrent tasks
#   ONLY="mumford0 mumford0_bypass" bash scripts/submit_all_sweeps.sh
#       # restrict to specific sweep slugs (matches the part after config_vss_)
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PARALLELISM="${PARALLELISM:-40}"
ONLY="${ONLY:-}"  # space-separated list of slugs to include; empty = all
JOB_LOG="${REPO_ROOT}/.all_sweeps_jobs"

# All 12 sweep slugs. Each one maps to:
#   IN_CSV  = Data/config_vss_<slug>.csv   (or Data/config_vss.csv for "sf_main")
#   OUT_CSV = Data/config_vss_<slug>_expanded.csv
SWEEPS=(
    "sf_main"
    "sf_bypass"
    "sf_overdemand"
    "sf_pfail"
    "sf_replcost"
    "sf_traincap"
    "sf_failurestructures"
    "mumford0"
    "mumford0_bypass"
    "mumford0_overdemand"
    "mumford0_pfail"
    "mumford0_replcost"
    "mumford0_traincap"
    "mumford0_failurestructures"
)

# Map slug → IN_CSV. SF main is config_vss.csv (no suffix).
slug_to_in_csv() {
    case "$1" in
        sf_main)     echo "Data/config_vss.csv" ;;
        sf_*)        echo "Data/config_vss_${1#sf_}.csv" ;;
        mumford0)    echo "Data/config_vss_mumford0.csv" ;;
        mumford0_*)  echo "Data/config_vss_${1}.csv" ;;
        *)           echo "" ;;
    esac
}

> "${JOB_LOG}.new"

for slug in "${SWEEPS[@]}"; do
    if [[ -n "${ONLY}" ]]; then
        skip=1
        for s in ${ONLY}; do
            if [[ "${s}" == "${slug}" ]]; then skip=0; fi
        done
        if [[ ${skip} -eq 1 ]]; then continue; fi
    fi

    IN_CSV="$(slug_to_in_csv "${slug}")"
    if [[ -z "${IN_CSV}" || ! -f "${IN_CSV}" ]]; then
        echo "[SKIP] ${slug}: config '${IN_CSV}' not found" >&2
        continue
    fi
    OUT_CSV="Data/config_vss_${slug}_expanded.csv"

    echo "============================================================"
    echo "Sweep: ${slug}"
    echo "  IN : ${IN_CSV}"
    echo "  OUT: ${OUT_CSV}"
    echo "============================================================"

    # Remove stale .last_vss_job so we only pick up the file produced by *this*
    # submission, not an earlier one.
    rm -f "${REPO_ROOT}/.last_vss_job"

    IN_CSV="${IN_CSV}" \
    OUT_CSV="${OUT_CSV}" \
    PARALLELISM="${PARALLELISM}" \
        bash scripts/submit_vss_sweep.sh

    # submit_vss_sweep.sh writes .last_vss_job — capture for this sweep.
    if [[ -f "${REPO_ROOT}/.last_vss_job" ]]; then
        {
            echo "[${slug}]"
            cat "${REPO_ROOT}/.last_vss_job"
            echo
        } >> "${JOB_LOG}.new"
    else
        # Loud failure: silent skip would let the user think 12 sweeps ran
        # while only N did. Make the gap visible in the job log too.
        echo "[WARN] ${slug}: no .last_vss_job produced — submission likely failed" >&2
        {
            echo "[${slug}]"
            echo "status=FAILED_TO_SUBMIT"
            echo
        } >> "${JOB_LOG}.new"
    fi
done

mv "${JOB_LOG}.new" "${JOB_LOG}"
echo
echo "All submissions written to ${JOB_LOG}"
echo "Watch with: squeue -u \$USER"

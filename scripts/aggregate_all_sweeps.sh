#!/bin/bash
# ---------------------------------------------------------------------------
# Aggregate all 12 VSS sweeps (SF + Mumford0) into per-sweep `vss_map.csv`
# files. Reads slug → job_id from `.all_sweeps_jobs` produced by
# `submit_all_sweeps.sh`.
#
# Usage:
#   bash scripts/aggregate_all_sweeps.sh
#   OUT_PREFIX=redo bash scripts/aggregate_all_sweeps.sh   # write to Results/<prefix>_<slug>/
#
# Per-slug output goes to Results/<OUT_PREFIX>_<slug>/vss_map.csv (default
# prefix: empty → Results/<slug>/, matching the existing layout but with the
# slug as folder name).
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

JOB_LOG="${REPO_ROOT}/.all_sweeps_jobs"
OUT_PREFIX="${OUT_PREFIX:-}"
# Optional filter: only aggregate these slugs (space-separated). Example:
#   ONLY="sf_failurestructures mumford0_failurestructures" bash scripts/aggregate_all_sweeps.sh
ONLY="${ONLY:-}"
VENV_PATH="${VENV_PATH:-${HOME}/envs/lpp}"

# Activate the project venv so pandas/numpy are on PYTHONPATH. Safe to call
# even when already activated.
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_PATH}/bin/activate"
else
    echo "[WARN] venv at ${VENV_PATH} not found — aggregator will fail if pandas not on path" >&2
fi

if [[ ! -f "${JOB_LOG}" ]]; then
    echo "[ERR] ${JOB_LOG} not found — run submit_all_sweeps.sh first" >&2
    exit 1
fi

# Parse .all_sweeps_jobs: blocks of "[slug]\nkey=value\n...\n".
# Emit "slug job_id" pairs on stdout (one per line). Skip slugs without job_id.
parse_jobs() {
    python3 - <<'PY' "${JOB_LOG}"
import re, sys
text = open(sys.argv[1]).read()
blocks = re.split(r"\n(?=\[)", text.strip())
for b in blocks:
    m = re.match(r"\[([^\]]+)\]", b.strip())
    if not m: continue
    slug = m.group(1)
    jm = re.search(r"job_id=(\d+)", b)
    if not jm:
        print(f"# SKIP {slug}: no job_id", file=sys.stderr)
        continue
    print(f"{slug} {jm.group(1)}")
PY
}

mapfile -t PAIRS < <(parse_jobs)

if [[ ${#PAIRS[@]} -eq 0 ]]; then
    echo "[ERR] no slug→job_id pairs parsed from ${JOB_LOG}" >&2
    exit 1
fi

echo "Aggregating ${#PAIRS[@]} sweeps..."
for pair in "${PAIRS[@]}"; do
    slug=$(echo "${pair}" | awk '{print $1}')
    job=$(echo "${pair}" | awk '{print $2}')

    if [[ -n "${ONLY}" ]]; then
        skip=1
        for s in ${ONLY}; do
            if [[ "${s}" == "${slug}" ]]; then skip=0; fi
        done
        if [[ ${skip} -eq 1 ]]; then continue; fi
    fi

    if [[ -n "${OUT_PREFIX}" ]]; then
        OUT_DIR="Results/${OUT_PREFIX}_${slug}"
    else
        OUT_DIR="Results/vss_${slug}_redo"  # don't overwrite OLD vss_<slug>/
    fi

    echo "------------------------------------------------------------"
    echo "[${slug}] job=${job} → ${OUT_DIR}"
    echo "------------------------------------------------------------"

    # Single-job aggregation per sweep. --tasks is auto-derived from the
    # job folder layout when only one job is passed.
    python scripts/aggregate_vss_map.py \
        --job "${job}" \
        --out-dir "${OUT_DIR}" \
        || echo "[WARN] aggregation failed for ${slug} (job ${job})"
done

echo
echo "Done. Per-sweep vss_map.csv files:"
for pair in "${PAIRS[@]}"; do
    slug=$(echo "${pair}" | awk '{print $1}')
    if [[ -n "${OUT_PREFIX}" ]]; then
        f="Results/${OUT_PREFIX}_${slug}/vss_map.csv"
    else
        f="Results/vss_${slug}_redo/vss_map.csv"
    fi
    if [[ -f "${f}" ]]; then
        n=$(($(grep -c . "${f}") - 1))
        echo "  ${f}  (${n} cases)"
    else
        echo "  ${f}  [MISSING]"
    fi
done

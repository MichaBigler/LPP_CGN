#!/bin/bash
# ---------------------------------------------------------------------------
# Aggregate the full matched program. Reads slug -> job_id from .matched_jobs
# (written by submit_all_matched.sh) and writes Results/vss_matched_<slug>/.
#
# Usage:
#   bash scripts/aggregate_all_matched.sh
#   ONLY="main_sf pfail_sf" bash scripts/aggregate_all_matched.sh
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
JOB_LOG="${REPO_ROOT}/.matched_jobs"
ONLY="${ONLY:-}"
VENV_PATH="${VENV_PATH:-${HOME}/envs/lpp}"

[[ -f "${VENV_PATH}/bin/activate" ]] && source "${VENV_PATH}/bin/activate"
[[ -f "${JOB_LOG}" ]] || { echo "[ERR] ${JOB_LOG} not found" >&2; exit 1; }

mapfile -t PAIRS < <(python3 - "${JOB_LOG}" <<'PY'
import re,sys
t=open(sys.argv[1]).read()
for b in re.split(r"\n(?=\[)",t.strip()):
    m=re.match(r"\[([^\]]+)\]",b.strip()); jm=re.search(r"job_id=(\d+)",b)
    if m and jm: print(f"{m.group(1)} {jm.group(1)}")
PY
)

for pair in "${PAIRS[@]}"; do
    slug=$(echo "$pair" | awk '{print $1}'); job=$(echo "$pair" | awk '{print $2}')
    if [[ -n "${ONLY}" ]]; then
        skip=1; for s in ${ONLY}; do [[ "$s" == "$slug" ]] && skip=0; done
        [[ $skip -eq 1 ]] && continue
    fi
    OUT="Results/vss_matched_${slug}"
    echo "[${slug}] job=${job} -> ${OUT}"
    python scripts/aggregate_vss_map.py --job "${job}" --out-dir "${OUT}" \
        || echo "[WARN] aggregation failed for ${slug}"
done
echo
echo "Done. Matched results:"
for pair in "${PAIRS[@]}"; do
    slug=$(echo "$pair" | awk '{print $1}')
    f="Results/vss_matched_${slug}/vss_map.csv"
    if [[ -f "$f" ]]; then echo "  $f  ($(($(grep -c . "$f")-1)) cases)"; else echo "  $f  [MISSING]"; fi
done

#!/usr/bin/env python3
# scripts/aggregate_vss_map.py
# -*- coding: utf-8 -*-
"""
Aggregate the per-task `bounds_summary.csv` files from a VSS sweep array job
into one tidy `vss_map.csv` keyed by case_id.

A SLURM array job submitted via `submit_vss_sweep.sh` writes one run folder
per task, named `Results/<datetime>_<arrayjob>_<task>/`. After the array
completes, this script:

  1. globs all matching run folders for the given SLURM array job ID
  2. reads each folder's `bounds_summary.csv`
  3. concatenates them
  4. writes `Results/vss_<arrayjob>/vss_map.csv` with case_* metadata + bounds
  5. emits a small `missing_tasks.txt` listing tasks whose summary is absent

Defaults make the typical case trivial — after `bash submit_vss_sweep.sh`:

    python scripts/aggregate_vss_map.py

reads `.last_vss_job` (written by the wrapper) for the job ID and expected
task count, and auto-detects everything else.

Explicit overrides:

    python scripts/aggregate_vss_map.py --job 12141700
    python scripts/aggregate_vss_map.py --job 12141700 --tasks 0-127
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Iterable, List, Optional, Set

import pandas as pd


_RE_TASK = re.compile(r"_(\d+)_(\d+)$")


def _parse_task_spec(spec: Optional[str]) -> Optional[Set[int]]:
    """Parse '0-127' or '0,1,5,7' into a set of task IDs. None on missing spec."""
    if not spec:
        return None
    out: Set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            for i in range(int(lo), int(hi) + 1):
                out.add(i)
        else:
            out.add(int(part))
    return out


def _collect_task_summaries(results_root: str, job_id: int) -> List[tuple]:
    """Return list of (task_id, df) for every run folder belonging to `job_id`."""
    pattern = os.path.join(results_root, f"*_{job_id}_*")
    dirs = sorted(p for p in glob.glob(pattern) if os.path.isdir(p))
    out = []
    for d in dirs:
        m = _RE_TASK.search(os.path.basename(d))
        if not m:
            continue
        # The glob can match folders whose timestamp segment coincidentally
        # contains `_<job_id>_` (e.g. minute = job_id). Verify the regex
        # captured the same job_id we asked for; otherwise the folder belongs
        # to a different array job and must be skipped.
        if int(m.group(1)) != job_id:
            continue
        task_id = int(m.group(2))
        bounds_path = os.path.join(d, "bounds_summary.csv")
        if not os.path.isfile(bounds_path):
            print(f"[WARN] missing bounds_summary in {d}", file=sys.stderr)
            continue
        try:
            df = pd.read_csv(bounds_path, sep=";")
        except Exception as exc:
            print(f"[WARN] could not read {bounds_path}: {exc}", file=sys.stderr)
            continue
        df = df.copy()
        df.insert(0, "task_id", task_id)
        df.insert(0, "array_job_id", job_id)
        out.append((task_id, df))
    return out


def aggregate(job_id: int, results_root: str = "Results",
              expected_tasks: Optional[Iterable[int]] = None,
              out_dir: Optional[str] = None) -> Optional[str]:
    found = _collect_task_summaries(results_root, job_id)
    if not found:
        print(f"[ERR] no bounds_summary files found for job {job_id} "
              f"under {results_root}/*_{job_id}_*", file=sys.stderr)
        return None

    found_ids = {t for t, _ in found}
    dfs = [df for _, df in sorted(found, key=lambda kv: kv[0])]
    combined = pd.concat(dfs, ignore_index=True)

    # Tidy column order: keys, case meta, bounds.
    case_cols = [c for c in combined.columns if c.startswith("case_")]
    bound_cols = ["WS", "RP", "EEV_nom", "EEV_mean", "EVPI", "VSS_nom", "VSS_mean"]
    head_cols = ["array_job_id", "task_id", "run_group_id"]
    other = [c for c in combined.columns
             if c not in head_cols + case_cols + bound_cols + ["procedures_present"]]
    ordered = (
        [c for c in head_cols if c in combined.columns]
        + case_cols
        + [c for c in bound_cols if c in combined.columns]
        + (["procedures_present"] if "procedures_present" in combined.columns else [])
        + other
    )
    combined = combined.reindex(columns=ordered)

    out_dir = out_dir or os.path.join(results_root, f"vss_{job_id}")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "vss_map.csv")
    combined.to_csv(out_csv, sep=";", index=False)
    print(f"Wrote {len(combined)} rows to {out_csv}")

    if expected_tasks is not None:
        missing = sorted(set(expected_tasks) - found_ids)
        miss_path = os.path.join(out_dir, "missing_tasks.txt")
        with open(miss_path, "w", encoding="utf-8") as fh:
            for t in missing:
                fh.write(f"{t}\n")
        if missing:
            print(f"[WARN] {len(missing)} task(s) missing; see {miss_path}",
                  file=sys.stderr)

    return out_csv


def _read_last_job_meta(repo_root: str = ".") -> dict:
    """Parse the `.last_vss_job` file written by submit_vss_sweep.sh."""
    path = os.path.join(repo_root, ".last_vss_job")
    if not os.path.isfile(path):
        return {}
    meta: dict = {}
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            meta[k.strip()] = v.strip()
    return meta


def _detect_most_recent_job(results_root: str) -> Optional[int]:
    """Fallback: derive a job ID from the most recently modified run folder
    that matches `<datetime>_<JOB>_<TASK>`."""
    candidates = []
    for entry in os.listdir(results_root) if os.path.isdir(results_root) else []:
        full = os.path.join(results_root, entry)
        if not os.path.isdir(full):
            continue
        m = _RE_TASK.search(entry)
        if not m:
            continue
        candidates.append((os.path.getmtime(full), int(m.group(1))))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", type=int, default=None,
                    help="SLURM array job ID. If omitted, read from "
                         "`.last_vss_job` (written by submit_vss_sweep.sh) or "
                         "fall back to the most recent run folder.")
    ap.add_argument("--results-root", default="Results",
                    help="Directory containing per-task run folders")
    ap.add_argument("--tasks", default=None,
                    help='Task ID spec for missing-task detection, '
                         'e.g. "0-127" or "0,1,5". If omitted, derived from '
                         '`.last_vss_job` (tasks=N → "0-(N-1)").')
    ap.add_argument("--out-dir", default=None,
                    help="Output directory (default: Results/vss_<job>)")
    ns = ap.parse_args()

    # Auto-detect job ID if not given.
    job_id = ns.job
    tasks_spec = ns.tasks
    meta = _read_last_job_meta()
    if job_id is None and meta.get("job_id"):
        job_id = int(meta["job_id"])
        print(f"Using job_id={job_id} from .last_vss_job")
    if job_id is None:
        job_id = _detect_most_recent_job(ns.results_root)
        if job_id is None:
            print("[ERR] could not determine job id; pass --job explicitly",
                  file=sys.stderr)
            sys.exit(1)
        print(f"Using job_id={job_id} from most recent run folder")
    if tasks_spec is None and meta.get("job_id") and int(meta["job_id"]) == job_id:
        n = int(meta.get("tasks", 0) or 0)
        if n > 0:
            tasks_spec = f"0-{n - 1}"

    expected = _parse_task_spec(tasks_spec)
    path = aggregate(job_id, ns.results_root, expected, ns.out_dir)
    sys.exit(0 if path else 1)


if __name__ == "__main__":
    _cli()

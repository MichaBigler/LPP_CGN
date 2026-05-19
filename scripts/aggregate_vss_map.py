#!/usr/bin/env python3
# scripts/aggregate_vss_map.py
# -*- coding: utf-8 -*-
"""
Aggregate the per-task `bounds_summary.csv` files from a VSS sweep array job
(or several) into one tidy `vss_map.csv` keyed by case_id.

A SLURM array job submitted via `submit_vss_sweep.sh` writes one run folder
per task, named `Results/<datetime>_<arrayjob>_<task>/`. After the array
completes, this script:

  1. globs all matching run folders for each given SLURM array job ID
  2. reads each folder's `bounds_summary.csv`
  3. concatenates them, deduplicating by case_id when --jobs lists several
     (the later job wins, useful when re-running OOM/TIMEOUT cases)
  4. writes `Results/vss_<job>/vss_map.csv` with case_* metadata + bounds
  5. emits a small `missing_tasks.txt` listing task IDs whose summary is
     absent (single-job mode only)

Defaults make the typical case trivial — after `bash submit_vss_sweep.sh`:

    python scripts/aggregate_vss_map.py

reads `.last_vss_job` (written by the wrapper) for the job ID and expected
task count.

Single job:

    python scripts/aggregate_vss_map.py --job 12141700
    python scripts/aggregate_vss_map.py --job 12141700 --tasks 0-127

Multiple jobs (merge a re-run into the original sweep):

    python scripts/aggregate_vss_map.py --jobs 12206136,12212522
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Iterable, List, Optional, Sequence, Set, Tuple

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


def _parse_job_list(spec: str) -> List[int]:
    """Parse '12141700' or '12141700,12150000' into a list of job IDs."""
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _collect_task_summaries(results_root: str, job_id: int) -> List[Tuple[int, pd.DataFrame, float]]:
    """Return list of (task_id, df, folder_mtime) for every run folder belonging to `job_id`."""
    pattern = os.path.join(results_root, f"*_{job_id}_*")
    dirs = sorted(p for p in glob.glob(pattern) if os.path.isdir(p))
    out: List[Tuple[int, pd.DataFrame, float]] = []
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
        out.append((task_id, df, os.path.getmtime(d)))
    return out


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    case_cols = [c for c in df.columns if c.startswith("case_")]
    bound_cols = ["WS", "RP", "EEV_nom", "EEV_mean", "EVPI", "VSS_nom", "VSS_mean"]
    head_cols = ["array_job_id", "task_id", "run_group_id"]
    other = [c for c in df.columns
             if c not in head_cols + case_cols + bound_cols + ["procedures_present"]]
    ordered = (
        [c for c in head_cols if c in df.columns]
        + case_cols
        + [c for c in bound_cols if c in df.columns]
        + (["procedures_present"] if "procedures_present" in df.columns else [])
        + other
    )
    return df.reindex(columns=ordered)


def aggregate(job_ids: Sequence[int],
              results_root: str = "Results",
              expected_tasks: Optional[Iterable[int]] = None,
              out_dir: Optional[str] = None) -> Optional[str]:
    """
    Aggregate one or several VSS sweep array jobs into a single `vss_map.csv`.

    When `job_ids` lists multiple IDs, the per-task DataFrames from all jobs
    are concatenated and deduplicated by `case_id`, keeping the row from the
    job folder with the latest modification time. This makes it trivial to
    merge a re-run that fills in OOM/TIMEOUT gaps of an earlier sweep.

    Single-job mode (len(job_ids) == 1) preserves the original missing-tasks
    detection via `expected_tasks`.
    """
    if not job_ids:
        print("[ERR] no job IDs given", file=sys.stderr)
        return None

    per_job_found: dict[int, Set[int]] = {}
    rows: List[Tuple[int, pd.DataFrame, float]] = []
    for jid in job_ids:
        found = _collect_task_summaries(results_root, jid)
        per_job_found[jid] = {t for t, _, _ in found}
        if not found:
            print(f"[WARN] no bounds_summary files for job {jid}", file=sys.stderr)
            continue
        print(f"  job {jid}: {len(found)} task folders collected")
        rows.extend(found)

    if not rows:
        print(f"[ERR] nothing to aggregate for {job_ids}", file=sys.stderr)
        return None

    # Concat all DataFrames; preserve a `_folder_mtime` helper so we can
    # decide which row wins on duplicate case_ids.
    enriched = []
    for _task_id, df, mtime in rows:
        df = df.copy()
        df["_folder_mtime"] = mtime
        enriched.append(df)
    combined = pd.concat(enriched, ignore_index=True)

    # Deduplicate by case_id: pick the row with the latest folder mtime.
    if "case_id" in combined.columns:
        before = len(combined)
        combined = (
            combined.sort_values("_folder_mtime")
                    .drop_duplicates("case_id", keep="last")
                    .sort_values("case_id")
                    .reset_index(drop=True)
        )
        if before != len(combined):
            print(f"  resolved {before - len(combined)} duplicate case_id rows "
                  f"(kept the most recent run)")

    combined = combined.drop(columns="_folder_mtime", errors="ignore")
    combined = _order_columns(combined)

    target_job = job_ids[-1]
    out_dir = out_dir or os.path.join(results_root, f"vss_{target_job}")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "vss_map.csv")
    combined.to_csv(out_csv, sep=";", index=False)
    print(f"Wrote {len(combined)} rows to {out_csv}")

    # Missing-tasks file: only meaningful for single-job mode where we know
    # the task ID space. Multi-job merges report per-job coverage instead.
    if expected_tasks is not None and len(job_ids) == 1:
        found_ids = per_job_found[job_ids[0]]
        missing = sorted(set(expected_tasks) - found_ids)
        miss_path = os.path.join(out_dir, "missing_tasks.txt")
        with open(miss_path, "w", encoding="utf-8") as fh:
            for t in missing:
                fh.write(f"{t}\n")
        if missing:
            print(f"[WARN] {len(missing)} task(s) missing; see {miss_path}",
                  file=sys.stderr)
    elif len(job_ids) > 1:
        for jid in job_ids:
            ids = per_job_found.get(jid, set())
            print(f"  job {jid}: {len(ids)} tasks contributed")

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
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--job", type=int, default=None,
                       help="SLURM array job ID (single-job mode). If omitted, "
                            "read from `.last_vss_job` or fall back to the most "
                            "recent run folder.")
    group.add_argument("--jobs", type=str, default=None,
                       help="Comma-separated SLURM array job IDs to merge "
                            "(e.g. '12206136,12212522'). Later IDs override "
                            "earlier ones on duplicate case_id.")
    ap.add_argument("--results-root", default="Results",
                    help="Directory containing per-task run folders")
    ap.add_argument("--tasks", default=None,
                    help='Task ID spec for missing-task detection (single-job '
                         'mode only), e.g. "0-127" or "0,1,5". If omitted, '
                         'derived from `.last_vss_job` (tasks=N → "0-(N-1)").')
    ap.add_argument("--out-dir", default=None,
                    help="Output directory. Default: Results/vss_<job> "
                         "(or Results/vss_<last_job> for --jobs).")
    ns = ap.parse_args()

    # Resolve target job(s).
    if ns.jobs:
        job_ids = _parse_job_list(ns.jobs)
        if not job_ids:
            print("[ERR] --jobs parsed to empty list", file=sys.stderr)
            sys.exit(1)
    else:
        job_id = ns.job
        meta = _read_last_job_meta()
        if job_id is None and meta.get("job_id"):
            job_id = int(meta["job_id"])
            print(f"Using job_id={job_id} from .last_vss_job")
        if job_id is None:
            job_id = _detect_most_recent_job(ns.results_root)
            if job_id is None:
                print("[ERR] could not determine job id; pass --job/--jobs "
                      "explicitly", file=sys.stderr)
                sys.exit(1)
            print(f"Using job_id={job_id} from most recent run folder")
        job_ids = [job_id]

    tasks_spec = ns.tasks
    if tasks_spec is None and len(job_ids) == 1:
        meta = _read_last_job_meta()
        if meta.get("job_id") and int(meta["job_id"]) == job_ids[0]:
            n = int(meta.get("tasks", 0) or 0)
            if n > 0:
                tasks_spec = f"0-{n - 1}"

    expected = _parse_task_spec(tasks_spec)
    path = aggregate(job_ids, ns.results_root, expected, ns.out_dir)
    sys.exit(0 if path else 1)


if __name__ == "__main__":
    _cli()

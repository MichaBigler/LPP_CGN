#!/usr/bin/env python3
# scripts/reorganize_results.py
# -*- coding: utf-8 -*-
"""
Move flat Results/<datetime>_<JOB>_<TASK>/ folders into a grouped layout:

    Results/job_<JOB>/<datetime>_task_<TASK>/

Idempotent and dry-run by default. Pre-array timestamped folders
(`Results/<datetime>/`) and aggregated outputs (`Results/vss_*/`) are
left untouched.

Usage:
    # Show what would happen
    python scripts/reorganize_results.py
    # Actually move folders
    python scripts/reorganize_results.py --apply
    # Only a specific job
    python scripts/reorganize_results.py --apply --job 12206136
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from typing import List, Optional, Tuple


_RE_FLAT = re.compile(r"^(\d{4}_\d{2}_\d{2}_\d{2}_\d{2})_(\d+)_(\d+)$")


def _scan(results_root: str, only_job: Optional[int] = None) -> List[Tuple[str, str, str, str]]:
    """Return list of (src_path, dst_dir, dst_path, basename) for each
    folder that matches the legacy flat layout."""
    if not os.path.isdir(results_root):
        return []
    out: List[Tuple[str, str, str, str]] = []
    for entry in sorted(os.listdir(results_root)):
        m = _RE_FLAT.match(entry)
        if not m:
            continue
        stamp, job, task = m.group(1), int(m.group(2)), int(m.group(3))
        if only_job is not None and job != only_job:
            continue
        src = os.path.join(results_root, entry)
        if not os.path.isdir(src):
            continue
        dst_dir = os.path.join(results_root, f"job_{job}")
        dst = os.path.join(dst_dir, f"{stamp}_task_{task}")
        out.append((src, dst_dir, dst, entry))
    return out


def _move_or_report(items, apply: bool) -> Tuple[int, int]:
    moved = 0
    skipped = 0
    for src, dst_dir, dst, base in items:
        if os.path.exists(dst):
            print(f"SKIP (target exists): {base}", file=sys.stderr)
            skipped += 1
            continue
        if not apply:
            print(f"WOULD MOVE: {src}  ->  {dst}")
            moved += 1
            continue
        os.makedirs(dst_dir, exist_ok=True)
        try:
            shutil.move(src, dst)
        except OSError as exc:
            print(f"FAIL: {base}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        moved += 1
    return moved, skipped


def _cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default="Results",
                    help="Root containing the flat folders (default: Results)")
    ap.add_argument("--apply", action="store_true",
                    help="Actually move folders. Without this, dry-run only.")
    ap.add_argument("--job", type=int, default=None,
                    help="Restrict reorganisation to one array job id.")
    ns = ap.parse_args()
    items = _scan(ns.results_root, only_job=ns.job)
    if not items:
        print("Nothing to reorganise.")
        return
    print(f"{'DRY-RUN' if not ns.apply else 'APPLY'}: {len(items)} folders match the legacy layout")
    moved, skipped = _move_or_report(items, apply=ns.apply)
    verb = "would move" if not ns.apply else "moved"
    print(f"\n{verb}: {moved}, skipped: {skipped}")
    if not ns.apply:
        print("Re-run with --apply to actually move.")


if __name__ == "__main__":
    _cli()

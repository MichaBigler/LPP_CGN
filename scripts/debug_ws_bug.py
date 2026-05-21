#!/usr/bin/env python3
# scripts/debug_ws_bug.py
# -*- coding: utf-8 -*-
"""
Diagnose the WS > RP bug observed on Mumford0 k>=2 cases.

For a given SLURM array job id, walks each task folder and decomposes
per-procedure obj into (stage1, stage2_exp, total). Reports the cases
where WS > RP and prints a breakdown so we can see which component
overshoots.

Usage:
    python scripts/debug_ws_bug.py --job 12329281 --top 10
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd


def _load_task_base_logs(job_id: int, results_root: str = "Results") -> pd.DataFrame:
    """Concatenate all base_log.csv files for a given array job into one frame."""
    grouped = os.path.join(results_root, f"job_{job_id}", "*", "base_log.csv")
    legacy = os.path.join(results_root, f"*_{job_id}_*", "base_log.csv")
    files = sorted(set(glob.glob(grouped)) | set(glob.glob(legacy)))
    if not files:
        print(f"[ERR] no base_log.csv found for job {job_id}", file=sys.stderr)
        sys.exit(1)
    rows = []
    for f in files:
        df = pd.read_csv(f, sep=";")
        df["_task_folder"] = os.path.dirname(f)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", type=int, required=True)
    ap.add_argument("--results-root", default="Results")
    ap.add_argument("--top", type=int, default=10,
                    help="Print decomposition for the top-N worst (WS - RP) cases.")
    ns = ap.parse_args()

    df = _load_task_base_logs(ns.job, ns.results_root)

    if "run_group_id" not in df.columns or "procedure" not in df.columns:
        print("[ERR] missing run_group_id or procedure columns", file=sys.stderr)
        sys.exit(1)

    # Pivot by (run_group_id, procedure)
    keep = ["run_group_id", "procedure", "objective",
            "obj_stage1", "obj_stage2_exp", "repl_cost_exp",
            "case_edges", "case_k", "case_per_run",
            "cost_bypass", "cost_time", "cost_wait", "cost_oper",
            "opt_gap", "runtime_s", "status"]
    keep = [c for c in keep if c in df.columns]
    df2 = df[keep].copy()

    pivot = df2.pivot_table(
        index=["run_group_id"] + [c for c in ("case_k", "case_per_run", "case_edges") if c in df2.columns],
        columns="procedure", values=[c for c in ["objective", "obj_stage1", "obj_stage2_exp",
                                                  "repl_cost_exp", "cost_bypass", "cost_time",
                                                  "cost_wait", "cost_oper",
                                                  "opt_gap", "runtime_s"] if c in df2.columns],
        aggfunc="first",
    ).reset_index()

    pivot.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c
                     for c in pivot.columns]

    # Diff: WS vs RP
    if "objective_ws" in pivot.columns and "objective_integrated" in pivot.columns:
        pivot["WS_minus_RP"] = pivot["objective_ws"] - pivot["objective_integrated"]
    else:
        print("[ERR] missing ws or integrated columns", file=sys.stderr)
        print("available cols:", list(pivot.columns))
        sys.exit(1)

    violations = pivot[pivot["WS_minus_RP"] > 1].copy()
    print(f"Cases with WS > RP: {len(violations)} / {len(pivot)}")
    if len(violations) == 0:
        return

    violations = violations.nlargest(ns.top, "WS_minus_RP")
    cols = [c for c in [
        "case_k", "case_per_run", "case_edges",
        "objective_integrated", "objective_ws",
        "obj_stage1_integrated", "obj_stage1_ws",
        "obj_stage2_exp_integrated", "obj_stage2_exp_ws",
        "cost_bypass_integrated", "cost_bypass_ws",
        "cost_time_integrated", "cost_time_ws",
        "opt_gap_integrated", "opt_gap_ws",
        "runtime_s_integrated", "runtime_s_ws",
        "WS_minus_RP",
    ] if c in violations.columns]

    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    print("\n=== TOP violations (WS > RP) ===")
    print(violations[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # Also show a contrast: top non-violating cases with the same case_k for comparison
    if "case_k" in pivot.columns and len(violations):
        worst_k = violations["case_k"].iloc[0]
        ok = pivot[(pivot["WS_minus_RP"] <= 1) & (pivot["case_k"] == worst_k)].copy()
        if len(ok):
            ok = ok.nsmallest(min(5, len(ok)), "WS_minus_RP")
            print(f"\n=== Comparison: top non-violating cases (case_k={worst_k}) ===")
            print(ok[[c for c in cols if c in ok.columns]].to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()

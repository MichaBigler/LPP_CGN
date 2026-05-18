# compute_bounds.py
# -*- coding: utf-8 -*-
"""
Aggregate WS / RP / EEV_nom / EEV_mean per run group and compute the
classical stochastic-programming bounds:

    EVPI     = RP        - WS
    VSS_nom  = EEV_nom   - RP
    VSS_mean = EEV_mean  - RP

Source: `base_log.csv` produced by `RunBatchLogger`. Sibling runs that
were expanded from the same input row carry the same `run_group_id` and
are joined on that column. For legacy logs without `run_group_id`, the
grouping falls back to a parameter fingerprint (all config columns except
`procedure` and the KPI columns).

Used in two ways:
  - as an end-of-batch hook called from `run.py main()`
  - as a standalone CLI:   python compute_bounds.py <results-dir-or-glob>
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Iterable, List, Optional

import pandas as pd

# Procedure → conventional symbol for the bound.
_PROC_TO_BOUND = {
    "ws": "WS",
    "wait_and_see": "WS",
    "integrated": "RP",
    "joint": "RP",
    "separated": "EEV_nom",
    "sequential": "EEV_nom",
    "eev": "EEV_mean",
    "expected_value": "EEV_mean",
}

# KPI / status columns produced by the logger; never part of the parameter fingerprint.
_KPI_COLS = {
    "status_code", "status", "objective", "runtime_s", "opt_gap",
    "cost_time", "cost_time_base", "cost_time_over", "cost_bypass", "cost_wait", "cost_oper",
    "obj_stage1", "obj_stage2_exp",
    "repl_cost_freq_exp", "repl_cost_path_exp", "repl_cost_exp",
}


def _read_base_log(results_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(results_dir, "base_log.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path, sep=";")


def _fingerprint_columns(df: pd.DataFrame) -> List[str]:
    """Columns identifying a 'case' — all config columns except procedure and KPIs."""
    return [c for c in df.columns if c not in _KPI_COLS and c not in ("procedure", "run_group_id")]


def _group_key_columns(df: pd.DataFrame) -> List[str]:
    if "run_group_id" in df.columns and df["run_group_id"].notna().any():
        return ["run_group_id"]
    return _fingerprint_columns(df)


def _bounds_for_group(group: pd.DataFrame) -> dict:
    """Compute bound values for one group. Missing bounds → NaN."""
    out = {"WS": pd.NA, "RP": pd.NA, "EEV_nom": pd.NA, "EEV_mean": pd.NA}
    for _, row in group.iterrows():
        proc = str(row.get("procedure") or "").strip().lower()
        bound = _PROC_TO_BOUND.get(proc)
        if bound is None:
            continue
        out[bound] = row.get("objective")

    def _f(x):
        try:
            return float(x) if x is not None and not pd.isna(x) else None
        except Exception:
            return None

    ws, rp, ev_nom, ev_mean = _f(out["WS"]), _f(out["RP"]), _f(out["EEV_nom"]), _f(out["EEV_mean"])
    out["EVPI"]     = (rp - ws) if (rp is not None and ws is not None) else pd.NA
    out["VSS_nom"]  = (ev_nom - rp) if (ev_nom is not None and rp is not None) else pd.NA
    out["VSS_mean"] = (ev_mean - rp) if (ev_mean is not None and rp is not None) else pd.NA
    return out


def compute_bounds(results_dir: str, *, out_csv: Optional[str] = None) -> Optional[str]:
    """
    Build `bounds_summary.csv` for one results directory.

    Returns the written path, or None if there is no `base_log.csv` or
    nothing could be grouped.
    """
    df = _read_base_log(results_dir)
    if df is None or df.empty:
        return None

    key_cols = _group_key_columns(df)
    if not key_cols:
        return None

    # Columns whose value is constant within a group and worth carrying through
    # so downstream sweep-aggregators (e.g. aggregate_vss_map.py) can join on
    # case metadata without re-reading base_log.csv.
    # ASSUMPTION: case_* columns are config columns shared by every procedure
    # sub-row of a group. If a future schema introduces a case_* column that
    # varies across rows in a group, the first non-null value wins silently.
    carry_cols = [c for c in df.columns if c.startswith("case_") and c not in key_cols]

    # One row per group with key columns + carry-through + bound columns.
    out_rows = []
    for keys, group in df.groupby(key_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(key_cols, keys))
        for c in carry_cols:
            # Constant within a group by construction; first non-null wins.
            vals = group[c].dropna()
            row[c] = vals.iloc[0] if not vals.empty else None
        row.update(_bounds_for_group(group))
        # Carry a representative procedure list so the user can see which
        # bounds are available in this group at a glance.
        row["procedures_present"] = ",".join(sorted({
            str(p).strip().lower() for p in group["procedure"].dropna().unique()
        }))
        out_rows.append(row)

    if not out_rows:
        return None

    bounds_df = pd.DataFrame(out_rows)
    # Stable column order: keys, case meta, bound objectives, bound differences, audit.
    ordered = (
        key_cols
        + carry_cols
        + ["WS", "RP", "EEV_nom", "EEV_mean", "EVPI", "VSS_nom", "VSS_mean", "procedures_present"]
    )
    bounds_df = bounds_df[[c for c in ordered if c in bounds_df.columns]]

    out_path = out_csv or os.path.join(results_dir, "bounds_summary.csv")
    bounds_df.to_csv(out_path, sep=";", index=False)
    return out_path


def _resolve_targets(args: Iterable[str]) -> List[str]:
    """Expand CLI args (paths or globs) into a list of results directories."""
    targets: List[str] = []
    for a in args:
        if any(ch in a for ch in "*?["):
            targets.extend(sorted(p for p in glob.glob(a) if os.path.isdir(p)))
        elif os.path.isdir(a):
            targets.append(a)
    return targets


def _cli():
    ap = argparse.ArgumentParser(description="Aggregate SP bounds (EVPI, VSS) from base_log.csv files.")
    ap.add_argument("results_dirs", nargs="+", help="Result directories or glob patterns")
    ns = ap.parse_args()
    targets = _resolve_targets(ns.results_dirs)
    if not targets:
        print("No matching result directories.", file=sys.stderr)
        sys.exit(1)
    for d in targets:
        path = compute_bounds(d)
        if path:
            print(f"wrote {path}")
        else:
            print(f"skipped {d} (no base_log.csv or no groupable rows)")


if __name__ == "__main__":
    _cli()

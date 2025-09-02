# -*- coding: utf-8 -*-
"""Logging utilities for LPP-CGN runs.

Creates a timestamped Results/<YYYY_MM_DD_HH_MM>/ folder and manages:
- base_log.csv (one row per run; config.csv columns + KPIs)
- row_XXX_freq.csv (per-run line frequencies)

All CSVs use ';' as separator to match config.csv.
"""

from __future__ import annotations
import os
from datetime import datetime
from typing import Dict, Any, Iterable
import pandas as pd


BASE_EXTRA_COLS = [
    "status_code", "status", "objective",
    "cost_time", "cost_wait", "cost_oper",
    "runtime_s", "N", "E_dir", "L", "S",
]


class RunBatchLogger:
    """Handles creation of output dir and per-run logging."""

    def __init__(self, data_root: str, cfg_df: pd.DataFrame, *, stamp: str | None = None):
        self.data_root = data_root
        self.stamp = stamp or datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.out_dir = os.path.join(self.data_root, "Results", self.stamp)
        os.makedirs(self.out_dir, exist_ok=True)

        # Base log schema = all config.csv columns + KPI columns
        self.base_columns = list(cfg_df.columns) + BASE_EXTRA_COLS
        self.base_log_path = os.path.join(self.out_dir, "base_log.csv")

        # Initialise header once
        if not os.path.exists(self.base_log_path):
            pd.DataFrame(columns=self.base_columns).to_csv(self.base_log_path, sep=';', index=False)

    # -------- helpers --------
    def base_row_template(self, cfg_row: pd.Series) -> Dict[str, Any]:
        """Prepares a dict with config columns filled from cfg_row and KPI columns set to None."""
        base = {k: cfg_row.get(k, None) for k in self.base_columns if k not in BASE_EXTRA_COLS}
        for k in BASE_EXTRA_COLS:
            base[k] = None
        return base

    def append_base_row(self, row_dict: Dict[str, Any]) -> None:
        """Appends a single line to base_log.csv (flushes immediately)."""
        # Ensure correct column order and presence
        df = pd.DataFrame([row_dict], columns=self.base_columns)
        df.to_csv(self.base_log_path, sep=';', index=False, mode='a', header=False)

    def write_freq_file(self, run_index: int, chosen_freq: Dict[int, int] | Dict[str, int]) -> str:
        """Writes per-run frequencies to row_XXX_freq.csv; returns path."""
        # Normalise keys to int if possible
        items: Iterable[tuple[int, int]] = []
        for k, v in chosen_freq.items():
            try:
                items = list(chosen_freq.items())  # type: ignore[assignment]
                break
            except Exception:
                pass
        # Sort by line id
        rows = [{"line": int(k), "freq": int(v)} for k, v in sorted(chosen_freq.items(), key=lambda kv: int(kv[0]))]
        df = pd.DataFrame(rows, columns=["line", "freq"])
        path = os.path.join(self.out_dir, f"row_{int(run_index):03d}_freq.csv")
        df.to_csv(path, sep=';', index=False)
        return path

    # -------- convenience --------
    def summary_paths(self) -> Dict[str, str]:
        return {
            "out_dir": self.out_dir,
            "base_log": self.base_log_path,
        }

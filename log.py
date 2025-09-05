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
    "status", "objective",
    "cost_time", "cost_wait", "cost_oper",
    "runtime_s", 
    # new (optional) two-stage KPIs:
    "obj_stage1", "obj_stage2_exp", "repl_cost_exp"
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
        # keys robust in int wandeln, sortieren, schreiben
        rows = []
        for k, v in chosen_freq.items():
            try:
                rows.append({"line": int(k), "freq": int(v)})
            except Exception:
                # falls key/val mal nicht-castbar sind, überspringen
                continue
        rows.sort(key=lambda r: r["line"])
        df = pd.DataFrame(rows, columns=["line", "freq"])
        path = os.path.join(self.out_dir, f"row_{int(run_index):03d}_freq.csv")
        df.to_csv(path, sep=';', index=False)
        return path

    def write_freqs_two_stage(self, run_index: int, model, nominal: Dict[int, int],
                          scenarios: list[dict], nominal_costs: Dict[str, float] | None = None) -> str:
        """
        Breite Frequenz-Tabelle:
        - oben: 6 Zeilen mit prob/objective/cost_time/cost_wait/cost_oper/cost_repl
        - darunter: Frequenzen je Linie (nominal + je Szenario)
        """
        scen_ids = [int(s["id"]) for s in scenarios]
        cols = ["line", "group", "nominal"] + [f"scenario {sid}" for sid in scen_ids]

        nom = nominal_costs or {}
        def _get(s, key):
            v = s.get(key) if s is not None else None
            return "" if v is None else v

        meta_rows = []

        # 1) prob (nominal leer)
        row = {"line": "prob", "group": "", "nominal": ""}
        for sid, s in zip(scen_ids, scenarios):
            row[f"scenario {sid}"] = _get(s, "prob")
        meta_rows.append(row)

        # 2) objective
        row = {"line": "objective", "group": "", "nominal": _get(nom, "objective")}
        for sid, s in zip(scen_ids, scenarios):
            row[f"scenario {sid}"] = _get(s, "objective")
        meta_rows.append(row)

        # 3) cost_time
        row = {"line": "cost_time", "group": "", "nominal": _get(nom, "time")}
        for sid, s in zip(scen_ids, scenarios):
            row[f"scenario {sid}"] = _get(s, "cost_time")
        meta_rows.append(row)

        # 4) cost_wait (gewichtet)
        row = {"line": "cost_wait", "group": "", "nominal": _get(nom, "wait")}
        for sid, s in zip(scen_ids, scenarios):
            row[f"scenario {sid}"] = _get(s, "cost_wait")
        meta_rows.append(row)

        # 5) cost_oper
        row = {"line": "cost_oper", "group": "", "nominal": _get(nom, "oper")}
        for sid, s in zip(scen_ids, scenarios):
            row[f"scenario {sid}"] = _get(s, "cost_oper")
        meta_rows.append(row)

        # 6) cost_repl (nominal leer)
        row = {"line": "cost_repl", "group": "", "nominal": ""}
        for sid, s in zip(scen_ids, scenarios):
            row[f"scenario {sid}"] = _get(s, "cost_repl")
        meta_rows.append(row)

        df_meta = pd.DataFrame(meta_rows, columns=cols)

        # ---- Frequenzen je Linie ----
        freq_rows = []
        for ell in range(model.L):
            r = {
                "line": ell,
                "group": int(model.line_idx_to_group[ell]),
                "nominal": int(nominal.get(ell, 0)),
            }
            for sid, s in zip(scen_ids, scenarios):
                r[f"scenario {sid}"] = int((s.get("freq") or {}).get(ell, 0))
            freq_rows.append(r)
        df_freq = pd.DataFrame(freq_rows, columns=cols)

        # ---- schreiben ----
        df_out = pd.concat([df_meta, df_freq], ignore_index=True)
        path = os.path.join(self.out_dir, f"row_{int(run_index):03d}_freq.csv")
        df_out.to_csv(path, sep=';', index=False)
        return path
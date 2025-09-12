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
    "status_code","status","objective","runtime_s",
    "cost_time","cost_wait","cost_oper",
    "obj_stage1","obj_stage2_exp",
    "repl_cost_freq_exp","repl_cost_path_exp","repl_cost_exp",
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

    def write_freqs_two_stage(
        self,
        run_index: int,
        model,
        nominal: dict[int, int],
        scenarios: list[dict],
        nominal_costs: dict[str, float] | None = None,
        *,
        cand_selected: dict[int, dict[int, int]] | None = None,   # {s: {g: k}}
        cand_all: dict[int, dict[int, list[dict]]] | None = None, # {s: {g: [cand...]}}
        use_stop_ids: bool = True,  # True: externe Stop-IDs, False: 0..N-1 Indizes
    ) -> str:
        """
        Breite Frequenz-Tabelle erweitert um Pfadspalten:
        - oben: 6 Zeilen mit prob/objective/cost_time/cost_wait/cost_oper/cost_repl
                (Werte stehen in den *_freq-Spalten; *_path bleibt leer)
        - darunter: je Linie: nominal_freq, nominal_path, und je Szenario: scenario <id>_freq/_path

        Pfade pro Szenario: falls cand_selected & cand_all vorhanden -> gewählter Kandidat;
                            sonst nominaler Pfad.
        """
        import os
        import pandas as pd

        # --- Hilfsfunktionen ------------------------------------------------
        idx_to_node_id = getattr(model, "idx_to_node_id", None)  # Liste: idx -> ext. ID
        idx_to_arc_uv  = getattr(model, "idx_to_arc_uv", None)   # Liste: a -> (u_id, v_id)

        def _to_id_list(node_idx_seq: list[int]) -> list[int]:
            if not node_idx_seq:
                return []
            if use_stop_ids and idx_to_node_id is not None:
                return [int(idx_to_node_id[i]) for i in node_idx_seq]
            return [int(i) for i in node_idx_seq]

        def nominal_path_nodes_for_line(ell: int) -> list[int]:
            seq_idx = list(map(int, model.line_idx_to_stops[ell]))
            return _to_id_list(seq_idx)

        def nodes_from_arcs_ids(arcs: list[int]) -> list[int]:
            """Erzeugt eine ID-Knotenfolge aus gerichteten Arc-IDs (nutzt idx_to_arc_uv)."""
            if not arcs:
                return []
            if idx_to_arc_uv is None:
                # Fallback: kein arcs->IDs Mapping, gib leere Liste
                return []
            u0, _ = idx_to_arc_uv[int(arcs[0])]
            seq = [int(u0)]
            for a in arcs:
                _, v = idx_to_arc_uv[int(a)]
                seq.append(int(v))
            return seq

        def path_for_scenario_line(s: int, ell: int) -> list[int]:
            """Gewählter Kandidatenpfad je Szenario/Gruppe; sonst nominal."""
            g = int(model.line_idx_to_group[ell])
            if cand_selected and cand_all:
                k = (cand_selected.get(s) or {}).get(g, None)
                cand_list = (cand_all.get(s) or {}).get(g, None)
                if k is not None and cand_list and 0 <= k < len(cand_list):
                    arcs = list(map(int, cand_list[k].get("arcs", [])))
                    nodes = nodes_from_arcs_ids(arcs)
                    if nodes:  # nur wenn wir sinnvoll rekonstruieren konnten
                        return nodes
            # fallback
            return nominal_path_nodes_for_line(ell)

        def _fmt_nodes(seq: list[int]) -> str:
            return ",".join(map(str, seq))

        # --- Spaltenlayout ---------------------------------------------------
        scen_ids = [int(s["id"]) for s in scenarios]
        cols = ["line", "group", "nominal_freq", "nominal_path"]
        for sid in scen_ids:
            cols += [f"scenario {sid}_freq", f"scenario {sid}_path"]

        # --- Kopfzeilen (Kennzahlen) ----------------------------------------
        nom = nominal_costs or {}

        def _get(s: dict | None, key: str):
            if s is None:
                return ""
            v = s.get(key)
            if v is None:
                # cost_repl evtl. als Summe aus freq+path
                if key == "cost_repl":
                    cf = s.get("cost_repl_freq"); cp = s.get("cost_repl_path")
                    if cf is None and cp is None:
                        return ""
                    return (cf or 0.0) + (cp or 0.0)
                return ""
            return v

        def _meta_row(label: str) -> dict:
            row = {"line": label, "group": "", "nominal_freq": (_get(nom, "objective") if label == "objective"
                                                                else _get(nom, "time") if label == "cost_time"
                                                                else _get(nom, "wait") if label == "cost_wait"
                                                                else _get(nom, "oper") if label == "cost_oper"
                                                                else "")}
            row["nominal_path"] = ""
            for sid, s in zip(scen_ids, scenarios):
                row[f"scenario {sid}_freq"] = (
                    _get(s, "prob")       if label == "prob" else
                    _get(s, "objective")  if label == "objective" else
                    _get(s, "cost_time")  if label == "cost_time" else
                    _get(s, "cost_wait")  if label == "cost_wait" else
                    _get(s, "cost_oper")  if label == "cost_oper" else
                    _get(s, "cost_repl")  if label == "cost_repl" else
                    ""
                )
                row[f"scenario {sid}_path"] = ""
            return row

        meta_rows = [
            _meta_row("prob"),
            _meta_row("objective"),
            _meta_row("cost_time"),
            _meta_row("cost_wait"),
            _meta_row("cost_oper"),
            _meta_row("cost_repl"),
        ]
        df_meta = pd.DataFrame(meta_rows, columns=cols)

        # --- Frequenzen + Pfade je Linie ------------------------------------
        freq_rows = []
        for ell in range(model.L):
            g = int(model.line_idx_to_group[ell])
            row = {
                "line": int(ell),
                "group": g,
                "nominal_freq": int(nominal.get(ell, 0)),
                "nominal_path": _fmt_nodes(nominal_path_nodes_for_line(ell)),
            }
            # WICHTIG: enumerate(scenarios) -> s_idx = interner Index, sid = (externe) Szenario-ID
            for s_idx, s in enumerate(scenarios):
                sid = scen_ids[s_idx]
                f = int((s.get("freq") or {}).get(ell, 0))
                row[f"scenario {sid}_freq"] = f
                # hier jetzt s_idx statt sid!
                row[f"scenario {sid}_path"] = _fmt_nodes(path_for_scenario_line(int(s_idx), ell))
            freq_rows.append(row)

        df_freq = pd.DataFrame(freq_rows, columns=cols)
        # --- schreiben -------------------------------------------------------
        df_out = pd.concat([df_meta, df_freq], ignore_index=True)
        path = os.path.join(self.out_dir, f"row_{int(run_index):03d}_freq.csv")
        df_out.to_csv(path, sep=';', index=False)
        return path
    
    def write_candidates_per_line(
        self,
        run_index: int,
        model,
        cand_all_lines: dict,                 # {s: {ell: [cand, ...]}}
        *,
        c_repl_line: float = 0.0,
        selected: dict | None = None,         # {s: {ell: k}}
    ) -> str:
        import pandas as pd, os

        idx_to_arc_uv = getattr(model, "idx_to_arc_uv")

        def nodes_from_arcs(arcs: list[int]) -> list[int]:
            if not arcs:
                return []
            u0, _ = idx_to_arc_uv[int(arcs[0])]
            seq = [int(u0)]
            for a in arcs:
                _, v = idx_to_arc_uv[int(a)]
                seq.append(int(v))
            return seq

        rows = []
        for s, per_line in (cand_all_lines or {}).items():
            chosen = (selected or {}).get(s, {})  # Dict[ell] -> k
            for ell, cand_list in (per_line or {}).items():
                sel_k = chosen.get(ell, None)

                if not cand_list:
                    # Optional: Dummy-Zeile, falls keine Kandidaten existieren
                    rows.append({
                        "run": int(run_index), "scenario": int(s), "line": int(ell),
                        "cand_id": "", "kind": "none",
                        "start_id": "", "end_id": "", "nodes": "", "arcs": "",
                        "path_len": "", "delta_len_vs_nom": "", "unit_repl_cost_per_freq": "",
                        "selected": 0, "is_nominal": "", "is_base": ""
                    })
                    continue

                for k, cand in enumerate(cand_list):
                    arcs = [int(a) for a in cand.get("arcs", [])]
                    nodes = nodes_from_arcs(arcs)
                    arcs_uv = ";".join(f"{u}->{v}" for (u, v) in (idx_to_arc_uv[a] for a in arcs))
                    nodes_id = ",".join(map(str, nodes))

                    path_len = float(cand.get("len", 0.0))
                    delta_len = float(
                        cand.get("delta_len_nom",
                                float(cand.get("add_len", 0.0)) + float(cand.get("rem_len", 0.0)))
                    )
                    unit_cost = delta_len * float(c_repl_line)

                    # Labels/Flags wenn vorhanden
                    is_nominal = bool(cand.get("is_nominal", False))
                    is_base    = bool(cand.get("is_base", False))
                    gen        = str(cand.get("gen", ""))  # "detour" | "ksp" | "" (optional)
                    kind = ("nominal" if is_nominal else
                            ("base" if is_base else str(cand.get("kind", gen or "alt"))))

                    rows.append({
                        "run": int(run_index),
                        "scenario": int(s),
                        "line": int(ell),
                        "cand_id": int(k),
                        "kind": kind,
                        "start_id": nodes[0] if nodes else "",
                        "end_id": nodes[-1] if nodes else "",
                        "nodes": nodes_id,
                        "arcs": arcs_uv,
                        "path_len": path_len,
                        "delta_len_vs_nom": delta_len,
                        "unit_repl_cost_per_freq": unit_cost,
                        "selected": 1 if sel_k == k else 0,
                    })

        df = pd.DataFrame(rows, columns=[
            "run","scenario","line","cand_id","kind","start_id","end_id",
            "nodes","arcs","path_len","delta_len_vs_nom","unit_repl_cost_per_freq",
            "selected","is_nominal","is_base"
        ])
        path = os.path.join(self.out_dir, f"candidates_run_{int(run_index):03d}.csv")
        df.to_csv(path, sep=';', index=False)
        return path
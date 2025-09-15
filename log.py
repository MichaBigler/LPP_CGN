#log.py
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
from typing import Dict, Any, Iterable, List, Optional
import pandas as pd
import csv

BASE_EXTRA_COLS = [
    "status_code","status","objective","runtime_s",
    "cost_time","cost_wait","cost_oper",
    "obj_stage1","obj_stage2_exp",
    "repl_cost_freq_exp","repl_cost_path_exp","repl_cost_exp",
]

def _arc_seq_to_nodes_and_arcs(model, arc_seq: List[int]) -> tuple[str, str, Optional[int], Optional[int]]:
        """Erzeuge (nodes_str, arcs_str, start_id, end_id) aus einer Arc-Sequenz."""
        if not arc_seq:
            return "", "", None, None
        # Original Stop-IDs aus dem Datensatz (nicht die indexbasierten)
        uv_pairs = [model.idx_to_arc_uv[a] for a in arc_seq]  # [(u_id, v_id), ...]
        node_ids = [uv_pairs[0][0]] + [v for (_, v) in uv_pairs]
        nodes_str = ",".join(str(n) for n in node_ids)
        arcs_str  = ";".join(f"{u}->{v}" for (u, v) in uv_pairs)
        return nodes_str, arcs_str, node_ids[0], node_ids[-1]

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

        def _get_selected_k(s_idx: int, ell: int) -> Optional[int]:
            """Liefert gewählten Kandidatenindex k – akzeptiert per-Line ODER per-Group Keys."""
            if not cand_selected:
                return None
            sel_s = cand_selected.get(s_idx, {})
            if ell in sel_s:               # per-line key
                return sel_s[ell]
            g = int(model.line_idx_to_group[ell])
            return sel_s.get(g, None)      # per-group fallback

        def _get_cand_list(s_idx: int, ell: int) -> List[Dict[str, Any]]:
            """Liefert Kandidatenliste – akzeptiert per-Line ODER per-Group Keys."""
            if not cand_all:
                return []
            per_s = cand_all.get(s_idx, {})
            if ell in per_s:                # per-line key
                return per_s.get(ell, [])
            g = int(model.line_idx_to_group[ell])
            return per_s.get(g, [])         # per-group fallback

        def path_for_scenario_line(s_idx: int, ell: int) -> list[int]:
            """Gewählter Kandidatenpfad je Szenario/Line; sonst nominal."""
            k = _get_selected_k(s_idx, ell)
            cand_list = _get_cand_list(s_idx, ell)
            if k is not None and 0 <= k < len(cand_list):
                arcs = list(map(int, cand_list[k].get("arcs", [])))
                nodes = nodes_from_arcs_ids(arcs)
                if nodes:
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
                row[f"scenario {sid}_path"] = _fmt_nodes(path_for_scenario_line(s_idx, ell))
            freq_rows.append(row)

        df_freq = pd.DataFrame(freq_rows, columns=cols)
        # --- schreiben -------------------------------------------------------
        df_out = pd.concat([df_meta, df_freq], ignore_index=True)
        path = os.path.join(self.out_dir, f"row_{int(run_index):03d}_freq.csv")
        df_out.to_csv(path, sep=';', index=False)
        return path
    
    def write_candidates(
        self,
        run_id: int,
        model,
        *,
        candidates_per_s: Dict[int, Dict[int, List[Dict[str, Any]]]],
        out_csv: str = None,
        c_repl_line: float = 0.0,
        selected: Optional[Dict[int, Dict[int, int]]] = None,   # {s: {ell or group: k}}
        freqs_per_s: Optional[List[Dict[int, int]]] = None,      # [ {ell: f}, ... ]
    ) -> str:
        import csv, os
        if out_csv is None:
            out_csv = os.path.join("Results", f"candidates_run{run_id}.csv")
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)

        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh, delimiter=";")   # ← Semikolon wie in den anderen Logs
            wr.writerow([
                "run","scenario","line","cand_id","kind",
                "start_id","end_id","nodes","arcs",
                "path_len","delta_len_vs_nom","unit_repl_cost_per_freq",
                "selected","freq"  # ← NEU
            ])

            for s, per_line in sorted(candidates_per_s.items()):
                freq_by_line = (freqs_per_s[s] if freqs_per_s and s < len(freqs_per_s) else {}) or {}
                sel_s = (selected.get(s, {}) if selected else {})

                for ell, cand_list in sorted(per_line.items()):
                    f_line = int(freq_by_line.get(ell, 0))  # Frequenz der Linie im Szenario s
                    for k, cand in enumerate(cand_list or []):
                        nodes_str, arcs_str, u0, v1 = _arc_seq_to_nodes_and_arcs(model, cand.get("arcs", []))
                        path_len = float(cand.get("len", 0.0))
                        delta    = float(cand.get("delta_len_nom", cand.get("add_len", 0.0) + cand.get("rem_len", 0.0)))
                        kind     = str(cand.get("kind", ""))
                        sel_flag = 1 if sel_s.get(ell, -1) == k else 0
                        unit_cost = float(c_repl_line) * float(delta)

                        # NUR der gewählte Kandidat bekommt die Frequenz, alle anderen 0
                        f_row = f_line if sel_flag == 1 else 0

                        wr.writerow([
                            run_id, s, ell, k, kind,
                            u0, v1, nodes_str, arcs_str,
                            path_len, delta, unit_cost,
                            sel_flag, f_row
                        ])
        return out_csv
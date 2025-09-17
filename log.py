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
     "cost_time","cost_time_base","cost_time_over","cost_bypass","cost_wait","cost_oper",
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
    
    def run_dir(self, run_index: int, *, name: str | None = None) -> str:
        """
        Liefert den Ordner für eine konkrete config-Zeile.
        Standard: Results/<stamp>/row_XXX/
        Optional 'name' könnte man später als Suffix nutzen (z.B. row_003_metroA).
        """
        dirname = f"row_{int(run_index):03d}" if not name else f"row_{int(run_index):03d}_{name}"
        path = os.path.join(self.out_dir, dirname)
        os.makedirs(path, exist_ok=True)
        return path
    
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
        rows = []
        for k, v in chosen_freq.items():
            try:
                rows.append({"line": int(k), "freq": int(v)})
            except Exception:
                continue
        rows.sort(key=lambda r: r["line"])
        df = pd.DataFrame(rows, columns=["line", "freq"])

        base_dir = self.run_dir(run_index)
        path = os.path.join(base_dir, "freq.csv")   # <-- jetzt im Run-Unterordner
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
            if label == "objective":
                nom_val = _get(nom, "objective")
            elif label == "cost_time":
                nom_val = _get(nom, "time")
            elif label == "cost_bypass":
                nom_val = _get(nom, "bypass")
            elif label == "cost_wait":
                nom_val = _get(nom, "wait")
            elif label == "cost_oper":
                nom_val = _get(nom, "oper")
            else:
                nom_val = ""

            row = {"line": label, "group": "", "nominal_freq": nom_val, "nominal_path": ""}

            for sid, s in zip(scen_ids, scenarios):
                row[f"scenario {sid}_freq"] = (
                    _get(s, "prob") if label == "prob" else
                    _get(s, "objective") if label == "objective" else
                    _get(s, "cost_time") if label == "cost_time" else
                    _get(s, "cost_bypass") if label == "cost_bypass" else
                    _get(s, "cost_wait") if label == "cost_wait" else
                    _get(s, "cost_oper") if label == "cost_oper" else
                    _get(s, "cost_repl") if label == "cost_repl" else
                    ""
                )
                row[f"scenario {sid}_path"] = ""
            return row

        meta_rows = [
            _meta_row("prob"),
            _meta_row("objective"),
            _meta_row("cost_time"),
            _meta_row("cost_bypass"),
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
        base_dir = self.run_dir(run_index)
        path = os.path.join(base_dir, "freq.csv")   # <-- jetzt im Run-Unterordner
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
        if out_csv is None:
            base_dir = self.run_dir(run_id)
            out_csv = os.path.join(base_dir, "candidates.csv")   # <-- jetzt im Run-Unterordner
        else:
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
    
    def write_edge_passenger_flows(
        self,
        run_index: int,
        model,
        cgn,
        x_vars,
        *,
        arc_to_keys: dict | None = None,
        filename_suffix: str = "",
    ) -> str:
        from collections import defaultdict
        import os, csv
        import numpy as np

        try:
            import gurobipy as _gp
        except Exception:
            _gp = None

        out_path = os.path.join(self.run_dir(run_index), f"edge_flows{filename_suffix}.csv")
        DEBUG = True

        # ---------- Helpers ----------
        def _val_of(v) -> float:
            if v is None:
                return 0.0
            # Gurobi Var/MVar
            if hasattr(v, "X"):
                # Var: skalar
                try:
                    return float(v.X)
                except Exception:
                    pass
                # MVar: ndarray
                try:
                    return float(np.sum(v.X))
                except Exception:
                    return 0.0
            # LinExpr
            if hasattr(v, "getValue"):
                try:
                    return float(v.getValue())
                except Exception:
                    return 0.0
            # NumPy array
            if isinstance(v, np.ndarray):
                try:
                    return float(np.sum(v))
                except Exception:
                    return 0.0
            # Container rekursiv
            if isinstance(v, dict):
                return sum(_val_of(x) for x in v.values())
            if isinstance(v, (list, tuple, set)):
                return sum(_val_of(x) for x in v)
            # Skalar
            try:
                return float(v)
            except Exception:
                return 0.0

        def _get_from_mvar_by_index(mv, idx) -> float:
            try:
                return _val_of(mv[int(idx)])
            except Exception:
                return 0.0

        def _sum_mvar_indices(mv, idxs) -> float:
            s = 0.0
            for i in idxs:
                s += _get_from_mvar_by_index(mv, i)
            return s

        def _get_from_x(loc):
            """
            loc kann sein:
            - direkter Var/MVar/LinExpr/Skalar
            - int-Index (für MVar oder list/tuple)
            - atomarer Key (inkl. tuple) in dict/tupledict
            - Container (list/tuple/set/dict) von obigen
            """
            if loc is None:
                return 0.0

            # Wenn loc schon ein Var/MVar/LinExpr ist:
            if hasattr(loc, "X") or hasattr(loc, "getValue"):
                return _val_of(loc)

            # Container der Locators – WICHTIG: tuple kann auch ein atomarer Key sein
            if isinstance(loc, dict):
                return sum(_get_from_x(v) for v in loc.values())

            # Zugriff per Index/Indices auf MVar
            if _gp is not None and isinstance(x_vars, getattr(_gp, "MVar", tuple())):
                if isinstance(loc, (int, np.integer)):
                    return _get_from_mvar_by_index(x_vars, loc)
                if isinstance(loc, (list, set)) and all(isinstance(i, (int, np.integer)) for i in loc):
                    return _sum_mvar_indices(x_vars, loc)
                if isinstance(loc, tuple):
                    # 1) Wenn das tuple atomarer Key in dict/tupledict ist (weiter unten)
                    # 2) Wenn es reine Indizes enthält -> als Liste von Indizes interpretieren
                    if all(isinstance(i, (int, np.integer)) for i in loc):
                        return _sum_mvar_indices(x_vars, loc)

            # list/tuple/set von Locators (allg. Fall)
            if isinstance(loc, (list, set)):
                return sum(_get_from_x(k) for k in loc)

            # Atomare tuple-Keys in dict/tupledict
            if isinstance(loc, tuple):
                # dict-ähnliche Strukturen
                if isinstance(x_vars, dict) and loc in x_vars:
                    return _val_of(x_vars[loc])
                # gurobipy.tupledict: hat __contains__ und __getitem__
                if hasattr(x_vars, "__contains__") and hasattr(x_vars, "__getitem__"):
                    try:
                        if loc in x_vars:
                            return _val_of(x_vars[loc])
                    except Exception:
                        pass
                # sonst als Container interpretieren (falls gemischte Locators)
                return sum(_get_from_x(k) for k in loc)

            # positionsbasierter Zugriff auf list/tuple
            if isinstance(x_vars, (list, tuple)) and isinstance(loc, (int, np.integer)):
                if 0 <= int(loc) < len(x_vars):
                    return _val_of(x_vars[int(loc)])
                return 0.0

            # dict / tupledict atomarer Key
            if isinstance(x_vars, dict):
                try:
                    v = x_vars.get(loc, None)
                    if v is not None:
                        return _val_of(v)
                except Exception:
                    pass
            if hasattr(x_vars, "__getitem__"):
                try:
                    v = x_vars[loc]
                    return _val_of(v)
                except Exception:
                    pass

            # Fallback
            return _val_of(loc)

        def _flow_for_arc(a: int) -> float:
            # 1) Mapping nutzen, wenn vorhanden
            if arc_to_keys is not None:
                if a in arc_to_keys:
                    return _get_from_x(arc_to_keys[a])
                try:
                    loc = arc_to_keys.get(a)
                    if loc is not None:
                        return _get_from_x(loc)
                except Exception:
                    pass
            # 2) Fallback: positionsbasierter Zugriff
            if isinstance(x_vars, (list, tuple)) and 0 <= a < len(x_vars):
                return _val_of(x_vars[a])
            if _gp is not None and isinstance(x_vars, getattr(_gp, "MVar", tuple())):
                return _get_from_mvar_by_index(x_vars, a)
            return 0.0

        # ---------- Sammeln ----------
        total_by_edge = defaultdict(float)
        by_edge_line  = defaultdict(lambda: defaultdict(float))
        len_a = getattr(model, "len_a", None)

        arc_edge = getattr(cgn, "arc_edge", None)
        arc_line = getattr(cgn, "arc_line", None)
        arc_kind = getattr(cgn, "arc_kind", None)
        A = len(arc_edge) if arc_edge is not None else 0

        if DEBUG:
            print(f"[edgeflow-debug] x_vars type={type(x_vars).__name__}, arc_to_keys type={type(arc_to_keys).__name__}")
            if arc_to_keys:
                # Zeige 5 Beispiel-Locators (Typ & kurzer Inhalt)
                cnt = 0
                for a, loc in arc_to_keys.items():
                    print(f"[edgeflow-debug] locator sample a={a}: type={type(loc).__name__}, repr={repr(loc)[:120]}")
                    cnt += 1
                    if cnt >= 5: break

            # Probe: Zähle nonzero aus Locators
            nz_probe, samples = 0, []
            if arc_to_keys:
                for a, loc in list(arc_to_keys.items())[:200]:
                    val = _get_from_x(loc)
                    if val > 1e-9:
                        nz_probe += 1
                        if len(samples) < 5:
                            samples.append((a, val))
            else:
                # Fallback: direkt aus x_vars ein paar Werte prüfen
                if isinstance(x_vars, dict):
                    for k, v in list(x_vars.items())[:200]:
                        val = _val_of(v)
                        if val > 1e-9:
                            nz_probe += 1; samples.append((k, val))
                            if len(samples) >= 5: break
                elif isinstance(x_vars, (list, tuple)):
                    for idx, v in enumerate(x_vars[:200]):
                        val = _val_of(v)
                        if val > 1e-9:
                            nz_probe += 1; samples.append((idx, val))
                            if len(samples) >= 5: break
                elif _gp is not None and isinstance(x_vars, getattr(_gp, "MVar", tuple())):
                    try:
                        arr = np.asarray(x_vars.X).ravel()
                        nz_idx = np.nonzero(arr)[0]
                        nz_probe = len(nz_idx)
                        for idx in nz_idx[:5]:
                            samples.append((int(idx), float(arr[idx])))
                    except Exception:
                        pass
            print(f"[edgeflow-debug] probe nonzero: {nz_probe}, sample: {samples}")

        # Ride-Arcs: arc_edge[a] >= 0
        for a in range(A):
            if arc_edge[a] is None:
                continue
            # optional: wenn arc_kind existiert, nur ride
            if arc_kind is not None and arc_kind[a] != "ride":
                continue
            e = int(arc_edge[a])
            if e < 0:
                continue
            ell = int(arc_line[a]) if arc_line is not None else -1
            val = _flow_for_arc(a)
            if val <= 0.0:
                continue
            total_by_edge[e] += val
            by_edge_line[e][ell] += val
            if DEBUG and total_by_edge[e] == val:
                print(f"[edgeflow-debug] first flow on edge {e}: arc {a}, line {ell}, val={val}")

        all_lines = sorted({ell for mp in by_edge_line.values() for ell in mp})

        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh, delimiter=";")
            wr.writerow(["edge_id", "edge_len", "flow_total"] + [f"flow_line_{ell}" for ell in all_lines])
            E = getattr(model, "E_dir", 0)
            for e in range(E):
                L = float(len_a[e]) if len_a is not None else 0.0
                tot = total_by_edge.get(e, 0.0)
                wr.writerow([e, L, tot] + [by_edge_line[e].get(ell, 0.0) for ell in all_lines])

        if DEBUG:
            tot_sum = sum(total_by_edge.values())
            print(f"[edgeflow-debug] wrote {out_path} with total flow sum = {tot_sum}")
        return out_path

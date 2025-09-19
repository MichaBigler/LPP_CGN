# log.py
# -*- coding: utf-8 -*-
"""
Logging utilities for LPP-CGN runs.

Creates a timestamped Results/<YYYY_MM_DD_HH_MM>/ folder and manages:
- base_log.csv (one row per run; config.csv columns + KPIs)
- row_XXX/freq.csv (per-run line frequencies and paths)
- row_XXX/candidates.csv (candidate paths, when applicable)
- row_XXX/edge_flows*.csv (passenger flows per infrastructure edge)

All CSVs use ';' as separator to match config.csv.
"""

from __future__ import annotations
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import csv
from collections import defaultdict


# ---------- Extra KPI columns that we append to base_log.csv ----------
BASE_EXTRA_COLS = [
    "status_code", "status", "objective", "runtime_s",
    "cost_time", "cost_time_base", "cost_time_over", "cost_bypass", "cost_wait", "cost_oper",
    "obj_stage1", "obj_stage2_exp",
    "repl_cost_freq_exp", "repl_cost_path_exp", "repl_cost_exp",
]


# ---------- Small helpers (pure formatting / mapping) ----------

def _arc_seq_to_nodes_and_arcs(model, arc_seq: List[int]) -> Tuple[str, str, Optional[int], Optional[int]]:
    """
    Convert a directed arc sequence (CGN arc indices) into:
      - nodes_str: "n0,n1,n2,..."
      - arcs_str : "u->v;u->v;..."
      - start_id, end_id (stop IDs), if available
    Uses model.idx_to_arc_uv (list of (u_id, v_id)) to resolve IDs.
    """
    if not arc_seq:
        return "", "", None, None
    uv_pairs = [model.idx_to_arc_uv[a] for a in arc_seq]  # [(u_id, v_id), ...]
    node_ids = [uv_pairs[0][0]] + [v for (_, v) in uv_pairs]
    nodes_str = ",".join(str(n) for n in node_ids)
    arcs_str  = ";".join(f"{u}->{v}" for (u, v) in uv_pairs)
    return nodes_str, arcs_str, node_ids[0], node_ids[-1]


def _expand_line_frequencies(model, freq_map: Dict[int, int] | None) -> Dict[int, int]:
    """
    Expand a frequency mapping that may be indexed by line or by group into a pure per-line mapping.
    Priority: per-line key wins; if missing, use its group's value; fallback to 0.
    """
    freq_map = freq_map or {}
    L = int(getattr(model, "L", 0))
    line_to_group = getattr(model, "line_idx_to_group", None)
    if L == 0 or line_to_group is None:
        # Nothing to expand; return numeric keys only
        return {int(k): int(v) for k, v in freq_map.items() if v is not None}
    out = {}
    for ell in range(L):
        g = int(line_to_group[ell])
        val = freq_map.get(ell, freq_map.get(g, 0))
        out[ell] = int(val)
    return out


def _nodes_from_arc_ids(model, arcs: List[int], *, use_stop_ids: bool = True) -> List[int]:
    """
    Convert CGN arc indices into node IDs sequence (n0..nk).
    Uses idx_to_arc_uv. If missing, returns [].
    """
    if not arcs:
        return []
    idx_to_arc_uv = getattr(model, "idx_to_arc_uv", None)
    if idx_to_arc_uv is None:
        return []
    u0, _ = idx_to_arc_uv[int(arcs[0])]
    seq = [int(u0)]
    for a in arcs:
        _, v = idx_to_arc_uv[int(a)]
        seq.append(int(v))
    return seq


# ---------- Main logger class ----------

class RunBatchLogger:
    """Creates run folders and writes all CSV logs for each config row."""

    def __init__(self, data_root: str, cfg_df: pd.DataFrame, *, stamp: str | None = None):
        self.data_root = data_root
        self.stamp = stamp or datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.out_dir = os.path.join(self.data_root, "Results", self.stamp)
        os.makedirs(self.out_dir, exist_ok=True)

        # Base log schema = all config.csv columns + KPI columns
        self.base_columns = list(cfg_df.columns) + BASE_EXTRA_COLS
        self.base_log_path = os.path.join(self.out_dir, "base_log.csv")

        # Initialize base_log.csv header once
        if not os.path.exists(self.base_log_path):
            pd.DataFrame(columns=self.base_columns).to_csv(self.base_log_path, sep=';', index=False)

    # ---------- directory helpers ----------

    def run_dir(self, run_index: int, *, name: str | None = None) -> str:
        """
        Returns the folder for a specific config row.
        Default: Results/<stamp>/row_XXX/
        Optional 'name' could be used later to suffix (e.g., row_003_metroA).
        """
        dirname = f"row_{int(run_index):03d}" if not name else f"row_{int(run_index):03d}_{name}"
        path = os.path.join(self.out_dir, dirname)
        os.makedirs(path, exist_ok=True)
        return path

    # ---------- base_log.csv ----------

    def base_row_template(self, cfg_row: pd.Series) -> Dict[str, Any]:
        """Prepare a dict with config columns from cfg_row and KPI columns set to None."""
        base = {k: cfg_row.get(k, None) for k in self.base_columns if k not in BASE_EXTRA_COLS}
        for k in BASE_EXTRA_COLS:
            base[k] = None
        return base

    def append_base_row(self, row_dict: Dict[str, Any]) -> None:
        """Append a single line to base_log.csv (flush immediately)."""
        df = pd.DataFrame([row_dict], columns=self.base_columns)
        df.to_csv(self.base_log_path, sep=';', index=False, mode='a', header=False)

    # ---------- frequencies & paths ----------

    def write_freq_file(self, run_index: int, chosen_freq: Dict[int, int] | Dict[str, int]) -> str:
        """
        Minimal frequency file (line;freq). Prefer write_freqs_two_stage for richer output.
        """
        rows = []
        for k, v in (chosen_freq or {}).items():
            try:
                rows.append({"line": int(k), "freq": int(v)})
            except Exception:
                continue
        rows.sort(key=lambda r: r["line"])
        df = pd.DataFrame(rows, columns=["line", "freq"])
        path = os.path.join(self.run_dir(run_index), "freq.csv")
        df.to_csv(path, sep=';', index=False)
        return path

    def write_freqs_two_stage(
        self,
        run_index: int,
        model,
        nominal: Dict[int, int],
        scenarios: List[Dict],
        nominal_costs: Dict[str, float] | None = None,
        *,
        cand_selected: Dict[int, Dict[int, int]] | None = None,   # {s: {ell or group: k}}
        cand_all: Dict[int, Dict[int, List[Dict]]] | None = None, # {s: {ell or group: [cand...]}}
        use_stop_ids: bool = True,
    ) -> str:
        """
        Wide frequency table with nominal and (optionally) scenario frequencies and paths.
        Top section adds meta rows: prob, objective, cost_time, cost_bypass, cost_wait, cost_oper, cost_repl.

        cand_selected / cand_all let us write the chosen candidate path per line+scenario.
        If unavailable, we fall back to the nominal path.
        """
        # Normalize nominal and scenario 'freq' to per-line mappings
        nominal = _expand_line_frequencies(model, nominal)
        scenarios = list(scenarios or [])
        for s in scenarios:
            s["freq"] = _expand_line_frequencies(model, s.get("freq", {}))

        # Helpers for node/arc rendering
        idx_to_node_id = getattr(model, "idx_to_node_id", None)

        def _to_id_list(node_idx_seq: List[int]) -> List[int]:
            if not node_idx_seq:
                return []
            if use_stop_ids and idx_to_node_id is not None:
                return [int(idx_to_node_id[i]) for i in node_idx_seq]
            return [int(i) for i in node_idx_seq]

        def nominal_path_nodes_for_line(ell: int) -> List[int]:
            seq_idx = list(map(int, model.line_idx_to_stops[ell]))
            return _to_id_list(seq_idx)

        def _get_selected_k(s_idx: int, ell: int) -> Optional[int]:
            """Return selected candidate index for (scenario s, line ell). Accept per-line or per-group keys."""
            if not cand_selected:
                return None
            sel_s = cand_selected.get(s_idx, {})
            if ell in sel_s:
                return sel_s[ell]
            g = int(model.line_idx_to_group[ell])
            return sel_s.get(g, None)

        def _get_cand_list(s_idx: int, ell: int) -> List[Dict[str, Any]]:
            """Return candidate list for (scenario s, line ell). Accept per-line or per-group keys."""
            if not cand_all:
                return []
            per_s = cand_all.get(s_idx, {})
            if ell in per_s:
                return per_s.get(ell, [])
            g = int(model.line_idx_to_group[ell])
            return per_s.get(g, [])

        def path_for_scenario_line(s_idx: int, ell: int) -> List[int]:
            """Chosen candidate path, else nominal."""
            k = _get_selected_k(s_idx, ell)
            cand_list = _get_cand_list(s_idx, ell)
            if k is not None and 0 <= k < len(cand_list):
                arcs = list(map(int, cand_list[k].get("arcs", [])))
                nodes = _nodes_from_arc_ids(model, arcs, use_stop_ids=use_stop_ids)
                if nodes:
                    return nodes
            return nominal_path_nodes_for_line(ell)

        def _fmt_nodes(seq: List[int]) -> str:
            return ",".join(map(str, seq))

        # Column layout
        scen_ids = [int(s["id"]) for s in scenarios]
        cols = ["line", "group", "nominal_freq", "nominal_path"]
        for sid in scen_ids:
            cols += [f"scenario {sid}_freq", f"scenario {sid}_path"]

        # Meta rows (top section)
        nom = nominal_costs or {}

        def _get(s: Dict | None, key: str):
            if s is None:
                return ""
            v = s.get(key)
            if v is None:
                if key == "cost_repl":
                    cf = s.get("cost_repl_freq"); cp = s.get("cost_repl_path")
                    if cf is None and cp is None:
                        return ""
                    return (cf or 0.0) + (cp or 0.0)
                return ""
            return v

        def _meta_row(label: str) -> Dict[str, Any]:
            nom_val = (
                _get(nom, "objective") if label == "objective" else
                _get(nom, "time")      if label == "cost_time" else
                _get(nom, "bypass")    if label == "cost_bypass" else
                _get(nom, "wait")      if label == "cost_wait" else
                _get(nom, "oper")      if label == "cost_oper" else
                ""
            )
            row = {"line": label, "group": "", "nominal_freq": nom_val, "nominal_path": ""}
            for sid, s in zip(scen_ids, scenarios):
                row[f"scenario {sid}_freq"] = (
                    _get(s, "prob")       if label == "prob" else
                    _get(s, "objective")  if label == "objective" else
                    _get(s, "cost_time")  if label == "cost_time" else
                    _get(s, "cost_bypass") if label == "cost_bypass" else
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
            _meta_row("cost_bypass"),
            _meta_row("cost_wait"),
            _meta_row("cost_oper"),
            _meta_row("cost_repl"),
        ]
        df_meta = pd.DataFrame(meta_rows, columns=cols)

        # Per-line block
        freq_rows = []
        for ell in range(model.L):
            g = int(model.line_idx_to_group[ell])
            row = {
                "line": int(ell),
                "group": g,
                "nominal_freq": int(nominal.get(ell, 0)),
                "nominal_path": _fmt_nodes(nominal_path_nodes_for_line(ell)),
            }
            for s_idx, s in enumerate(scenarios):
                sid = scen_ids[s_idx]
                f = int((s.get("freq") or {}).get(ell, 0))
                row[f"scenario {sid}_freq"] = f
                row[f"scenario {sid}_path"] = _fmt_nodes(path_for_scenario_line(s_idx, ell))
            freq_rows.append(row)

        df_freq = pd.DataFrame(freq_rows, columns=cols)

        # Write file
        df_out = pd.concat([df_meta, df_freq], ignore_index=True)
        path = os.path.join(self.run_dir(run_index), "freq.csv")
        df_out.to_csv(path, sep=';', index=False)
        return path

    # ---------- candidates ----------

    def write_candidates(
        self,
        run_id: int,
        model,
        *,
        candidates_per_s: Dict[int, Dict[int, List[Dict[str, Any]]]],
        out_csv: str = None,
        c_repl_line: float = 0.0,
        selected: Optional[Dict[int, Dict[int, int]]] = None,   # {s: {ell or group: k}}
        freqs_per_s: Optional[List[Dict[int, int]]] = None,     # [ {ell: f}, ... ]
    ) -> str:
        """
        Write one CSV with all candidate paths per scenario and line.
        Only the selected candidate (if any) gets the scenario line frequency in the 'freq' column.
        """
        if out_csv is None:
            out_csv = os.path.join(self.run_dir(run_id), "candidates.csv")
        else:
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)

        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh, delimiter=";")
            wr.writerow([
                "run", "scenario", "line", "cand_id", "kind",
                "start_id", "end_id", "nodes", "arcs",
                "path_len", "delta_len_vs_nom", "unit_repl_cost_per_freq",
                "selected", "freq"
            ])

            for s, per_line in sorted((candidates_per_s or {}).items()):
                freq_by_line = (freqs_per_s[s] if freqs_per_s and s < len(freqs_per_s) else {}) or {}
                sel_s = (selected.get(s, {}) if selected else {})

                for ell, cand_list in sorted((per_line or {}).items()):
                    f_line = int(freq_by_line.get(ell, 0))
                    for k, cand in enumerate(cand_list or []):
                        nodes_str, arcs_str, u0, v1 = _arc_seq_to_nodes_and_arcs(model, cand.get("arcs", []))
                        path_len = float(cand.get("len", 0.0))
                        delta    = float(cand.get("delta_len_nom", cand.get("add_len", 0.0) + cand.get("rem_len", 0.0)))
                        kind     = str(cand.get("kind", ""))
                        sel_flag = 1 if sel_s.get(ell, -1) == k else 0
                        unit_cost = float(c_repl_line) * float(delta)

                        # Only the selected candidate gets the frequency; others: 0
                        f_row = f_line if sel_flag == 1 else 0

                        wr.writerow([
                            run_id, s, ell, k, kind,
                            u0, v1, nodes_str, arcs_str,
                            path_len, delta, unit_cost,
                            sel_flag, f_row
                        ])
        return out_csv

    # ---------- passenger flows per infrastructure edge ----------

    def write_edge_passenger_flows(
        self,
        run_index: int,
        model,
        cgn,
        x_vars,
        *,
        arc_to_keys: dict | None = None,
        filename_suffix: str = "",
        aggregate_undirected: bool = False,
    ) -> str:
        """
        Write per-edge passenger flows:
          - edge_id: directed edge id (or undirected id if aggregate_undirected=True)
          - edge_len: length of the (directed) edge (or first-seen length for undirected aggregate)
          - flow_total: sum of flows over all lines on that edge
          - flow_line_<ell>: line-wise flow contribution

        Inputs:
          cgn.arc_kind[a] in {"ride","change","board","alight"}
          cgn.arc_edge[a] = directed infra-edge id (>=0 for 'ride', else -1)
          cgn.arc_line[a] = line index for 'ride' arcs
          x_vars: Gurobi tupledict / dict / list of flow variables/values for CGN arcs
          arc_to_keys: map CGN arc a -> key/tuple/list of keys inside x_vars if x_vars is indexed
        """
        out_path = os.path.join(self.run_dir(run_index), f"edge_flows{filename_suffix}.csv")

        # --- Read flow value for a CGN arc index `a` ------------------------
        def _safe_val(v) -> float:
            try:
                return float(getattr(v, "X", v))
            except Exception:
                return 0.0

        def _td_get(td, key):
            try:
                return td[key]
            except Exception:
                return None

        def _flow_on_arc(a: int) -> float:
            """
            Return scalar flow for CGN arc `a`, summing sub-keys when arc_to_keys[a] is a list.
            Supports:
              - tupledict: x_vars[(a, ...)] or x_vars[a] depending on formulation
              - dict/list: x_vars[a] or fallback variants
            """
            # tupledict / dict-like
            if hasattr(x_vars, "items") and hasattr(x_vars, "__getitem__"):
                if arc_to_keys is not None:
                    loc = arc_to_keys.get(a, None)
                    if isinstance(loc, list):
                        s = 0.0
                        for sub in loc:
                            key = sub if isinstance(sub, tuple) else (a, sub)
                            v = _td_get(x_vars, key)
                            s += _safe_val(v)
                        return s
                    if isinstance(loc, tuple):
                        v = _td_get(x_vars, loc)
                        return _safe_val(v)
                    if loc is not None:
                        # try (a, loc) and then plain a
                        v = _td_get(x_vars, (a, loc))
                        if v is not None:
                            return _safe_val(v)
                        v = _td_get(x_vars, a)
                        return _safe_val(v)
                # no mapping → try 1D and (a,0)
                v = _td_get(x_vars, a)
                if v is not None:
                    return _safe_val(v)
                v = _td_get(x_vars, (a, 0))
                return _safe_val(v)

            # list/tuple container
            if isinstance(x_vars, (list, tuple)):
                return _safe_val(x_vars[a]) if 0 <= a < len(x_vars) else 0.0

            # dict fallback
            if isinstance(x_vars, dict):
                return _safe_val(x_vars.get(a))

            return 0.0

        # --- Optional mapping to undirected edge IDs ------------------------
        # If aggregate_undirected=True and a mapping exists, flows are grouped per undirected edge.
        undir_map = None
        if aggregate_undirected:
            for name in (
                "edge_dir_to_undir_id", "dir_to_undir_edge_id",
                "e_dir_to_e_undir", "edir_to_eund", "undir_id_of_dir_edge"
            ):
                undir_map = getattr(model, name, None)
                if undir_map is not None:
                    break
        idx_to_uv_infra = getattr(model, "idx_to_arc_uv_infra", None) or getattr(model, "idx_to_arc_uv", None)

        def _undir_id(e_dir: int):
            if not aggregate_undirected:
                return int(e_dir)  # keep directed IDs (default behavior)
            if undir_map is not None:
                try:
                    return int(undir_map[e_dir])
                except Exception:
                    pass
            if idx_to_uv_infra is not None and 0 <= e_dir < len(idx_to_uv_infra):
                u, v = idx_to_uv_infra[e_dir]
                return f"{min(int(u), int(v))}-{max(int(u), int(v))}"
            return f"dir-{int(e_dir)}"

        # --- Aggregate flows -------------------------------------------------
        total_by_edge = defaultdict(float)                    # edge_id -> total flow
        by_edge_line  = defaultdict(lambda: defaultdict(float))  # edge_id -> line -> flow
        len_by_edge   = {}  # remembered length (directed length, or first-seen if undirected)

        len_a = getattr(model, "len_a", None)  # directed edge lengths
        A = len(cgn.arc_kind)

        for a in range(A):
            if cgn.arc_kind[a] != "ride":
                continue
            e_dir = int(cgn.arc_edge[a])
            if e_dir < 0:
                continue
            ell = int(cgn.arc_line[a])
            val = _flow_on_arc(a)
            if val == 0.0:
                continue

            e_key = _undir_id(e_dir)
            total_by_edge[e_key] += val
            by_edge_line[e_key][ell] += val

            if e_key not in len_by_edge:
                L = float(len_a[e_dir]) if (len_a is not None and 0 <= e_dir < len(len_a)) else 0.0
                len_by_edge[e_key] = L

        # --- Write CSV -------------------------------------------------------
        all_lines = sorted({ell for mp in by_edge_line.values() for ell in mp})
        out_path = os.path.join(self.run_dir(run_index), f"edge_flows{filename_suffix}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh, delimiter=";")
            wr.writerow(["edge_id", "edge_len", "flow_total"] + [f"flow_line_{ell}" for ell in all_lines])

            # keep stable ordering; works for int or str IDs
            for e_key in sorted(total_by_edge.keys(), key=lambda k: (isinstance(k, str), str(k))):
                row = [e_key, len_by_edge.get(e_key, 0.0), total_by_edge[e_key]] \
                      + [by_edge_line[e_key].get(ell, 0.0) for ell in all_lines]
                wr.writerow(row)

        return out_path

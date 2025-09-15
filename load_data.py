#load_data.py
# -*- coding: utf-8 -*-
"""
load_data.py
------------
Liest Netzwerk- und Szenariodateien ein und baut kompakte Modellstrukturen.

Eingang:
- data_root: Ordner, der <source>/<network>/ und Data/<scenario_line_data>/ enthält
- cfg_row:   eine Zeile aus Data/config.csv als dict; MUSS enthalten:
             source, network, scenario_line_data
  Optional in cfg_row:
             num_od (int; begrenzt Zahl der berücksichtigten OD-Paare nach Demand)

Ausgang:
- DomainData:  rohe eingelesene Tabellen/Listen + config
- ModelData:   indexierte, kompakte Arrays/CSR-Matrizen für das Optimierungsmodell
"""
from typing import Dict, Tuple, List
import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from data_model import LineDef, DomainData, ModelData, Config
from typing import Any, cast, Optional
from dataclasses import dataclass, asdict
from data_model import CandidateConfig

# ---------- Config-Parsing ----------

def _as_bool(x, default=False):
    if x is None: return default
    s = str(x).strip().lower()
    if s in ("1","true","yes","y"): return True
    if s in ("0","false","no","n"): return False
    return default

def _as_int(x, default=0):
    try:    return int(x)
    except: return default

def _as_float(x, default=0.0):
    try:    return float(x)
    except: return default

def _as_int_list(x) -> Optional[list]:
    if x is None: return None
    s = str(x).strip()
    if not s: return None
    parts = s.replace(",", " ").split()
    out = []
    for p in parts:
        try: out.append(int(p))
        except: pass
    return out or None

# ----------------------------- smmall Utils -----------------------------

def _must(path: str) -> str:
    """Fail fast if file/dir doesn't exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path

def _cfg_key(d: dict, name: str) -> str:
    """Case-insensitive Zugriff auf Spaltennamen in cfg_row."""
    for k in d.keys():
        if k.lower() == name.lower():
            return k
    raise KeyError(f"Missing '{name}' in config row")

def _cfg_get_int(d: dict, name: str, default: int = 0) -> int:
    """Case-insensitive int aus cfg_row lesen (mit Default)."""
    for k in d.keys():
        if k.lower() == name.lower():
            try:
                return int(d[k])
            except Exception:
                return default
    return default


# -------------------------- Datei-Leser (roh) ---------------------------

def read_stop_giv(path: str) -> pd.DataFrame:
    """Stop.giv: id;short;long;x;y"""
    df = pd.read_csv(_must(path), sep=';', comment='#', header=None,
                     names=['id', 'short', 'long', 'x', 'y'])
    return df.astype({'id': int, 'short': str, 'long': str, 'x': float, 'y': float})

def read_edge_giv(path: str) -> pd.DataFrame:
    """Edge.giv: id;a;b;length;t_min;t_max  (ungeri chtetes Netz, später zu gerichteten Arcs dupliziert)"""
    df = pd.read_csv(_must(path), sep=';', comment='#', header=None,
                     names=['id', 'a', 'b', 'length', 't_min', 't_max'])
    return df.astype({'id': int, 'a': int, 'b': int, 'length': float, 't_min': float, 't_max': float})

def read_od_giv(path: str) -> pd.DataFrame:
    """OD.giv: i;j;demand"""
    df = pd.read_csv(_must(path), sep=';', comment='#', header=None,
                     names=['i', 'j', 'demand'])
    return df.astype({'i': int, 'j': int, 'demand': float})

def read_lines_csv(path: str) -> List[LineDef]:
    """
    lines.csv: erwartet mind. Spalten: property, line_group, value_1
    - property == 'line' (eine oder zwei Zeilen je Gruppe)
    - value_1 = kommaseparierte Stop-IDs (z. B. '1,2,3')
    Gibt eine Liste LineDef(group, direction, stops) zurück.
    """
    raw = pd.read_csv(_must(path), sep=';')
    raw = raw[raw['property'].str.lower() == 'line']

    def parse_seq(val: Any) -> List[int]:
        parts = [s.strip() for s in str(val).split(',') if s and s.strip() != '']
        return [int(s) for s in parts]

    out: List[LineDef] = []
    for g_obj, grp in raw.groupby('line_group', sort=True):
        group_id: int = int(cast(Any, g_obj))  # sauberer Cast für Pylance
        rows = grp.sort_index()
        seqs: List[List[int]] = [parse_seq(rows.iloc[k]['value_1']) for k in range(len(rows))]
        if len(seqs) == 1:
            out.append(LineDef(group_id, +1, seqs[0]))
        else:
            out.append(LineDef(group_id, +1, seqs[0]))
            out.append(LineDef(group_id, -1, seqs[1]))
    return out

def parse_config_row(cfg_row: dict) -> Config:
    # Pflichtfelder (case-insensitive Zugriff über _cfg_key)
    src  = str(cfg_row[_cfg_key(cfg_row, 'source')])
    net  = str(cfg_row[_cfg_key(cfg_row, 'network')])
    scen = str(cfg_row[_cfg_key(cfg_row, 'scenario_line_data')])

    cand_default = _as_int(cfg_row.get('cand_count'), 0)

    return Config(
        source=src,
        network=net,
        scenario_line_data=scen,
        procedure=cfg_row.get('procedure'),

        optimize_lines=_as_bool(cfg_row.get('optimize_lines'), False),
        routing_agg=_as_bool(cfg_row.get('routing_agg'), False),
        eliminate_subtours=_as_bool(cfg_row.get('eliminate_subtours'), False),
        line_repl_allowed=_as_bool(cfg_row.get('line_repl_allowed'), False),

        waiting_time_frequency=_as_bool(cfg_row.get('waiting_time_frequency'), True),

        gap=_as_float(cfg_row.get('gap'), 0.0),
        travel_time_cost_mult=_as_float(cfg_row.get('travel_time_cost_mult'), 1.0),
        waiting_time_cost_mult=_as_float(cfg_row.get('waiting_time_cost_mult'), 1.0),
        line_operation_cost_mult=_as_float(cfg_row.get('line_operation_cost_mult'), 1.0),

        num_od=_as_int(cfg_row.get('num_od'), 0),
        train_capacity=_as_int(cfg_row.get('train_capacity'), 200),
        infrastructure_capacity=_as_int(cfg_row.get('infrastructure_capacity'), 10),
        max_frequency=_as_int(cfg_row.get('max_frequency'), 5),
        num_scenarios=_as_int(cfg_row.get('num_scenarios'), 1),

        cost_repl_freq=_as_float(cfg_row.get('cost_repl_freq'), 0.0),
        cost_repl_line=_as_float(cfg_row.get('cost_repl_line'), 0.0),
        repl_budget=_as_float(cfg_row.get('repl_budget'), 0.0),

        freq_values=_as_int_list(cfg_row.get('freq_values')),

        cand_detour_count=_as_int(cfg_row.get('cand_detour_count'), cand_default),
        cand_ksp_count=_as_int(cfg_row.get('cand_ksp_count'), cand_default),
    )

def read_scenario_prob_csv(path: str) -> pd.DataFrame:
    """
    scenario_prob.csv: erwartet Zeilen mit property == 'scenario'
    und Spalten value_1 (id), value_2 (prob)
    -> DataFrame mit Spalten: id:int, prob:float
    """
    df = pd.read_csv(_must(path), sep=';')
    df = df[df['property'].str.lower() == 'scenario'][['value_1', 'value_2']]
    df = df.rename(columns={'value_1': 'id', 'value_2': 'prob'})
    return df.astype({'id': int, 'prob': float})

def read_scenario_infra_csv(path: str) -> pd.DataFrame:
    """
    scenario_infra.csv: Spalten u=left-stop, v=right-stop, cap=infrastructure_capacity, scenario
    -> DataFrame: scenario:int, u:int, v:int, cap:int
    """
    df = pd.read_csv(_must(path), sep=';')
    df = df.rename(columns={'left-stop': 'u', 'right-stop': 'v', 'infrastructure_capacity': 'cap'})
    return df.astype({'scenario': int, 'u': int, 'v': int, 'cap': int})


# ----------------------- kompakte Build-Helfer --------------------------

def build_node_indexing(stops_df: pd.DataFrame):
    """Mappe Stop-IDs -> dichte 0..N-1 Indizes + inverse Liste."""
    node_ids = stops_df['id'].astype(int).tolist()
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    idx_to_node_id = node_ids[:]  # inverse
    return node_id_to_idx, idx_to_node_id, len(node_ids)

def build_directed_arcs(links_df: pd.DataFrame):
    """Dupliziere ungerichtete Kanten in gerichtete Arcs; sammle Längen/Zeitfenster."""
    idx_to_arc_uv: List[tuple] = []
    len_list: List[float] = []; tmin_list: List[float] = []; tmax_list: List[float] = []
    for _, r in links_df.iterrows():
        a, b = int(r['a']), int(r['b'])
        for (u, v) in ((a, b), (b, a)):
            idx_to_arc_uv.append((u, v))
            len_list.append(float(r['length']))
            tmin_list.append(float(r['t_min']))
            tmax_list.append(float(r['t_max']))
    arc_uv_to_idx = {uv: i for i, uv in enumerate(idx_to_arc_uv)}
    E_dir = len(idx_to_arc_uv)
    return (np.array(len_list), np.array(tmin_list), np.array(tmax_list),
            idx_to_arc_uv, arc_uv_to_idx, E_dir)

def build_coordinates(stops_df: pd.DataFrame, node_id_to_idx: Dict[int, int], N: int):
    """Koordinaten als dichte Arrays (Indexraum 0..N-1)."""
    coord_x = np.zeros(N); coord_y = np.zeros(N)
    for _, r in stops_df.iterrows():
        i = node_id_to_idx[int(r['id'])]
        coord_x[i] = float(r['x']); coord_y[i] = float(r['y'])
    return coord_x, coord_y

def build_od_matrix(od_df: pd.DataFrame, node_id_to_idx: Dict[int, int], N: int, zero_od_diagonal: bool = False):
    """OD-Matrix D[i,j] (dicht)."""
    D = np.zeros((N, N))
    for _, r in od_df.iterrows():
        i = node_id_to_idx[int(r['i'])]; j = node_id_to_idx[int(r['j'])]
        D[i, j] = float(r['demand'])
    if zero_od_diagonal:
        np.fill_diagonal(D, 0.0)
    return D

def build_scenarios(scen_prob_df: pd.DataFrame):
    """Szenario-Wahrscheinlichkeiten + id->index Map."""
    scen_ids = scen_prob_df['id'].astype(int).tolist()
    p_s = scen_prob_df['prob'].astype(float).to_numpy()
    total = float(p_s.sum())
    if not (0.0 < total):
        raise ValueError("scenario_prob: Sum of probabilities must be > 0.")
    # normalise if needed (tolerant)
    if abs(total - 1.0) > 1e-9:
        p_s = p_s / total
    scen_id_to_idx = {sid: s for s, sid in enumerate(scen_ids)}
    return p_s, scen_id_to_idx

def build_scenario_capacities(
    scen_infra_df: pd.DataFrame,
    scen_id_to_idx: Dict[int, int],
    arc_uv_to_idx: Dict[tuple, int],
    E_dir: int,
    cap_std: int = 10,
    symmetrise_infra: bool = False
):
    """
    Kapazität je (Szenario, gerichteter Arc). Startwert = cap_std (aus config),
    wird durch scenario_infra.csv überschrieben, wenn vorhanden.
    """
    S = len(scen_id_to_idx)
    cap_sa = np.full((S, E_dir), float(cap_std), dtype=np.float64)
    for _, r in scen_infra_df.iterrows():
        s = scen_id_to_idx[int(r['scenario'])]
        u, v, cap = int(r['u']), int(r['v']), float(r['cap'])
        if (u, v) in arc_uv_to_idx:
            cap_sa[s, arc_uv_to_idx[(v, u)]] = float(cap)
        if symmetrise_infra and (v, u) in arc_uv_to_idx:
            cap_sa[s, arc_uv_to_idx[(v, u)]] = float(cap)
    cap_sa = np.maximum(cap_sa, 0.0)
    return cap_sa

def build_lines(
    lines: List[LineDef],
    node_id_to_idx: Dict[int, int],
    arc_uv_to_idx: Dict[tuple, int]
):
    """Erweitere Liniendefinitionen in Ketten aus Knoten-/Arc-Indizes; gruppiere Vor/Rückrichtung."""
    line_idx_to_group: List[int] = []
    line_group_to_lines: Dict[int, tuple] = {}
    line_idx_to_stops: List[List[int]] = []
    line_idx_to_arcs:  List[List[int]] = []

    # nach Gruppe sammeln
    by_group: Dict[int, List[LineDef]] = {}
    for ld in lines:
        by_group.setdefault(ld.group, []).append(ld)

    ell = 0
    for g, items in sorted(by_group.items()):
        # +1 zuerst, dann -1
        dir_order = sorted(items, key=lambda t: -t.direction)
        ids_here: List[int] = []
        for ld in dir_order:
            stops_idx = [node_id_to_idx[sid] for sid in ld.stops]
            arcs_idx: List[int] = []
            for u_id, v_id in zip(ld.stops[:-1], ld.stops[1:]):
                if (u_id, v_id) not in arc_uv_to_idx:
                    raise ValueError(f"Line group {g} missing link {(u_id, v_id)}")
                arcs_idx.append(arc_uv_to_idx[(u_id, v_id)])
            line_idx_to_group.append(g)
            line_idx_to_stops.append(stops_idx)
            line_idx_to_arcs.append(arcs_idx)
            ids_here.append(ell); ell += 1
        line_group_to_lines[g] = (ids_here[0], ids_here[1] if len(ids_here) > 1 else -1)

    L = len(line_idx_to_group)
    return line_idx_to_group, line_group_to_lines, line_idx_to_stops, line_idx_to_arcs, L

def build_adjacency(idx_to_arc_uv: List[tuple], node_id_to_idx: Dict[int, int], N: int):
    """Adjazenzlisten (ein-/ausgehende Arc-IDs je physischem Knoten)."""
    adj_out: List[List[int]] = [[] for _ in range(N)]
    adj_in:  List[List[int]] = [[] for _ in range(N)]
    for a_idx, (u, v) in enumerate(idx_to_arc_uv):
        ui, vi = node_id_to_idx[u], node_id_to_idx[v]
        adj_out[ui].append(a_idx); adj_in[vi].append(a_idx)
    return adj_out, adj_in

def build_incidence_matrices(
    line_idx_to_arcs: List[List[int]],
    line_idx_to_stops: List[List[int]],
    E_dir: int, L: int, N: int
):
    """CSR-Inzidenzmatrizen: Edge–Line (E×L), Node–Line (N×L)."""
    # Edge–Line
    rows, cols, data = [], [], []
    for ell_idx, arc_list in enumerate(line_idx_to_arcs):
        for a in arc_list:
            rows.append(a); cols.append(ell_idx); data.append(1.0)
    A_edge_line = csr_matrix((data, (rows, cols)), shape=(E_dir, L))
    # Node–Line
    rows, cols, data = [], [], []
    for ell_idx, stops in enumerate(line_idx_to_stops):
        for i in stops:
            rows.append(i); cols.append(ell_idx); data.append(1.0)
    A_node_line = csr_matrix((data, (rows, cols)), shape=(N, L))
    return A_edge_line, A_node_line


# --------------------------- Top-Level Loader ---------------------------

def load_and_build(
    data_root: str,
    cfg_row: dict,
    *,
    symmetrise_infra: bool = False,
    zero_od_diagonal: bool = False
) -> Tuple[DomainData, ModelData]:
    """
    Liest alle Rohdateien ein, optional beschneidet OD-Paare (num_od),
    und baut daraus die kompakten Modelldaten.
    """

    # 1) komplette config-Zeile robust parsen
    cfg = parse_config_row(cfg_row)

    # 2) Pfade aus der typisierten Config
    net_dir  = os.path.join(data_root, cfg.source, cfg.network)
    scen_dir = os.path.join(data_root, "Data", cfg.scenario_line_data)

    # ---- Rohdaten lesen (genau die gewünschten Dateien) ----
    stops_df = read_stop_giv(os.path.join(net_dir, "Stop.giv"))
    links_df = read_edge_giv(os.path.join(net_dir, "Edge.giv"))
    od_df    = read_od_giv(  os.path.join(net_dir, "OD.giv"))

    lines    = read_lines_csv(         os.path.join(scen_dir, "lines.csv"))
    scen_p   = read_scenario_prob_csv( os.path.join(scen_dir, "scenario_prob.csv"))
    scen_i   = read_scenario_infra_csv(os.path.join(scen_dir, "scenario_infra.csv"))

    # 4) OD-Kürzung
    if cfg.num_od and cfg.num_od > 0:
        cand = od_df[od_df['demand'] > 0].copy()
        before = len(cand)
        cand = cand.sort_values('demand', ascending=False)
        od_df = cand.head(cfg.num_od).copy()
        print(f"[od-limit] kept {len(od_df)} of {before} positive-demand ODs (num_od={cfg.num_od}).")

    # 5) DomainData mit *vollständiger* (typisierter) Config
    domain = DomainData(
        stops_df=stops_df,
        links_df=links_df,
        od_df=od_df,
        lines=lines,
        include_sets={},
        scen_prob_df=scen_p,
        scen_infra_df=scen_i,
        props={},                 # properties_general fällt weg
        config=cfg.to_dict()      # ganze, bereits typisierte Row
    )

    # 6) Kompaktstrukturen bauen (unverändert, aber cap_std aus cfg)
    node_id_to_idx, idx_to_node_id, N = build_node_indexing(domain.stops_df)
    (len_a, t_min_a, t_max_a, idx_to_arc_uv, arc_uv_to_idx, E_dir) = build_directed_arcs(domain.links_df)
    coord_x, coord_y = build_coordinates(domain.stops_df, node_id_to_idx, N)
    D = build_od_matrix(domain.od_df, node_id_to_idx, N, zero_od_diagonal=zero_od_diagonal)
    p_s, scen_id_to_idx = build_scenarios(domain.scen_prob_df)
    cap_sa = build_scenario_capacities(
        domain.scen_infra_df, scen_id_to_idx, arc_uv_to_idx, E_dir,
        cap_std=cfg.infrastructure_capacity,  # <- aus Config
        symmetrise_infra=symmetrise_infra
    )
    (line_idx_to_group, line_group_to_lines, line_idx_to_stops, line_idx_to_arcs, L) = \
        build_lines(domain.lines, node_id_to_idx, arc_uv_to_idx)
    adj_out, adj_in = build_adjacency(idx_to_arc_uv, node_id_to_idx, N)
    A_edge_line, A_node_line = build_incidence_matrices(line_idx_to_arcs, line_idx_to_stops, E_dir, L, N)

    model = ModelData(
        N=N, E_dir=E_dir, L=L, S=len(p_s),
        node_id_to_idx=node_id_to_idx, idx_to_node_id=idx_to_node_id,
        arc_uv_to_idx=arc_uv_to_idx, idx_to_arc_uv=idx_to_arc_uv,
        line_idx_to_group=line_idx_to_group, line_group_to_lines=line_group_to_lines,
        line_idx_to_stops=line_idx_to_stops, line_idx_to_arcs=line_idx_to_arcs,
        coord_x=coord_x, coord_y=coord_y,
        len_a=len_a, t_min_a=t_min_a, t_max_a=t_max_a,
        D=D, p_s=p_s, cap_sa=cap_sa,
        exclude_nodes=set(),
        include_sets=domain.include_sets,
        adj_out=adj_out, adj_in=adj_in,
        A_edge_line=A_edge_line, A_node_line=A_node_line
    )
    return domain, model

# load_candidate_config.py


_BOOL_TRUE = {"1","true","y","yes"}
def _as_bool(x, default=False):
    if x is None: return default
    s = str(x).strip().lower()
    return s in _BOOL_TRUE if s in _BOOL_TRUE|{"0","false","n","no"} else default

def load_candidate_config(data_root: str) -> CandidateConfig:
    path = os.path.join(data_root, "Data", "config_candidates.csv")
    if not os.path.exists(path):
        return CandidateConfig()  # Defaults

    df = pd.read_csv(path, sep=';')
    if df.empty:
        return CandidateConfig()

    row = df.iloc[0].to_dict()
    return CandidateConfig(
        k_loc_detour          = int(row.get("k_loc_detour", 3)),
        k_sp_global           = int(row.get("k_sp_global", 8)),
        max_candidates_per_line = int(row.get("max_candidates_per_line", 20)),
        div_min_edges         = int(row.get("div_min_edges", 1)),
        w_len                 = (None if pd.isna(row.get("w_len", None)) else float(row.get("w_len"))),
        w_repl                = (None if pd.isna(row.get("w_repl", None)) else float(row.get("w_repl"))),
        corr_eps              = float(row.get("corr_eps", 0.25)),
        generate_only_if_disrupted = _as_bool(row.get("generate_only_if_disrupted", True), True),
        mirror_backward       = str(row.get("mirror_backward", "auto")).strip().lower(),
    )

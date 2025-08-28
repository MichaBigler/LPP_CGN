from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

@dataclass(frozen=True)
class LineDef:
    group: int
    direction: int  # +1 or -1
    stops: List[int]  # original stop IDs (as in files), ordered

@dataclass
class DomainData:
    stops_df: pd.DataFrame
    links_df: pd.DataFrame   # undirected links
    od_df: pd.DataFrame
    lines: List[LineDef]
    include_sets: Dict[int, Set[int]]
    scen_prob_df: pd.DataFrame
    scen_infra_df: pd.DataFrame
    props: dict
    config: dict

@dataclass
class ModelData:
    # Index sets and mappings
    N: int
    E_dir: int
    L: int
    S: int
    node_id_to_idx: Dict[int, int]
    idx_to_node_id: List[int]
    arc_uv_to_idx: Dict[Tuple[int,int], int]
    idx_to_arc_uv: List[Tuple[int,int]]
    line_idx_to_group: List[int]
    line_group_to_lines: Dict[int, Tuple[int,int]]
    line_idx_to_stops: List[List[int]]  # node indices
    line_idx_to_arcs: List[List[int]]   # arc indices
    # Parameters
    coord_x: np.ndarray
    coord_y: np.ndarray
    len_a: np.ndarray
    t_min_a: np.ndarray
    t_max_a: np.ndarray
    D: np.ndarray
    p_s: np.ndarray
    cap_sa: np.ndarray
    exclude_nodes: Set[int]
    include_sets: Dict[int, Set[int]]
    # Incidence / adjacency
    adj_out: List[List[int]]
    adj_in: List[List[int]]
    A_edge_line: csr_matrix
    A_node_line: csr_matrix
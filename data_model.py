from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Tuple, Optional, Literal
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
class Config:
    # Pflichtpfade
    source: str
    network: str
    scenario_line_data: str

    # Schalter/Parameter aus deiner neuen config.csv
    procedure: Optional[str] = None
    optimize_lines: bool = False
    routing_agg: bool = False
    eliminate_subtours: bool = False
    line_repl_allowed: bool = False

    # Wartezeit: Frequenz-abhängig (True) vs. pauschal (False)
    waiting_time_frequency: bool = True

    # Solver/Weights
    gap: float = 0.0
    travel_time_cost_mult: float = 1.0
    waiting_time_cost_mult: float = 1.0
    line_operation_cost_mult: float = 1.0

    # Ressourcen/Größen
    num_od: int = 0
    train_capacity: int = 200
    infrastructure_capacity: int = 10
    max_frequency: int = 5
    num_scenarios: int = 1

    # (noch unbenutzt, aber mitgenommen)
    cost_repl_freq: float = 0.0
    cost_repl_line: float = 0.0
    repl_budget: float = 0.0

    


    # optional: Frequenzwerte explizit (z. B. "1,2,3,4,6")
    freq_values: Optional[List[int]] = None

    cand_detour_count: int = 0
    cand_ksp_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class CandidateConfig:
    # Mengensteuerung
    k_loc_detour: int = 3
    k_sp_global: int = 8
    max_candidates_per_line: int = 20
    div_min_edges: int = 1

    # Kosten-Gewichte für Ranking (None => aus Haupt-Config spiegeln)
    w_len: Optional[float] = None
    w_repl: Optional[float] = None

    # Korridor (zulässige Längenausweitung ggü. Referenz)
    corr_eps: float = 0.25

    # wann generieren & Richtungsspiegelung
    generate_only_if_disrupted: bool = True
    mirror_backward: str = "auto"   # "auto" | "force" | "off"

    def resolve_weights(self, main_cfg: Config) -> Tuple[float, float]:
        """Gewichte auflösen (falls None → aus main config spiegeln)."""
        w_len = self.w_len if self.w_len is not None else float(main_cfg.line_operation_cost_mult)
        w_repl = self.w_repl if self.w_repl is not None else float(main_cfg.cost_repl_line)
        return float(w_len), float(w_repl)

    def to_dict(self) -> dict:
        return asdict(self)

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


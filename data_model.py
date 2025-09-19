# data_model.py
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Tuple, Optional, Literal
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


# -----------------------------------------------------------------------------
# Basic line definition (input-side, human-friendly)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class LineDef:
    """Single nominal line description coming from input data.

    Attributes:
        group: Logical group id (e.g., forward/backward pair share a group).
        direction: +1 or -1. Semantic is up to the loader (e.g., forward/backward).
        stops: Ordered list of original stop IDs (as they appear in the raw files).
    """
    group: int
    direction: int  # +1 or -1
    stops: List[int]  # original stop IDs (as in files), ordered


# -----------------------------------------------------------------------------
# All raw domain inputs bundled (before modeling/graph expansion)
# -----------------------------------------------------------------------------
@dataclass
class DomainData:
    """Container holding raw input tables and configuration for a run.

    DataFrames should contain the raw IDs/values as parsed from disk.
    The modeling code is responsible for deriving index-based arrays.

    Attributes:
        stops_df: Table of stops (original IDs, coordinates, etc.).
        links_df: Undirected links at the infrastructure level.
        od_df: OD matrix/records (demand per origin-destination).
        lines: List of nominal lines (LineDef).
        include_sets: Optional "allowed edges per group" constraints (by group id).
        scen_prob_df: Scenario id/probability mapping.
        scen_infra_df: Scenario-dependent infrastructure info (capacities/closures).
        props: Free-form metadata.
        config: Raw config dict (values from config.csv).
    """
    stops_df: pd.DataFrame
    links_df: pd.DataFrame   # undirected links
    od_df: pd.DataFrame
    lines: List[LineDef]
    include_sets: Dict[int, Set[int]]
    scen_prob_df: pd.DataFrame
    scen_infra_df: pd.DataFrame
    props: dict
    config: dict


# -----------------------------------------------------------------------------
# Flat configuration exposed to the solver (values from config.csv)
# -----------------------------------------------------------------------------
@dataclass
class Config:
    """Run-time configuration parameters.

    Notes:
        - Booleans default to conservative behavior (off) unless stated.
        - Frequencies can be provided explicitly via freq_values; otherwise we
          use a standard range up to max_frequency in the solver layer.
    """

    # Required identifiers (used to locate data on disk)
    source: str
    network: str
    scenario_line_data: str

    # Procedure / switches
    procedure: Optional[str] = None                 # 'one', 'integrated', 'separated', ...
    optimize_lines: bool = False                    # if False: keep nominal lines
    routing_agg: bool = False                       # aggregated routing (faster; less detail)
    eliminate_subtours: bool = False                # optional routing sanity constraint
    line_repl_allowed: bool = False                 # allow line path replacements (stage-2)

    # Waiting time model: if True → frequency-dependent waiting, else flat waiting
    waiting_time_frequency: bool = True

    # Solver / weighting coefficients
    gap: float = 0.0
    travel_time_cost_mult: float = 1.0
    waiting_time_cost_mult: float = 1.0
    line_operation_cost_mult: float = 1.0

    # Resource limits / sizes
    num_od: int = 0
    train_capacity: int = 200
    infrastructure_capacity: int = 10
    max_frequency: int = 5
    num_scenarios: int = 1

    # Replanning costs/budgets (used in two-stage variants)
    cost_repl_freq: float = 0.0
    cost_repl_line: float = 0.0
    repl_budget: float = 0.0

    # Bypass disutility: <0 disables; >=0 enables (cost = length * multiplier)
    bypass_multiplier: float = -1.0

    # Over-demand travel-time inflation (>= threshold share gets multiplier)
    overdemand_threshold: float = 1.0
    overdemand_multiplier: float = 1.0

    # Optional: explicit frequency values (e.g., [1,2,3,4,6])
    freq_values: Optional[List[int]] = None

    # Candidate generation knobs
    cand_detour_count: int = 0
    cand_ksp_count: int = 0

    def to_dict(self) -> dict:
        """Return a plain dictionary (useful for logging/serialization)."""
        return asdict(self)


# -----------------------------------------------------------------------------
# Candidate path generation parameters (pre-solver heuristic)
# -----------------------------------------------------------------------------
@dataclass
class CandidateConfig:
    """Parameters that control candidate path generation and ranking.

    Attributes:
        k_loc_detour: Max number of local detours per line (per disrupted segment).
        k_sp_global: Max number of global k-shortest paths considered.
        max_candidates_per_line: Hard cap per line.
        div_min_edges: Diversity minimum (distinct edges) between candidates.

        w_len: Weight for path length in ranking (None → mirror line_operation_cost_mult).
        w_repl: Weight for replacement cost in ranking (None → mirror cost_repl_line).

        corr_eps: Length corridor: candidate length ≤ (1 + corr_eps) * reference length.

        generate_only_if_disrupted: If True, generate only for disrupted lines.
        mirror_backward: "auto" = mirror reverse direction if meaningful,
                         "force" = always mirror,
                         "off"   = never mirror.
    """
    # Cardinalities / limits
    k_loc_detour: int = 3
    k_sp_global: int = 8
    max_candidates_per_line: int = 20
    div_min_edges: int = 1

    # Ranking weights (None => mirror from main Config)
    w_len: Optional[float] = None          # mirrors line_operation_cost_mult
    w_repl: Optional[float] = None         # mirrors cost_repl_line

    # Corridor for candidate path length
    corr_eps: float = 0.25

    # Generation policies
    generate_only_if_disrupted: bool = True
    mirror_backward: Literal["auto", "force", "off"] = "auto"

    def resolve_weights(self, main_cfg: "Config") -> Tuple[float, float]:
        """Resolve ranking weights, falling back to main Config when unset."""
        w_len = self.w_len if self.w_len is not None else float(main_cfg.line_operation_cost_mult)
        w_repl = self.w_repl if self.w_repl is not None else float(main_cfg.cost_repl_line)
        return float(w_len), float(w_repl)

    def to_dict(self) -> dict:
        """Return a plain dictionary (useful for logging/serialization)."""
        return asdict(self)


# -----------------------------------------------------------------------------
# Model-ready, index-based data (solver input after preprocessing)
# -----------------------------------------------------------------------------
@dataclass
class ModelData:
    """Index-based model data used by the solver.

    Index sets / mappings:
        N: number of nodes
        E_dir: number of directed infrastructure edges
        L: number of lines (layers)
        S: number of scenarios

        node_id_to_idx / idx_to_node_id: external <-> internal node mapping
        arc_uv_to_idx / idx_to_arc_uv: (u,v) directed infra-arc mapping
        line_idx_to_group: line -> group id
        line_group_to_lines: group id -> (forward_line, backward_line) indices (may use -1/None)
        line_idx_to_stops: per line, node indices in order (internal indices)
        line_idx_to_arcs: per line, directed arc indices along the line

    Parameters (arrays aligned to model indices):
        coord_x, coord_y: node coordinates
        len_a: directed edge lengths
        t_min_a, t_max_a: lower/upper travel time bounds per edge
        D: demand (format defined in loader; typically OD array)
        p_s: scenario probabilities (length S)
        cap_sa: per-scenario, per-edge capacities (shape S x E_dir)
        exclude_nodes: node indices to exclude from routing
        include_sets: optional constraints (e.g., allowable arcs per group)

    Incidence / adjacency:
        adj_out, adj_in: adjacency lists over directed edges
        A_edge_line: sparse incidence of edges to lines
        A_node_line: sparse incidence of nodes to lines
    """
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


# -----------------------------------------------------------------------------
# CGN (Composite Graph Network) representation
# -----------------------------------------------------------------------------
@dataclass
class CGN:
    """Composite Graph Network (layered graph for line-routing and passenger flow).

    Nodes:
        V: number of CGN nodes
        in_arcs / out_arcs: per CGN node incoming/outgoing CGN arc indices
        ground_of: map CGN node -> original node index for ground layer nodes

    Arcs (aligned lists; same length A):
        A: number of CGN arcs
        arc_tail / arc_head: endpoints (CGN node indices)
        arc_kind: one of {"ride", "change", "board", "alight"}
        arc_line: line id of the arc's layer
                 - "ride": operating line
                 - "board"/"change": source line
                 - otherwise -1 if not applicable
        arc_edge: directed infrastructure edge id for "ride" arcs; -1 otherwise
        arc_variant: candidate index per layer arc; -1 if no candidate is active

    Node/variant helpers (for candidate-enabled CGNs):
        node_line: per CGN node, line id on that node (-1 for ground nodes)
        node_variant: per CGN node, candidate index (-1 for ground nodes)
        arc_line_to: for waiting/transfer arcs: the target line id; -1 otherwise
    """
    V: int
    A: int
    in_arcs: List[List[int]]
    out_arcs: List[List[int]]
    ground_of: List[int]

    arc_tail: List[int]
    arc_head: List[int]
    arc_kind: List[str]            # "ride" | "change" | "board" | "alight"
    arc_line: List[int]            # see class docstring
    arc_edge: List[int]            # infra arc id (only for ride, else -1)
    arc_variant: List[int]         # candidate index per layer arc; -1 if none

    # NEW (candidate-enabled CGNs):
    node_line: List[int]           # per CGN node: line id at node (-1 for ground)
    node_variant: List[int]        # per CGN node: candidate index (-1 for ground)
    arc_line_to: List[int]         # for waiting/transfer: target line (else -1)

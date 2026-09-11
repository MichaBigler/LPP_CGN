from __future__ import annotations

import argparse
import heapq
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from gurobipy import GRB


# Add repository root to Python's import path so that this analysis
# script can reuse the existing model modules
ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from load_data import load_and_build, load_candidate_config
from data_model import DomainData, ModelData, CGN
from solve_cgn_one_stage import solve_one_stage
from solve_cgn_integrated import solve_two_stage_integrated
from optimisation import cgn_reachable_for_od


FLOW_EPS = 1e-8
CHECK_TOL = 1e-6
ATTRACTIVE_THRESHOLD = 1.2

#decide which first-stage solution to check
SOLUTION_TYPE = "stochastic"   # "deterministic" or "stochastic"
SCENARIO_INFRA_IDS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                      16,17,18,19,20,21,22,23,24,25,26,27,
                      28,29,30,31,32,33,34,35,36,37,38]

TEST_SCENARIO_INFRA_ID = 18
EXAMPLE_EDGE = 18

# Load the model for the edge-redundancy analysis.
def load_analysis_model(
    data_root: Path,
    config_path: Path,
    row: int = 0,
    scenario_infra_id: int | None = None,
) -> tuple[DomainData, ModelData, dict]:
    cfg_df = pd.read_csv(config_path, sep=";")

    if row < 0 or row >= len(cfg_df):
        raise IndexError(
            f"Config row {row} outside range 0..{len(cfg_df) - 1}"
        )

    cfg_row = cfg_df.iloc[row].to_dict()
    if scenario_infra_id is not None:
        cfg_row["scenario_infra_id"] = scenario_infra_id

    #set correct procedure-type for analysis
    if SOLUTION_TYPE == "deterministic":
        cfg_row["procedure"] = "one"

    elif SOLUTION_TYPE == "stochastic":
        cfg_row["procedure"] = "integrated"

    else:
        raise ValueError(
            f"Unknown SOLUTION_TYPE '{SOLUTION_TYPE}'. "
            "Use 'deterministic' or 'stochastic'."
        )

    # This analysis always needs OD-disaggregated passenger routing.
    cfg_row["routing_agg"] = False

    domain, model = load_and_build(
        data_root=str(data_root),
        cfg_row=cfg_row,
        symmetrise_infra=False,
        zero_od_diagonal=False,
    )

    if SOLUTION_TYPE == "stochastic":
        domain.cand_cfg = load_candidate_config(str(data_root))

    return domain, model, cfg_row


# Solve the OD-disaggregated solution used for the redundancy analysis.
def solve_analysis_solution(
    domain: DomainData,
    model: ModelData,
) -> dict:

    if SOLUTION_TYPE == "deterministic":
        gurobi_model, solution, artifacts = solve_one_stage(
            domain,
            model,
        )

        chosen_freq = solution["chosen_freq"]

    elif SOLUTION_TYPE == "stochastic":
        gurobi_model, solution, artifacts = solve_two_stage_integrated(
            domain,
            model,
        )

        chosen_freq = solution["chosen_freq_stage1"]

    else:
        raise ValueError(
            f"Unknown SOLUTION_TYPE '{SOLUTION_TYPE}'."
        )

    if gurobi_model.SolCount <= 0:
        raise RuntimeError(
            f"Analysis problem has no solution. "
            f"Gurobi status = {gurobi_model.Status}"
        )

    if gurobi_model.Status != GRB.OPTIMAL:
        print(
            f"[WARNING] Solution is not proven optimal "
            f"(status={gurobi_model.Status})."
        )

    return {
        "gurobi_model": gurobi_model,
        "cgn": artifacts["cgn_stage1"],
        "x": artifacts["x_stage1"],
        "arc_to_ods": artifacts["arc_to_keys_stage1"],
        "chosen_freq": chosen_freq,
        "solution": solution,
    }


# Map each physical edge to its two directed infrastructure arcs.
def build_physical_edge_maps(
    domain: DomainData,
    model: ModelData,
) -> tuple[dict[int, set[int]], dict[int, int]]:
    edge_to_infra = {}
    infra_to_edge = {}

    for _, row in domain.links_df.iterrows():
        edge_id = int(row["id"])
        u = int(row["a"])
        v = int(row["b"])

        uv = int(model.arc_uv_to_idx[(u, v)])
        vu = int(model.arc_uv_to_idx[(v, u)])

        edge_to_infra[edge_id] = {uv, vu}
        infra_to_edge[uv] = edge_id
        infra_to_edge[vu] = edge_id

    return edge_to_infra, infra_to_edge

# Extract nominal OD-specific passenger flow on each physical edge.
def extract_nominal_od_edge_flows(
    model: ModelData,
    cgn,
    x,
    arc_to_ods: dict,
    infra_to_edge: dict[int, int],
) -> pd.DataFrame:
    flow_by_edge_od = defaultdict(float)

    for a in range(cgn.A):
        if cgn.arc_kind[a] != "ride":
            continue

        infra_arc = int(cgn.arc_edge[a])
        edge_id = infra_to_edge[infra_arc]

        for od in arc_to_ods.get(a, []):
            flow = float(x[a, od].X)

            if flow <= FLOW_EPS:
                continue

            o, d = od

            flow_by_edge_od[
                (edge_id, int(o), int(d))
            ] += flow

    rows = []

    for (edge_id, o, d), flow in flow_by_edge_od.items():
        rows.append(
            {
                "edge_id": edge_id,
                "origin_idx": o,
                "destination_idx": d,
                "origin": int(model.idx_to_node_id[o]),
                "destination": int(model.idx_to_node_id[d]),
                "od_demand": float(model.D[o, d]),
                "nominal_flow_on_edge": flow,
            }
        )

    od_edge_flows = pd.DataFrame(rows)

    if not od_edge_flows.empty:
        invalid = (
                od_edge_flows["nominal_flow_on_edge"]
                > od_edge_flows["od_demand"] + CHECK_TOL
        )

        if invalid.any():
            raise AssertionError(
                "Nominal OD flow on an edge exceeds total OD demand."
            )

        od_edge_flows = od_edge_flows.sort_values(
            ["edge_id", "origin", "destination"]
        ).reset_index(drop=True)

    return od_edge_flows

# Build static passenger costs for all CGN arcs. Decide here whether bypasses are included.
def build_arc_costs(
    domain: DomainData,
    model: ModelData,
    cgn: CGN,
    include_model_bypass: bool = False,
) -> np.ndarray:

    # For now assumed waiting time independent of frequency
    if bool(domain.config.get("waiting_time_frequency", False)):
        raise NotImplementedError(
            "Edge redundancy currently assumes "
            "waiting_time_frequency=False."
        )

    travel_weight = float(
        domain.config.get("travel_time_cost_mult", 1.0)
    )
    waiting_weight = float(
        domain.config.get("waiting_time_cost_mult", 1.0)
    )
    bypass_multiplier = float(
        domain.config.get("bypass_multiplier", -1.0)
    )

    arc_costs = np.full(cgn.A, np.inf, dtype=float)

    for a in range(cgn.A):
        kind = cgn.arc_kind[a]

        if kind == "ride":
            infra_arc = int(cgn.arc_edge[a])
            arc_costs[a] = (
                travel_weight * float(model.t_min_a[infra_arc])
            )

        elif kind in ("board", "change"):
            arc_costs[a] = waiting_weight

        elif kind == "alight":
            arc_costs[a] = 0.0

        elif kind == "bypass":
            if include_model_bypass and bypass_multiplier >= 0:
                infra_arc = int(cgn.arc_edge[a])
                arc_costs[a] = (
                    bypass_multiplier
                    * float(model.len_a[infra_arc])
                )

    return arc_costs

# Check whether a line is operated in the nominal solution.
def line_is_active(
    chosen_freq: dict[int, int],
    line: int,
) -> bool:
    return line >= 0 and chosen_freq.get(line, 0) > 0


# Check whether a CGN arc is available in the pool or nominal plan.
def arc_available(
    cgn: CGN,
    a: int,
    mode: str,
    chosen_freq: dict[int, int],
) -> bool:
    kind = cgn.arc_kind[a]

    if mode == "pool":
        return True

    if mode != "plan":
        raise ValueError(
            f"Unknown redundancy mode '{mode}'. "
            "Expected 'pool' or 'plan'."
        )

    if kind == "ride":
        return line_is_active(
            chosen_freq,
            int(cgn.arc_line[a]),
        )

    if kind == "board":
        return line_is_active(
            chosen_freq,
            int(cgn.arc_line_to[a]),
        )

    if kind == "change":
        return (
            line_is_active(
                chosen_freq,
                int(cgn.arc_line[a]),
            )
            and line_is_active(
                chosen_freq,
                int(cgn.arc_line_to[a]),
            )
        )

    if kind == "alight":
        return line_is_active(
            chosen_freq,
            int(cgn.arc_line[a]),
        )

    if kind == "bypass":
        return True

    return False

# Check whether an arc can be used when rerouting around one physical edge.
def rerouting_arc_available(
    cgn: CGN,
    a: int,
    mode: str,
    chosen_freq: dict[int, int],
    forbidden_infra_arcs: set[int],
) -> bool:
    kind = cgn.arc_kind[a]

    # Remove normal line travel over the disrupted physical edge.
    if (
        kind == "ride"
        and int(cgn.arc_edge[a]) in forbidden_infra_arcs
    ):
        return False

    return arc_available(
        cgn=cgn,
        a=a,
        mode=mode,
        chosen_freq=chosen_freq,
    )

# Compute the unrestricted shortest generalized passenger cost for one OD pair.
def shortest_od_cost(
    cgn: CGN,
    o: int,
    d: int,
    allowed_arcs: set[int],
    arc_costs: np.ndarray,
    arc_filter=None,
) -> float:
    start = int(cgn.ground_of[o])
    goal = int(cgn.ground_of[d])

    distances = [math.inf] * cgn.V
    distances[start] = 0.0

    queue = [(0.0, start)]

    while queue:
        current_cost, v = heapq.heappop(queue)

        if current_cost > distances[v] + CHECK_TOL:
            continue

        if v == goal:
            return current_cost

        for a in cgn.out_arcs[v]:
            if a not in allowed_arcs:
                continue

            if arc_filter is not None and not arc_filter(a):
                continue

            cost = float(arc_costs[a])

            if not math.isfinite(cost):
                continue

            w = int(cgn.arc_head[a])
            new_cost = current_cost + cost

            if new_cost + CHECK_TOL < distances[w]:
                distances[w] = new_cost

                heapq.heappush(
                    queue,
                    (new_cost, w),
                )

    return math.inf


# Compute unrestricted shortest costs for all ODs used in the analysis.
def compute_unrestricted_od_costs(
    model: ModelData,
    cgn: CGN,
    arc_costs: np.ndarray,
    od_edge_flows: pd.DataFrame,
) -> pd.DataFrame:
    ods = (
        od_edge_flows[
            ["origin_idx", "destination_idx"]
        ]
        .drop_duplicates()
        .sort_values(
            ["origin_idx", "destination_idx"]
        )
    )

    rows = []

    for row in ods.itertuples(index=False):
        o = int(row.origin_idx)
        d = int(row.destination_idx)

        _, allowed_arcs = cgn_reachable_for_od(
            cgn,
            model,
            o,
            d,
        )

        t_star = shortest_od_cost(
            cgn=cgn,
            o=o,
            d=d,
            allowed_arcs=allowed_arcs,
            arc_costs=arc_costs,
        )

        rows.append(
            {
                "origin_idx": o,
                "destination_idx": d,
                "origin": int(model.idx_to_node_id[o]),
                "destination": int(model.idx_to_node_id[d]),
                "t_star": t_star,
            }
        )

    return pd.DataFrame(rows)

# Compute edge-avoiding shortest costs for each relevant OD-edge pair.
def compute_edge_avoiding_costs(
    model: ModelData,
    cgn: CGN,
    chosen_freq: dict[int, int],
    arc_costs: np.ndarray,
    edge_to_infra: dict[int, set[int]],
    od_edge_flows: pd.DataFrame,
) -> pd.DataFrame:
    result = od_edge_flows.copy()

    allowed_arcs_by_od = {}

    t_avoid_pool = []
    t_avoid_plan = []

    for row in result.itertuples(index=False):
        edge_id = int(row.edge_id)
        o = int(row.origin_idx)
        d = int(row.destination_idx)

        od = (o, d)

        if od not in allowed_arcs_by_od:
            _, allowed_arcs = cgn_reachable_for_od(
                cgn,
                model,
                o,
                d,
            )
            allowed_arcs_by_od[od] = allowed_arcs

        allowed_arcs = allowed_arcs_by_od[od]
        forbidden_infra_arcs = edge_to_infra[edge_id]

        pool_cost = shortest_od_cost(
            cgn=cgn,
            o=o,
            d=d,
            allowed_arcs=allowed_arcs,
            arc_costs=arc_costs,
            arc_filter=lambda a: rerouting_arc_available(
                cgn=cgn,
                a=a,
                mode="pool",
                chosen_freq=chosen_freq,
                forbidden_infra_arcs=forbidden_infra_arcs,
            ),
        )

        plan_cost = shortest_od_cost(
            cgn=cgn,
            o=o,
            d=d,
            allowed_arcs=allowed_arcs,
            arc_costs=arc_costs,
            arc_filter=lambda a: rerouting_arc_available(
                cgn=cgn,
                a=a,
                mode="plan",
                chosen_freq=chosen_freq,
                forbidden_infra_arcs=forbidden_infra_arcs,
            ),
        )

        t_avoid_pool.append(pool_cost)
        t_avoid_plan.append(plan_cost)

    result["t_avoid_pool"] = t_avoid_pool
    result["t_avoid_plan"] = t_avoid_plan

    return result

# Add detour ratios and attractive-route classifications.
def classify_attractive_alternatives(
    od_edge_redundancy: pd.DataFrame,
) -> pd.DataFrame:
    result = od_edge_redundancy.copy()

    if (result["t_star"] <= 0).any():
        raise AssertionError(
            "Unrestricted shortest cost t_star must be positive."
        )

    result["ratio_pool"] = (
        result["t_avoid_pool"] / result["t_star"]
    )

    result["ratio_plan"] = (
        result["t_avoid_plan"] / result["t_star"]
    )

    result["attractive_pool"] = (
        result["ratio_pool"]
        <= ATTRACTIVE_THRESHOLD + CHECK_TOL
    )

    result["attractive_plan"] = (
        result["ratio_plan"]
        <= ATTRACTIVE_THRESHOLD + CHECK_TOL
    )

    return result

# Aggregate OD-level redundancy information to physical edges.
def build_edge_summary(
    domain: DomainData,
    od_edge_redundancy: pd.DataFrame,
) -> pd.DataFrame:
    data = od_edge_redundancy.copy()

    data["attractive_flow_pool"] = np.where(
        data["attractive_pool"],
        data["nominal_flow_on_edge"],
        0.0,
    )

    data["attractive_flow_plan"] = np.where(
        data["attractive_plan"],
        data["nominal_flow_on_edge"],
        0.0,
    )

    aggregated = (
        data.groupby("edge_id", as_index=False)
        .agg(
            nominal_flow=("nominal_flow_on_edge", "sum"),
            n_od_using_edge=("nominal_flow_on_edge", "size"),
            attractive_flow_pool=("attractive_flow_pool", "sum"),
            attractive_flow_plan=("attractive_flow_plan", "sum"),
            n_od_attractive_pool=("attractive_pool", "sum"),
            n_od_attractive_plan=("attractive_plan", "sum"),
        )
    )

    # Start from all physical edges so zero-flow edges are retained.
    summary = domain.links_df[
        ["id", "a", "b", "length", "t_min", "t_max"]
    ].copy()

    summary = summary.rename(
        columns={
            "id": "edge_id",
            "a": "u",
            "b": "v",
        }
    )

    summary = summary.merge(
        aggregated,
        on="edge_id",
        how="left",
    )

    zero_fill_columns = [
        "nominal_flow",
        "n_od_using_edge",
        "attractive_flow_pool",
        "attractive_flow_plan",
        "n_od_attractive_pool",
        "n_od_attractive_plan",
    ]

    summary[zero_fill_columns] = (
        summary[zero_fill_columns].fillna(0)
    )

    summary["n_od_using_edge"] = (
        summary["n_od_using_edge"].astype(int)
    )
    summary["n_od_attractive_pool"] = (
        summary["n_od_attractive_pool"].astype(int)
    )
    summary["n_od_attractive_plan"] = (
        summary["n_od_attractive_plan"].astype(int)
    )

    summary["R_pool"] = np.nan
    summary["R_plan"] = np.nan

    positive_flow = summary["nominal_flow"] > FLOW_EPS

    summary.loc[positive_flow, "R_pool"] = (
        summary.loc[positive_flow, "attractive_flow_pool"]
        / summary.loc[positive_flow, "nominal_flow"]
    )

    summary.loc[positive_flow, "R_plan"] = (
        summary.loc[positive_flow, "attractive_flow_plan"]
        / summary.loc[positive_flow, "nominal_flow"]
    )

    summary["threshold"] = ATTRACTIVE_THRESHOLD

    return summary


def save_or_update_csv(
    new_data: pd.DataFrame,
    path: Path,
    key_columns: list[str],
) -> None:
    if path.exists():
        existing = pd.read_csv(path)

        new_keys = new_data[key_columns].drop_duplicates()

        existing = existing.merge(
            new_keys.assign(_replace=True),
            on=key_columns,
            how="left",
        )

        existing = existing[
            existing["_replace"].isna()
        ].drop(columns="_replace")

        combined = pd.concat(
            [existing, new_data],
            ignore_index=True,
        )
    else:
        combined = new_data.copy()

    combined.to_csv(path, index=False)

def print_debug_results(
    domain: DomainData,
    model: ModelData,
    cfg: dict,
    analysis_solution: dict,
    od_edge_flows: pd.DataFrame,
    arc_costs: np.ndarray,
    od_shortest_costs: pd.DataFrame,
    od_edge_redundancy: pd.DataFrame,
    edge_summary: pd.DataFrame,
) -> None:
    print("\n--- Debug information ---")

    print(
        f"\nObjective: "
        f"{analysis_solution['gurobi_model'].ObjVal:.4f}"
    )

    print(
        f"num_od={cfg.get('num_od')}, "
        f"bypass_multiplier={cfg.get('bypass_multiplier')}"
    )

    print(f"\nExtracted {len(od_edge_flows)} OD-edge flow rows.")

    # Physical edge totals
    edge_totals = (
        od_edge_flows.groupby("edge_id", as_index=False)
        ["nominal_flow_on_edge"]
        .sum()
        .rename(
            columns={
                "nominal_flow_on_edge": "nominal_flow"
            }
        )
    )

    print("\nNominal physical-edge flows:")
    print(edge_totals.to_string(index=False))

    # Example edge: nominal flow
    example = od_edge_flows[
        od_edge_flows["edge_id"] == EXAMPLE_EDGE
    ]

    print(
        f"\nOD-specific nominal flows on edge "
        f"{EXAMPLE_EDGE}:"
    )

    if example.empty:
        print("No nominal passenger flow.")
    else:
        print(
            example[
                [
                    "origin",
                    "destination",
                    "od_demand",
                    "nominal_flow_on_edge",
                ]
            ].to_string(index=False)
        )

    # Arc-cost diagnostics
    finite_costs = arc_costs[np.isfinite(arc_costs)]

    print(
        f"\nBuilt passenger costs for "
        f"{len(finite_costs)} of {len(arc_costs)} CGN arcs."
    )

    print(
        f"Cost range: "
        f"{finite_costs.min():.2f} to {finite_costs.max():.2f}"
    )

    for kind in ("ride", "board", "change", "alight", "bypass"):
        arcs = [
            a
            for a in range(analysis_solution["cgn"].A)
            if analysis_solution["cgn"].arc_kind[a] == kind
        ]

        finite = sum(
            np.isfinite(arc_costs[a])
            for a in arcs
        )

        print(
            f"{kind:>7}: {finite}/{len(arcs)} usable"
        )

    # Active lines / pool-plan availability
    chosen_freq = analysis_solution["chosen_freq"]

    active_lines = [
        line
        for line, freq in chosen_freq.items()
        if freq > 0
    ]

    print(
        f"\nActive lines: "
        f"{len(active_lines)} of {len(chosen_freq)}"
    )

    for mode in ("pool", "plan"):
        print(f"\nUsable arcs in '{mode}' mode:")

        for kind in ("ride", "board", "change", "alight", "bypass"):
            arcs = [
                a
                for a in range(analysis_solution["cgn"].A)
                if analysis_solution["cgn"].arc_kind[a] == kind
            ]

            usable = sum(
                arc_available(
                    analysis_solution["cgn"],
                    a,
                    mode,
                    chosen_freq,
                )
                and np.isfinite(arc_costs[a])
                for a in arcs
            )

            print(
                f"{kind:>7}: {usable}/{len(arcs)}"
            )

    # Example unrestricted shortest costs
    print("\nExample unrestricted OD costs:")

    print(
        od_shortest_costs[
            [
                "origin",
                "destination",
                "t_star",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # Example edge: rerouting
    example = od_edge_redundancy[
        od_edge_redundancy["edge_id"] == EXAMPLE_EDGE
    ]

    print(
        f"\nRedundancy details for edge {EXAMPLE_EDGE}:"
    )

    print(
        example[
            [
                "origin",
                "destination",
                "nominal_flow_on_edge",
                "t_star",
                "t_avoid_pool",
                "ratio_pool",
                "attractive_pool",
                "t_avoid_plan",
                "ratio_plan",
                "attractive_plan",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nEdge summary for edge {EXAMPLE_EDGE}:"
    )

    print(
        edge_summary[
            edge_summary["edge_id"] == EXAMPLE_EDGE
        ].to_string(index=False)
    )


def run_analysis_case(
    config_path: Path,
    scenario_infra_id: int | None,
    debug: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(
        f"\n=== {SOLUTION_TYPE} analysis "
        f"| scenario_infra_id={scenario_infra_id} ==="
    )

    # 1. Load model
    print("Loading model...")
    domain, model, cfg = load_analysis_model(
        data_root=ROOT,
        config_path=config_path,
        row=0,
        scenario_infra_id=scenario_infra_id,
    )

    # 2. Solve
    print("Solving OD-disaggregated model...")
    analysis_solution = solve_analysis_solution(
        domain,
        model,
    )

    # 3. Extract nominal OD-edge flows
    print("Extracting nominal OD-edge flows...")
    edge_to_infra, infra_to_edge = build_physical_edge_maps(
        domain,
        model,
    )

    od_edge_flows = extract_nominal_od_edge_flows(
        model=model,
        cgn=analysis_solution["cgn"],
        x=analysis_solution["x"],
        arc_to_ods=analysis_solution["arc_to_ods"],
        infra_to_edge=infra_to_edge,
    )

    # 4. Build generalized passenger costs
    print("Building passenger arc costs...")
    arc_costs = build_arc_costs(
        domain=domain,
        model=model,
        cgn=analysis_solution["cgn"],
        include_model_bypass=True,
    )

    chosen_freq = analysis_solution["chosen_freq"]

    # 5. Compute unrestricted shortest paths
    print("Computing unrestricted shortest paths...")
    od_shortest_costs = compute_unrestricted_od_costs(
        model=model,
        cgn=analysis_solution["cgn"],
        arc_costs=arc_costs,
        od_edge_flows=od_edge_flows,
    )

    n_unreachable = (
        ~np.isfinite(od_shortest_costs["t_star"])
    ).sum()

    if n_unreachable > 0:
        raise AssertionError(
            "Some nominally used ODs are unreachable "
            "in the unrestricted passenger network."
        )

    # 6. Compute edge-avoiding shortest paths
    print("Computing edge-avoiding shortest paths...")
    od_edge_redundancy = compute_edge_avoiding_costs(
        model=model,
        cgn=analysis_solution["cgn"],
        chosen_freq=chosen_freq,
        arc_costs=arc_costs,
        edge_to_infra=edge_to_infra,
        od_edge_flows=od_edge_flows,
    )

    od_edge_redundancy = od_edge_redundancy.merge(
        od_shortest_costs[
            [
                "origin_idx",
                "destination_idx",
                "t_star",
            ]
        ],
        on=["origin_idx", "destination_idx"],
        how="left",
    )

    # Validation: pool can never be worse than plan
    finite_both = (
        np.isfinite(od_edge_redundancy["t_avoid_pool"])
        & np.isfinite(od_edge_redundancy["t_avoid_plan"])
    )

    invalid = (
        od_edge_redundancy.loc[
            finite_both,
            "t_avoid_pool",
        ]
        >
        od_edge_redundancy.loc[
            finite_both,
            "t_avoid_plan",
        ]
        + CHECK_TOL
    )

    if invalid.any():
        raise AssertionError(
            "Pool rerouting cost exceeds plan rerouting cost."
        )

    # Validation: removing an edge cannot improve the shortest path
    finite_pool = np.isfinite(
        od_edge_redundancy["t_avoid_pool"]
    )

    invalid = (
        od_edge_redundancy.loc[
            finite_pool,
            "t_avoid_pool",
        ]
        + CHECK_TOL
        <
        od_edge_redundancy.loc[
            finite_pool,
            "t_star",
        ]
    )

    if invalid.any():
        raise AssertionError(
            "Removing an edge produced a shorter route than "
            "the unrestricted shortest path."
        )

    # 7. Classify attractive alternatives
    print("Classifying attractive alternatives...")
    od_edge_redundancy = classify_attractive_alternatives(
        od_edge_redundancy
    )

    invalid = (
        od_edge_redundancy["attractive_plan"]
        & ~od_edge_redundancy["attractive_pool"]
    )

    if invalid.any():
        raise AssertionError(
            "Plan alternative classified as attractive "
            "while pool alternative is not."
        )

    # 8. Aggregate to edge level
    print("Building edge summary...")
    edge_summary = build_edge_summary(
        domain=domain,
        od_edge_redundancy=od_edge_redundancy,
    )

    edge_summary["solution_type"] = SOLUTION_TYPE
    od_edge_redundancy["solution_type"] = SOLUTION_TYPE

    edge_summary["scenario_infra_id"] = scenario_infra_id
    od_edge_redundancy["scenario_infra_id"] = scenario_infra_id

    if debug:
        print_debug_results(
            domain=domain,
            model=model,
            cfg=cfg,
            analysis_solution=analysis_solution,
            od_edge_flows=od_edge_flows,
            arc_costs=arc_costs,
            od_shortest_costs=od_shortest_costs,
            od_edge_redundancy=od_edge_redundancy,
            edge_summary=edge_summary,
        )

    return edge_summary, od_edge_redundancy


def test_main():
    config_path = ROOT / "Data" / "config.csv"

    if SOLUTION_TYPE == "deterministic":
        scenario_infra_id = None
    else:
        scenario_infra_id = TEST_SCENARIO_INFRA_ID

    run_analysis_case(
        config_path=config_path,
        scenario_infra_id=scenario_infra_id,
        debug=True,
    )


def main():
    config_path = ROOT / "Data" / "config.csv"

    if SOLUTION_TYPE == "deterministic":
        scenario_ids = [None]
    else:
        scenario_ids = SCENARIO_INFRA_IDS

    output_dir = (
        ROOT
        / "Analysis"
        / "role_of_links"
        / "edge_redundancy"
        / "results"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = (
        output_dir
        / f"edge_redundancy_summary_{SOLUTION_TYPE}.csv"
    )

    detail_path = (
        output_dir
        / f"edge_redundancy_od_detail_{SOLUTION_TYPE}.csv"
    )

    for scenario_infra_id in scenario_ids:
        edge_summary, od_edge_redundancy = run_analysis_case(
            config_path=config_path,
            scenario_infra_id=scenario_infra_id,
            debug=False,
        )

        save_or_update_csv(
            new_data=edge_summary,
            path=summary_path,
            key_columns=[
                "solution_type",
                "scenario_infra_id",
                "edge_id",
            ],
        )

        save_or_update_csv(
            new_data=od_edge_redundancy,
            path=detail_path,
            key_columns=[
                "solution_type",
                "scenario_infra_id",
                "edge_id",
                "origin",
                "destination",
            ],
        )

        print(
            f"Finished scenario_infra_id="
            f"{scenario_infra_id}."
        )

    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
    #use following for debugging individual runs:
    #test_main()

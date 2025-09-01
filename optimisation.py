# model_line_planning.py
# -*- coding: utf-8 -*-
"""
Baseline Change-and-Go line planning model using given candidate lines.
- Objective = passenger in-vehicle time + waiting time + operating costs
- Constraints = flow conservation, vehicle capacity, infrastructure capacity, line selection/frequency
No path-construction constraints (no branches/subtours), because lines are pre-defined paths.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
from data_model import DomainData, ModelData

def build_line_planning_model(domain: DomainData,
                              data: ModelData,
                              params: Optional[Dict] = None) -> Tuple[gp.Model, Dict]:
    """
    Build a Gurobi model from DomainData/ModelData.
    params keys (optional):
      - select_lines: bool (if False, all lines forced active)
      - f_max: int (max frequency)
      - freq_values: List[int] (allowed frequencies; default 1..f_max)
      - scenario_index: int (index for infra capacity, default 0)
      - vehicle_capacity: int (train seat capacity, default  data from config/properties or 200)
      - line_cost_mult: float (operating cost multiplier, default properties_general)
      - time_weight: float (weight for in-vehicle time in objective, default 1.0)
      - wait_weight: float (weight for waiting time, default 1.0)
      - op_cost_weight: float (weight for operating cost, default 1.0)
      - bigM_flow: float (upper bound for flows to tie to z_l; default sum(D))
    Returns (model, var_dict)
    """
    if params is None:
        params = {}
    select_lines   = bool(params.get("select_lines", True))
    f_max          = int(params.get("f_max", int(domain.config.get("max_frequency", 5))))
    freq_values    = list(params.get("freq_values", list(range(1, f_max + 1))))
    scenario_index = int(params.get("scenario_index", 0))
    vehicle_cap    = int(params.get("vehicle_capacity", int(domain.config.get("train_capacity", 200))))
    line_cost_mult = float(params.get("line_cost_mult", float(domain.props.get("line_cost_mult", 1.0))))
    time_w         = float(params.get("time_weight", 1.0))
    wait_w         = float(params.get("wait_weight", 1.0))
    op_w           = float(params.get("op_cost_weight", 1.0))

    # Sets and handy data
    N = data.N
    E = data.E_dir
    L = data.L

    # OD pairs with positive demand
    K: List[Tuple[int, int]] = [(i, j) for i in range(N) for j in range(N) if data.D[i, j] > 0]
    D_map: Dict[Tuple[int, int], float] = {(i, j): float(data.D[i, j]) for (i, j) in K}
    total_demand = float(np.sum(data.D))
    bigM_flow = float(params.get("bigM_flow", max(1.0, total_demand)))

    # Per-line arc/node sets and line lengths
    arcs_on_line = [set(lst) for lst in data.line_idx_to_arcs]      # list[set[int]]
    nodes_on_line = [set(lst) for lst in data.line_idx_to_stops]    # list[set[int]]
    line_length = [float(sum(data.len_a[a] for a in arcs_on_line[ell])) for ell in range(L)]

    # Infrastructure capacity per arc for chosen scenario (or standard if not provided)
    if data.cap_sa is not None and data.cap_sa.shape[0] > scenario_index:
        cap_a = [int(data.cap_sa[scenario_index, a]) for a in range(E)]
    else:
        cap_std = int(domain.props.get("infra_cap_std", 10))
        cap_a = [cap_std for _ in range(E)]

    # Build model
    m = gp.Model("line_planning")
    m.Params.OutputFlag = 1
    # TIP: set NonConvex=2 only if you later add bilinear/quad terms
    # m.Params.NonConvex = 2

    # ---- Variables ----
    # Line selection
    z = {}
    if select_lines:
        z = m.addVars(L, vtype=GRB.BINARY, name="z")
    else:
        # Represent 'always on' by a dict-like that returns 1
        z = {ell: 1.0 for ell in range(L)}

    # Frequency choice via binaries δ_{ell,r}, avoids bilinear headway * boardings
    freq_vals = freq_values  # allowed integer frequencies (>=1)
    delta = m.addVars(L, len(freq_vals), vtype=GRB.BINARY, name="delta")

    # Derived expressions for frequency and headway
    # f_ell = sum_r r * delta_ellr ; h_ell = sum_r (1/r) * delta_ellr
    f_expr = {ell: gp.LinExpr(quicksum(freq_vals[r] * delta[ell, r] for r in range(len(freq_vals))))
              for ell in range(L)}
    h_expr = {ell: gp.LinExpr(quicksum((1.0 / freq_vals[r]) * delta[ell, r] for r in range(len(freq_vals))))
              for ell in range(L)}

    # Passenger flow on arcs per OD per line (only where arc ∈ line)
    x = {}
    for ell in range(L):
        for a in arcs_on_line[ell]:
            for (i, j) in K:
                x[(a, i, j, ell)] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                              name=f"x[a{a},k({i},{j}),l{ell}]")

    # Boardings per node/OD/line (for waiting)
    b = {}
    for ell in range(L):
        for i in nodes_on_line[ell]:
            for (o, d) in K:
                b[(i, o, d, ell)] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                             name=f"b[i{i},k({o},{d}),l{ell}]")

    m.update()

    # ---- Linking constraints for delta and z ----
    # Exactly one frequency level if active; 0 if inactive.
    if select_lines:
        for ell in range(L):
            m.addConstr(quicksum(delta[ell, r] for r in range(len(freq_vals))) == z[ell],
                        name=f"freq_sel[{ell}]")
    else:
        for ell in range(L):
            m.addConstr(quicksum(delta[ell, r] for r in range(len(freq_vals))) == 1,
                        name=f"freq_sel[{ell}]")

    # Optional: restrict number of active lines (if provided)
    if select_lines:
        if "num_lines (opt)" in domain.config:
            m.addConstr(quicksum(z[ell] for ell in range(L)) <= int(domain.config["num_lines (opt)"]),
                        name="line_budget")

    # ---- Flow conservation per OD and node ----
    for (o, d), dem in D_map.items():
        for i in range(N):
            out_terms = []
            in_terms = []
            # collect all arc flows across all lines that leave/enter node i
            for ell in range(L):
                # out arcs from i on this line
                for a in data.adj_out[i]:
                    if a in arcs_on_line[ell]:
                        out_terms.append(x[(a, o, d, ell)])
                # in arcs to i on this line
                for a in data.adj_in[i]:
                    if a in arcs_on_line[ell]:
                        in_terms.append(x[(a, o, d, ell)])
            rhs = 0.0
            if i == o:
                rhs = dem
            elif i == d:
                rhs = -dem
            m.addConstr(quicksum(out_terms) - quicksum(in_terms) == rhs,
                        name=f"flow[{o},{d},i{i}]")

    # ---- Shut flows if line is inactive (only needed when select_lines=True) ----
    if select_lines:
        for ell in range(L):
            for a in arcs_on_line[ell]:
                # bound sum of x over all OD on this arc by bigM * z[ell]
                m.addConstr(quicksum(x[(a, i, j, ell)] for (i, j) in K) <= bigM_flow * z[ell],
                            name=f"line_off_block[{a},{ell}]")

    # ---- Vehicle capacity on each line-arc: sum_k x <= Q * f_ell ----
    for ell in range(L):
        for a in arcs_on_line[ell]:
            m.addConstr(quicksum(x[(a, i, j, ell)] for (i, j) in K) <= vehicle_cap * f_expr[ell],
                        name=f"veh_cap[{a},{ell}]")

    # ---- Infrastructure capacity per arc: sum_ell f_ell * A[a,ell] <= cap_a ----
    for a in range(E):
        # Only lines that use arc a contribute
        m.addConstr(quicksum(f_expr[ell] for ell in range(L) if a in arcs_on_line[ell]) <= cap_a[a],
                    name=f"infra_cap[a{a}]")

    # ---- Boarding detection for waiting cost (positive increases along line chain) ----
    # For each line ell, order of arcs along the path is data.line_idx_to_arcs[ell]
    for ell in range(L):
        arc_chain = data.line_idx_to_arcs[ell]  # [a0, a1, ...]
        stop_chain = data.line_idx_to_stops[ell]  # [s0, s1, ...] with len = len(arcs)+1

        for (o, d) in K:
            # first node s0: b >= x on first arc
            first_arc = arc_chain[0]
            first_node = stop_chain[0]
            m.addConstr(b[(first_node, o, d, ell)] >= x[(first_arc, o, d, ell)],
                        name=f"board_first[l{ell},k({o},{d})]")
            # interior nodes s_m (m>=1): b >= x[a_m] - x[a_{m-1}]
            for pos in range(1, len(arc_chain)):
                cur_arc = arc_chain[pos]
                prev_arc = arc_chain[pos - 1]
                node_i = stop_chain[pos]  # boarding occurs at the upstream node of cur_arc
                m.addConstr(b[(node_i, o, d, ell)] >= x[(cur_arc, o, d, ell)] - x[(prev_arc, o, d, ell)],
                            name=f"board_increase[l{ell},k({o},{d}),pos{pos}]")

    # ---- Objective components ----
    # In-vehicle time: sum x * t_min_a (or use length if preferred)
    use_t_min = bool(domain.config.get("use_t_min_time", True))
    time_coeff = data.t_min_a if use_t_min else data.len_a
    obj_invehicle = quicksum(time_coeff[a] * x[(a, i, j, ell)]
                             for ell in range(L)
                             for a in arcs_on_line[ell]
                             for (i, j) in K)

    # Waiting time: 0.5 * sum_ell (h_ell * total_boardings_on_ell)
    # total_boardings_on_ell = sum_{i in line, (o,d)} b[i,o,d,ell]
    b_total = {ell: quicksum(b[(i, o, d, ell)] for i in nodes_on_line[ell] for (o, d) in K)
               for ell in range(L)}
    obj_wait = 0.5 * quicksum(h_expr[ell] * b_total[ell] for ell in range(L))
    # NOTE: This term is linear because h_expr is linear in binaries delta.

    # Operating cost: sum_ell f_ell * line_length * line_cost_mult
    obj_oper = quicksum(f_expr[ell] * line_length[ell] * line_cost_mult for ell in range(L))

    m.setObjective(time_w * obj_invehicle + wait_w * obj_wait + op_w * obj_oper, GRB.MINIMIZE)

    # ---- Tidy bounds: if select_lines=False, enforce exactly one frequency level per line
    if not select_lines:
        # already enforced via freq_sel == 1; nothing else to do
        pass

    # Optional: symmetry breaking — prefer lower index frequency if equal cost (not required)
    # m.addConstrs((delta[ell, r] >= delta[ell, r+1] for ell in range(L) for r in range(len(freq_vals)-1)),
    #              name="freq_monotone")  # comment out unless helpful

    m.update()
    varpack = {
        "z": z,
        "delta": delta,
        "f_expr": f_expr,
        "h_expr": h_expr,
        "x": x,
        "b": b,
        "b_total": b_total,
        "line_length": line_length,
        "K": K,
        "cap_a": cap_a,
    }
    return m, varpack

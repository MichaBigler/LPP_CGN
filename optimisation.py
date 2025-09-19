# optimisation.py
# -----------------------------------------------------------------------------
# Pure model-building blocks: no config parsing and no I/O here.
# Every numeric value (capacities, weights, penalties, etc.) is injected
# from the caller. This keeps modelling code small, testable and reusable.
# -----------------------------------------------------------------------------

from collections import deque
from typing import Dict, List, Tuple, Iterable, Any, Optional, TypeGuard, Union
import numpy as np
import gurobipy as gp
from prepare_cgn import CGN


# =============================== Utilities ================================

def od_pairs(data) -> List[Tuple[int, int]]:
    """
    Enumerate all OD pairs (o, d) with positive demand in index space 0..N-1.
    """
    return [(o, d) for o in range(data.N) for d in range(data.N) if data.D[o, d] > 0]


ODKey = Tuple[int, int]
OriginKey = int
FlowKey = Union[ODKey, OriginKey]


def _is_od_key(x: object) -> TypeGuard[ODKey]:
    """Type guard: True if x looks like an OD key tuple (int, int)."""
    return (
        isinstance(x, tuple)
        and len(x) == 2
        and all(isinstance(t, (int, np.integer)) for t in x)
    )


def cgn_reachable_for_od(cgn: CGN, data, o: int, d: int):
    """
    Compute the CGN subgraph reachable for OD=(o,d) under:
      - boarding only allowed at origin ground node,
      - alighting only allowed at destination ground node.

    Returns:
        (visited_nodes, allowed_arcs) where
            visited_nodes is a set of CGN node indices,
            allowed_arcs   is a set of CGN arc indices.
    """
    start = cgn.ground_of[o]
    goal = cgn.ground_of[d]

    seen = [False] * cgn.V
    seen[start] = True
    allowed_arcs = set()

    Q = deque([start])
    while Q:
        v = Q.popleft()
        for a in cgn.out_arcs[v]:
            kind = cgn.arc_kind[a]
            w = cgn.arc_head[a]

            # OD-specific boarding/alighting rules:
            if kind == "board" and v != start:
                continue
            if kind == "alight" and w != goal:
                continue

            allowed_arcs.add(a)
            if not seen[w]:
                seen[w] = True
                Q.append(w)

    visited_nodes = {i for i, b in enumerate(seen) if b}
    return visited_nodes, allowed_arcs


def group_x_keys_by_arc(x: gp.tupledict) -> Dict[int, List[Any]]:
    """
    Build a mapping arc_id -> [flow_key,...] for existing x[arc_id, key] entries.
    'key' is (o,d) in OD-mode or 'o' in origin-aggregated mode.
    """
    arc_to_keys: Dict[int, List[Any]] = {}
    for a, key in x.keys():
        arc_to_keys.setdefault(a, []).append(key)
    return arc_to_keys


# ============================ Flow Conservation ============================

def add_flow_conservation_by_origin(m: gp.Model, data, cgn: CGN):
    """
    Origin-aggregated routing:
      - One commodity per origin o, variables x[a, o].
      - Boarding only at ground(o), alighting only at any ground(d) with D[o,d] > 0.
      - Flow conservation is enforced on the reachable subgraph per origin.

    Returns:
        x           : tupledict with variables x[a, o] for allowed arcs
        arc_to_keys: dict mapping arc_id -> list of origins used in x
    """
    # valid destinations per origin
    dests = {o: [d for d in range(data.N) if data.D[o, d] > 0] for o in range(data.N)}
    origins = [o for o, L in dests.items() if L]

    # Reachability per origin (board only at origin, alight only at any valid destination)
    def reach_nodes_and_arcs_for_origin(o: int, dest_set: Iterable[int]):
        start = cgn.ground_of[o]
        goal_grounds = {cgn.ground_of[d] for d in dest_set}

        seen = [False] * cgn.V
        seen[start] = True
        allowed = set()

        Q = deque([start])
        while Q:
            v = Q.popleft()
            for a in cgn.out_arcs[v]:
                kind = cgn.arc_kind[a]
                w = cgn.arc_head[a]
                if kind == "board" and v != start:
                    continue
                if kind == "alight" and w not in goal_grounds:
                    continue
                allowed.add(a)
                if not seen[w]:
                    seen[w] = True
                    Q.append(w)

        nodes = {i for i, b in enumerate(seen) if b}
        return nodes, allowed

    # build variable keys only for allowed arcs
    keys = []
    nodes_by_o: Dict[int, set] = {}
    arcs_by_o: Dict[int, set] = {}
    for o in origins:
        nodes, arcs = reach_nodes_and_arcs_for_origin(o, dests[o])
        nodes_by_o[o] = nodes
        arcs_by_o[o] = arcs
        keys.extend((a, o) for a in arcs)

    x = m.addVars(keys, lb=0.0, name="x")

    # flow conservation on reachable nodes
    for o in origins:
        rhs = [0.0] * cgn.V
        supply = float(sum(data.D[o, d] for d in dests[o]))
        rhs[cgn.ground_of[o]] -= supply
        for d in dests[o]:
            rhs[cgn.ground_of[d]] += float(data.D[o, d])

        nodes = nodes_by_o[o]
        allowed = arcs_by_o[o]
        for v in nodes:
            m.addConstr(
                gp.quicksum(x[a, o] for a in cgn.in_arcs[v] if a in allowed)
                - gp.quicksum(x[a, o] for a in cgn.out_arcs[v] if a in allowed)
                == rhs[v],
                name=f"flow[v{v},o{o}]",
            )

    # a -> [o,...] for which x[a,o] exists
    arc_to_keys: Dict[int, List[int]] = {}
    for a, o in x.keys():
        arc_to_keys.setdefault(a, []).append(o)

    return x, arc_to_keys


def add_flow_conservation(m: gp.Model, data, cgn: CGN, K: List[Tuple[int, int]]):
    """
    Standard OD routing:
      - One commodity per OD pair with positive demand.
      - Variables x[a,(o,d)] exist only on arcs that are reachable for (o,d).

    Returns:
        x          : tupledict with variables x[a, (o,d)]
        od_allowed : dict (o,d) -> set of allowed arcs
        od_nodes   : dict (o,d) -> set of reachable nodes
        arc_to_ods : dict arc_id -> list of (o,d) keys present in x
    """
    od_nodes: Dict[Tuple[int, int], set] = {}
    od_allowed: Dict[Tuple[int, int], set] = {}
    keys = []

    for (o, d) in K:
        nodes, arcs = cgn_reachable_for_od(cgn, data, o, d)
        od_nodes[(o, d)] = nodes
        od_allowed[(o, d)] = arcs
        keys.extend((a, (o, d)) for a in arcs)

    x = m.addVars(keys, lb=0.0, name="x")

    # flow conservation per OD on its reachable nodes
    for (o, d) in K:
        rhs = [0.0] * cgn.V
        dem = float(data.D[o, d])
        rhs[cgn.ground_of[o]] -= dem
        rhs[cgn.ground_of[d]] += dem

        nodes = od_nodes[(o, d)]
        allowed = od_allowed[(o, d)]
        for v in nodes:
            m.addConstr(
                gp.quicksum(x[a, (o, d)] for a in cgn.in_arcs[v] if a in allowed)
                - gp.quicksum(x[a, (o, d)] for a in cgn.out_arcs[v] if a in allowed)
                == rhs[v],
                name=f"flow[v{v},od({o},{d})]",
            )

    arc_to_ods = group_x_keys_by_arc(x)
    return x, od_allowed, od_nodes, arc_to_ods


# ============================== Frequencies ===============================

def add_frequency_grouped(m: gp.Model, model, freq_vals: List[int]):
    """
    Group-coupled frequency choice with on/off per group.

    For each group g:
      - z_g ∈ {0,1} activates the group,
      - δ_{g,r} ∈ {0,1} selects exactly one frequency value f_r when z_g=1,
      - f_g = Σ_r f_r · δ_{g,r},
      - h_g = Σ_r (1/f_r) · δ_{g,r}.

    Each line ℓ in group g inherits f_ℓ := f_g and h_ℓ := h_g.

    Returns:
        z_g        : gp.tupledict keyed by group id
        delta_line : dict keyed by (ℓ, r) (reuse-friendly handle for waiting-time code)
        f_expr     : dict ℓ -> LinExpr frequency expression
        h_expr     : dict ℓ -> LinExpr headway expression (1/f)
    """
    groups = sorted(model.line_group_to_lines.keys())
    R = len(freq_vals)

    # on/off per group
    z_g = m.addVars(groups, vtype=gp.GRB.BINARY, name="z_group")

    # frequency pick per group
    delta_g = m.addVars(((g, r) for g in groups for r in range(R)),
                        vtype=gp.GRB.BINARY, name="delta_g")

    # exactly one frequency if on; zero if off
    m.addConstrs(
        (gp.quicksum(delta_g[g, r] for r in range(R)) == z_g[g] for g in groups),
        name="pick_or_off_group"
    )

    # group-level f and h
    f_g = {g: gp.quicksum(freq_vals[r] * delta_g[g, r] for r in range(R)) for g in groups}
    h_g = {g: gp.quicksum((1.0 / freq_vals[r]) * delta_g[g, r] for r in range(R)) for g in groups}

    # per-line expressions inherit from their group
    f_expr: Dict[int, gp.LinExpr] = {}
    h_expr: Dict[int, gp.LinExpr] = {}
    delta_line: Dict[Tuple[int, int], gp.Var] = {}

    for ell in range(model.L):
        g = model.line_idx_to_group[ell]
        f_expr[ell] = f_g[g]
        h_expr[ell] = h_g[g]
        for r in range(R):
            # Expose per-(ell,r) picks to re-use waiting-time linearisation without changing its API
            delta_line[(ell, r)] = delta_g[g, r]

    return z_g, delta_line, f_expr, h_expr


# ============================== Capacities ================================

def add_passenger_capacity(
    m: gp.Model,
    data,
    cgn: CGN,
    x: gp.tupledict,
    f_expr: Dict[int, gp.LinExpr],
    arc_to_keys: Dict[int, List[Any]],
    Q: int,
):
    """
    Vehicle capacity on ride arcs:
        sum_key x[a, key] ≤ Q * f_ℓ   for each ride arc a that belongs to line ℓ.
    """
    # collect ride arcs by line
    ride_arcs_by_line: List[List[int]] = [[] for _ in range(data.L)]
    for a in range(cgn.A):
        if cgn.arc_kind[a] == "ride":
            ride_arcs_by_line[cgn.arc_line[a]].append(a)

    for ell in range(data.L):
        f_ell = f_expr[ell]
        for a in ride_arcs_by_line[ell]:
            keys = arc_to_keys.get(a, [])
            if not keys:
                continue
            m.addConstr(
                gp.quicksum(x[a, key] for key in keys) <= Q * f_ell,
                name=f"veh_cap[a{a},l{ell}]",
            )


def add_infrastructure_capacity(
    m: gp.Model,
    data,
    f_expr: Dict[int, gp.LinExpr],
    cap_std: Optional[int] = None,
    *,
    cap_per_arc: Optional[Iterable] = None,
    name: str = "infra_cap",
):
    """
    Infrastructure capacity per directed infra arc.

    Use EITHER:
      - a global scalar (cap_std), OR
      - a per-arc array-like (cap_per_arc) of length E_dir.
    If cap_per_arc is provided, it overrides cap_std.
    """
    # lines using each directed infra arc
    lines_per_arc: List[List[int]] = [[] for _ in range(data.E_dir)]
    for ell, arc_list in enumerate(data.line_idx_to_arcs):
        for a in arc_list:
            lines_per_arc[a].append(ell)

    # RHS provider
    if cap_per_arc is not None:
        cap_arr = np.asarray(cap_per_arc, dtype=float).reshape(-1)
        if len(cap_arr) != data.E_dir:
            raise ValueError(f"cap_per_arc length {len(cap_arr)} != E_dir {data.E_dir}")

        def rhs_a(a: int) -> float:
            return float(cap_arr[a])
    else:
        if cap_std is None:
            raise ValueError("Either cap_std or cap_per_arc must be provided.")
        cap_val = float(cap_std)

        def rhs_a(a: int) -> float:
            return cap_val

    for a in range(data.E_dir):
        if not lines_per_arc[a]:
            continue
        m.addConstr(
            gp.quicksum(f_expr[ell] for ell in lines_per_arc[a]) <= rhs_a(a),
            name=f"{name}[a{a}]",
        )


# ============================== Cost Blocks ===============================

def build_obj_invehicle(
    m: gp.Model,
    data,
    cgn: CGN,
    x: gp.tupledict,
    arc_to_keys: Dict[int, List[Any]],
    *,
    use_t_min_time: bool = True,
):
    """
    In-vehicle time bucket (ride arcs only).
    (Bypass arcs are priced separately in build_obj_bypass().)
    """
    time_a = data.t_min_a if use_t_min_time else data.len_a
    ride = [a for a in range(cgn.A) if cgn.arc_kind[a] == "ride"]
    expr = gp.quicksum(
        time_a[cgn.arc_edge[a]] * x[a, key]
        for a in ride
        for key in arc_to_keys.get(a, [])
    )
    return expr


def build_obj_invehicle_with_overdemand(
    m: gp.Model,
    data,
    cgn: CGN,
    x: gp.tupledict,
    arc_to_keys: Dict[int, List[Any]],
    f_expr: Dict[int, gp.LinExpr],
    Q: int,
    *,
    threshold: float = 1.0,  # τ in [0,1]
    multiplier: float = 1.0, # μ >= 1
    use_t_min_time: bool = True,
):
    """
    Over-demand aware in-vehicle time:
      Returns (time_raw, over_raw) as linear expressions.

      time_raw = Σ_a t_a * (Σ_keys x[a,key])
      over_raw = Σ_a t_a * s_a,  with
        s_a ≥ (Σ_keys x[a,key]) - τ * Q * f_ℓ(a)
        s_a ≥ 0

    Caller forms:
      total_time = time_raw + max(multiplier-1, 0) * over_raw
      and multiplies by the time weight externally.

    Args:
      threshold  τ: fraction of Q·f considered "not overloaded" (e.g. 1.0 → no overcharge).
      multiplier μ: surcharge factor ≥ 1; if μ==1 or τ==1 no surcharge is applied.
    """
    tau = max(0.0, min(1.0, float(threshold)))
    mu  = float(multiplier)
    use_over = (mu > 1.0) and (tau < 1.0)

    time_a = data.t_min_a if use_t_min_time else data.len_a
    ride = [a for a in range(cgn.A) if cgn.arc_kind[a] == "ride"]

    # sum of flows per arc
    def flow_sum_on_arc(a: int):
        keys = arc_to_keys.get(a, [])
        return gp.quicksum(x[a, key] for key in keys) if keys else gp.LinExpr(0.0)

    time_raw = gp.LinExpr(0.0)
    over_raw = gp.LinExpr(0.0)

    if not use_over:
        # No surcharge; just the base time term
        for a in ride:
            time_raw += float(time_a[cgn.arc_edge[a]]) * flow_sum_on_arc(a)
        return time_raw, over_raw

    # With over-demand hinge
    for a in ride:
        ell = int(cgn.arc_line[a])
        t   = float(time_a[cgn.arc_edge[a]])
        F_a = flow_sum_on_arc(a)
        T_a = tau * float(Q) * f_expr[ell]

        s = m.addVar(lb=0.0, name=f"overdemand_s[a{a}]")
        m.addConstr(s >= F_a - T_a, name=f"overdemand_hinge[a{a}]")

        time_raw += t * F_a
        over_raw += t * s

    return time_raw, over_raw


def build_obj_bypass(m, data, cgn, x, arc_to_keys):
    """
    Bypass cost bucket:
      Σ bypass_arcs (bypass_multiplier * len_a[infra(a)] * Σ_key x[a,key])

    The returned expression is already “fully weighted” by bypass_multiplier
    (unlike time/wait/oper which are weighted at the top-level objective).
    """
    bypass_mult = float(getattr(data, "config", {}).get("bypass_multiplier", -1.0))
    if bypass_mult < 0.0:
        return gp.LinExpr(0.0)

    bypass = [a for a in range(cgn.A) if cgn.arc_kind[a] == "bypass"]
    if not bypass:
        return gp.LinExpr(0.0)

    return gp.quicksum(
        bypass_mult * float(data.len_a[cgn.arc_edge[a]]) * x[a, key]
        for a in bypass
        for key in arc_to_keys.get(a, [])
    )


def build_obj_waiting(
    m, data, cgn, x, arc_to_keys, freq_vals, delta,
    include_origin_wait=False,
    waiting_time_frequency=True
):
    """
    Waiting-time bucket.

    Returns:
        (wait_expr_raw, y_vars_or_None)

    Two modes:
      1) waiting_time_frequency == True:
         Half-headway linearisation driven by selected frequencies on target lines.
         Uses cgn.arc_line_to[a] on "board" / "change" arcs to pick the target line.
         Introduces split variables y[a,r].

      2) waiting_time_frequency == False:
         Flat penalty: 1.0 per passenger traversing a change-like arc.
         (Optionally also counts the initial "board" if include_origin_wait==True.)
    """
    # pick the change-like arcs to charge waiting on
    change_like = [a for a in range(cgn.A) if cgn.arc_kind[a] == "change"]
    if include_origin_wait:
        change_like += [a for a in range(cgn.A) if cgn.arc_kind[a] == "board"]

    if not waiting_time_frequency:
        wait_expr_raw = gp.quicksum(
            x[a, key] for a in change_like for key in arc_to_keys.get(a, [])
        )
        return wait_expr_raw, None

    # frequency-driven half-headway: split x[a,*] into y[a,r] by chosen frequency index r
    R = len(freq_vals)
    y = m.addVars(((a, r) for a in change_like for r in range(R)), lb=0.0, name="chg_split")

    # demand per key (supports OD mode and origin-aggregated mode)
    def _dem_for_key(key: object) -> float:
        if isinstance(key, tuple) and len(key) == 2:
            o, d = key
            return float(data.D[int(o), int(d)])
        elif isinstance(key, (int, np.integer)):
            o = int(key)
            return float(np.asarray(data.D[o, :]).sum())
        else:
            raise TypeError(f"Unsupported flow key type: {type(key)} -> {key}")

    # tight big-M per arc equals total flow over this arc (by demand)
    M_arc = [0.0] * cgn.A
    for a in change_like:
        M_arc[a] = sum(_dem_for_key(key) for key in arc_to_keys.get(a, []))

    # split constraints and activation by target line ℓ_to
    for a in change_like:
        keys = arc_to_keys.get(a, [])
        m.addConstr(
            gp.quicksum(y[a, r] for r in range(R)) ==
            gp.quicksum(x[a, key] for key in keys),
            name=f"chg_split_sum[a{a}]"
        )

        ell_to = int(cgn.arc_line_to[a])
        if ell_to < 0:
            # Shouldn't happen (we exclude 'alight'): deactivate the split if no target line.
            for r in range(R):
                m.addConstr(y[a, r] == 0.0, name=f"chg_split_off[a{a},r{r}]")
            continue

        for r in range(R):
            m.addConstr(
                y[a, r] <= M_arc[a] * delta[(ell_to, r)],
                name=f"chg_split_on[a{a},r{r}]"
            )

    wait_expr_raw = 0.5 * gp.quicksum((1.0 / freq_vals[r]) * y[a, r] for (a, r) in y.keys())
    return wait_expr_raw, y


def add_candidate_choice_per_line(m, model, z_g, cand_by_line, name="cand_line"):
    """
    Add candidate-selection variables per line (direction).
    For each line ℓ with candidates k=0..K-1:
        Sum_k y_{ℓ,k} = z_{g(ℓ)}

    This ties the "line on/off" group decision z_g to choosing exactly one
    candidate path for that line when the group is active.

    Returns:
        y_line dict with Boolean vars keyed by (ℓ, k)
    """
    y = {}
    for ell, cand_list in cand_by_line.items():
        g = int(model.line_idx_to_group[ell])
        if not cand_list:
            # No candidates → leave group logic to other constraints; nothing to add here.
            continue
        vars_ell = []
        for k in range(len(cand_list)):
            y[ell, k] = m.addVar(vtype=gp.GRB.BINARY, name=f"{name}_y[l{ell},k{k}]")
            vars_ell.append(y[ell, k])
        m.addConstr(gp.quicksum(vars_ell) == z_g[g], name=f"{name}_oneof[l{ell}]")
    return y


def add_passenger_capacity_with_candidates_per_line(
    m, model, cgn, x, f_expr, arc_to_keys, Q, y_line, name="pass_cap"
):
    """
    Vehicle capacity with candidate gating:
        For every CGN ride arc r belonging to line ℓ and variant k_r:
            sum_key x[r, key] ≤ Q * f_ℓ(s) * y_{ℓ, k_r}

    This ensures flow can only use the arcs of the chosen candidate variant.
    """
    for r in range(cgn.A):
        if cgn.arc_kind[r] != "ride":
            continue

        ell = int(cgn.arc_line[r])
        k_r = int(cgn.arc_variant[r])   # candidate (variant) index for this ride arc

        keys = arc_to_keys.get(r, [])
        if not keys:
            continue

        flow_sum = gp.quicksum(x[r, key] for key in keys)

        m.addConstr(
            flow_sum <= Q * f_expr[ell] * y_line[ell, k_r],
            name=f"{name}[r{r}]"
        )


def add_infrastructure_capacity_with_candidates_per_line(
    m, model, f_expr, y_line, cand_by_line, cap_per_arc, name="infra_cap"
):
    """
    Infrastructure capacity with candidate gating:
      For each directed infra arc a, sum frequencies of all chosen
      candidate paths that include a, and bound by cap_per_arc[a].
    """
    # Pre-compute coverage: for each infra arc a -> list of (ℓ, k) whose candidate path uses a
    A = int(model.E_dir)
    cover = [[] for _ in range(A)]
    for ell, cand_list in (cand_by_line or {}).items():
        for k, cand in enumerate(cand_list or []):
            for a in cand.get("arcs", []):
                cover[int(a)].append((ell, k))

    for a in range(A):
        if not cover[a]:
            continue
        m.addConstr(
            gp.quicksum(f_expr[ell] * y_line[ell, k] for (ell, k) in cover[a]) <= float(cap_per_arc[a]),
            name=f"{name}[a{a}]"
        )


def build_obj_operating(
    data,
    f_expr: Dict[int, gp.LinExpr],
):
    """
    Operating cost (unweighted):
        Σ_ℓ f_ℓ * (infrastructure length of line ℓ)

    The caller applies the outer weight (op_w) when forming the objective.
    Returns:
        (oper_expr, line_len_list)
    """
    line_len = [
        float(sum(data.len_a[a] for a in data.line_idx_to_arcs[ell]))
        for ell in range(data.L)
    ]
    oper_expr = gp.quicksum(f_expr[ell] * line_len[ell] for ell in range(data.L))
    return oper_expr, line_len


def build_obj_operating_with_candidates_per_line(
    model,
    f_expr: Dict[int, gp.LinExpr],
    y_line: Dict[tuple, gp.Var],                 # y_line[(ell,k)]
    candidates_per_line: Dict[int, List[Dict]],  # {ell: [ {len: ...}, ... ]}
):
    """
    Operating cost with per-line candidate paths:
        Σ_ℓ f_ℓ * (Σ_k y_{ℓ,k} * len_{ℓ,k})

    The caller applies the outer weight (op_w) when forming the objective.
    """
    terms = []
    for ell, cand_list in (candidates_per_line or {}).items():
        if not cand_list:
            continue
        len_expr = gp.quicksum(
            float(c.get("len", 0.0)) * y_line[ell, k]
            for k, c in enumerate(cand_list)
            if (ell, k) in y_line
        )
        terms.append(f_expr[ell] * len_expr)
    return gp.quicksum(terms) if terms else gp.LinExpr(0.0)


def add_path_replanning_cost_linear_per_line(
    m: gp.Model,
    model,
    y: Dict[tuple, gp.Var],                     # y[(ell,k)] ∈ {0,1}
    candidates_per_line: Dict[int, List[Dict]], # {ell: [{arcs,len,add_len,rem_len,delta_len_nom?}, ...]}
    f_expr: Dict[int, gp.LinExpr],              # scenario frequency f_ℓ(s)
    cost_repl_line: float,
    freq_vals: List[int] | None = None,
    *,
    name: str = "repl_path_line",
):
    """
    Linear path-replanning cost within a single scenario s:

        Σ_{ℓ,k} cost_repl_line * Δ_len_{ℓ,k} * ( y_{ℓ,k} * f_ℓ(s) )

    where Δ_len_{ℓ,k} := add_len + rem_len (or provided delta_len_nom),
    always relative to the nominal path of line ℓ.

    The bilinear term y * f is linearized via standard McCormick envelopes.
    Returns:
        LinExpr
    """
    # Upper bound for frequencies to tighten McCormick envelopes
    Fmax = 0.0
    if freq_vals and hasattr(freq_vals, "__iter__"):
        try:
            Fmax = float(max(freq_vals))
        except Exception:
            pass
    if Fmax <= 0.0:
        # Fallback: try model.config["max_frequency"], else 10.0
        mf = getattr(model, "config", {}).get("max_frequency", None) if hasattr(model, "config") else None
        try:
            Fmax = float(mf) if mf is not None else 10.0
        except Exception:
            Fmax = 10.0

    terms = []
    for ell, cand_list in (candidates_per_line or {}).items():
        if not cand_list:
            continue

        # skip if ℓ has no frequency expression in this scenario
        if ell not in f_expr:
            continue
        Fell = f_expr[ell]

        for k, cand in enumerate(cand_list):
            # Δ_len: prefer explicit delta_len_nom, otherwise add_len + rem_len
            delta_len = float(
                cand.get("delta_len_nom",
                         float(cand.get("add_len", 0.0)) + float(cand.get("rem_len", 0.0)))
            )
            if delta_len <= 0.0:
                continue

            y_ellk = y.get((ell, k))
            if y_ellk is None:
                # not modelled → skip silently
                continue

            # McCormick w ≈ Fell * y_ellk
            w = m.addVar(lb=0.0, ub=Fmax, name=f"{name}_w[l{ell},k{k}]")
            m.addConstr(w <= Fmax * y_ellk,               name=f"{name}_w_le_Fy[l{ell},k{k}]")
            m.addConstr(w <= Fell,                        name=f"{name}_w_le_F[l{ell},k{k}]")
            m.addConstr(w >= Fell - Fmax * (1 - y_ellk),  name=f"{name}_w_ge_F_Fy[l{ell},k{k}]")

            terms.append(cost_repl_line * delta_len * w)

    return gp.quicksum(terms) if terms else gp.LinExpr(0.0)


# ============================== Objective ================================

def set_objective(
    m: gp.Model,
    time_expr,
    wait_expr,
    oper_expr,
    *,
    time_w: float = 1.0,
    wait_w: float = 1.0,
    op_w: float = 1.0,
):
    """
    Convenience wrapper to set:
        Minimize  time_w * time_expr  +  wait_w * wait_expr  +  op_w * oper_expr
    """
    m.setObjective(time_w * time_expr + wait_w * wait_expr + op_w * oper_expr, gp.GRB.MINIMIZE)

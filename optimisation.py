from cgn import CGN
import gurobipy as gp
from collections import deque

def od_pairs(data):
    """List of (o,d) with positive demand in index space 0..N-1."""
    return [(o, d) for o in range(data.N) for d in range(data.N) if data.D[o, d] > 0]

def cgn_reachable_for_od(cgn, data, o, d):
    start = cgn.ground_of[o]; goal = cgn.ground_of[d]
    seen = [False]*cgn.V; seen[start] = True
    allowed_arcs = set()
    Q = deque([start])
    while Q:
        v = Q.popleft()
        for a in cgn.out_arcs[v]:
            kind = cgn.arc_kind[a]; w = cgn.arc_head[a]
            if kind == "board"  and v != start:    continue
            if kind == "alight" and w != goal:     continue
            allowed_arcs.add(a)
            if not seen[w]:
                seen[w] = True; Q.append(w)
    visited_nodes = {i for i,b in enumerate(seen) if b}
    return visited_nodes, allowed_arcs

def group_x_keys_by_arc(x):
    """
    Returns a dict a -> [ (o,d), ... ] for which x[a, (o,d)] exists.
    Works with tupledict keys created via addVars(((a,(o,d)) ...)).
    """
    arc_to_ods = {}
    for a, od in x.keys():
        arc_to_ods.setdefault(a, []).append(od)
    return arc_to_ods

def add_flow_conservation_by_origin(m, data, cgn):
    """Aggregate all destinations per origin into one commodity x[a,o]."""
    # Collect destinations per origin (index space 0..N-1)
    dests = {o: [d for d in range(data.N) if data.D[o, d] > 0] for o in range(data.N)}
    origins = [o for o, L in dests.items() if L]

    # Local reachability: allow board only at ground(o), alight only at ground(d) for d in dests[o]
    def reach_nodes_and_arcs_for_origin(o, dest_set):
        start = cgn.ground_of[o]
        goal_grounds = {cgn.ground_of[d] for d in dest_set}
        seen = [False] * cgn.V; seen[start] = True
        allowed = set()
        Q = deque([start])
        while Q:
            v = Q.popleft()
            for a in cgn.out_arcs[v]:
                kind = cgn.arc_kind[a]; w = cgn.arc_head[a]
                if kind == "board"  and v != start:          continue
                if kind == "alight" and w not in goal_grounds: continue
                allowed.add(a)
                if not seen[w]:
                    seen[w] = True; Q.append(w)
        nodes = {i for i, b in enumerate(seen) if b}
        return nodes, allowed

    # Build keys only where reachable
    keys = []
    nodes_by_o = {}
    arcs_by_o  = {}
    for o in origins:
        nodes, arcs = reach_nodes_and_arcs_for_origin(o, dests[o])
        nodes_by_o[o] = nodes
        arcs_by_o[o]  = arcs
        keys.extend((a, o) for a in arcs)

    # Variables x[a,o]
    x = m.addVars(keys, lb=0.0, name="x")

    # Flow conservation: source = ground(o) with supply sum_d D[o,d], sinks = ground(d) with demand D[o,d]
    for o in origins:
        rhs = [0.0] * cgn.V
        supply = float(sum(data.D[o, d] for d in dests[o]))
        rhs[cgn.ground_of[o]] -= supply
        for d in dests[o]:
            rhs[cgn.ground_of[d]] += float(data.D[o, d])

        nodes   = nodes_by_o[o]
        allowed = arcs_by_o[o]
        for v in nodes:
            m.addConstr(
                gp.quicksum(x[a, o] for a in cgn.in_arcs[v]  if a in allowed) -
                gp.quicksum(x[a, o] for a in cgn.out_arcs[v] if a in allowed) == rhs[v],
                name=f"flow[v{v},o{o}]"
            )

    # Helper: a -> list of second indices (here: origins) for which x[a,*] exists
    arc_to_keys = {}
    for a, o in x.keys():
        arc_to_keys.setdefault(a, []).append(o)

    return x, arc_to_keys

def add_flow_conservation(m, data, cgn, K):
    # 1) collect only admissible x-keys
    od_nodes = {}
    od_allowed = {}
    keys = []
    for (o, d) in K:
        nodes, arcs = cgn_reachable_for_od(cgn, data, o, d)
        od_nodes[(o, d)] = nodes
        od_allowed[(o, d)] = arcs
        keys.extend((a, (o, d)) for a in arcs)

    # 2) create x only for allowed pairs  -> no forbid_* constraints needed
    x = m.addVars(keys, lb=0.0, name="x")

    # 3) flow conservation only on visited nodes for this OD
    for (o, d) in K:
        rhs = [0.0] * cgn.V
        dem = float(data.D[o, d])
        rhs[cgn.ground_of[o]] -= dem   # source
        rhs[cgn.ground_of[d]] += dem   # sink
        nodes   = od_nodes[(o, d)]
        allowed = od_allowed[(o, d)]
        for v in nodes:
            m.addConstr(
                gp.quicksum(x[a, (o, d)] for a in cgn.in_arcs[v]  if a in allowed) -
                gp.quicksum(x[a, (o, d)] for a in cgn.out_arcs[v] if a in allowed) == rhs[v],
                name=f"flow[v{v},od({o},{d})]"
            )
    arc_to_ods = group_x_keys_by_arc(x)
    return x, od_allowed, od_nodes, arc_to_ods

def add_frequency(m, L, freq_vals, z=None):
    """Discrete frequency choice per line; returns (delta, f_expr, h_expr)."""
    R = len(freq_vals)
    delta = m.addVars(L, R, vtype=gp.GRB.BINARY, name="delta")
    if z is None:
        m.addConstrs((gp.quicksum(delta[ell, r] for r in range(R)) == 1 for ell in range(L)),
                     name="pick1")
    else:
        m.addConstrs((gp.quicksum(delta[ell, r] for r in range(R)) == z[ell] for ell in range(L)),
                     name="pick_or_off")
    f_expr = {ell: gp.quicksum(freq_vals[r] * delta[ell, r] for r in range(R)) for ell in range(L)}
    h_expr = {ell: gp.quicksum((1.0 / freq_vals[r]) * delta[ell, r] for r in range(R)) for ell in range(L)}
    return delta, f_expr, h_expr


def add_passenger_capacity(m, domain, data, cgn: CGN, x, f_expr, arc_to_ods):
    Q = int(domain.config.get("train_capacity", 200))
    ride_arcs_by_line = [[] for _ in range(data.L)]
    for a in range(cgn.A):
        if cgn.arc_kind[a] == "ride":
            ride_arcs_by_line[cgn.arc_line[a]].append(a)
    for ell in range(data.L):
        f_ell = f_expr[ell]
        for a in ride_arcs_by_line[ell]:
            ods = arc_to_ods.get(a, [])
            if not ods:
                continue
            m.addConstr(
                gp.quicksum(x[a, od] for od in ods) <= Q * f_ell,
                name=f"veh_cap[a{a},l{ell}]"
            )


def add_infrastructure_capacity(m, domain, data, f_expr, cap_std=None, name="infra_cap"):
    """
    Sum of line frequencies using each directed infra arc <= cap_std.
    Inputs:
      - f_expr[ell]: linear frequency expression from add_frequency(...)
      - cap_std: override; if None, read from properties_general
    """
    cap = int(cap_std if cap_std is not None else domain.props.get("infra_cap_std", 10))

    # Precompute: which lines use which directed infra arc a
    lines_per_arc = [[] for _ in range(data.E_dir)]  # a -> [ell,...]
    for ell, arc_list in enumerate(data.line_idx_to_arcs):
        for a in arc_list:
            lines_per_arc[a].append(ell)

    # Capacity per directed arc
    for a in range(data.E_dir):
        if not lines_per_arc[a]:
            continue  # no candidate line uses this arc; skip constraint row
        m.addConstr(
            gp.quicksum(f_expr[ell] for ell in lines_per_arc[a]) <= cap,
            name=f"{name}[a{a}]"
        )

def build_obj_invehicle(m, data, cgn, x, arc_to_ods, use_t_min_time=True):
    time_a = data.t_min_a if use_t_min_time else data.len_a
    ride = [a for a in range(cgn.A) if cgn.arc_kind[a] == "ride"]
    return gp.quicksum(
        time_a[cgn.arc_edge[a]] * x[a, od]
        for a in ride
        for od in arc_to_ods.get(a, [])
    )


def build_obj_waiting(
    m, data, cgn, x, arc_to_ods, freq_vals, delta,
    include_origin_wait=False,
    waiting_time: float = -1.0
):
    """
    Returns (wait_expr, y_vars_or_None).

    waiting_time >= 0  -> fixed waiting per change/boarding (falls include_origin_wait=True)
    waiting_time < 0   -> half headway via linearisation with freq_vals & delta
    """
    # collect affected arcs
    change_like = [a for a in range(cgn.A) if cgn.arc_kind[a] == "change"]
    if include_origin_wait:
        change_like += [a for a in range(cgn.A) if cgn.arc_kind[a] == "board"]

    # CASE A: fixed waiting time
    if waiting_time is not None and float(waiting_time) >= 0.0:
        w = float(waiting_time)
        wait_expr = w * gp.quicksum(
            x[a, key] for a in change_like for key in arc_to_ods.get(a, [])
        )
        return wait_expr, None

    # ---- CASE B: half-headway (linearised) ----
    R = len(freq_vals)
    y = m.addVars(((a, r) for a in change_like for r in range(R)), lb=0.0, name="chg_split")

    # helper: demand weight for a key (od or origin)
    def _dem_for_key(key) -> float:
        # key is either (o,d) or just o
        if isinstance(key, tuple) and len(key) == 2:
            o, d = key
            return float(data.D[o, d])
        else:
            o = int(key)
            return float(data.D[o, :].sum())  # total demand from origin o

    # tight Big-M per arc = sum of relevant demand over keys that actually have x[a,*]
    M_arc = [0.0] * cgn.A
    for a in change_like:
        M_arc[a] = sum(_dem_for_key(key) for key in arc_to_ods.get(a, []))

    # sum_r y[a,r] equals total flow on arc a (only existing x-keys)
    for a in change_like:
        keys = arc_to_ods.get(a, [])
        m.addConstr(
            gp.quicksum(y[a, r] for r in range(R)) ==
            gp.quicksum(x[a, key] for key in keys),
            name=f"chg_split_sum[a{a}]"
        )
        ell_to = cgn.arc_line_to[a]
        for r in range(R):
            m.addConstr(
                y[a, r] <= M_arc[a] * delta[ell_to, r],
                name=f"chg_split_on[{a},{r}]"
            )

    wait_expr = 0.5 * gp.quicksum((1.0 / freq_vals[r]) * y[a, r] for (a, r) in y.keys())
    return wait_expr, y


def build_obj_operating(domain, data, f_expr):
    """Return linear expression for operating costs and per-line lengths."""
    line_len = [float(sum(data.len_a[a] for a in data.line_idx_to_arcs[ell])) for ell in range(data.L)]
    mult = float(domain.props.get("line_cost_mult", 1.0))
    oper_expr = gp.quicksum(f_expr[ell] * line_len[ell] * mult for ell in range(data.L))
    return oper_expr, line_len

def set_objective(m, time_expr, wait_expr, oper_expr, time_w=1.0, wait_w=1.0, op_w=1.0):
    m.setObjective(time_w * time_expr + wait_w * wait_expr + op_w * oper_expr, gp.GRB.MINIMIZE)


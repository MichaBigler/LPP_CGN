# optimisation.py
# -----------------------------------------------------------------------------
# Reine Modellbausteine: keine Config- oder I/O-Logik.
# Alle "Werte" (Kapazitäten, Gewichte) werden von außen hereingegeben.
# -----------------------------------------------------------------------------

from collections import deque
from typing import Dict, List, Tuple, Iterable, Any, Optional
import numpy as np
import gurobipy as gp
from cgn import CGN


# --------------------------- Hilfsfunktionen ---------------------------

def od_pairs(data) -> List[Tuple[int, int]]:
    """Liste aller (o,d) mit positiver Nachfrage im Indexraum 0..N-1."""
    return [(o, d) for o in range(data.N) for d in range(data.N) if data.D[o, d] > 0]


def cgn_reachable_for_od(cgn: CGN, data, o: int, d: int):
    """
    Erlaube Boarding nur am Origin, Alighting nur am Destination.
    Gibt (besuchte Knoten, erlaubte Arcs) für dieses OD zurück.
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
    Erzeuge Mapping a -> [key,...] für existierende x[a,key].
    'key' ist (o,d) im OD-Fall oder 'o' im Origin-Aggregat-Fall.
    """
    arc_to_keys: Dict[int, List[Any]] = {}
    for a, key in x.keys():
        arc_to_keys.setdefault(a, []).append(key)
    return arc_to_keys


# ----------------------------- Flüsse -----------------------------

def add_flow_conservation_by_origin(m: gp.Model, data, cgn: CGN):
    """
    Aggregiertes Routing: eine Commodity pro Origin.
    Variablen x[a,o]; Flusserhaltung mit Quelle ground(o) und Senken ground(d) für alle d mit D[o,d] > 0.
    """
    # Ziele je Origin
    dests = {o: [d for d in range(data.N) if data.D[o, d] > 0] for o in range(data.N)}
    origins = [o for o, L in dests.items() if L]

    # Erreichbarkeit je Origin (Board nur am Origin, Alight nur an beliebigen gültigen Destinationen)
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

    # Nur zulässige Schlüssel erzeugen
    keys = []
    nodes_by_o: Dict[int, set] = {}
    arcs_by_o: Dict[int, set] = {}
    for o in origins:
        nodes, arcs = reach_nodes_and_arcs_for_origin(o, dests[o])
        nodes_by_o[o] = nodes
        arcs_by_o[o] = arcs
        keys.extend((a, o) for a in arcs)

    # Variablen
    x = m.addVars(keys, lb=0.0, name="x")

    # Flusserhaltung
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

    # a -> [o,...], für die x[a,o] existiert
    arc_to_keys: Dict[int, List[int]] = {}
    for a, o in x.keys():
        arc_to_keys.setdefault(a, []).append(o)

    return x, arc_to_keys


def add_flow_conservation(m: gp.Model, data, cgn: CGN, K: List[Tuple[int, int]]):
    """
    Standard-Routing: eine Commodity je OD-Paar.
    Variablen x[a,(o,d)] nur auf arcs, die für (o,d) erreichbar sind.
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


# ----------------------------- Frequenzen -----------------------------

def add_frequency(m: gp.Model, L: int, freq_vals: List[int], z: gp.MVar | None = None):
    """
    Diskrete Frequenzwahl pro Linie:
      delta[ell,r] ∈ {0,1}, Sum_r delta[ell,r] = 1 (oder = z[ell], falls Linienaktivierung modelliert wird)
      f_expr[ell] = Sum_r f_r * delta[ell,r]
      h_expr[ell] = Sum_r (1/f_r) * delta[ell,r]   (falls benötigt)
    """
    R = len(freq_vals)
    delta = m.addVars(L, R, vtype=gp.GRB.BINARY, name="delta")
    if z is None:
        m.addConstrs(
            (gp.quicksum(delta[ell, r] for r in range(R)) == 1 for ell in range(L)),
            name="pick1",
        )
    else:
        m.addConstrs(
            (gp.quicksum(delta[ell, r] for r in range(R)) == z[ell] for ell in range(L)),
            name="pick_or_off",
        )

    f_expr = {
        ell: gp.quicksum(freq_vals[r] * delta[ell, r] for r in range(R)) for ell in range(L)
    }
    h_expr = {
        ell: gp.quicksum((1.0 / freq_vals[r]) * delta[ell, r] for r in range(R))
        for ell in range(L)
    }
    return delta, f_expr, h_expr

def add_frequency_grouped(m: gp.Model, model, freq_vals: List[int]):
    """
    Group-coupled frequency choice with on/off per group.
    - For each group g: z_g ∈ {0,1}, δ_{g,r} ∈ {0,1},  Σ_r δ_{g,r} = z_g
    - f_g = Σ_r f_r · δ_{g,r},  h_g = Σ_r (1/f_r) · δ_{g,r}
    - For each line ℓ in group g: f_ℓ := f_g, h_ℓ := h_g
    Returns:
      z_g:         gp.tupledict keyed by group id
      delta_line:  dict keyed by (ℓ, r) to reuse existing waiting-time code
      f_expr:      dict ℓ -> LinExpr
      h_expr:      dict ℓ -> LinExpr
    """
    groups = sorted(model.line_group_to_lines.keys())
    R = len(freq_vals)

    # on/off per group
    z_g = m.addVars(groups, vtype=gp.GRB.BINARY, name="z_group")

    # frequency pick per group
    delta_g = m.addVars(((g, r) for g in groups for r in range(R)),
                        vtype=gp.GRB.BINARY, name="delta_g")

    # exactly one freq if on; zero if off
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
            # reuse existing waiting-time linearisation without touching its signature
            delta_line[(ell, r)] = delta_g[g, r]

    return z_g, delta_line, f_expr, h_expr
# ----------------------------- Kapazitäten -----------------------------

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
    Fahrzeugkapazität je Ride-Arc a einer Linie ℓ:
      Sum_key x[a,key] ≤ Q * f_ℓ
    """
    # Ride-Arcs pro Linie sammeln
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
    Use EITHER a global scalar (cap_std) OR per-arc capacities (cap_per_arc).
    If cap_per_arc is provided, it overrides cap_std.
    """
    # lines per directed infra arc
    lines_per_arc: List[List[int]] = [[] for _ in range(data.E_dir)]
    for ell, arc_list in enumerate(data.line_idx_to_arcs):
        for a in arc_list:
            lines_per_arc[a].append(ell)

    # RHS selector
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


# ----------------------------- Kostenblöcke -----------------------------

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
    In-Vehicle-Zeit: Sum_{Ride-Arcs a, keys} t_a * x[a,key]
    t_a = t_min_a (oder Länge)
    """
    time_a = data.t_min_a if use_t_min_time else data.len_a
    ride = [a for a in range(cgn.A) if cgn.arc_kind[a] == "ride"]
    return gp.quicksum(
        time_a[cgn.arc_edge[a]] * x[a, key]
        for a in ride
        for key in arc_to_keys.get(a, [])
    )


def build_obj_waiting(
    m, data, cgn, x, arc_to_ods, freq_vals, delta,
    include_origin_wait=False,
    waiting_time_frequency=True
):
    """
    Returns (wait_expr_raw, y_vars_or_None).

    waiting_time_frequency == True  -> half-headway via linearisation (depends on chosen frequencies)
    waiting_time_frequency == False -> flat per-change penalty: 1.0 * sum flow on change-like arcs
                                      (the multiplier comes from waiting_time_cost_mult in the objective)
    """
    # betroffene Arcs einsammeln
    change_like = [a for a in range(cgn.A) if cgn.arc_kind[a] == "change"]
    if include_origin_wait:
        change_like += [a for a in range(cgn.A) if cgn.arc_kind[a] == "board"]

    if not waiting_time_frequency:
        # --- FLAT: keine y-Variablen, 1.0 pro "Ereignis" (Board/Change) * Flow
        wait_expr_raw = gp.quicksum(
            x[a, key] for a in change_like for key in arc_to_ods.get(a, [])
        )
        return wait_expr_raw, None

    # --- FREQ-abhängig: linearisierte half-headway-Kosten
    R = len(freq_vals)
    y = m.addVars(((a, r) for a in change_like for r in range(R)), lb=0.0, name="chg_split")

    # Demand je Key (funktioniert für (o,d) und für o)
    def _dem_for_key(key) -> float:
        if isinstance(key, tuple) and len(key) == 2:
            o, d = key
            return float(data.D[o, d])
        else:
            o = int(key)
            return float(data.D[o, :].sum())

    # enges Big-M pro Arc
    M_arc = [0.0] * cgn.A
    for a in change_like:
        M_arc[a] = sum(_dem_for_key(key) for key in arc_to_ods.get(a, []))

    # Splitting-Bedingungen
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

    wait_expr_raw = 0.5 * gp.quicksum((1.0 / freq_vals[r]) * y[a, r] for (a, r) in y.keys())
    return wait_expr_raw, y


def build_obj_operating(
    data,
    f_expr: Dict[int, gp.LinExpr],
):
    """
    Betriebsausdruck ohne Gewichtung:
      Sum_ℓ f_ℓ * (Linienlänge_ℓ)
    (Gewichtung passiert außen in set_objective via op_w.)
    """
    line_len = [
        float(sum(data.len_a[a] for a in data.line_idx_to_arcs[ell]))
        for ell in range(data.L)
    ]
    oper_expr = gp.quicksum(f_expr[ell] * line_len[ell] for ell in range(data.L))
    return oper_expr, line_len


# ----------------------------- Zielfunktion -----------------------------

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
    """Setze Zielfunktion: time_w * time + wait_w * wait + op_w * oper."""
    m.setObjective(time_w * time_expr + wait_w * wait_expr + op_w * oper_expr, gp.GRB.MINIMIZE)

# optimisation.py
# -----------------------------------------------------------------------------
# Reine Modellbausteine: keine Config- oder I/O-Logik.
# Alle "Werte" (Kapazitäten, Gewichte) werden von außen hereingegeben.
# -----------------------------------------------------------------------------

from collections import deque
from typing import Dict, List, Tuple, Iterable, Any, Optional, TypeGuard, Union
import numpy as np
import gurobipy as gp
from prepare_cgn import CGN


# --------------------------- Hilfsfunktionen ---------------------------

def od_pairs(data) -> List[Tuple[int, int]]:
    """Liste aller (o,d) mit positiver Nachfrage im Indexraum 0..N-1."""
    return [(o, d) for o in range(data.N) for d in range(data.N) if data.D[o, d] > 0]

ODKey = Tuple[int, int]
OriginKey = int
FlowKey = Union[ODKey, OriginKey]

def _is_od_key(x: object) -> TypeGuard[ODKey]:
    return (
        isinstance(x, tuple)
        and len(x) == 2
        and all(isinstance(t, (int, np.integer)) for t in x)
    )


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
    m, data, cgn, x, arc_to_keys, freq_vals, delta,
    include_origin_wait=False,
    waiting_time_frequency=True
):
    """
    Returns (wait_expr_raw, y_vars_or_None).

    waiting_time_frequency == True:
       half-headway via linearisation using selected frequencies per target line.
       Uses cgn.arc_line_to[a] for board/change arcs.
    waiting_time_frequency == False:
       flat penalty: 1.0 * sum flow on change-like arcs (board optional).
    """
    # collect change-like arcs
    change_like = [a for a in range(cgn.A) if cgn.arc_kind[a] == "change"]
    if include_origin_wait:
        change_like += [a for a in range(cgn.A) if cgn.arc_kind[a] == "board"]

    if not waiting_time_frequency:
        wait_expr_raw = gp.quicksum(
            x[a, key] for a in change_like for key in arc_to_keys.get(a, [])
        )
        return wait_expr_raw, None

    R = len(freq_vals)
    y = m.addVars(((a, r) for a in change_like for r in range(R)), lb=0.0, name="chg_split")

    # demand per key, supports (o,d) and o
    def _dem_for_key(key: object) -> float:
        if isinstance(key, tuple) and len(key) == 2:
            o, d = key
            return float(data.D[int(o), int(d)])
        elif isinstance(key, (int, np.integer)):
            o = int(key)
            return float(np.asarray(data.D[o, :]).sum())
        else:
            raise TypeError(f"Unsupported flow key type: {type(key)} -> {key}")

    # tight Big-M per arc (sum flows on that arc)
    M_arc = [0.0] * cgn.A
    for a in change_like:
        M_arc[a] = sum(_dem_for_key(key) for key in arc_to_keys.get(a, []))

    # split constraints and activation by target line ell_to
    for a in change_like:
        keys = arc_to_keys.get(a, [])
        m.addConstr(
            gp.quicksum(y[a, r] for r in range(R)) ==
            gp.quicksum(x[a, key] for key in keys),
            name=f"chg_split_sum[a{a}]"
        )

        ell_to = int(cgn.arc_line_to[a])
        if ell_to < 0:
            # shouldn't happen (we excluded 'alight')
            # bind y[a,r] = 0 if no target line
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
    z_g: gp.tupledict group on/off (aus add_frequency_grouped)
    cand_by_line[ell] = Liste Kandidaten
    Returns: y_line[(ell,k)] ∈ {0,1},  Sum_k y_line[(ell,k)] = z_g[g(ell)]
    """
    y = {}
    for ell, cand_list in cand_by_line.items():
        g = int(model.line_idx_to_group[ell])
        if not cand_list:
            # keine Kandidaten -> Linie muss aus sein
            #m.addConstr(z_g[g] == 0, name=f"{name}_force_off[g{g},l{ell}]")
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
    Für jeden CGN-Ride-Arc r (gehört zu Linie ell und Variante k_r):
        Sum_{flows auf r} x[r,·]  ≤  Q * f_ell(s) * y_{ell,k_r}

    - Damit fließt nur dann etwas auf genau diesem Varianten-Arc, wenn GENAU diese Variante gewählt ist.
    - f_expr[ell] ist die (Szenario-)Frequenz der Linie ell.
    """
    for r in range(cgn.A):
        if cgn.arc_kind[r] != "ride":
            continue

        ell = int(cgn.arc_line[r])
        k_r = int(cgn.arc_variant[r])   # Variantenindex dieses Ride-Arcs

        # x ist als (r, key) indiziert; arc_to_keys[r] liefert die flow-keys auf r
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
    # Precompute: für jeden infra-Arc a -> Liste von (ell,k), deren Kandidat a enthält
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

def build_obj_operating_with_candidates_per_line(
    model,
    f_expr: Dict[int, gp.LinExpr],
    y_line: Dict[tuple, gp.Var],                 # y_line[(ell,k)]
    candidates_per_line: Dict[int, List[Dict]],  # {ell: [ {len: ...}, ... ]}
):
    """
    Betriebskosten mit Kandidatenpfaden (per Linie):
      Sum_ell f_ell * ( Sum_k y_{ell,k} * len_{ell,k} )
    """
    terms = []
    for ell, cand_list in (candidates_per_line or {}).items():
        if not cand_list:
            continue
        len_expr = gp.quicksum(float(c.get("len", 0.0)) * y_line[ell, k]
                               for k, c in enumerate(cand_list)
                               if (ell, k) in y_line)
        terms.append(f_expr[ell] * len_expr)
    return gp.quicksum(terms) if terms else gp.LinExpr(0.0)



def add_path_replanning_cost_linear_per_line(
    m: gp.Model,
    model,
    y: Dict[tuple, gp.Var],                   # y[(ell,k)] ∈ {0,1}
    candidates_per_line: Dict[int, List[Dict]],  # {ell: [ {arcs,len,add_len,rem_len,delta_len_nom?}, ... ]}
    f_expr: Dict[int, gp.LinExpr],            # ***Szenario***-Frequenz je Linie ℓ
    cost_repl_line: float,
    freq_vals: List[int] | None = None,
    *,
    name: str = "repl_path_line",
):
    """
    Path-Penalty in EINEM Szenario s (pro Linie):
      Sum_{ell,k} cost_repl_line * (add_len + rem_len)_{ell,k} * (y_{ell,k} * f_ell(s))

    - (add_len + rem_len) ist absoluter Umbau (entfernte + hinzugefügte Kantenlängen),
      immer relativ zum NOMINALPFAD DIESER LINIE ℓ.
    - f_ell(s) ist die ***Szenario***-Frequenz dieser Linie.
    - y_{ell,k} wählt genau einen Kandidaten pro Linie (oder 0, falls Linie aus).
    - Das Produkt y * f wird via McCormick linearisiert.

    Rückgabe: LinExpr (Summe der Path-Replanning-Kosten in diesem Szenario).
    """
    # obere Schranke für Frequenzen
    Fmax = 0.0
    if freq_vals and hasattr(freq_vals, "__iter__"):
        try:
            Fmax = float(max(freq_vals))   # sicher aus Liste/Tuple/np.array
        except Exception:
            pass
    if Fmax <= 0.0:
        # Fallback: benutze max_frequency aus dem Model, wenn vorhanden
        mf = getattr(model, "config", {}).get("max_frequency", None) if hasattr(model, "config") else None
        try:
            Fmax = float(mf) if mf is not None else 10.0
        except Exception:
            Fmax = 10.0

    terms = []
    for ell, cand_list in (candidates_per_line or {}).items():
        if not cand_list:
            continue
        # Szenario-Frequenz dieser Linie
        if ell not in f_expr:
            # Linie existiert im Szenario nicht (oder wurde entkoppelt) -> nichts addieren
            continue
        Fell = f_expr[ell]

        for k, cand in enumerate(cand_list):
            # Delta-Länge (immer nominal vs. Kandidat). Bevorzugt explizit, sonst add+rem.
            delta_len = float(
                cand.get("delta_len_nom",
                         float(cand.get("add_len", 0.0)) + float(cand.get("rem_len", 0.0)))
            )
            if delta_len <= 0.0:
                continue

            y_ellk = y.get((ell, k))
            if y_ellk is None:
                # für Sicherheit: überspringen, falls diese Kombination nicht modelliert wurde
                continue

            # McCormick-Linearisation von w ≈ Fell * y_ellk
            w = m.addVar(lb=0.0, ub=Fmax, name=f"{name}_w[l{ell},k{k}]")
            m.addConstr(w <= Fmax * y_ellk,               name=f"{name}_w_le_Fy[l{ell},k{k}]")
            m.addConstr(w <= Fell,                        name=f"{name}_w_le_F[l{ell},k{k}]")
            m.addConstr(w >= Fell - Fmax * (1 - y_ellk),  name=f"{name}_w_ge_F_Fy[l{ell},k{k}]")

            terms.append(cost_repl_line * delta_len * w)

    return gp.quicksum(terms) if terms else gp.LinExpr(0.0)

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

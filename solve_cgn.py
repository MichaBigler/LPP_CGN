# solve_cgn.py
import gurobipy as gp
from gurobipy import GRB
from cgn import make_cgn
from debug_cgn import precheck_all_od_connectivity
from debug_iis import dump_iis
from optimisation import (  # <- ersetze durch deinen Modulnamen/Datei
    od_pairs, add_flow_conservation, add_flow_conservation_by_origin, add_frequency,
    add_passenger_capacity, add_infrastructure_capacity,
    build_obj_invehicle, build_obj_waiting, build_obj_operating, set_objective
)

def build_and_solve(domain, data,
                    freq_vals=None,
                    include_origin_wait=False,
                    select_lines=False, waiting_time=-1.0,
                    routing_agg=False,
                    time_w=1.0, wait_w=1.0, op_w=1.0,
                    gurobi_params=None):
    """
    End-to-end: build CGN model, add constraints/objective, solve.
    Returns (model, artifacts dict).
    """
    # --- defaults from config/properties ---
    if freq_vals is None:
        fmax = int(domain.config.get("max_frequency", 5))
        freq_vals = list(range(1, fmax + 1))

    # --- model + CGN + OD set ---
    m = gp.Model("LPP_CGN")
    m.Params.Threads     = 0
    cgn = make_cgn(data)
    K = od_pairs(data)

    precheck_all_od_connectivity(cgn, m, K)

    # --- passenger flow conservation on CGN ---

    if routing_agg:
        # one commodity per origin
        x, arc_to_keys = add_flow_conservation_by_origin(m, data, cgn)
    else:
        # default: per-OD commodities
        x, _, _, arc_to_keys = add_flow_conservation(m, data, cgn, K)

    # --- frequency assignment (optionally with line selection) ---
    z = None
    if select_lines:
        z = m.addVars(data.L, vtype=GRB.BINARY, name="z")
    delta, f_expr, h_expr = add_frequency(m, data.L, freq_vals, z=z)


    # capacities
    add_passenger_capacity(m, domain, data, cgn, x, f_expr, arc_to_keys)
    add_infrastructure_capacity(m, domain, data, f_expr)

    # --- costs / objective ---
    time_expr = build_obj_invehicle(m, data, cgn, x, arc_to_keys, use_t_min_time=True)
    wait_expr, y = build_obj_waiting(
        m, data, cgn, x, arc_to_keys, freq_vals, delta,
        include_origin_wait=include_origin_wait,
        waiting_time=waiting_time
    )
    oper_expr, line_len = build_obj_operating(domain, data, f_expr)

    # Dynamische Wartekosten-Gewichtung:
    # waiting_time < 0  -> half-headway UND Gewicht = abs(waiting_time)
    # waiting_time >= 0 -> fixe Wartezeit, Gewicht = 1.0
    wait_weight = 1.0
    try:
        wt = float(waiting_time)
        if wt < 0.0:
            wait_weight = abs(wt)
    except Exception:
        pass

    set_objective(m, time_expr, wait_expr, oper_expr,
                time_w=time_w, wait_w=wait_weight, op_w=op_w)

    # --- parameters + solve ---
    if gurobi_params:
        for k, v in gurobi_params.items():
            setattr(m.Params, k, v)
    # sensible defaults from config
    if "time_limit" in domain.config:
        m.Params.TimeLimit = int(domain.config["time_limit"])
    if "mip_gap" in domain.config:
        m.Params.MIPGap = float(domain.config["mip_gap"])

    m.optimize()
    if m.Status == gp.GRB.INFEASIBLE:
        dump_iis(m)

    # --- basic results ---
    chosen_freq = {}
    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        # decode chosen frequency per line
        for ell in range(data.L):
            f = sum(freq_vals[r] for r in range(len(freq_vals)) if delta[ell, r].X > 0.5)
            chosen_freq[ell] = f

    # Kosten-Breakdown (wenn Lösung vorhanden)
    costs = None
    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
        try:
            costs = dict(
                time=float(time_expr.getValue()),
                wait=float(wait_expr.getValue()),
                oper=float(oper_expr.getValue()),
                objective=float(m.ObjVal)
            )
        except Exception:
            # Falls kein Wert evaluierbar (extrem selten): None lassen
            pass

    artifacts = dict(
        routing_agg=routing_agg, cgn=cgn, x=x, delta=delta, f_expr=f_expr, h_expr=h_expr, y=y,
        time_expr=time_expr, wait_expr=wait_expr, oper_expr=oper_expr,
        line_len=line_len, chosen_freq=chosen_freq, costs=costs
    )
    return m, artifacts

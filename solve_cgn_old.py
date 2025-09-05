# solve_cgn.py
import gurobipy as gp
from gurobipy import GRB
from cgn import make_cgn
from optimisation import (
    od_pairs, add_flow_conservation, add_flow_conservation_by_origin, add_frequency, add_frequency_grouped,
    add_passenger_capacity, add_infrastructure_capacity,
    build_obj_invehicle, build_obj_waiting, build_obj_operating, set_objective
)

def build_and_solve(
    domain, model,
    *,
    freq_vals=None,
    include_origin_wait=True,       # default = True
    select_lines=False,
    waiting_time_frequency=True,    # <— NEU: bool aus Config
    routing_agg=False,
    gurobi_params=None
):
    # --- Frequenzwerte
    if freq_vals is None:
        fmax = int(domain.config.get("max_frequency", 5))
        freq_vals = list(range(1, fmax + 1))

    # --- Model + CGN + OD-Set
    m = gp.Model("LPP_CGN")
    m.Params.Threads = 0
    cgn = make_cgn(model)
    K   = od_pairs(model)

    # --- Flüsse
    if str(routing_agg).lower() in ("1","true","origin","by_origin","o","yes","y"):
        x, arc_to_keys = add_flow_conservation_by_origin(m, model, cgn)
    else:
        x, _, _, arc_to_keys = add_flow_conservation(m, model, cgn, K)

    # --- Frequenzen
    if freq_vals is None:
        freq_vals = domain.config.get("freq_values")
    if not freq_vals:
        fmax = int(domain.config.get("max_frequency", 5))
        freq_vals = list(range(1, fmax + 1))

    # --- Frequenzen: gruppen-gekoppelt + on/off pro Gruppe
    z_g, delta_line, f_expr, h_expr = add_frequency_grouped(m, model, freq_vals)

    # --- Kapazitäten
    Q_cap   = int(domain.config.get("train_capacity", 200))
    cap_std = int(domain.config.get("infrastructure_capacity", 10))
    add_passenger_capacity(m, model, cgn, x, f_expr, arc_to_keys, Q=Q_cap)
    add_infrastructure_capacity(m, model, f_expr, cap_std=cap_std)

    # --- Kosten
    time_expr = build_obj_invehicle(m, model, cgn, x, arc_to_keys, use_t_min_time=True)

    # WICHTIG: hier das neue Flag nutzen
    wait_expr, y = build_obj_waiting(
        m, model, cgn, x, arc_to_keys, freq_vals, delta_line,  # <- hier delta_line statt delta
        include_origin_wait=include_origin_wait,
        waiting_time_frequency=bool(domain.config.get("waiting_time_frequency", True))
    )

    oper_expr, line_len = build_obj_operating(model, f_expr)

    # Gewichte aus Config
    time_w = float(domain.config.get("travel_time_cost_mult", 1.0))
    wait_w = float(domain.config.get("waiting_time_cost_mult", 1.0))
    op_w   = float(domain.config.get("line_operation_cost_mult", 1.0))

    set_objective(m, time_expr, wait_expr, oper_expr, time_w=time_w, wait_w=wait_w, op_w=op_w)

    # --- Gurobi-Parameter
    if gurobi_params:
        for k, v in gurobi_params.items():
            setattr(m.Params, k, v)
    if "time_limit" in domain.config:
        m.Params.TimeLimit = int(domain.config["time_limit"])
    if "mip_gap" in domain.config:
        m.Params.MIPGap = float(domain.config["mip_gap"])

    # --- Optimieren
    m.optimize()

    # --- Frequenzen dekodieren
    chosen_freq = {}
    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        R = len(freq_vals)
        for ell in range(model.L):
            f = 0
            for r in range(R):
                var = delta_line.get((ell, r))
                if var is not None and var.X > 0.5:
                    f = freq_vals[r]
                    break
            chosen_freq[ell] = f

    # --- Kosten-Breakdown
    costs = {}
    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
        v_time = float(time_expr.getValue())
        v_wait_raw = float(wait_expr.getValue())
        v_oper = float(oper_expr.getValue())
        v_wait = wait_w * v_wait_raw
        obj_val = time_w * v_time + v_wait + op_w * v_oper
        costs = {
            "time": v_time,
            "wait_raw": v_wait_raw,
            "wait": v_wait,               # GEWICHTET
            "oper": v_oper,
            "objective": obj_val,
            "wait_mode": "freq" if waiting_time_frequency else "flat",
        }

    # kompakter Solution-Block (falls du ihn im run.py nutzt)
    solution = {
        "status_code": int(m.Status),
        "status": m.Status,
        "objective": costs.get("objective"),
        "runtime_s": getattr(m, "Runtime", None),
        "chosen_freq": chosen_freq,
        "costs": costs,
    }

    artifacts = dict(
        cgn=cgn, K=K, x=x, delta=delta_line, f_expr=f_expr, h_expr=h_expr, y=y,
        time_expr=time_expr, wait_expr=wait_expr, oper_expr=oper_expr,
        line_len=line_len, chosen_freq=chosen_freq, costs=costs
    )
    return m, solution, artifacts

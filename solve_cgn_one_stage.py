# solve_cgn.py
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from prepare_cgn import make_cgn
from optimisation import (
    add_passenger_capacity, add_infrastructure_capacity,
    build_obj_invehicle, build_obj_waiting, build_obj_operating, set_objective,
    add_frequency_grouped,
)
from solve_utils import (
    _freq_values_from_config, _routing_is_aggregated,
    _waiting_mode, _add_flows
)


# ---------- ONE STAGE (nominal, with global infra cap) ----------

def solve_one_stage(domain, model, *, gurobi_params=None):
    m = gp.Model("LPP_ONE_STAGE")
    cgn = make_cgn(model)

    freq_vals = _freq_values_from_config(domain)
    aggregated = _routing_is_aggregated(domain)
    wait_freq = _waiting_mode(domain)

    # flows
    x0, arc_to0 = _add_flows(m, model, cgn, aggregated)

    # grouped frequencies (with on/off via z_g)
    z0, delta0, f0_expr, h0_expr = add_frequency_grouped(m, model, freq_vals)

    # capacities
    Q = int(domain.config.get("train_capacity", 200))
    add_passenger_capacity(m, model, cgn, x0, f0_expr, arc_to0, Q=Q)
    cap_std = int(domain.config.get("infrastructure_capacity", 10))
    add_infrastructure_capacity(m, model, f0_expr, cap_std=cap_std)

    # costs
    time0 = build_obj_invehicle(m, model, cgn, x0, arc_to0, use_t_min_time=True)
    wait0, y0 = build_obj_waiting(m, model, cgn, x0, arc_to0, freq_vals, delta0,
                                  include_origin_wait=True,
                                  waiting_time_frequency=wait_freq)
    oper0, line_len = build_obj_operating(model, f0_expr)

    # weights
    time_w = float(domain.config.get("travel_time_cost_mult", 1.0))
    wait_w = float(domain.config.get("waiting_time_cost_mult", 1.0))
    op_w   = float(domain.config.get("line_operation_cost_mult", 1.0))

    set_objective(m, time0, wait0, oper0, time_w=time_w, wait_w=wait_w, op_w=op_w)

    # params
    if gurobi_params:
        for k, v in gurobi_params.items():
            setattr(m.Params, k, v)
    if "time_limit" in domain.config:
        m.Params.TimeLimit = int(domain.config["time_limit"])
    if "mip_gap" in domain.config:
        m.Params.MIPGap = float(domain.config["mip_gap"])
    elif "gap" in domain.config:
        m.Params.MIPGap = float(domain.config["gap"])

    m.optimize()

    # decode line frequencies (inherit group decision)
    chosen_freq0 = {}
    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        for ell in range(model.L):
            f = 0
            for r, _ in enumerate(freq_vals):
                if delta0[(ell, r)].X > 0.5:
                    f = freq_vals[r]; break
            chosen_freq0[ell] = f

    # costs
    costs0 = {}
    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
        v_time0_raw = float(time0.getValue())
        v_wait0_raw = float(wait0.getValue())
        v_oper0_raw = float(oper0.getValue())

        v_time0 = time_w * v_time0_raw
        v_wait0 = wait_w * v_wait0_raw
        v_oper0 =   op_w * v_oper0_raw

        obj0 = v_time0 + v_wait0 + v_oper0
        costs0 = dict(
            time=v_time0, wait=v_wait0, oper=v_oper0, objective=obj0,
            time_raw=v_time0_raw, wait_raw=v_wait0_raw, oper_raw=v_oper0_raw
        )

    solution = dict(
        status_code=int(m.Status),
        status=m.Status,
        objective=costs0.get("objective"),
        runtime_s=getattr(m, "Runtime", None),
        chosen_freq=chosen_freq0,
        costs_0=costs0
    )
    artifacts = dict(model=m, cgn=cgn, delta0=delta0, f0_expr=f0_expr,
                     x0=x0, y0=y0, line_len=line_len)
    return m, solution, artifacts


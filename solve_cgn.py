# solve_cgn.py
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from cgn import make_cgn
from optimisation import (
    od_pairs, add_flow_conservation, add_flow_conservation_by_origin,
    add_passenger_capacity, add_infrastructure_capacity,
    build_obj_invehicle, build_obj_waiting, build_obj_operating, set_objective,
    add_frequency_grouped
)

# ---------- small utilities ----------

def _freq_values_from_config(domain):
    vals = domain.config.get("freq_values")
    if vals:
        return list(map(int, vals))
    fmax = int(domain.config.get("max_frequency", 5))
    return list(range(1, fmax + 1))

def _routing_is_aggregated(domain, override=None):
    if override is not None:
        return bool(override)
    return bool(domain.config.get("routing_agg", False))

def _waiting_mode(domain, override=None):
    if override is not None:
        return bool(override)
    return bool(domain.config.get("waiting_time_frequency", True))

def _add_flows(m, model, cgn, aggregated: bool):
    if aggregated:
        x, arc_to_keys = add_flow_conservation_by_origin(m, model, cgn)
    else:
        K = od_pairs(model)
        x, _, _, arc_to_keys = add_flow_conservation(m, model, cgn, K)
    return x, arc_to_keys

def _line_lengths(model):
    # pure lengths per line (sum of infra arc lengths)
    return [
        float(sum(model.len_a[a] for a in model.line_idx_to_arcs[ell]))
        for ell in range(model.L)
    ]

def _group_lengths(model, line_len):
    # sum of both directions' lengths per group
    gl = {}
    for ell in range(model.L):
        g = model.line_idx_to_group[ell]
        gl[g] = gl.get(g, 0.0) + line_len[ell]
    return gl  # dict g -> length

def _rep_line_of_group(model):
    # pick the first line index as group representative
    rep = {}
    for g, (fwd, bwd) in model.line_group_to_lines.items():
        rep[g] = fwd if fwd >= 0 else (bwd if bwd >= 0 else None)
    return rep  # dict g -> ell


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
        v_time = float(time0.getValue())
        v_wait_raw = float(wait0.getValue())
        v_oper = float(oper0.getValue())
        obj_val = time_w * v_time + wait_w * v_wait_raw + op_w * v_oper
        costs0 = dict(time=v_time, wait_raw=v_wait_raw, wait=wait_w * v_wait_raw,
                      oper=v_oper, objective=obj_val)

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


# ---------- TWO-STAGE INTEGRATED (joint) ----------

def solve_two_stage_integrated(domain, model, *, gurobi_params=None):
    m = gp.Model("LPP_TWO_STAGE_INTEGRATED")
    cgn = make_cgn(model)

    freq_vals = _freq_values_from_config(domain)
    aggregated = _routing_is_aggregated(domain, override=domain.config.get("routing_agg", True))  # suggest True for scale
    wait_freq = _waiting_mode(domain)

    S = len(model.p_s)
    line_len = _line_lengths(model)
    glen = _group_lengths(model, line_len)
    rep = _rep_line_of_group(model)

    # Stage 1: flows + grouped freq
    x0, arc_to0 = _add_flows(m, model, cgn, aggregated)
    z0, delta0, f0_expr, h0_expr = add_frequency_grouped(m, model, freq_vals)

    # Stage 1 capacities: global cap_std (not scenario-specific)
    Q = int(domain.config.get("train_capacity", 200))
    add_passenger_capacity(m, model, cgn, x0, f0_expr, arc_to0, Q=Q)
    cap_std = int(domain.config.get("infrastructure_capacity", 10))
    add_infrastructure_capacity(m, model, f0_expr, cap_std=cap_std)

    # Stage 1 costs
    time0 = build_obj_invehicle(m, model, cgn, x0, arc_to0, use_t_min_time=True)
    wait0, y0 = build_obj_waiting(m, model, cgn, x0, arc_to0, freq_vals, delta0,
                                  include_origin_wait=True,
                                  waiting_time_frequency=wait_freq)
    oper0, _ = build_obj_operating(model, f0_expr)

    # Stage 2 per scenario: flows + grouped freq + non-anticipativity on z
    xs, arcs_to_s = [], []
    zs, deltas, f_expr_s, y_s, time_s, wait_s, oper_s = [], [], [], [], [], [], []

    for s in range(S):
        xi, arc_toi = _add_flows(m, model, cgn, aggregated)
        zs_i, delta_i, f_expr_i, h_expr_i = add_frequency_grouped(m, model, freq_vals)

        # non-anticipativity: on/off identical to Stage 1
        #for g in zs_i.keys():
        #    m.addConstr(zs_i[g] == z0[g], name=f"nonant_z[g{g},s{s}]")

        # capacities scenario-specific
        add_passenger_capacity(m, model, cgn, xi, f_expr_i, arc_toi, Q=Q)
        add_infrastructure_capacity(m, model, f_expr_i, cap_per_arc=model.cap_sa[s, :])

        # costs
        ti = build_obj_invehicle(m, model, cgn, xi, arc_toi, use_t_min_time=True)
        wi, yi = build_obj_waiting(m, model, cgn, xi, arc_toi, freq_vals, delta_i,
                                   include_origin_wait=True,
                                   waiting_time_frequency=wait_freq)
        oi, _ = build_obj_operating(model, f_expr_i)

        xs.append(xi); arcs_to_s.append(arc_toi)
        zs.append(zs_i); deltas.append(delta_i); f_expr_s.append(f_expr_i)
        y_s.append(yi); time_s.append(ti); wait_s.append(wi); oper_s.append(oi)

    # Replanning deviation variables: d_{g,s} >= |f_g^(s) - f_g^(0)|
    d = {}
    for s in range(S):
        for g, ell_rep in rep.items():
            if ell_rep is None: 
                continue
            d[g, s] = m.addVar(lb=0.0, name=f"d[g{g},s{s}]")
            m.addConstr(d[g, s] >= f_expr_s[s][ell_rep] - f0_expr[ell_rep],
                        name=f"d_pos[g{g},s{s}]")
            m.addConstr(d[g, s] >= -(f_expr_s[s][ell_rep] - f0_expr[ell_rep]),
                        name=f"d_neg[g{g},s{s}]")

    # weights and probabilities
    time_w = float(domain.config.get("travel_time_cost_mult", 1.0))
    wait_w = float(domain.config.get("waiting_time_cost_mult", 1.0))
    op_w   = float(domain.config.get("line_operation_cost_mult", 1.0))
    c_repl = float(domain.config.get("cost_repl_freq", 0.0))
    p = model.p_s  # numpy array of length S

    # objective
    stage1_obj = time_w * time0 + wait_w * wait0 + op_w * oper0

    stage2_exp = gp.quicksum(
        float(p[s]) * (time_w * time_s[s] + wait_w * wait_s[s] + op_w * oper_s[s])
        for s in range(S)
    )

    repl_exp = gp.quicksum(
        float(p[s]) * gp.quicksum(c_repl * float(glen[g]) * d[g, s] for g in glen.keys())
        for s in range(S)
    )

    m.setObjective(stage1_obj + stage2_exp + repl_exp, GRB.MINIMIZE)

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

    # decode frequencies
    def _decode(delta_map):
        out = {}
        for ell in range(model.L):
            f = 0
            for r, _ in enumerate(freq_vals):
                if delta_map[(ell, r)].X > 0.5:
                    f = freq_vals[r]; break
            out[ell] = f
        return out

    chosen_freq0 = _decode(delta0)
    chosen_freq_s = [ _decode(deltas[s]) for s in range(S) ]

    # cost extraction
    res = dict()
    costs0 = {}
    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
        v_time0 = float(time0.getValue())
        v_wait0_raw = float(wait0.getValue())
        v_oper0 = float(oper0.getValue())
        obj0 = time_w * v_time0 + wait_w * v_wait0_raw + op_w * v_oper0
        costs0 = dict(time=v_time0, wait_raw=v_wait0_raw, wait=wait_w * v_wait0_raw, oper=v_oper0, objective=obj0)

    # --- per-scenario details (unweighted components + repl + per-s objective)
    scenario_ids = domain.scen_prob_df["id"].astype(int).tolist()
    scenarios = []
    for s in range(S):
        v_time = float(time_s[s].getValue())
        v_wait_raw = float(wait_s[s].getValue())
        v_oper = float(oper_s[s].getValue())
        v_repl = sum(float(c_repl) * float(glen[g]) * float(d[g, s].X) for g in glen.keys())
        obj_s = time_w * v_time + wait_w * v_wait_raw + op_w * v_oper + v_repl
        scenarios.append({
            "id": int(scenario_ids[s]),
            "prob": float(model.p_s[s]),
            "freq": chosen_freq_s[s],           # per-line freq dict for scenario s
            "cost_time": v_time,                 # unweighted
            "cost_wait": wait_w * v_wait_raw,    # weighted, konsistent zu bisher
            "cost_oper": v_oper,                 # unweighted
            "cost_repl": v_repl,
            "objective": obj_s                   # unweighted time/oper + weighted wait + repl
        })

    # existing aggregates
    v1  = float(stage1_obj.getValue()) if m.SolCount else None
    v2e = float(stage2_exp.getValue()) if m.SolCount else None
    vre = float(repl_exp.getValue())   if m.SolCount else None
    tot = (v1 or 0.0) + (v2e or 0.0) + (vre or 0.0) if m.SolCount else None

    solution = dict(
        status_code=int(m.Status),
        status=m.Status,
        runtime_s=getattr(m, "Runtime", None),
        chosen_freq_stage1=chosen_freq0,
        chosen_freq_stage2=chosen_freq_s,
        scenarios=scenarios,            # <<— NEU
        obj_stage1=v1, obj_stage2_exp=v2e, repl_cost_exp=vre, objective=tot,
        costs_0=costs0
    )
    artifacts = dict(model=m, cgn=cgn, delta0=delta0, deltas=deltas,
                     f0_expr=f0_expr, f_expr_s=f_expr_s, d=d,
                     line_len=line_len, group_len=glen, probs=p)
    return m, solution, artifacts


# ---------- TWO-STAGE SEPARATED (sequential) ----------

def solve_two_stage_separated(domain, model, *, gurobi_params=None):
    # 1) Stage 1
    m0, sol0, art0 = solve_one_stage(domain, model, gurobi_params=gurobi_params)

    # representative by group to reference f_expr
    freq_vals = _freq_values_from_config(domain)
    rep = _rep_line_of_group(model)
    line_len = art0["line_len"]
    glen = _group_lengths(model, line_len)
    Q = int(domain.config.get("train_capacity", 200))
    time_w = float(domain.config.get("travel_time_cost_mult", 1.0))
    wait_w = float(domain.config.get("waiting_time_cost_mult", 1.0))
    op_w   = float(domain.config.get("line_operation_cost_mult", 1.0))
    c_repl = float(domain.config.get("cost_repl_freq", 0.0))
    p = model.p_s
    wait_freq = _waiting_mode(domain)
    aggregated = _routing_is_aggregated(domain)

    chosen_freq0 = sol0.get("chosen_freq", {}) or sol0.get("chosen_freq_stage1", {})
    # turn line -> group frequency by reading any line in the group
    f0_by_group = {}
    for g, ell_rep in rep.items():
        f0_by_group[g] = int(chosen_freq0.get(ell_rep, 0))

    # 2) Stage 2 per scenario
    S = len(model.p_s)
    per_s = []
    for s in range(S):
        m = gp.Model(f"LPP_TWO_STAGE_S{str(s)}")
        cgn = make_cgn(model)

        # flows
        x, arc_to = _add_flows(m, model, cgn, aggregated)

        # grouped frequencies
        zs, delta_s, f_s_expr, h_s_expr = add_frequency_grouped(m, model, freq_vals)

        # FIX 1: on/off equal to Stage 1 via group switch (not sum over deltas)
        #for g in zs.keys():
        #    m.addConstr(zs[g] == (1 if f0_by_group[g] > 0 else 0), name=f"fix_z_g{g}")

        # capacities for scenario s
        add_passenger_capacity(m, model, cgn, x, f_s_expr, arc_to, Q=Q)
        add_infrastructure_capacity(m, model, f_s_expr, cap_per_arc=model.cap_sa[s, :])

        # costs (stage-2)
        time = build_obj_invehicle(m, model, cgn, x, arc_to, use_t_min_time=True)
        wait, y = build_obj_waiting(m, model, cgn, x, arc_to, freq_vals, delta_s,
                                    include_origin_wait=True,
                                    waiting_time_frequency=wait_freq)
        oper, _ = build_obj_operating(model, f_s_expr)

        # FIX 2: replanning with explicit deviation variables (no gp.abs_)
        # d_{g,s} >= | f_g^(s) - f_g^(0) |
        repl_terms = []
        d_vars = {}
        for g, ell_rep in rep.items():
            if ell_rep is None:
                continue
            d = m.addVar(lb=0.0, name=f"d_g{g}_s{s}")
            m.addConstr(d >=  f_s_expr[ell_rep] - float(f0_by_group[g]), name=f"dpos_g{g}_s{s}")
            m.addConstr(d >= -f_s_expr[ell_rep] + float(f0_by_group[g]), name=f"dneg_g{g}_s{s}")
            d_vars[g] = d
            coef = c_repl * float(glen[g])              # weight by group length
            repl_terms.append(coef * d)

        repl = gp.quicksum(repl_terms)

        # stage-2 objective
        m.setObjective(time_w * time + wait_w * wait + op_w * oper + repl, GRB.MINIMIZE)

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

        chosen = {}
        if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            for ell in range(model.L):
                f = 0
                for r, _ in enumerate(freq_vals):
                    if delta_s[(ell, r)].X > 0.5:
                        f = freq_vals[r]; break
                chosen[ell] = f

        # extract components
        time_val = float(time.getValue()) if m.SolCount else None
        wait_raw = float(wait.getValue()) if m.SolCount else None
        oper_val = float(oper.getValue()) if m.SolCount else None
        repl_val = float(repl.getValue()) if m.SolCount else None
        # Note: repl was part of objective; reconstruktionsformel ist robust

        obj_val = float(m.ObjVal) if m.SolCount else None

        per_s.append(dict(
            status=int(m.Status),
            objective=obj_val,
            chosen_freq=chosen,
            cost_time=time_val,
            cost_wait=wait_w * (wait_raw or 0.0) if wait_raw is not None else None,
            cost_oper=oper_val,
            cost_repl=repl_val
        ))


    # expected second-stage objective
    obj2_exp = 0.0
    repl_cost_exp = 0.0

    for s in range(S):
        repl_cost_exp += float(p[s]) * float(per_s[s].get("cost_repl") or 0.0)
        if per_s[s]["objective"] is not None:
            obj2_exp += float(p[s]) * float(per_s[s]["objective"])


    scenario_ids = domain.scen_prob_df["id"].astype(int).tolist()
    scenarios = [{
        "id": int(scenario_ids[s]),
        "prob": float(model.p_s[s]),
        "freq": per_s[s]["chosen_freq"],
        "cost_time": per_s[s]["cost_time"],
        "cost_wait": per_s[s]["cost_wait"],
        "cost_oper": per_s[s]["cost_oper"],
        "cost_repl": per_s[s]["cost_repl"],
        "objective": per_s[s]["objective"]
    } for s in range(S)]

    solution = dict(
        status_code=int(sol0["status_code"]),
        status=sol0["status"],
        runtime_s=sol0.get("runtime_s"),
        chosen_freq_stage1=chosen_freq0,
        chosen_freq_stage2=[ps["chosen_freq"] for ps in per_s],
        scenarios=scenarios,   # <<— NEU
        obj_stage1=sol0["costs_0"]["objective"],
        obj_stage2_exp=obj2_exp,
        repl_cost_exp=repl_cost_exp,
        objective=sol0["costs_0"]["objective"] + obj2_exp,
        costs_0=sol0.get("costs_0")
    )
    artifacts = dict(per_s=per_s, line_len=line_len, group_len=glen, probs=p)
    return None, solution, artifacts  # main model returned as None here (we solved submodels)

# solve_cgn.py
import os
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from prepare_cgn_candidates import make_cgn_with_candidates_per_line
from find_candidates import build_candidates_all_scenarios_per_line_cfg
from optimisation import (
    od_pairs, add_flow_conservation, add_flow_conservation_by_origin,
    add_passenger_capacity, add_infrastructure_capacity,
    build_obj_invehicle, build_obj_waiting, build_obj_operating, build_obj_operating_with_candidates_per_line, set_objective,
    add_frequency_grouped,
    add_candidate_choice_per_line,
    add_passenger_capacity_with_candidates_per_line,
    add_infrastructure_capacity_with_candidates_per_line,
    add_path_replanning_cost_linear_per_line,
)
from solve_utils import (
    _freq_values_from_config, _routing_is_aggregated, _waiting_mode,
    _add_flows, _line_lengths, _group_lengths, _rep_line_of_group, _cand_counts
)

from solve_cgn_one_stage import solve_one_stage
from data_model import CandidateConfig

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
    detour_cnt, ksp_cnt = _cand_counts(domain)

    cand_cfg = getattr(domain, "cand_cfg", None)
    if cand_cfg is None:
        
        cand_cfg = CandidateConfig()  # Fallback-Defaults

    cand_all_lines = build_candidates_all_scenarios_per_line_cfg(model, cand_cfg, domain.config)
    lines_by_group = {}
    for g, (fwd, bwd) in model.line_group_to_lines.items():
        lines_by_group[g] = [ell for ell in (fwd, bwd) if ell is not None and ell >= 0]
    selected = {}
    S = len(model.p_s)
    per_s = []
    for s in range(S):
        m = gp.Model(f"LPP_TWO_STAGE_S{str(s)}")
        m.Params.Threads = os.cpu_count()
        cgn = make_cgn_with_candidates_per_line(model, cand_all_lines[s])
        

        # Flüsse + Frequenzen
        x, arc_to_keys = _add_flows(m, model, cgn, aggregated)
        zs, delta_s, f_s_expr, h_s_expr = add_frequency_grouped(m, model, freq_vals)

        for g, Lg in lines_by_group.items():
            has_any = any(cand_all_lines[s].get(ell) for ell in Lg)
            if not has_any:
                m.addConstr(zs[g] == 0, name=f"noCand_forceOff[g{g},s{s}]")

        # Kandidatenauswahl y (an z gebunden)
        y = add_candidate_choice_per_line(m, model, zs, cand_all_lines[s], name=f"cand_s{s}")

        # Kapazitäten ersetzen:
        add_passenger_capacity_with_candidates_per_line(
            m, model, cgn,
            x=x,
            f_expr=f_s_expr,
            arc_to_keys=arc_to_keys,
            Q=Q,
            y_line=y,
            name=f"pass_cap_s{s}",
        )
        add_infrastructure_capacity_with_candidates_per_line(m, model, f_s_expr, y, cand_all_lines[s],
                                                    cap_per_arc=model.cap_sa[s, :], name=f"infra_s{s}")

        # Kosten
        time = build_obj_invehicle(m, model, cgn, x, arc_to_keys, use_t_min_time=True)
        wait, y_wait = build_obj_waiting(m, model, cgn, x, arc_to_keys, freq_vals, delta_s,
                                         include_origin_wait=True,
                                         waiting_time_frequency=wait_freq)
        oper = build_obj_operating_with_candidates_per_line(model, f_s_expr, y, cand_all_lines[s])
        c_repl_line = float(domain.config.get("cost_repl_line", 0.0))

        repl_path = add_path_replanning_cost_linear_per_line(
            m, model,
            y=y,
            candidates_per_line=cand_all_lines[s],
            f_expr=f_s_expr,
            cost_repl_line=float(domain.config.get("cost_repl_line", 0.0)),
            freq_vals=freq_vals,
            name=f"repl_path_s{s}"
        )

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
        m.setObjective(time_w * time + wait_w * wait + op_w * oper + repl + repl_path, GRB.MINIMIZE)


        if gurobi_params:
            for k, v in gurobi_params.items():
                setattr(m.Params, k, v)
        if "time_limit" in domain.config:
            m.Params.TimeLimit = int(domain.config["time_limit"])
        if "threads" in domain.config:
            m.Params.Threads = int(domain.config["threads"])
        if "seed" in domain.config:
            m.Params.Seed = int(domain.config["seed"])
        if "mip_gap" in domain.config:
            m.Params.MIPGap = float(domain.config["mip_gap"])
        elif "gap" in domain.config:
            m.Params.MIPGap = float(domain.config["gap"])

        m.optimize()

        chosen_k = {}
        for (ell, k), var in y.items():
            if var.X > 0.5:
                chosen_k[int(ell)] = int(k)
        selected[s] = chosen_k  # {ell: k}

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

        repl_freq_val = float(repl.getValue()) if m.SolCount else None
        repl_path_val = float(repl_path.getValue()) if m.SolCount else None
        repl_total    = ((repl_freq_val or 0.0) + (repl_path_val or 0.0)) if m.SolCount else None

        obj_val = float(m.ObjVal) if m.SolCount else None

        # weighted for logging
        time_w_val = (time_w * time_val) if time_val is not None else None
        wait_w_val = (wait_w * wait_raw) if wait_raw is not None else None
        oper_w_val = (op_w   * oper_val) if oper_val is not None else None

        per_s.append(dict(
            status=int(m.Status),
            objective=obj_val,                 # enthält time_w+wait_w+op_w + repl + repl_path
            chosen_freq=chosen,
            cost_time=time_w_val,
            cost_wait=wait_w_val,
            cost_oper=oper_w_val,
            cost_repl_freq=repl_freq_val,
            cost_repl_path=repl_path_val,
            cost_repl=repl_total,            # Summe für einfache Auswertung
            # raw:
            cost_time_raw=time_val,
            cost_wait_raw=wait_raw,
            cost_oper_raw=oper_val,
        ))


    # expected second-stage objective
    obj2_exp = 0.0
    repl_cost_freq_exp = 0.0
    repl_cost_path_exp = 0.0

    for s in range(S):
        if per_s[s]["objective"] is not None:
            obj2_exp += float(p[s]) * float(per_s[s]["objective"])
        repl_cost_freq_exp += float(p[s]) * float(per_s[s].get("cost_repl_freq") or 0.0)
        repl_cost_path_exp += float(p[s]) * float(per_s[s].get("cost_repl_path") or 0.0)

    repl_cost_exp = repl_cost_freq_exp + repl_cost_path_exp


    scenario_ids = domain.scen_prob_df["id"].astype(int).tolist()
    scenarios = [{
        "id": int(scenario_ids[s]),
        "prob": float(model.p_s[s]),
        "freq": per_s[s]["chosen_freq"],
        "cost_time": per_s[s]["cost_time"],
        "cost_wait": per_s[s]["cost_wait"],
        "cost_oper": per_s[s]["cost_oper"],
        "cost_repl_freq": per_s[s]["cost_repl_freq"],
        "cost_repl_path": per_s[s]["cost_repl_path"],
        "cost_repl": per_s[s]["cost_repl"],
        "objective": per_s[s]["objective"],
    } for s in range(S)]

    solution = dict(
        status_code=int(sol0["status_code"]),
        status=sol0["status"],
        runtime_s=sol0.get("runtime_s"),
        chosen_freq_stage1=chosen_freq0,
        chosen_freq_stage2=[ps["chosen_freq"] for ps in per_s],
        scenarios=scenarios,
        obj_stage1=sol0["costs_0"]["objective"],
        obj_stage2_exp=obj2_exp,
        repl_cost_freq_exp=repl_cost_freq_exp,   # <— NEU
        repl_cost_path_exp=repl_cost_path_exp,   # <— NEU
        repl_cost_exp=repl_cost_exp,             # <— Summe
        objective=sol0["costs_0"]["objective"] + obj2_exp,
        costs_0=sol0.get("costs_0"),
    )
    artifacts = dict(
        per_s=per_s,
        line_len=line_len,
        group_len=glen,
        probs=p,
        candidates_lines=cand_all_lines,      # <- neuer Key (klarer)
        cand_selected_lines=selected,         # <- neuer Key (per Linie)
    )
    return None, solution, artifacts  # main model returned as None here (we solved submodels)

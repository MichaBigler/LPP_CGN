# solve_cgn.py

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from prepare_cgn import make_cgn
from prepare_cgn_candidates import make_cgn_with_candidates_per_line
from find_candidates import build_candidates_all_scenarios_per_line
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
    _cand_counts, _freq_values_from_config, _routing_is_aggregated,
    _waiting_mode, _line_lengths, _group_lengths, _rep_line_of_group, _add_flows
)




# ---------- TWO-STAGE INTEGRATED (joint) ----------

def solve_two_stage_integrated(domain, model, *, gurobi_params=None):
    import gurobipy as gp
    from gurobipy import GRB

    # FIX: falsche/überflüssige Imports oben entfernen
    # from xml.parsers.expat import model   # <-- raus!
    # from preprare_cgn import make_cgn     # <-- Tippfehler
    # korrekt:
    # from prepare_cgn import make_cgn

    m = gp.Model("LPP_TWO_STAGE_INTEGRATED")

    # Kandidaten (per Linie, per Szenario)
    detour_cnt, ksp_cnt = _cand_counts(domain)
    cand_all_lines = build_candidates_all_scenarios_per_line(model, detour_cnt, ksp_cnt)

    # Nominal-CGN (Stage 1)
    cgn = make_cgn(model)
    freq_vals = _freq_values_from_config(domain)
    aggregated = _routing_is_aggregated(domain, override=domain.config.get("routing_agg", True))
    wait_freq = _waiting_mode(domain)

    S = len(model.p_s)
    line_len = _line_lengths(model)
    glen = _group_lengths(model, line_len)
    rep = _rep_line_of_group(model)

    # group -> [lines]
    lines_by_group = {
        g: [ell for ell in (fwd, bwd) if ell is not None and ell >= 0]
        for g, (fwd, bwd) in model.line_group_to_lines.items()
    }

    # ---------- Stage 1 ----------
    x0, arc_to0 = _add_flows(m, model, cgn, aggregated)
    z0, delta0, f0_expr, h0_expr = add_frequency_grouped(m, model, freq_vals)

    Q = int(domain.config.get("train_capacity", 200))
    add_passenger_capacity(m, model, cgn, x0, f0_expr, arc_to0, Q=Q)
    cap_std = int(domain.config.get("infrastructure_capacity", 10))
    add_infrastructure_capacity(m, model, f0_expr, cap_std=cap_std)

    time0 = build_obj_invehicle(m, model, cgn, x0, arc_to0, use_t_min_time=True)
    wait0, y0 = build_obj_waiting(
        m, model, cgn, x0, arc_to0, freq_vals, z0,  # delta0? Falls deine waiting-Funktion delta erwartet, hier delta0.
        include_origin_wait=True,
        waiting_time_frequency=wait_freq
    )
    oper0, _ = build_obj_operating(model, f0_expr)

    # ---------- Stage 2 (pro Szenario) ----------
    xs, arcs_to_s = [], []
    zs, deltas, f_expr_s, y_s, time_s, wait_s, oper_s = [], [], [], [], [], [], []
    repl_path_s = []

    c_repl_line = float(domain.config.get("cost_repl_line", 0.0))
    c_repl_freq = float(domain.config.get("cost_repl_freq", 0.0))

    for s in range(S):
        # CGN mit per-Linie-Kandidaten (Szenario s)
        cgn_s = make_cgn_with_candidates_per_line(model, cand_all_lines[s])

        xi, arc_toi = _add_flows(m, model, cgn_s, aggregated)
        zs_i, delta_i, f_expr_i, h_expr_i = add_frequency_grouped(m, model, freq_vals)

        # Gruppen ohne Kandidaten deaktivieren
        for g, Lg in lines_by_group.items():
            has_any = any(cand_all_lines[s].get(ell) for ell in Lg)
            if not has_any:
                m.addConstr(zs_i[g] == 0, name=f"noCand_forceOff[g{g},s{s}]")

        # Kandidaten-Auswahl: y[(ell,k)] und Kopplung an z_g (in der Funktion)
        y_i = add_candidate_choice_per_line(m, model, zs_i, cand_all_lines[s], name=f"cand_s{s}")

        # Kapazitäten mit Kandidaten-Gates
        add_passenger_capacity_with_candidates_per_line(
            m, model, cgn_s, xi, f_expr_i, arc_toi, Q, y_i,
            name=f"pass_cap_s{s}",
        )
        add_infrastructure_capacity_with_candidates_per_line(
            m, model, f_expr_i, y_i, cand_all_lines[s],
            cap_per_arc=model.cap_sa[s, :], name=f"infra_s{s}"
        )

        # Kosten
        ti = build_obj_invehicle(m, model, cgn_s, xi, arc_toi, use_t_min_time=True)
        wi, yi_wait = build_obj_waiting(
            m, model, cgn_s, xi, arc_toi, freq_vals, delta_i,
            include_origin_wait=True, waiting_time_frequency=wait_freq
        )
        oi   = build_obj_operating_with_candidates_per_line(model, f_expr_i, y_i, cand_all_lines[s])

        # Pfad-Replanning (linear, per Linie): freq(s) * (add_len + rem_len)
        repl_path = add_path_replanning_cost_linear_per_line(
            m, model,
            y=y_i,                                 # <- statt y_line=y_i
            candidates_per_line=cand_all_lines[s],
            f_expr=f_expr_i,
            cost_repl_line=c_repl_line,
            freq_vals=freq_vals,
            name=f"repl_path_s{s}",
        )

        xs.append(xi); arcs_to_s.append(arc_toi)
        zs.append(zs_i); deltas.append(delta_i); f_expr_s.append(f_expr_i)
        y_s.append(y_i); time_s.append(ti); wait_s.append(wi); oper_s.append(oi)
        repl_path_s.append(repl_path)

    # Frequenz-Replanning: d_{g,s} >= | f_g^(s) - f_g^(0) |
    d = {}
    for s in range(S):
        for g, ell_rep in rep.items():
            if ell_rep is None:
                continue
            d[g, s] = m.addVar(lb=0.0, name=f"d[g{g},s{s}]")
            m.addConstr(d[g, s] >= f_expr_s[s][ell_rep] - f0_expr[ell_rep],  name=f"d_pos[g{g},s{s}]")
            m.addConstr(d[g, s] >= -(f_expr_s[s][ell_rep] - f0_expr[ell_rep]), name=f"d_neg[g{g},s{s}]")

    # Gewichte
    time_w = float(domain.config.get("travel_time_cost_mult", 1.0))
    wait_w = float(domain.config.get("waiting_time_cost_mult", 1.0))
    op_w   = float(domain.config.get("line_operation_cost_mult", 1.0))
    p = model.p_s  # numpy array

    # Ziel: Stage1 + Erwartungswert Stage2 (ohne Repl) + Pfad-Repl + Frequenz-Repl
    stage1_obj = time_w * time0 + wait_w * wait0 + op_w * oper0

    stage2_norepl_exp = gp.quicksum(
        float(p[s]) * (time_w * time_s[s] + wait_w * wait_s[s] + op_w * oper_s[s])
        for s in range(S)
    )
    repl_path_exp = gp.quicksum(float(p[s]) * repl_path_s[s] for s in range(S))
    repl_freq_exp = gp.quicksum(
        float(p[s]) * gp.quicksum(c_repl_freq * float(glen[g]) * d[g, s] for g in glen.keys())
        for s in range(S)
    )

    m.setObjective(stage1_obj + stage2_norepl_exp + repl_path_exp + repl_freq_exp, GRB.MINIMIZE)

    # Params
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

    # Auswahl y → pro Linie
    selected = {}
    for s in range(S):
        chosen = {}
        for (ell, k), var in y_s[s].items():
            if var.X > 0.5:
                chosen[int(ell)] = int(k)
        selected[s] = chosen

    # Frequenzen decodieren
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

    # Stage-1 Komponentenkosten (gewichtet)
    costs0 = {}
    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
        v_time0_raw = float(time0.getValue())
        v_wait0_raw = float(wait0.getValue())
        v_oper0_raw = float(oper0.getValue())
        costs0 = dict(
            time=time_w * v_time0_raw,
            wait=wait_w * v_wait0_raw,
            oper=op_w   * v_oper0_raw,
            objective=time_w * v_time0_raw + wait_w * v_wait0_raw + op_w * v_oper0_raw,
            time_raw=v_time0_raw, wait_raw=v_wait0_raw, oper_raw=v_oper0_raw
        )

    # Szenario-Details
    scenario_ids = domain.scen_prob_df["id"].astype(int).tolist()
    scenarios = []
    for s in range(S):
        v_time = float(time_s[s].getValue())
        v_wait_raw = float(wait_s[s].getValue())
        v_oper = float(oper_s[s].getValue())

        repl_freq_val = sum(float(c_repl_freq) * float(glen[g]) * float(d[g, s].X) for g in glen.keys())
        repl_path_val = float(repl_path_s[s].getValue())

        scenarios.append({
            "id": int(scenario_ids[s]),
            "prob": float(model.p_s[s]),
            "freq": chosen_freq_s[s],
            "cost_time": time_w * v_time,
            "cost_wait": wait_w * v_wait_raw,
            "cost_oper": op_w   * v_oper,
            "cost_repl_freq": repl_freq_val,
            "cost_repl_path": repl_path_val,
            "cost_repl": repl_freq_val + repl_path_val,
            "cost_time_raw": v_time,
            "cost_wait_raw": v_wait_raw,
            "cost_oper_raw": v_oper,
            "objective": (time_w * v_time + wait_w * v_wait_raw + op_w * v_oper
                          + repl_freq_val + repl_path_val),
        })

    # Aggregates
    obj_total = float(m.ObjVal) if m.SolCount else None
    obj_stage1_val = float(stage1_obj.getValue()) if m.SolCount else None
    obj_stage2_exp_val = float((stage2_norepl_exp + repl_path_exp + repl_freq_exp).getValue()) if m.SolCount else None
    repl_cost_path_exp_val = float(repl_path_exp.getValue()) if m.SolCount else None
    repl_cost_freq_exp_val = float(repl_freq_exp.getValue()) if m.SolCount else None

    solution = dict(
        status_code=int(m.Status),
        status=m.Status,
        runtime_s=getattr(m, "Runtime", None),
        chosen_freq_stage1=chosen_freq0,
        chosen_freq_stage2=chosen_freq_s,
        scenarios=scenarios,
        obj_stage1=obj_stage1_val,
        obj_stage2_exp=obj_stage2_exp_val,      # enthält Replanning (freq + path)
        repl_cost_freq_exp=repl_cost_freq_exp_val,
        repl_cost_path_exp=repl_cost_path_exp_val,
        repl_cost_exp=(repl_cost_freq_exp_val or 0.0) + (repl_cost_path_exp_val or 0.0),
        objective=obj_total,
        costs_0=costs0,
    )

    artifacts = dict(
        model=m, cgn=cgn, delta0=delta0, deltas=deltas,
        f0_expr=f0_expr, f_expr_s=f_expr_s, d=d,
        line_len=line_len, group_len=glen, probs=p,
        candidates_lines=cand_all_lines,         # <- per Linie
        cand_selected_lines=selected,            # <- per Linie
    )
    return m, solution, artifacts
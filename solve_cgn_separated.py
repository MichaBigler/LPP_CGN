# solve_cgn_separated.py
import os
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from prepare_cgn import make_cgn_with_candidates_per_line
from find_candidates import build_candidates_all_scenarios_per_line_cfg
from optimisation import (
    build_obj_invehicle_with_overdemand, build_obj_bypass, build_obj_waiting,
    build_obj_operating_with_candidates_per_line, add_frequency_grouped,
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


def _x_to_1d(cgn, x_vars, arc_to_keys):
    """
    Convert a (possibly) multi-key flow tupledict x[a, key] into a simple
    per-arc aggregation: X1D[a] = sum_key x[a, key], but only for ride arcs.
    - Uses arc_to_keys[a] to list the exact keys (no global scans).
    - If there is no mapping for a, tries x[a] and x[(a,0)] as conservative fallbacks.
    """
    A = len(cgn.arc_kind)

    def _val(v):
        try:
            return float(getattr(v, "X", v))
        except Exception:
            return 0.0

    def _keys_for(a):
        if arc_to_keys is None:
            return []
        loc = arc_to_keys.get(a, None)
        if loc is None:
            return []
        subs = loc if isinstance(loc, list) else [loc]
        out = []
        for sub in subs:
            out.append((a, *sub) if isinstance(sub, tuple) else (a, sub))
        return out

    x1d = {}
    for a in range(A):
        if cgn.arc_kind[a] != "ride":
            continue
        s = 0.0
        for k in _keys_for(a):
            try:
                s += _val(x_vars[k])
            except Exception:
                pass
        if s == 0.0 and (arc_to_keys is None or a not in arc_to_keys):
            # Fallbacks only; no global scans
            v = None
            try:
                v = x_vars[a]
            except Exception:
                v = x_vars.get((a, 0), None) if hasattr(x_vars, "get") else None
            s = _val(v)
        if s:
            x1d[a] = s
    return x1d


# ---------- TWO-STAGE SEPARATED (solve stage-1 first, then one submodel per scenario) ----------

def solve_two_stage_separated(domain, model, *, gurobi_params=None):
    """
    Two-stage (separated) procedure:
      Stage 1 (nominal):
        - Solve one-stage model to obtain base frequencies and nominal flows.
      Stage 2 (per scenario s):
        - Build CGN with per-line candidates (for scenario s).
        - Re-optimise flows + per-group frequency choice + candidate selection.
        - Add replanning penalties (frequency deviation from stage-1 and path changes).

    Returns:
      (None, solution, artifacts)
        - We return None for the main model because each scenario has its own sub-model.
        - solution includes stage-1 summary, expected stage-2 objective and per-scenario breakdowns.
        - artifacts include CGNs and flows (flattened to per-arc) for logging.
    """
    # ----------------------------- Stage 1 -----------------------------
    m0, sol0, art0 = solve_one_stage(domain, model, gurobi_params=gurobi_params)

    # Flatten nominal flows to per-arc for logging downstream
    cgn0 = art0["cgn_stage1"]
    x0   = art0["x_stage1"]
    a2k0 = art0.get("arc_to_keys_stage1") or art0.get("arc_to_keys0") or art0.get("arc_to_keys")
    x0_1d = _x_to_1d(cgn0, x0, a2k0)

    # Group representative and geometry
    freq_vals = _freq_values_from_config(domain)
    rep = _rep_line_of_group(model)            # g -> representative line ell
    line_len = art0["line_len"]
    glen = _group_lengths(model, line_len)     # g -> length (sum of both directions)
    Q = int(domain.config.get("train_capacity", 200))

    # Cost weights
    time_w = float(domain.config.get("travel_time_cost_mult", 1.0))
    wait_w = float(domain.config.get("waiting_time_cost_mult", 1.0))
    op_w   = float(domain.config.get("line_operation_cost_mult", 1.0))
    c_repl = float(domain.config.get("cost_repl_freq", 0.0))
    p = model.p_s
    wait_freq = _waiting_mode(domain)
    aggregated = _routing_is_aggregated(domain)

    # Stage-1 chosen frequency per group (using representative)
    chosen_freq0 = sol0.get("chosen_freq", {}) or sol0.get("chosen_freq_stage1", {})
    f0_by_group = {g: int(chosen_freq0.get(ell_rep, 0)) for g, ell_rep in rep.items()}

    # ----------------------------- Candidate generation -----------------------------
    # Build per-scenario, per-line candidate lists
    cand_cfg = getattr(domain, "cand_cfg", None) or CandidateConfig()
    cand_all_lines = build_candidates_all_scenarios_per_line_cfg(model, cand_cfg, domain.config)

    # Helper: lines per group (both directions if present)
    lines_by_group = {
        g: [ell for ell in (fwd, bwd) if ell is not None and ell >= 0]
        for g, (fwd, bwd) in model.line_group_to_lines.items()
    }

    # Accumulators
    selected = {}      # scenario -> {ell: k}
    S = len(model.p_s)
    per_s = []         # scenario summaries

    cgn_s_list = []    # store CGN per scenario for logging
    x_s_list = []      # store flows (flattened later)
    arc_to_keys_s_list = []

    # ----------------------------- Stage 2 (per scenario) -----------------------------
    for s in range(S):
        m = gp.Model(f"LPP_TWO_STAGE_S{str(s)}")
        m.Params.Threads = os.cpu_count()

        # Build CGN for scenario s using per-line candidates
        cgn = make_cgn_with_candidates_per_line(model, cand_all_lines[s])

        # Flows + grouped frequencies
        x, arc_to_keys = _add_flows(m, model, cgn, aggregated)
        cgn_s_list.append(cgn)
        x_s_list.append(x)                # keep raw here; we will flatten after solve
        arc_to_keys_s_list.append(arc_to_keys)

        zs, delta_s, f_s_expr, h_s_expr = add_frequency_grouped(m, model, freq_vals)

        # Turn off groups with no candidates in this scenario
        for g, Lg in lines_by_group.items():
            has_any = any(cand_all_lines[s].get(ell) for ell in Lg)
            if not has_any:
                m.addConstr(zs[g] == 0, name=f"noCand_forceOff[g{g},s{s}]")

        # Candidate selection per line, tied to group on/off
        y = add_candidate_choice_per_line(m, model, zs, cand_all_lines[s], name=f"cand_s{s}")

        # Replace capacities with candidate-aware versions
        add_passenger_capacity_with_candidates_per_line(
            m, model, cgn, x, f_s_expr, arc_to_keys, Q, y, name=f"pass_cap_s{s}"
        )
        add_infrastructure_capacity_with_candidates_per_line(
            m, model, f_s_expr, y, cand_all_lines[s],
            cap_per_arc=model.cap_sa[s, :], name=f"infra_s{s}"
        )

        # Cost buckets (scenario s)
        tau = float(domain.config.get("overdemand_threshold", 1.0))
        mu  = float(domain.config.get("overdemand_multiplier", 1.0))
        time_base_raw, time_over_raw = build_obj_invehicle_with_overdemand(
            m, model, cgn, x, arc_to_keys, f_s_expr, Q, threshold=tau, multiplier=mu, use_t_min_time=True
        )
        time   = time_base_raw + max(mu - 1.0, 0.0) * time_over_raw
        bypass = build_obj_bypass(m, model, cgn, x, arc_to_keys)
        wait, _y_wait = build_obj_waiting(
            m, model, cgn, x, arc_to_keys, freq_vals, delta_s,
            include_origin_wait=True, waiting_time_frequency=wait_freq
        )
        oper = build_obj_operating_with_candidates_per_line(model, f_s_expr, y, cand_all_lines[s])

        # Path replanning (per line): freq(s) * (add_len + rem_len) for the chosen candidate
        repl_path = add_path_replanning_cost_linear_per_line(
            m, model,
            y=y,
            candidates_per_line=cand_all_lines[s],
            f_expr=f_s_expr,
            cost_repl_line=float(domain.config.get("cost_repl_line", 0.0)),
            freq_vals=freq_vals,
            name=f"repl_path_s{s}"
        )

        # Frequency replanning vs. stage-1: d_{g,s} >= | f_g^(s) - f_g^(0) |, weighted by group length
        repl_terms = []
        d_vars = {}
        for g, ell_rep in rep.items():
            if ell_rep is None:
                continue
            d = m.addVar(lb=0.0, name=f"d_g{g}_s{s}")
            m.addConstr(d >=  f_s_expr[ell_rep] - float(f0_by_group[g]), name=f"dpos_g{g}_s{s}")
            m.addConstr(d >= -f_s_expr[ell_rep] + float(f0_by_group[g]), name=f"dneg_g{g}_s{s}")
            d_vars[g] = d
            coef = c_repl * float(glen[g])   # scale by group length
            repl_terms.append(coef * d)
        repl = gp.quicksum(repl_terms)

        # Objective of scenario submodel
        m.setObjective(time_w * time + bypass + wait_w * wait + op_w * oper + repl + repl_path, GRB.MINIMIZE)

        # Solver params (inherit stage-1 settings)
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

        # ----------------------------- Extract flows (flatten to per-arc) -----------------------------
        A = len(cgn.arc_kind)

        def _val(v):
            try:
                return float(getattr(v, "X", v))
            except Exception:
                return 0.0

        def _keys_for_arc(a, a2k):
            """Build exactly the keys for arc a from arc_to_keys[a] (no fallbacks, no duplicates)."""
            if a2k is None:
                return []
            loc = a2k.get(a, None)
            if loc is None:
                return []
            subs = loc if isinstance(loc, list) else [loc]
            keys = []
            for sub in subs:
                keys.append((a, *sub) if isinstance(sub, tuple) else (a, sub))
            return keys

        x_by_arc = {}
        a2k = arc_to_keys  # scenario's arc_to_keys
        for a in range(A):
            if cgn.arc_kind[a] != "ride":
                continue
            ssum = 0.0
            used = set()
            for k in _keys_for_arc(a, a2k):
                if k in used:
                    continue
                used.add(k)
                try:
                    ssum += _val(x[k])
                except Exception:
                    pass

            # If there is no mapping for a, try simple 1D fallbacks
            if ssum == 0.0 and (a2k is None or a not in a2k):
                v = None
                try:
                    v = x[a]
                except Exception:
                    try:
                        v = x[(a, 0)]
                    except Exception:
                        v = None
                ssum = _val(v)

            if ssum != 0.0:
                x_by_arc[a] = ssum

        # Replace raw tupledict with flattened dict for logging; clear mapper
        x_s_list[-1] = x_by_arc
        arc_to_keys_s_list[-1] = {}

        # Selected candidate per line
        chosen_k = {}
        for (ell, k), var in y.items():
            if var.X > 0.5:
                chosen_k[int(ell)] = int(k)
        selected[s] = chosen_k  # {ell: k}

        # Chosen frequencies in scenario s (per line)
        chosen = {}
        if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            for ell in range(model.L):
                f = 0
                for r, _ in enumerate(freq_vals):
                    if delta_s[(ell, r)].X > 0.5:
                        f = freq_vals[r]
                        break
                chosen[ell] = f

        # Component values
        time_val       = float(time.getValue()) if m.SolCount else None
        time_base_raw_ = float(time_base_raw.getValue()) if m.SolCount else None
        time_over_raw_ = float(time_over_raw.getValue()) if m.SolCount else None
        bypass_raw     = float(bypass.getValue()) if m.SolCount else None
        wait_raw       = float(wait.getValue()) if m.SolCount else None
        oper_val       = float(oper.getValue()) if m.SolCount else None

        repl_freq_val  = float(repl.getValue()) if m.SolCount else None
        repl_path_val  = float(repl_path.getValue()) if m.SolCount else None
        repl_total     = ((repl_freq_val or 0.0) + (repl_path_val or 0.0)) if m.SolCount else None

        obj_val        = float(m.ObjVal) if m.SolCount else None

        # Weighted components for convenient reporting
        time_w_val   = (time_w * time_val) if time_val is not None else None
        time_base_w  = (time_w * time_base_raw_) if time_base_raw_ is not None else None
        time_over_w  = (time_w * max(mu - 1.0, 0.0) * time_over_raw_) if time_over_raw_ is not None else None
        bypass_w_val = bypass_raw if bypass_raw is not None else None
        wait_w_val   = (wait_w * wait_raw) if wait_raw is not None else None
        oper_w_val   = (op_w   * oper_val) if oper_val is not None else None

        per_s.append(dict(
            status=int(m.Status),
            objective=obj_val,                   # includes time_w + wait_w + op_w + repl + repl_path
            chosen_freq=chosen,
            cost_time=time_w_val,
            cost_time_base=time_base_w,
            cost_time_over=time_over_w,
            cost_bypass=bypass_w_val,
            cost_wait=wait_w_val,
            cost_oper=oper_w_val,
            cost_repl_freq=repl_freq_val,
            cost_repl_path=repl_path_val,
            cost_repl=repl_total,                # convenience sum
            # raw components (unweighted)
            cost_time_raw=time_val,
            cost_time_base_raw=time_base_raw_,
            cost_time_over_raw=time_over_raw_,
            cost_bypass_raw=bypass_raw,
            cost_wait_raw=wait_raw,
            cost_oper_raw=oper_val,
        ))

    # ----------------------------- Expected values over scenarios -----------------------------
    obj2_exp = 0.0
    repl_cost_freq_exp = 0.0
    repl_cost_path_exp = 0.0
    for s in range(S):
        if per_s[s]["objective"] is not None:
            obj2_exp += float(p[s]) * float(per_s[s]["objective"])
        repl_cost_freq_exp += float(p[s]) * float(per_s[s].get("cost_repl_freq") or 0.0)
        repl_cost_path_exp += float(p[s]) * float(per_s[s].get("cost_repl_path") or 0.0)
    repl_cost_exp = repl_cost_freq_exp + repl_cost_path_exp

    # ----------------------------- Assemble per-scenario details -----------------------------
    scenario_ids = domain.scen_prob_df["id"].astype(int).tolist()
    scenarios = [dict(
        id=int(scenario_ids[s]),
        prob=float(model.p_s[s]),
        freq=per_s[s]["chosen_freq"],
        cost_time=per_s[s]["cost_time"],
        cost_time_base=per_s[s].get("cost_time_base"),
        cost_time_over=per_s[s].get("cost_time_over"),
        cost_bypass=per_s[s]["cost_bypass"],
        cost_wait=per_s[s]["cost_wait"],
        cost_oper=per_s[s]["cost_oper"],
        cost_repl_freq=per_s[s]["cost_repl_freq"],
        cost_repl_path=per_s[s]["cost_repl_path"],
        cost_repl=per_s[s]["cost_repl"],
        objective=per_s[s]["objective"],
        # raw components
        cost_time_raw=per_s[s]["cost_time_raw"],
        cost_time_base_raw=per_s[s]["cost_time_base_raw"],
        cost_time_over_raw=per_s[s]["cost_time_over_raw"],
        cost_bypass_raw=per_s[s]["cost_bypass_raw"],
        cost_wait_raw=per_s[s]["cost_wait_raw"],
        cost_oper_raw=per_s[s]["cost_oper_raw"],
    ) for s in range(S)]

    # ----------------------------- Return payloads -----------------------------
    solution = dict(
        status_code=int(sol0["status_code"]),
        status=sol0["status"],
        runtime_s=sol0.get("runtime_s"),
        chosen_freq_stage1=chosen_freq0,
        chosen_freq_stage2=[ps["chosen_freq"] for ps in per_s],
        scenarios=scenarios,
        obj_stage1=sol0["costs_0"]["objective"],
        obj_stage2_exp=obj2_exp,
        repl_cost_freq_exp=repl_cost_freq_exp,
        repl_cost_path_exp=repl_cost_path_exp,
        repl_cost_exp=repl_cost_exp,
        objective=sol0["costs_0"]["objective"] + obj2_exp,
        costs_0=sol0.get("costs_0"),
    )

    artifacts = dict(
        # stage-1 (normalized)
        cgn_stage1=cgn0,
        x_stage1=x0_1d,
        arc_to_keys_stage1={},   # explicitly flattened

        # stage-2 (already normalized above)
        cgn_stage2_list=cgn_s_list,
        x_stage2_list=x_s_list,
        arc_to_keys_stage2_list=arc_to_keys_s_list,

        line_len=line_len, group_len=glen, probs=p,
        candidates_lines=cand_all_lines,
        cand_selected_lines=selected,
    )

    # We solved separate submodels; return None for the "main model" handle.
    return None, solution, artifacts

# solve_cgn_wait_and_see.py
# -*- coding: utf-8 -*-
"""
Wait-and-See (WS) solver.

For each scenario s in S, solve an independent one-stage LPP with the
scenario's per-arc infrastructure capacity treated as the *nominal*
capacity. Aggregate:

    WS_total = sum_s p_s * z_s

where z_s is the optimal objective of the one-stage subproblem under
scenario s.

WS is a lower bound to the recourse problem (RP) and is required to
compute EVPI = RP - WS. By construction the resulting "policy" is not
implementable (each scenario gets its own first-stage plan), so the
returned artifacts carry per-scenario plans in the stage-2 slots and
leave the stage-1 slot empty.

Bypass / overdemand handling is inherited from `solve_one_stage` — if
the config enables either, the subsolves use them. This keeps WS finite
even for heavy restriction scenarios.
"""

from typing import Any, Dict, List, Optional
import gurobipy as gp
from gurobipy import GRB

from solve_cgn_one_stage import solve_one_stage


def _agg_status_code(per_scen_codes: List[int]) -> int:
    """Aggregate Gurobi status codes across scenarios.

    OPTIMAL only if every subsolve is OPTIMAL. Else surface the first
    non-OPTIMAL code so the operator sees the real failure mode.
    """
    if not per_scen_codes:
        return -1
    if all(c == int(GRB.OPTIMAL) for c in per_scen_codes):
        return int(GRB.OPTIMAL)
    for c in per_scen_codes:
        if c != int(GRB.OPTIMAL):
            return int(c)
    return int(per_scen_codes[0])


def solve_wait_and_see(domain, model, *, gurobi_params: Optional[Dict[str, Any]] = None):
    """
    Solve the Wait-and-See bound.

    Returns:
      m          : last gurobipy.Model (for inspection only — not the full WS policy)
      solution   : dict shaped like a two-stage solution but with empty stage 1.
                   Fields:
                     - objective         = WS_total = sum_s p_s * z_s
                     - obj_stage1        = None
                     - obj_stage2_exp    = WS_total (everything is "stage 2" for WS)
                     - repl_cost_*_exp   = 0
                     - scenarios         = [{id, prob, freq, cost_*, objective, ...}, ...]
                     - chosen_freq_stage1= {}
                     - chosen_freq_stage2= [per-scenario freq dict]
                     - costs_0           = {}
      artifacts  : dict carrying per-scenario CGN/flows in stage-2-shaped lists.
    """
    S = int(model.S)
    p_s = model.p_s
    scenario_ids = domain.scen_prob_df["id"].astype(int).tolist()

    scen_dicts: List[Dict[str, Any]] = []
    chosen_freq_list: List[Dict[int, int]] = []
    cgn_list: List[Any] = []
    x_list: List[Any] = []
    a2k_list: List[Any] = []
    status_codes: List[int] = []
    gaps: List[float] = []
    runtimes: List[float] = []

    obj_agg: Optional[float] = 0.0
    last_m = None

    for s in range(S):
        cap_s = model.cap_sa[s, :]
        m_s, sol_s, art_s = solve_one_stage(
            domain, model,
            gurobi_params=gurobi_params,
            cap_per_arc_override=cap_s,
        )
        last_m = m_s

        z_s = sol_s.get("objective")
        prob = float(p_s[s])
        sc = int(sol_s.get("status_code", -1))
        rt = float(sol_s.get("runtime_s") or 0.0)
        gap = sol_s.get("opt_gap")
        c0 = sol_s.get("costs_0") or {}

        status_codes.append(sc)
        runtimes.append(rt)
        gaps.append(float(gap) if gap is not None else float("inf"))
        chosen_freq_list.append(sol_s.get("chosen_freq") or {})
        cgn_list.append(art_s.get("cgn_stage1"))
        x_list.append(art_s.get("x_stage1"))
        a2k_list.append(art_s.get("arc_to_keys_stage1"))

        if z_s is None:
            obj_agg = None  # one infeasible subsolve → WS undefined
        elif obj_agg is not None:
            obj_agg += prob * float(z_s)

        scen_dicts.append(dict(
            id=int(scenario_ids[s]),
            prob=prob,
            freq=sol_s.get("chosen_freq") or {},
            cost_time=c0.get("time"),
            cost_time_base=c0.get("time_base"),
            cost_time_over=c0.get("time_over"),
            cost_bypass=c0.get("bypass"),
            cost_wait=c0.get("wait"),
            cost_oper=c0.get("oper"),
            # WS has no replanning by construction
            cost_repl_freq=0.0,
            cost_repl_path=0.0,
            cost_repl=0.0,
            objective=z_s,
            # raw components (unweighted by cost multipliers)
            cost_time_raw=c0.get("time_raw"),
            cost_time_base_raw=c0.get("time_base_raw"),
            cost_time_over_raw=c0.get("time_over_raw"),
            cost_bypass_raw=c0.get("bypass_raw"),
            cost_wait_raw=c0.get("wait_raw"),
            cost_oper_raw=c0.get("oper_raw"),
        ))

    worst_gap = max(gaps) if gaps else None
    solution = dict(
        status_code=_agg_status_code(status_codes),
        status=_agg_status_code(status_codes),  # raw int, like solve_one_stage
        runtime_s=sum(runtimes),
        opt_gap=worst_gap,
        # No nominal stage in WS
        chosen_freq_stage1={},
        chosen_freq_stage2=chosen_freq_list,
        scenarios=scen_dicts,
        obj_stage1=None,
        obj_stage2_exp=obj_agg,
        repl_cost_freq_exp=0.0,
        repl_cost_path_exp=0.0,
        repl_cost_exp=0.0,
        objective=obj_agg,
        costs_0={},  # empty: WS has no nominal stage
    )

    artifacts = dict(
        # Stage-1 slot intentionally empty: WS has no nominal first-stage plan.
        cgn_stage1=None,
        x_stage1=None,
        arc_to_keys_stage1=None,
        # Per-scenario plans live in the stage-2 slots so the logger can iterate.
        cgn_stage2_list=cgn_list,
        x_stage2_list=x_list,
        arc_to_keys_stage2_list=a2k_list,
        line_len=None,
    )

    return last_m, solution, artifacts

# solve_cgn_wait_and_see.py
# -*- coding: utf-8 -*-
"""
Wait-and-See (WS) solver — proper SP semantics for the two-stage LPP.

For each scenario s in S, solve the FULL two-stage integrated problem with
only that single scenario active (p_s = 1). The resulting per-scenario cost

    z_s = min_{x, y_s} [ c_nominal(x) + Q(x, y_s, s) ]

is the optimum under perfect anticipation of s. Aggregate with the original
scenario probabilities:

    WS = sum_s p_s_orig * z_s

This matches the textbook SP definition of the wait-and-see bound and is the
appropriate lower bound for computing

    EVPI = RP - WS

A previous implementation solved only the stage-2 (disrupted-day) part per
scenario, which misses the nominal-day cost component that RP includes — that
yielded an artefact roughly equal to the nominal-day cost, not a meaningful
EVPI. The current implementation closes that gap.

Per-scenario plans are *not* implementable as a single policy by construction
(different scenarios may pick different nominal plans). The artifacts return
the per-scenario recourse flows so the logger can still inspect each plan.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from solve_cgn_integrated import solve_two_stage_integrated


def _agg_status_code(per_scen_codes: List[int]) -> int:
    """OPTIMAL only if every sub-solve was OPTIMAL; else surface the first
    non-OPTIMAL code so operators see the real failure mode."""
    if not per_scen_codes:
        return -1
    if all(c == int(GRB.OPTIMAL) for c in per_scen_codes):
        return int(GRB.OPTIMAL)
    for c in per_scen_codes:
        if c != int(GRB.OPTIMAL):
            return int(c)
    return int(per_scen_codes[0])


def _single_scenario_subview(domain, model, s: int):
    """Return shallow copies of (domain, model) restricted to scenario s with p_s=1.

    The originals are left untouched so the parent run / other threads cannot
    observe a partially-mutated state.
    """
    sub_scen_prob_df = domain.scen_prob_df.iloc[[s]].copy()
    sub_scen_prob_df["prob"] = 1.0
    sub_scen_prob_df = sub_scen_prob_df.reset_index(drop=True)

    sub_domain = dataclasses.replace(domain, scen_prob_df=sub_scen_prob_df)
    # `cand_cfg` is attached dynamically by run.py; dataclasses.replace drops
    # dynamic attributes, so propagate explicitly. NOTE: only `cand_cfg` is
    # propagated here — any other ad-hoc attribute hung on `domain` upstream
    # would be silently lost in the subview. Audit this list if new dynamic
    # attributes are introduced in run.py.
    cand_cfg_attr = getattr(domain, "cand_cfg", None)
    if cand_cfg_attr is not None:
        sub_domain.cand_cfg = cand_cfg_attr

    sub_model = dataclasses.replace(
        model,
        S=1,
        p_s=np.array([1.0], dtype=float),
        cap_sa=model.cap_sa[s:s + 1, :],
    )
    return sub_domain, sub_model


def solve_wait_and_see(domain, model, *, gurobi_params: Optional[Dict[str, Any]] = None):
    """
    Solve the WS bound by running the integrated two-stage solver once per
    scenario with that scenario as the sole reality.

    Returns (m, solution, artifacts) shaped like a two-stage solution with
    empty stage 1 (no single first-stage plan exists for WS by definition).
    Per-scenario totals (stage 1 + stage 2) are stored in `solution.scenarios`
    so the existing `_agg_components_two_stage` aggregator in run.py produces
    the correct expected cost decomposition.
    """
    S = int(len(model.p_s))
    p_orig = np.asarray(model.p_s, dtype=float)
    scen_ids = domain.scen_prob_df["id"].astype(int).tolist()

    scen_dicts: List[Dict[str, Any]] = []
    cgn_list: List[Any] = []
    x_list: List[Any] = []
    a2k_list: List[Any] = []
    chosen_freq_list: List[Dict[int, int]] = []
    status_codes: List[int] = []
    gaps: List[float] = []
    runtimes: List[float] = []

    obj_total: Optional[float] = 0.0
    obj_stage1_exp: float = 0.0
    obj_stage2_exp: float = 0.0
    repl_freq_exp: float = 0.0
    repl_path_exp: float = 0.0
    last_m = None

    def _add(a, b) -> float:
        """None-tolerant float sum used to fold stage-1 and stage-2 components."""
        return float(a or 0.0) + float(b or 0.0)

    for s in range(S):
        sub_domain, sub_model = _single_scenario_subview(domain, model, s)
        m_s, sol_s, art_s = solve_two_stage_integrated(
            sub_domain, sub_model, gurobi_params=gurobi_params,
        )
        last_m = m_s

        prob = float(p_orig[s])
        z_s = sol_s.get("objective")
        sub_obj_stage1 = sol_s.get("obj_stage1")
        sub_obj_stage2 = sol_s.get("obj_stage2_exp")
        sub_repl_freq = sol_s.get("repl_cost_freq_exp") or 0.0
        sub_repl_path = sol_s.get("repl_cost_path_exp") or 0.0

        status_codes.append(int(sol_s.get("status_code") or -1))
        rt = float(sol_s.get("runtime_s") or 0.0)
        runtimes.append(rt)
        gap = sol_s.get("opt_gap")
        gaps.append(float(gap) if gap is not None else float("inf"))

        # Aggregate expectations under original probabilities
        if z_s is None:
            obj_total = None
        elif obj_total is not None:
            obj_total += prob * float(z_s)
        if sub_obj_stage1 is not None:
            obj_stage1_exp += prob * float(sub_obj_stage1)
        if sub_obj_stage2 is not None:
            obj_stage2_exp += prob * float(sub_obj_stage2)
        repl_freq_exp += prob * float(sub_repl_freq)
        repl_path_exp += prob * float(sub_repl_path)

        # Build per-scenario report dict carrying TOTAL costs (stage 1 + stage 2)
        # so `_agg_components_two_stage` in run.py — which expects empty
        # `costs_0` and per-scenario raw values to be re-weighted by `prob` —
        # yields the right expected components for the base_log row.
        nom = sol_s.get("costs_0") or {}
        sub_scens = sol_s.get("scenarios") or []
        sub_scen0 = sub_scens[0] if sub_scens else {}

        chosen_freq_s = sub_scen0.get("freq") or {}
        chosen_freq_list.append(chosen_freq_s)

        scen_dicts.append(dict(
            id=int(scen_ids[s]),
            prob=prob,
            freq=chosen_freq_s,
            # Totals (anticipated nominal + recourse stage), to be p-weighted by run.py
            cost_time=_add(nom.get("time"), sub_scen0.get("cost_time")),
            cost_time_base=_add(nom.get("time_base"), sub_scen0.get("cost_time_base")),
            cost_time_over=_add(nom.get("time_over"), sub_scen0.get("cost_time_over")),
            cost_bypass=_add(nom.get("bypass"), sub_scen0.get("cost_bypass")),
            cost_wait=_add(nom.get("wait"), sub_scen0.get("cost_wait")),
            cost_oper=_add(nom.get("oper"), sub_scen0.get("cost_oper")),
            cost_repl_freq=float(sub_scen0.get("cost_repl_freq") or 0.0),
            cost_repl_path=float(sub_scen0.get("cost_repl_path") or 0.0),
            cost_repl=float(sub_scen0.get("cost_repl") or 0.0),
            objective=z_s,
            # Raw (unweighted by cost multipliers) — only stage-2 part for transparency
            cost_time_raw=sub_scen0.get("cost_time_raw"),
            cost_time_base_raw=sub_scen0.get("cost_time_base_raw"),
            cost_time_over_raw=sub_scen0.get("cost_time_over_raw"),
            cost_bypass_raw=sub_scen0.get("cost_bypass_raw"),
            cost_wait_raw=sub_scen0.get("cost_wait_raw"),
            cost_oper_raw=sub_scen0.get("cost_oper_raw"),
        ))

        # Carry the recourse-stage artifacts so the logger can emit one
        # edge-flows file per scenario, matching integrated/separated layout.
        sub_cgn_list = art_s.get("cgn_stage2_list") or []
        sub_x_list = art_s.get("x_stage2_list") or []
        sub_a2k_list = art_s.get("arc_to_keys_stage2_list") or []
        cgn_list.append(sub_cgn_list[0] if sub_cgn_list else None)
        x_list.append(sub_x_list[0] if sub_x_list else None)
        a2k_list.append(sub_a2k_list[0] if sub_a2k_list else None)

    worst_gap = max(gaps) if gaps else None

    solution = dict(
        status_code=_agg_status_code(status_codes),
        status=_agg_status_code(status_codes),
        runtime_s=sum(runtimes),
        opt_gap=worst_gap,
        # No single WS first-stage plan exists by construction.
        chosen_freq_stage1={},
        chosen_freq_stage2=chosen_freq_list,
        scenarios=scen_dicts,
        # E_s[c_nom(x_s)] and E_s[Q(x_s, y_s, s)] under perfect anticipation
        obj_stage1=obj_stage1_exp if obj_total is not None else None,
        obj_stage2_exp=obj_stage2_exp if obj_total is not None else None,
        repl_cost_freq_exp=repl_freq_exp if obj_total is not None else None,
        repl_cost_path_exp=repl_path_exp if obj_total is not None else None,
        repl_cost_exp=(repl_freq_exp + repl_path_exp) if obj_total is not None else None,
        objective=obj_total,
        # `costs_0` must stay empty so `_agg_components_two_stage` only sums
        # the per-scenario totals (already containing stage 1 + stage 2).
        costs_0={},
    )

    artifacts = dict(
        cgn_stage1=None,
        x_stage1=None,
        arc_to_keys_stage1=None,
        cgn_stage2_list=cgn_list,
        x_stage2_list=x_list,
        arc_to_keys_stage2_list=a2k_list,
        line_len=None,
    )

    return last_m, solution, artifacts

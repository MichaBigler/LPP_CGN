# solve_utils.py
"""
Small helper utilities used by the solvers.

This module stays intentionally lightweight and side-effect free.
It provides:
- simple config readers (frequency values, routing/waiting modes),
- wrappers to build flow variables (aggregated vs. OD-based),
- length helpers (per line / per group),
- representative line per group (for comparing group frequencies),
- candidate generation counters (detours/K-shortest).
"""

import numpy as np
from gurobipy import GRB

from optimisation import (
    od_pairs,
    add_flow_conservation,
    add_flow_conservation_by_origin,
)


# ---------- config readers ----------

def _freq_values_from_config(domain):
    """
    Return the list of admissible frequency values.
    If 'freq_values' is specified in the config, use it (cast to int).
    Otherwise, fall back to 1..max_frequency.
    """
    vals = domain.config.get("freq_values")
    if vals:
        return list(map(int, vals))
    fmax = int(domain.config.get("max_frequency", 5))
    return list(range(1, fmax + 1))


def _routing_is_aggregated(domain, override=None):
    """
    Return True if routing should be modeled with origin-aggregated commodities,
    False for standard OD commodities.

    If 'override' is provided (bool-like), it takes precedence over config.
    """
    if override is not None:
        return bool(override)
    return bool(domain.config.get("routing_agg", False))


def _waiting_mode(domain, override=None):
    """
    Return True if waiting time should depend on selected frequencies
    (half-headway linearisation), False for a flat per-change penalty.

    If 'override' is provided (bool-like), it takes precedence over config.
    """
    if override is not None:
        return bool(override)
    return bool(domain.config.get("waiting_time_frequency", True))


# ---------- flow construction ----------

def _add_flows(m, model, cgn, aggregated: bool):
    """
    Create flow variables and flow-conservation constraints on the given CGN.

    If aggregated == True:
        - One commodity per origin; variables x[a, o].
        - Boarding is only allowed at the origin's ground node,
          alighting only at any valid destination of that origin.

    If aggregated == False:
        - One commodity per OD pair with positive demand; variables x[a, (o, d)].

    Returns:
        x           : Gurobi tupledict of flow variables
        arc_to_keys: dict a -> [flow-key,...] for summing flows on each arc
    """
    if aggregated:
        x, arc_to_keys = add_flow_conservation_by_origin(m, model, cgn)
    else:
        K = od_pairs(model)  # list of (o, d) with D[o, d] > 0
        x, _, _, arc_to_keys = add_flow_conservation(m, model, cgn, K)
    return x, arc_to_keys


# ---------- length helpers ----------

def _line_lengths(model):
    """
    Pure infrastructure length per line (sum of directed infra-arc lengths).
    Returns a list of floats with length model.L.
    """
    return [
        float(sum(model.len_a[a] for a in model.line_idx_to_arcs[ell]))
        for ell in range(model.L)
    ]


def _group_lengths(model, line_len):
    """
    Total infrastructure length per line group
    (sum over all directions/lines in the group).

    Args:
        model    : ModelData
        line_len : list of per-line lengths (as returned by _line_lengths)

    Returns:
        dict g -> total length of group g
    """
    gl = {}
    for ell in range(model.L):
        g = model.line_idx_to_group[ell]
        gl[g] = gl.get(g, 0.0) + line_len[ell]
    return gl


def _rep_line_of_group(model):
    """
    Pick a deterministic representative line index per group.
    Preference is the forward line if present, otherwise the backward one.
    Used to reference f_expr[ell_rep] when comparing group frequencies.

    Returns:
        dict g -> representative ell (or None if the group is empty)
    """
    rep = {}
    for g, (fwd, bwd) in model.line_group_to_lines.items():
        rep[g] = fwd if fwd >= 0 else (bwd if bwd >= 0 else None)
    return rep


# ---------- scenario aggregation ----------

def _mean_scenario_capacity(model) -> np.ndarray:
    """
    Probability-weighted per-arc infrastructure capacity across scenarios.

        mean_cap[a] = sum_s p_s * cap_sa[s, a]

    Used as the deterministic surrogate for EEV. Unmodified arcs default to
    `infrastructure_capacity` in every scenario, so the weighted mean equals
    that nominal value; restricted arcs receive a fractional capacity
    reflecting the mix of scenarios. Length matches `model.E_dir`.
    """
    cap_sa = np.asarray(model.cap_sa, dtype=float)  # (S, E_dir)
    p_s = np.asarray(model.p_s, dtype=float).reshape(-1)
    if cap_sa.shape[0] != p_s.shape[0]:
        raise ValueError(
            f"cap_sa rows {cap_sa.shape[0]} != p_s length {p_s.shape[0]}"
        )
    return (p_s[:, None] * cap_sa).sum(axis=0)


# ---------- candidate counters ----------

def _cand_counts(domain):
    """
    Read candidate generation limits from config:
      - detours per line (cand_detour_count)
      - K-shortest alternatives per line (cand_ksp_count)

    Returns:
        (D, K) as non-negative integers.
    """
    D = int(domain.config.get("cand_detour_count", 0))
    K = int(domain.config.get("cand_ksp_count", 0))
    return max(0, D), max(0, K)

# ---------- solver status helpers ----------

_STATUS_PRIORITY = {
    int(GRB.OPTIMAL): 0,

    int(GRB.USER_OBJ_LIMIT): 10,
    int(GRB.SOLUTION_LIMIT): 20,

    int(GRB.ITERATION_LIMIT): 30,
    int(GRB.NODE_LIMIT): 35,
    int(GRB.WORK_LIMIT): 40,
    int(GRB.TIME_LIMIT): 50,
    int(GRB.MEM_LIMIT): 55,

    int(GRB.INTERRUPTED): 60,
    int(GRB.SUBOPTIMAL): 70,
    int(GRB.CUTOFF): 75,

    int(GRB.UNBOUNDED): 80,
    int(GRB.INF_OR_UNBD): 85,
    int(GRB.INFEASIBLE): 90,
    int(GRB.NUMERIC): 100,
}

_STATUS_NAMES = {
    int(GRB.OPTIMAL): "OPTIMAL",
    int(GRB.USER_OBJ_LIMIT): "USER_OBJ_LIMIT",
    int(GRB.SOLUTION_LIMIT): "SOLUTION_LIMIT",
    int(GRB.ITERATION_LIMIT): "ITERATION_LIMIT",
    int(GRB.NODE_LIMIT): "NODE_LIMIT",
    int(GRB.WORK_LIMIT): "WORK_LIMIT",
    int(GRB.TIME_LIMIT): "TIME_LIMIT",
    int(GRB.MEM_LIMIT): "MEM_LIMIT",
    int(GRB.INTERRUPTED): "INTERRUPTED",
    int(GRB.SUBOPTIMAL): "SUBOPTIMAL",
    int(GRB.CUTOFF): "CUTOFF",
    int(GRB.UNBOUNDED): "UNBOUNDED",
    int(GRB.INF_OR_UNBD): "INF_OR_UNBD",
    int(GRB.INFEASIBLE): "INFEASIBLE",
    int(GRB.NUMERIC): "NUMERIC",
}


def _aggregate_status_codes(status_codes):
    """
    Return the most serious status among all required solves.

    Returns:
        (status_code, status_name)
        e.g. (GRB.TIME_LIMIT, "TIME_LIMIT")
    """
    if not status_codes:
        return -1, "UNKNOWN"

    codes = [int(code) for code in status_codes]

    final_status_code = max(
        codes,
        key=lambda code: _STATUS_PRIORITY.get(code, 1000)
    )

    final_status = _STATUS_NAMES.get(
        final_status_code,
        str(final_status_code)
    )

    return final_status_code, final_status

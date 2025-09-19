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

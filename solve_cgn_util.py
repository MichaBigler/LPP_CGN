# solve_cgn.py
from optimisation import (
    od_pairs, add_flow_conservation, add_flow_conservation_by_origin,
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

def _cand_counts(domain):
    D = int(domain.config.get("cand_detour_count", 0))
    K = int(domain.config.get("cand_ksp_count", 0))
    return max(0, D), max(0, K)

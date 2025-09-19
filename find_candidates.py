# find_candidates.py
# -*- coding: utf-8 -*-
"""
Candidate path generation per line and per scenario for LPP-CGN.

Main entry points:
    - build_candidates_all_scenarios_per_line(...)
    - build_candidates_all_scenarios_per_line_cfg(...)

What this module does:
- For each scenario s and each directed line ℓ, generate alternative ride paths
  (nominal/base path, local detours, simple k-shortest variants).
- Optional composite path weights (length + “replanning” penalty vs nominal).
- Optional mirroring of forward-line candidates to the backward line.
- Optional “generate only if disrupted” policy.
- Optional diversity filter (minimum number of differing edges).
- Optional cap for max candidates per line.

Required model attributes (read-only):
- idx_to_arc_uv: List[Tuple[u_id, v_id]]   directed infra arcs in original IDs
- node_id_to_idx: Dict[node_id -> idx]     0..N-1 node index mapping
- len_a: np.ndarray                         length per directed infra arc
- cap_sa: np.ndarray (S × E_dir)            per-scenario capacity per directed arc (>0 means allowed)
- line_idx_to_stops: List[List[int]]        node indices (0..N-1) along each line
- line_idx_to_arcs:  List[List[int]]        directed arc ids along each line
- line_group_to_lines: Dict[group -> (fwd_id, bwd_id)]   (optional; used when mirroring)
- L: int                                    number of directed lines
- E_dir: int                                number of directed infra arcs
- p_s: np.ndarray                           scenario probabilities (used only for length S)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set, Iterable, Literal
import heapq

# Public enum-ish type for mirror behavior (kept for clarity/compatibility).
MirrorMode = Literal["auto", "force", "off"]


# ---------------------------------------------------------------------------
# Low-level helpers: graph indexing, arc properties, and tiny utilities
# ---------------------------------------------------------------------------

def _arc_endpoints(model) -> Tuple[List[int], List[int]]:
    """
    Return tails and heads (node indices 0..N-1) for every directed infra arc.
    Based on original IDs in model.idx_to_arc_uv and mapping model.node_id_to_idx.
    """
    tails: List[int] = []
    heads: List[int] = []
    nid2idx = model.node_id_to_idx
    for (u_id, v_id) in model.idx_to_arc_uv:
        tails.append(int(nid2idx[u_id]))
        heads.append(int(nid2idx[v_id]))
    return tails, heads


def _arc_length(model, a: int) -> float:
    """Length of directed infra arc a."""
    return float(model.len_a[a])


def _as_bool(x, default: bool = False) -> bool:
    """
    Lenient boolean parsing for convenience parameters.
    Accepts on/off variants; falls back to default otherwise.
    """
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _rev_map(model) -> List[Optional[int]]:
    """
    For each directed arc a, return a suitable reverse directed arc ar (if any).
    Robust to multi-edges: choose the reverse with the closest length to a.
    """
    tails, heads = _arc_endpoints(model)
    # Collect lists of arcs for each (u, v).
    by_uv: Dict[Tuple[int, int], List[int]] = {}
    for a in range(model.E_dir):
        by_uv.setdefault((tails[a], heads[a]), []).append(a)

    rev: List[Optional[int]] = [None] * model.E_dir
    for a in range(model.E_dir):
        u, v = tails[a], heads[a]
        candidates = by_uv.get((v, u), [])
        if not candidates:
            rev[a] = None
            continue
        len_a = float(model.len_a[a])
        best = min(candidates, key=lambda r: abs(float(model.len_a[r]) - len_a))
        rev[a] = int(best)
    return rev


def allowed_arcs_forward(model, s: int, eps: float = 1e-9) -> Set[int]:
    """
    Return the set of directed infra arcs whose capacity is strictly positive
    in scenario s (with small tolerance eps).
    """
    cap = model.cap_sa[s, :]
    return {a for a in range(model.E_dir) if float(cap[a]) > eps}


# ---------------------------------------------------------------------------
# Graph view and shortest-path routines on the directed infra graph
# ---------------------------------------------------------------------------

def _adj_directed_weighted(model, allowed: Set[int], rev: List[Optional[int]], weight_fn):
    """
    Build an adjacency view for directed movement. For each allowed arc a: u->v,
    add (v, a, ab, w_uv) where ab is the reverse arc if it exists and is allowed.
    Also add v->u using ab (not a), if ab exists and is allowed. This gives
    symmetric traversal only where both directions are actually usable.
    """
    tails, heads = _arc_endpoints(model)
    adj: Dict[int, List[Tuple[int, int, Optional[int], float]]] = {}
    for a in allowed:
        u, v = tails[a], heads[a]

        # u -> v via a
        w_uv = float(weight_fn(a))
        ab = rev[a] if (rev[a] is not None and rev[a] in allowed) else None
        adj.setdefault(u, []).append((v, a, ab, w_uv))

        # v -> u via ab if the reverse exists and is also allowed
        if ab is not None:
            w_vu = float(weight_fn(ab))
            adj.setdefault(v, []).append((u, ab, a, w_vu))
    return adj


def _shortest_path(model, adj, src: int, dst: int) -> Optional[List[int]]:
    """
    Dijkstra on the directed adjacency view.
    Returns the sequence of directed arc-IDs along the path (not node IDs).
    Empty list for src==dst, None if no path exists.
    """
    pq = [(0.0, src)]
    dist = {src: 0.0}
    prev_arc: Dict[int, Tuple[int, int]] = {}  # node -> (prev_node, prev_arc_forward)

    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            break
        if d > dist.get(u, float('inf')):
            continue
        for (v, a_fwd, _a_bwd, w) in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev_arc[v] = (u, a_fwd)
                heapq.heappush(pq, (nd, v))

    if src == dst:
        return []
    if dst not in prev_arc:
        return None

    # Reconstruct arc sequence from prev pointers.
    path: List[int] = []
    cur = dst
    while cur != src:
        u, a = prev_arc[cur]
        path.append(a)
        cur = u
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Convenience for “nominal/base” paths
# ---------------------------------------------------------------------------

def _mk_nominal_candidate(model, ell: int) -> Dict:
    """Build a nominal candidate dict for line ℓ."""
    arcs_nom = list(map(int, model.line_idx_to_arcs[ell]))
    L_nom = sum(float(model.len_a[a]) for a in arcs_nom)
    return {
        "arcs": arcs_nom,
        "len": L_nom,
        "add_len": 0.0, "rem_len": 0.0, "delta_len_nom": 0.0,
        "kind": "nominal", "is_nominal": True, "is_base": False,
    }


def _nominal_ok(model, s: int, ell: int, allowed: Optional[Set[int]] = None) -> bool:
    """Check whether all nominal arcs of line ℓ are allowed in scenario s."""
    if allowed is None:
        allowed = allowed_arcs_forward(model, s)
    return all(int(a) in allowed for a in model.line_idx_to_arcs[ell])


# ---------------------------------------------------------------------------
# Candidate families: local detours & simple KSP variants
# ---------------------------------------------------------------------------

def _path_src(model, path: List[int]) -> Optional[int]:
    """Return the source node index of a directed arc sequence."""
    tails, _ = _arc_endpoints(model)
    return int(tails[path[0]]) if path else None


def _path_dst(model, path: List[int]) -> Optional[int]:
    """Return the destination node index of a directed arc sequence."""
    tails, heads = _arc_endpoints(model)
    return int(heads[path[-1]]) if path else None


def _detour_candidates(model, adj, base_path: List[int], D: int) -> List[List[int]]:
    """
    Local detours: forbid one arc of the reference path at a time, recompute the
    shortest path, keep up to D unique alternatives.
    """
    if D <= 0 or not base_path:
        return []
    uniq: Dict[Tuple[int, ...], float] = {}

    def plen(p): return sum(_arc_length(model, a) for a in p)

    src = _path_src(model, base_path)
    dst = _path_dst(model, base_path)
    if src is None or dst is None:
        return []

    for idx, banned in enumerate(base_path):
        if len(uniq) >= D:
            break
        # Copy adjacency and drop the banned arc in both directions
        adj2 = {u: nbrs.copy() for u, nbrs in adj.items()}
        for u, nbrs in adj2.items():
            adj2[u] = [(v, af, ab, w) for (v, af, ab, w) in nbrs if af != banned and ab != banned]
        alt = _shortest_path(model, adj2, src, dst)
        if alt:
            key = tuple(alt)
            if key not in uniq:
                uniq[key] = plen(alt)

    return [list(k) for k, _ in sorted(uniq.items(), key=lambda kv: kv[1])]


def _yen_ksp(model, adj, src: int, dst: int, K: int) -> List[List[int]]:
    """
    Very lightweight “K-shortest” generator via arc bans applied to the last
    accepted solution. Returns up to K unique paths (including the best).
    This is *not* a full Yen implementation, but is sufficient for variety.
    """
    if K <= 0:
        return []
    base = _shortest_path(model, adj, src, dst)
    if not base:
        return []
    def plen(p): return sum(_arc_length(model, a) for a in p)
    A: List[List[int]] = [base]
    B: List[Tuple[float, List[int]]] = []
    tried: Set[Tuple[int, int]] = set()

    while len(A) < K:
        ref = A[-1]
        for i, banned in enumerate(ref):
            key = (i, banned)
            if key in tried:
                continue
            tried.add(key)
            adj2 = {u: nbrs.copy() for u, nbrs in adj.items()}
            for u, nbrs in adj2.items():
                adj2[u] = [(v, af, ab, w) for (v, af, ab, w) in nbrs if af != banned and ab != banned]
            alt = _shortest_path(model, adj2, src, dst)
            if alt and tuple(alt) not in map(tuple, A):
                heapq.heappush(B, (plen(alt), alt))
        if not B:
            break
        _, best = heapq.heappop(B)
        if tuple(best) not in map(tuple, A):
            A.append(best)
    return A


def _unique_paths(paths: Iterable[List[int]]) -> List[List[int]]:
    """Deduplicate a list of arc sequences while keeping order of first appearance."""
    out: List[List[int]] = []
    seen = set()
    for p in paths:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            out.append(p)
    return out


def _edge_diversity_ok(p: List[int], kept: List[List[int]], min_diff_edges: int) -> bool:
    """
    Check that path p differs by at least `min_diff_edges` from each already kept path.
    Distance metric: size of symmetric difference over arc-id sets.
    """
    if min_diff_edges <= 0:
        return True
    set_p = set(p)
    for q in kept:
        if len(set_p.symmetric_difference(set(q))) < min_diff_edges:
            return False
    return True


def _uniq_keep_order(cands: List[Dict]) -> List[Dict]:
    """Deduplicate candidates by their arc sequences while preserving order."""
    seen = set()
    out = []
    for c in cands:
        key = tuple(int(a) for a in c.get("arcs", []))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Per-line candidate builder (single scenario)
# ---------------------------------------------------------------------------

def _line_endpoints(model, ell: int) -> Tuple[int, int, List[int]]:
    """
    Return (src, dst, nominal_arc_seq) for a *directed* line ℓ.
    Note: line_idx_to_stops already stores node indices (0..N-1).
    """
    stops = list(map(int, model.line_idx_to_stops[ell]))
    arcs  = list(map(int, model.line_idx_to_arcs[ell]))
    return int(stops[0]), int(stops[-1]), arcs


def candidates_for_line_scenario(
    model,
    s: int,
    ell: int,
    detour_count: int,
    ksp_count: int,
    rev: List[Optional[int]],
    *,
    ksp_weight_mode: str = "length",   # "length" or "composite"
    w_len: float = 1.0,                 # length weight
    w_repl: float = 0.0,                # replanning penalty weight (used if composite)
    gamma_ksp: Optional[float] = None,  # override for replanning weight in KSP mode
    only_when_blocked: bool = False,    # generate alternatives only if nominal is disrupted
    min_edge_diff: int = 1,             # diversity threshold
    max_candidates_per_line: Optional[int] = None,  # cap
    corr_eps: Optional[float] = None,   # corridor: len(p) ≤ (1 + corr_eps) * len(reference)
) -> List[Dict]:
    """
    Generate candidate paths for a single directed line ℓ in scenario s.

    Returns a list of dicts:
        {
          "arcs": [directed arc ids],
          "len": float,
          "add_len": float, "rem_len": float, "delta_len_nom": float,
          "kind": "nominal" | "base" | "detour" | "ksp" | "alt",
          "is_nominal": bool,
          "is_base": bool
        }
    """
    # Allowed infra arcs in this scenario
    allowed = allowed_arcs_forward(model, s)

    # Endpoints and nominal arc chain for ℓ
    src, dst, nominal = _line_endpoints(model, ell)
    nominal_ok = all(a in allowed for a in nominal)

    nominal_set = set(nominal)

    # Weight function for arcs
    if ksp_weight_mode.lower() == "composite":
        # Composite: length + replanning penalty if arc not in nominal set
        gamma = float(w_repl if gamma_ksp is None else gamma_ksp)
        def weight_fn(a: int) -> float:
            L = _arc_length(model, a)
            return w_len * L + (gamma if a not in nominal_set else 0.0) * L
    else:
        # Pure length-based weights
        def weight_fn(a: int) -> float:
            return w_len * _arc_length(model, a)

    # Directed adjacency for current scenario
    adj = _adj_directed_weighted(model, allowed, rev, weight_fn)

    # Reference path: base shortest under current adjacency
    base = _shortest_path(model, adj, src, dst)
    if not base:
        # If no path under adjacency but the nominal is fully allowed, keep nominal
        if nominal_ok and nominal:
            return [{
                "arcs": nominal,
                "len": sum(_arc_length(model, a) for a in nominal),
                "add_len": 0.0, "rem_len": 0.0, "delta_len_nom": 0.0,
                "kind": "nominal", "is_nominal": True, "is_base": False
            }]
        # No valid path at all
        return []

    # Seed for deriving alternatives
    seed = nominal if nominal_ok else base

    # Build initial pool (always include nominal if it is valid)
    pool: List[List[int]] = []
    if nominal_ok and nominal:
        pool.append(nominal)
    if base and (not nominal_ok or base != nominal):
        pool.append(base)

    # Only add alternates if not restricted by policy
    dets: List[List[int]] = []
    ksps: List[List[int]] = []
    if not (only_when_blocked and nominal_ok):
        if detour_count > 0:
            dets = _detour_candidates(model, adj, seed, detour_count)
            pool += dets
        if ksp_count > 0:
            ksps = [p for p in _yen_ksp(model, adj, src, dst, ksp_count) if p and p != seed]
            pool += ksps

    # Deduplicate by arc sequences
    pool = _unique_paths(pool) or []

    # If empty (very unlikely), fall back to base
    if not pool:
        pool = [base]

    # Ensure nominal is at the front if it exists
    if nominal_ok:
        for i, p in enumerate(pool):
            if p == nominal and i != 0:
                pool.insert(0, pool.pop(i))
                break

    def plen(p: List[int]) -> float:
        return sum(_arc_length(model, a) for a in p)

    # Reference for corridor filtering
    ref_path = nominal if nominal_ok else (base if base else [])
    L_ref = plen(ref_path) if ref_path else float("inf")

    def within_corr(p: List[int]) -> bool:
        return True if corr_eps is None else (plen(p) <= (1.0 + float(corr_eps)) * L_ref)

    # Single filtering pass: corridor + diversity + optional cap
    filtered: List[List[int]] = []
    for p in pool:
        if within_corr(p) and _edge_diversity_ok(p, filtered, min_edge_diff):
            filtered.append(p)

    if not filtered:
        filtered = [nominal] if nominal_ok else [base]

    # Label sets to mark origins of each kept path
    detour_set = set(map(tuple, dets)) if dets else set()
    ksp_set    = set(map(tuple, ksps)) if ksps else set()

    # Build output dicts (compute add/rem/delta vs nominal set)
    out: List[Dict] = []
    ref_set = nominal_set
    for p in filtered:
        pset   = set(p)
        add_len = sum(_arc_length(model, a) for a in (pset - ref_set))
        rem_len = sum(_arc_length(model, a) for a in (ref_set - pset))
        delta   = add_len + rem_len

        if nominal_ok and p == nominal:
            kind, is_nom, is_base = "nominal", True, False
        elif (not nominal_ok) and p == base:
            kind, is_nom, is_base = "base", False, True
        elif tuple(p) in detour_set:
            kind, is_nom, is_base = "detour", False, False
        elif tuple(p) in ksp_set:
            kind, is_nom, is_base = "ksp", False, False
        else:
            kind, is_nom, is_base = "alt", False, False

        out.append({
            "arcs": p,
            "len": plen(p),
            "add_len": add_len,
            "rem_len": rem_len,
            "delta_len_nom": delta,
            "kind": kind,
            "is_nominal": is_nom,
            "is_base": is_base,
        })

    # Optional cap
    if max_candidates_per_line is not None and len(out) > int(max_candidates_per_line):
        out = out[:int(max_candidates_per_line)]

    return out


def _mirror_candidates_for_line(
    model,
    s: int,
    cand_list_fwd: List[Dict],
    rev: List[Optional[int]],
    ell_bwd: int,
) -> List[Dict]:
    """
    Mirror forward-line candidates to a backward line:

    - Reverse the arc sequence and map each arc via rev[a].
    - Recompute add/rem/delta vs. the backward line's *nominal* path.
    - Drop mirrored candidates if any reverse arc is missing or not allowed in s.
    """
    allowed = allowed_arcs_forward(model, s)
    _, _, nominal_bwd = _line_endpoints(model, ell_bwd)
    nominal_set_bwd = set(nominal_bwd)

    def plen(p): return sum(_arc_length(model, a) for a in p)

    mirrored: List[Dict] = []
    for cand in cand_list_fwd:
        arcs_fwd = list(map(int, cand.get("arcs", [])))
        ok = True
        arcs_bwd: List[int] = []
        for a in reversed(arcs_fwd):
            ra = rev[a]
            if ra is None or ra not in allowed:
                ok = False
                break
            arcs_bwd.append(ra)
        if not ok or not arcs_bwd:
            continue

        pset = set(arcs_bwd)
        add_len = sum(_arc_length(model, a) for a in (pset - nominal_set_bwd))
        rem_len = sum(_arc_length(model, a) for a in (nominal_set_bwd - pset))
        delta   = add_len + rem_len

        is_nom = (arcs_bwd == nominal_bwd)
        kind   = "nominal" if is_nom else cand.get("kind", "alt")

        mirrored.append({
            "arcs": arcs_bwd,
            "len": plen(arcs_bwd),
            "add_len": add_len,
            "rem_len": rem_len,
            "delta_len_nom": delta,
            "kind": kind,
            "is_nominal": bool(is_nom),
            "is_base": False,  # we don’t re-check “base” for the mirrored variant
        })
    return mirrored


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def build_candidates_all_scenarios_per_line(
    model,
    detour_count: int,
    ksp_count: int,
    *,
    ksp_weight_mode: str = "length",
    w_len: float = 1.0,
    w_repl: float = 0.0,
    gamma_ksp: Optional[float] = None,
    only_when_blocked: bool = False,
    min_edge_diff: int = 1,
    max_candidates_per_line: Optional[int] = None,
    mirror_backward: bool = False,
    corr_eps: Optional[float] = None,
) -> Dict[int, Dict[int, List[Dict]]]:
    """
    Build candidates for all scenarios and all directed lines.

    Returns:
        candidates[s][ell] -> List[Dict] (see candidates_for_line_scenario)

    Notes:
    - The current API only exposes `mirror_backward` as a boolean. A more
      expressive mirror mode (“auto|force|off”) is hinted in code but not
      passed through publicly. Keeping behavior identical to existing code.
      (TODO: Plumb a proper `mirror_mode` parameter if needed.)
    """
    rev = _rev_map(model)
    S   = len(model.p_s)
    results: Dict[int, Dict[int, List[Dict]]] = {}

    # Internal “mirror_mode” is currently “auto” only — retained for compatibility.
    mirror_mode = "auto"  # kept to preserve the original guard logic

    groups = getattr(model, "line_group_to_lines", None)

    for s in range(S):
        per_line: Dict[int, List[Dict]] = {}

        if mirror_backward and isinstance(groups, dict):
            # Treat forward/backward line pair as a unit to optionally mirror candidates.
            for g, pair in groups.items():
                ell_fwd, ell_bwd = pair

                # 1) Independently generate for both directions (unless “force” mirroring was intended).
                cand_fwd_gen: List[Dict] = []
                cand_bwd_gen: List[Dict] = []
                if ell_fwd is not None and ell_fwd >= 0 and mirror_mode != "force":
                    cand_fwd_gen = candidates_for_line_scenario(
                        model, s, int(ell_fwd),
                        detour_count, ksp_count, rev,
                        ksp_weight_mode=ksp_weight_mode,
                        w_len=w_len, w_repl=w_repl, gamma_ksp=gamma_ksp,
                        only_when_blocked=only_when_blocked,
                        min_edge_diff=min_edge_diff,
                        max_candidates_per_line=max_candidates_per_line,
                        corr_eps=corr_eps,
                    )
                if ell_bwd is not None and ell_bwd >= 0 and mirror_mode != "force":
                    cand_bwd_gen = candidates_for_line_scenario(
                        model, s, int(ell_bwd),
                        detour_count, ksp_count, rev,
                        ksp_weight_mode=ksp_weight_mode,
                        w_len=w_len, w_repl=w_repl, gamma_ksp=gamma_ksp,
                        only_when_blocked=only_when_blocked,
                        min_edge_diff=min_edge_diff,
                        max_candidates_per_line=max_candidates_per_line,
                        corr_eps=corr_eps,
                    )

                # 2) Mirror both ways
                cand_fwd_mir: List[Dict] = []
                cand_bwd_mir: List[Dict] = []
                if ell_fwd is not None and ell_fwd >= 0 and cand_bwd_gen:
                    cand_fwd_mir = _mirror_candidates_for_line(model, s, cand_bwd_gen, rev, int(ell_fwd))
                if ell_bwd is not None and ell_bwd >= 0 and cand_fwd_gen:
                    cand_bwd_mir = _mirror_candidates_for_line(model, s, cand_fwd_gen, rev, int(ell_bwd))

                # 3) For each direction: always include nominal (if allowed) + union + dedup + optional cap
                if ell_fwd is not None and ell_fwd >= 0:
                    out_fwd: List[Dict] = []
                    if _nominal_ok(model, s, int(ell_fwd), allowed_arcs_forward(model, s)):
                        out_fwd.append(_mk_nominal_candidate(model, int(ell_fwd)))
                    out_fwd += cand_fwd_mir + cand_fwd_gen
                    out_fwd = _uniq_keep_order(out_fwd)
                    if max_candidates_per_line is not None and len(out_fwd) > int(max_candidates_per_line):
                        out_fwd = out_fwd[:int(max_candidates_per_line)]
                    per_line[int(ell_fwd)] = out_fwd

                if ell_bwd is not None and ell_bwd >= 0:
                    out_bwd: List[Dict] = []
                    if _nominal_ok(model, s, int(ell_bwd), allowed_arcs_forward(model, s)):
                        out_bwd.append(_mk_nominal_candidate(model, int(ell_bwd)))
                    out_bwd += cand_bwd_mir + cand_bwd_gen
                    out_bwd = _uniq_keep_order(out_bwd)
                    if max_candidates_per_line is not None and len(out_bwd) > int(max_candidates_per_line):
                        out_bwd = out_bwd[:int(max_candidates_per_line)]
                    per_line[int(ell_bwd)] = out_bwd
        else:
            # Independent per directed line (no mirroring)
            for ell in range(model.L):
                per_line[int(ell)] = candidates_for_line_scenario(
                    model, s, int(ell),
                    detour_count, ksp_count, rev,
                    ksp_weight_mode=ksp_weight_mode,
                    w_len=w_len, w_repl=w_repl, gamma_ksp=gamma_ksp,
                    only_when_blocked=only_when_blocked,
                    min_edge_diff=min_edge_diff,
                    max_candidates_per_line=max_candidates_per_line,
                    corr_eps=corr_eps,
                )

        results[int(s)] = per_line

    return results


def build_candidates_all_scenarios_per_line_cfg(model, cand_cfg, main_cfg):
    """
    Convenience wrapper: read weights and policy knobs from CandidateConfig + main Config (dict-like).

    Mirrors original behavior:
    - If cand_cfg.w_len / w_repl is None, mirror from main_cfg (line_operation_cost_mult / cost_repl_line).
    - mirror_backward boolean is derived from cand_cfg.mirror_backward != "off".
      (Note: a potential “force” mode is *not* passed through; behavior unchanged.)
    """
    def _get(cfg, key, default):
        try:
            return getattr(cfg, key)
        except Exception:
            return cfg.get(key, default)

    w_len  = cand_cfg.w_len  if cand_cfg.w_len  is not None else float(_get(main_cfg, "line_operation_cost_mult", 1.0))
    w_repl = cand_cfg.w_repl if cand_cfg.w_repl is not None else float(_get(main_cfg, "cost_repl_line", 0.0))

    corr = None if getattr(cand_cfg, "corr_eps", None) is None else float(cand_cfg.corr_eps)
    mode: MirrorMode = getattr(cand_cfg, "mirror_backward", "auto") or "auto"
    mirror_mode = str(mode).lower()

    return build_candidates_all_scenarios_per_line(
        model,
        detour_count=cand_cfg.k_loc_detour,
        ksp_count=cand_cfg.k_sp_global,
        ksp_weight_mode=("composite" if abs(w_repl) > 1e-12 else "length"),
        w_len=w_len,
        w_repl=w_repl,
        gamma_ksp=None,
        only_when_blocked=bool(cand_cfg.generate_only_if_disrupted),
        min_edge_diff=int(cand_cfg.div_min_edges),
        max_candidates_per_line=int(cand_cfg.max_candidates_per_line),
        mirror_backward=(mirror_mode != "off"),  # “force” is not wired through here (unchanged behavior).
        corr_eps=corr,
    )

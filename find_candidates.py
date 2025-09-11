# candidates.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set
import heapq

# ---------- Helper ----------

def _arc_endpoints(model):
    """
    Liefert (tail, head) als Listen der Knoten-INDIZES (0..N-1) je gerichteter Arc.
    Basis sind die in load_data erzeugten Felder:
      - idx_to_arc_uv: List[Tuple[u_id, v_id]] (Original-IDs)
      - node_id_to_idx: Dict[id -> index]
    """
    tails: List[int] = []
    heads: List[int] = []
    nid2idx = model.node_id_to_idx
    for (u_id, v_id) in model.idx_to_arc_uv:       # Original-ID-Paare
        tails.append(int(nid2idx[u_id]))           # in Indexraum mappen
        heads.append(int(nid2idx[v_id]))
    return tails, heads

def _arc_length(model, a: int) -> float:
    # load_data liefert len_a als Längenarray für gerichtete Arcs
    return float(model.len_a[a])

def _rev_map(model) -> List[Optional[int]]:
    """
    Zu jedem gerichteten Arc a seine Gegenrichtung ar (falls vorhanden), sonst None.
    Arbeitet vollständig im Indexraum der Knoten.
    """
    tail, head = _arc_endpoints(model)
    by_uv: Dict[Tuple[int, int], int] = {}
    for a in range(model.E_dir):
        by_uv[(tail[a], head[a])] = a
    rev: List[Optional[int]] = [None] * model.E_dir
    for a in range(model.E_dir):
        rev[a] = by_uv.get((head[a], tail[a]))
    return rev


def allowed_arcs_forward(model, s: int) -> set[int]:
    """Zulässig sind forward-Arcs mit cap>0 im Szenario s."""
    cap = model.cap_sa[s, :]
    return {a for a in range(model.E_dir) if cap[a] > 0}

def _adj_undirected_loose(model, allowed: set[int], rev: list[int|None]):
    """Undirektionale Sicht: Kante (u,v) existiert, wenn (u->v) zulässig ist (Gegenkante optional)."""
    tail, head = _arc_endpoints(model)
    adj = {}
    seen = set()
    for a in allowed:
        u, v = tail[a], head[a]
        key = (min(u, v), max(u, v))
        if key in seen: continue
        seen.add(key)
        w = _arc_length(model, a)
        af = a
        ab = rev[a] if rev[a] in allowed else None
        adj.setdefault(u, []).append((v, af, ab, w))
        adj.setdefault(v, []).append((u, ab if ab is not None else af, af, w))
    return adj

def _shortest_path(model, adj, src: int, dst: int) -> Optional[List[int]]:
    """Dijkstra auf undirektionaler Sicht; Rückgabe: gerichtete Arc-ID Folge."""
    pq = [(0.0, src)]
    dist = {src: 0.0}
    prev: Dict[int, Tuple[int,int]] = {}  # node -> (prev_node, prev_arc_forward)
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            break
        if d > dist.get(u, float("inf")):
            continue
        for (v, a_fwd, _a_bwd, w) in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = (u, a_fwd)
                heapq.heappush(pq, (nd, v))
    if dst not in prev and src != dst:
        return None
    # rekonstruiere
    path: List[int] = []
    cur = dst
    while cur != src:
        u, a = prev[cur]
        path.append(a)
        cur = u
    path.reverse()
    return path

# ---------- Detour-Kandidaten (einfach: Kanten des Basisweges sperren) ----------

def _detour_candidates(model, adj, base_path: List[int], D: int) -> List[List[int]]:
    if D <= 0 or not base_path:
        return []
    # entferne nacheinander je eine Kante des Basisweges
    uniq: Dict[Tuple[int, ...], float] = {}
    def plen(p): return sum(_arc_length(model, a) for a in p)
    for idx, banned in enumerate(base_path):
        if len(uniq) >= D:
            break
        adj2 = {u: nbrs.copy() for u, nbrs in adj.items()}
        for u, nbrs in adj2.items():
            adj2[u] = [(v, af, ab, w) for (v, af, ab, w) in nbrs if af != banned and ab != banned]
        alt = _shortest_path(model, adj2, _path_src(model, base_path), _path_dst(model, base_path))
        if alt:
            key = tuple(alt)
            if key not in uniq:
                uniq[key] = plen(alt)
    return [list(k) for k, _ in sorted(uniq.items(), key=lambda kv: kv[1])]

def _path_src(model, path: List[int]) -> int:
    tail, _ = _arc_endpoints(model)
    return int(tail[path[0]]) if path else None

def _path_dst(model, path: List[int]) -> int:
    tail, head = _arc_endpoints(model)
    return int(head[path[-1]]) if path else None

# ---------- K-Shortest (vereinfachte Yen-Variante) ----------

def _yen_ksp(model, adj, src: int, dst: int, K: int) -> List[List[int]]:
    """Sehr schlanke KSP-Variante: erzeugt bis zu K eindeutige Pfade (inkl. kürzestem)."""
    if K <= 0:
        return []
    base = _shortest_path(model, adj, src, dst)
    if not base:
        return []
    # A: akzeptierte Pfade; B: Kandidaten (Dist, Pfad)
    def plen(p): return sum(_arc_length(model, a) for a in p)
    A: List[List[int]] = [base]
    B: List[Tuple[float, List[int]]] = []
    # generiere Kandidaten, indem je ein Arc der letzten akzeptierten Wege gesperrt wird
    tried: Set[Tuple[int, int]] = set()  # (index_in_path, arc_id) um Doppelarbeit zu vermeiden
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
    return A  # enthält base + weitere, max. K

# ---------- Gruppen-/Szenario-Kandidaten ----------

def _rep_line_of_group(model) -> Dict[int, Optional[int]]:
    rep = {}
    for g, (fwd, bwd) in model.line_group_to_lines.items():
        rep[g] = fwd if fwd >= 0 else (bwd if bwd >= 0 else None)
    return rep

def _line_endpoints(model, ell: int) -> tuple[int,int,list[int]]:
    """src,dst,nominal-arc-sequenz für genau diese Linie ℓ (eine Richtung)."""
    stops = model.line_idx_to_stops[ell]
    arcs  = model.line_idx_to_arcs[ell]
    return int(stops[0]), int(stops[-1]), list(map(int, arcs))

def _unique_by_tuple(paths: List[List[int]]) -> List[List[int]]:
    out = []
    seen = set()
    for p in paths:
        t = tuple(p)
        if t not in seen:
            seen.add(t); out.append(p)
    return out

def candidates_for_line_scenario(model, s: int, ell: int,
                                 detour_count: int, ksp_count: int,
                                 rev: list[int|None]) -> list[dict]:
    A   = allowed_arcs_forward(model, s)
    adj = _adj_undirected_loose(model, A, rev)
    src, dst, nominal = _line_endpoints(model, ell)
    nominal_ok = all(a in A for a in nominal)

    base = _shortest_path(model, adj, src, dst)
    if not base:
        return []

    seed = nominal if nominal_ok else base
    pool = [seed]
    if detour_count > 0:
        pool += _detour_candidates(model, adj, seed, detour_count)
    if ksp_count > 0:
        for p in _yen_ksp(model, adj, src, dst, ksp_count):
            if p and p != seed:
                pool.append(p)

    # unique + nominal nach vorn
    seen, uniq = set(), []
    for p in pool:
        t = tuple(p)
        if t not in seen:
            seen.add(t); uniq.append(p)
    pool = uniq
    if nominal_ok:
        for i,p in enumerate(pool):
            if p == nominal and i != 0:
                pool.insert(0, pool.pop(i))
                break

    # Kosten (immer ggü. nominal dieser Linie)
    ref = nominal
    ref_set = set(ref)
    res = []
    detour_set = set(map(tuple, _detour_candidates(model, adj, seed, max(0,detour_count)))) if detour_count>0 else set()
    ksp_set    = set(map(tuple, _yen_ksp(model, adj, src, dst, max(0,ksp_count)))) if ksp_count>0 else set()

    def plen(p): return sum(_arc_length(model,a) for a in p)

    for p in pool:
        pset = set(p)
        add_len = sum(_arc_length(model,a) for a in (pset - ref_set))
        rem_len = sum(_arc_length(model,a) for a in (ref_set - pset))
        delta   = add_len + rem_len
        if nominal_ok and p == nominal: kind, is_nom, is_base = "nominal", True,  False
        elif (not nominal_ok) and p == base: kind, is_nom, is_base = "base", False, True
        elif tuple(p) in detour_set: kind, is_nom, is_base = "detour", False, False
        elif tuple(p) in ksp_set:    kind, is_nom, is_base = "ksp",    False, False
        else:                        kind, is_nom, is_base = "alt",    False, False

        res.append({
            "arcs": p,
            "len": plen(p),
            "add_len": add_len, "rem_len": rem_len, "delta_len_nom": delta,
            "kind": kind, "is_nominal": is_nom, "is_base": is_base
        })
    return res

def build_candidates_all_scenarios_per_line(model, detour_count: int, ksp_count: int) -> dict[int, dict[int, list[dict]]]:
    """cands[s][ell] -> Kandidatenliste für Linie ℓ im Szenario s."""
    rev = _rev_map(model)
    S   = len(model.p_s)
    cands = {}
    for s in range(S):
        per_line = {}
        for ell in range(model.L):
            per_line[ell] = candidates_for_line_scenario(model, s, ell, detour_count, ksp_count, rev)
        cands[s] = per_line
    return cands

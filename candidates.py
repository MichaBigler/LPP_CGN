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


def allowed_arcs_bidir(model, s: int, rev: List[Optional[int]]) -> Set[int]:
    """Erlaubt nur Arcs mit Gegenbogen und cap>0 in BEIDEN Richtungen."""
    A: Set[int] = set()
    cap = model.cap_sa[s, :]
    for a in range(model.E_dir):
        ar = rev[a]
        if ar is not None and cap[a] > 0 and cap[ar] > 0:
            A.add(a)
    return A

def _adj_undirected(model, allowed: Set[int], rev: List[Optional[int]]):
    """Undirektionale Sicht (für einfache Pfade ohne Subtouren)."""
    tail, head = _arc_endpoints(model)
    adj: Dict[int, List[Tuple[int, int, int, float]]] = {}
    seen_pairs: Set[Tuple[int,int]] = set()
    for a in allowed:
        ar = rev[a]
        if ar is None or ar not in allowed: 
            continue
        u, v = tail[a], head[a]
        key = (min(u, v), max(u, v))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        w = _arc_length(model, a)
        adj.setdefault(u, []).append((v, a, ar, w))
        adj.setdefault(v, []).append((u, ar, a, w))
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

def _group_endpoints(model, g: int) -> Tuple[int, int, List[int]]:
    rep = _rep_line_of_group(model)
    ell = rep[g]
    if ell is None:
        raise ValueError(f"Group {g} hat keine repräsentative Linie.")
    stops = model.line_idx_to_stops[ell]
    arcs  = model.line_idx_to_arcs[ell]
    return int(stops[0]), int(stops[-1]), list(arcs)

def _unique_by_tuple(paths: List[List[int]]) -> List[List[int]]:
    out = []
    seen = set()
    for p in paths:
        t = tuple(p)
        if t not in seen:
            seen.add(t); out.append(p)
    return out

def candidates_for_group_scenario(model, s: int, g: int,
                                  detour_count: int, ksp_count: int,
                                  rev: List[Optional[int]]) -> List[Dict]:
    """
    Liefert Kandidaten inkl. nominal (falls zulässig), Detour (D) und KSP (K).
    Rückgabe-Item: {"arcs":[...], "len":..., "add_len":..., "rem_len":...}
    """
    A = allowed_arcs_bidir(model, s, rev)
    adj = _adj_undirected(model, A, rev)
    src, dst, nominal = _group_endpoints(model, g)

    nominal_ok = all(a in A for a in nominal)
    # Basisweg = kürzester im erlaubten Netz
    base = _shortest_path(model, adj, src, dst)
    if not base:
        return []

    # 1) nominal ggf. einfügen
    pool: List[List[int]] = []
    if nominal_ok:
        pool.append(nominal)
    else:
        # wenn nominal nicht zulässig, nimm den Basisweg als Start
        pool.append(base)

    # 2) Detour-Kandidaten basierend auf dem *nominalen* (falls ok, sonst base)
    seed = nominal if nominal_ok else base
    pool.extend(_detour_candidates(model, adj, seed, max(0, detour_count)))

    # 3) KSP-Kandidaten (inkl. base): wir nehmen daraus die Nicht-Seed-Kandidaten
    ksp = _yen_ksp(model, adj, src, dst, max(0, ksp_count))
    for p in ksp:
        if p != seed:
            pool.append(p)

    # Eindeutig + zuschneiden: nominal (falls ok) an erste Position
    pool = _unique_by_tuple(pool)

    # Replanning-Basiskosten IMMER ggü. nominal (auch wenn nominal nicht zulässig)
    rep = _rep_line_of_group(model)
    ell_rep = rep[g]
    nominal_arcs = list(model.line_idx_to_arcs[ell_rep])  # gerichtete Arc-IDs
    nom_set = set(nominal_arcs)

    res = []
    for p in pool:
        p_set = set(p)
        add_len_nom = sum(_arc_length(model, a) for a in (p_set - nom_set))
        rem_len_nom = sum(_arc_length(model, a) for a in (nom_set - p_set))
        res.append({
            "arcs": p,
            "len": sum(_arc_length(model, a) for a in p),
            # immer relativ zum nominalen Linienpfad
            "add_len_nom": add_len_nom,
            "rem_len_nom": rem_len_nom,
        })
    return res

def build_candidates_all_scenarios(model, detour_count: int, ksp_count: int) -> Dict[int, Dict[int, List[Dict]]]:
    """
    candidates[s][g] = Liste von Kandidaten (Dict mit 'arcs','len','add_len','rem_len')
    """
    rev = _rev_map(model)
    groups = sorted(model.line_group_to_lines.keys())
    S = len(model.p_s)
    result: Dict[int, Dict[int, List[Dict]]] = {}
    for s in range(S):
        per_g: Dict[int, List[Dict]] = {}
        for g in groups:
            per_g[g] = candidates_for_group_scenario(model, s, g, detour_count, ksp_count, rev)
        result[s] = per_g
    return result

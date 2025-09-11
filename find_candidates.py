# -*- coding: utf-8 -*-
"""
find_candidates.py
------------------
Per-Linie Kandidatengenerierung für LPP-CGN.

Hauptfunktion:
    build_candidates_all_scenarios_per_line(...)

Features:
- Kandidaten je Linie und Szenario (nominal/base + lokale Detours + K-Shortest)
- Optional: Composite-KSP-Gewichte (Länge + Replan-Penalty ggü. nominal)
- Optional: Spiegeln der Vorwärts-Kandidaten auf die Rückwärtslinie (mirror_backward)
- Optional: Kandidaten nur erzeugen, wenn nominal im Szenario verletzt ist (only_when_blocked)
- Optional: Diversitätsfilter (minimale Anzahl unterschiedlicher Kanten)
- Optional: Deckelung pro Linie (max_candidates_per_line)

Benötigte Model-Felder:
- idx_to_arc_uv: List[Tuple[u_id, v_id]] gerichtete Infrastruktur-Arcs (Original-ID)
- node_id_to_idx: Dict[id -> idx] Mapping auf 0..N-1
- len_a: Array[float] Länge je gerichteter Infrastruktur-Arc
- cap_sa: Array[S, E_dir] Kapazität je Szenario und gerichteter Arc (>0: erlaubt)
- line_idx_to_stops: List[List[int]] Knoten-INDIZES der Linie (0..N-1)
- line_idx_to_arcs: List[List[int]] gerichtete Arc-IDs der Linie
- line_group_to_lines: Dict[g] -> (ell_fwd, ell_bwd)  (optional, für mirror_backward)
- L: Anzahl Linien
- E_dir: Anzahl gerichteter Infrastruktur-Arcs
- p_s: Szenario-Wahrscheinlichkeiten (wird hier nicht benötigt, nur Länge S)

Autor: du :-)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set, Iterable
import heapq


# --------------------------- Low-level Helpers ---------------------------

def _arc_endpoints(model):
    """
    Liefert (tail, head) als Listen der Knoten-INDIZES (0..N-1) je gerichteter Arc,
    basierend auf Original-IDs in idx_to_arc_uv und node_id_to_idx.
    """
    tails: List[int] = []
    heads: List[int] = []
    nid2idx = model.node_id_to_idx
    for (u_id, v_id) in model.idx_to_arc_uv:
        tails.append(int(nid2idx[u_id]))
        heads.append(int(nid2idx[v_id]))
    return tails, heads


def _arc_length(model, a: int) -> float:
    return float(model.len_a[a])


def _rev_map(model) -> List[Optional[int]]:
    """
    Zu jedem gerichteten Arc a seine Gegenrichtung ar (falls vorhanden), sonst None.
    Arbeitet im Indexraum der Knoten.
    """
    tail, head = _arc_endpoints(model)
    by_uv: Dict[Tuple[int, int], int] = {}
    for a in range(model.E_dir):
        by_uv[(tail[a], head[a])] = a
    rev: List[Optional[int]] = [None] * model.E_dir
    for a in range(model.E_dir):
        rev[a] = by_uv.get((head[a], tail[a]))
    return rev


def allowed_arcs_forward(model, s: int) -> Set[int]:
    """Zulässig sind forward-Arcs mit cap>0 im Szenario s."""
    cap = model.cap_sa[s, :]
    return {a for a in range(model.E_dir) if cap[a] > 0}


# --------------------------- Graph Views & Shortest Path ---------------------------

def _adj_directed_weighted(model, allowed: Set[int], rev: List[Optional[int]], weight_fn):
    """
    Gerichtete Sicht mit richtungs-spezifischen Gewichten.
    adj[u] enthält Tupel (v, a_fwd, a_bwd, w_uv) für Kante u->v.
    """
    tail, head = _arc_endpoints(model)
    adj: Dict[int, List[Tuple[int, int, Optional[int], float]]] = {}
    seen_undirected = set()
    for a in allowed:
        u, v = tail[a], head[a]
        key = (min(u, v), max(u, v))
        if key in seen_undirected:
            continue
        seen_undirected.add(key)

        ab = rev[a] if (rev[a] is not None and rev[a] in allowed) else None

        # u -> v
        w_uv = float(weight_fn(a))
        adj.setdefault(u, []).append((v, a, ab, w_uv))

        # v -> u (falls Rückbogen; sonst fallback: gleicher a, aber in Gegenrichtung)
        a_back = ab if ab is not None else a
        w_vu = float(weight_fn(a_back))
        # In diesem Eintrag ist a_fwd = a_back (v->u), a_bwd = a (u->v)
        adj.setdefault(v, []).append((u, a_back, a, w_vu))
    return adj


def _shortest_path(model, adj, src: int, dst: int) -> Optional[List[int]]:
    """Dijkstra auf gerichteter Sicht; Rückgabe: Folge der **gerichteten Arc-IDs** entlang des Pfads."""
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
    # Rekonstruktion
    path: List[int] = []
    cur = dst
    while cur != src:
        u, a = prev_arc[cur]
        path.append(a)
        cur = u
    path.reverse()
    return path


# --------------------------- Candidate Families ---------------------------

def _path_src(model, path: List[int]) -> Optional[int]:
    tails, _ = _arc_endpoints(model)
    return int(tails[path[0]]) if path else None


def _path_dst(model, path: List[int]) -> Optional[int]:
    tails, heads = _arc_endpoints(model)
    return int(heads[path[-1]]) if path else None


def _detour_candidates(model, adj, base_path: List[int], D: int) -> List[List[int]]:
    """
    Lokale Umleitungen: jeweils eine Kante des Referenzpfades sperren und kürzesten Weg berechnen.
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
        # Kopie der Adjazenz mit gesperrtem Arc (in beiden Richtungen)
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
    Sehr einfache K-Shortest auf Basis von „Arc-Sperren“ der zuletzt akzeptierten Lösung.
    Liefert bis zu K eindeutige Pfade (inkl. dem besten).
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
    Prüft, ob Pfad p sich in mindestens 'min_diff_edges' Kanten von allen bereits
    behaltenen Pfaden unterscheidet (symmetrische Differenz).
    """
    if min_diff_edges <= 0:
        return True
    set_p = set(p)
    for q in kept:
        if len(set_p.symmetric_difference(set(q))) < min_diff_edges:
            return False
    return True


# --------------------------- Per-Line Candidate Builder ---------------------------

def _line_endpoints(model, ell: int) -> Tuple[int, int, List[int]]:
    """
    src, dst, nominal-Arc-Sequenz für genau diese Linie ℓ (eine Richtung).
    ACHTUNG: stops sind bereits Knoten-INDIZES (0..N-1).
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
    # KSP-Gewicht:
    ksp_weight_mode: str = "length",
    w_len: float = 1.0,
    w_repl: float = 0.0,
    gamma_ksp: Optional[float] = None,
    # Policies:
    only_when_blocked: bool = False,
    min_edge_diff: int = 1,
    max_candidates_per_line: Optional[int] = None,
    corr_eps: Optional[float] = None,   # <— NEU
) -> List[Dict]:
    """
    Erzeugt Kandidatenliste für Linie ℓ in Szenario s.

    Rückgabe-Einträge enthalten:
      {
        "arcs": [a0,a1,...],
        "len": float,
        "add_len": float,
        "rem_len": float,
        "delta_len_nom": float,  # = add_len + rem_len
        "kind": "nominal"|"base"|"detour"|"ksp"|"alt",
        "is_nominal": bool,
        "is_base": bool,
      }
    """
    allowed = allowed_arcs_forward(model, s)
    src, dst, nominal = _line_endpoints(model, ell)
    nominal_ok = all(a in allowed for a in nominal)
    nominal_set = set(nominal)

    # Gewichtsfunktion für KSP/Detour-Suche
    if ksp_weight_mode.lower() == "composite":
        gamma = float(w_repl if gamma_ksp is None else gamma_ksp)
        def weight_fn(a: int) -> float:
            L = _arc_length(model, a)
            pen = gamma if a not in nominal_set else 0.0
            return w_len * L + pen * L
    else:
        def weight_fn(a: int) -> float:
            return w_len * _arc_length(model, a)

    # gerichtete, gewichtete Sicht
    adj = _adj_directed_weighted(model, allowed, rev, weight_fn)

    # Referenzpfade
    base = _shortest_path(model, adj, src, dst)
    if not base:
        return []

    seed = nominal if nominal_ok else base

    # Falls Alternativen nur bei Blockade:
    if only_when_blocked and nominal_ok:
        pool = [nominal]
    else:
        pool: List[List[int]] = [seed]
        if detour_count > 0:
            pool += _detour_candidates(model, adj, seed, detour_count)
        if ksp_count > 0:
            pool += [p for p in _yen_ksp(model, adj, src, dst, ksp_count) if p and p != seed]
        pool = _unique_paths(pool)
        # nominal nach vorne heben, wenn zulässig
        if nominal_ok:
            for i, p in enumerate(pool):
                if p == nominal and i != 0:
                    pool.insert(0, pool.pop(i))
                    break
    def plen(p): return sum(_arc_length(model, a) for a in p)

    if nominal_ok:
        ref_path = nominal
    else:
        ref_path = base if base else []
    L_ref = plen(ref_path) if ref_path else float("inf")

    def within_corr(p: List[int]) -> bool:
        if corr_eps is None:
            return True
        return plen(p) <= (1.0 + float(corr_eps)) * L_ref
    

    # Diversität und Max-Limit anwenden
    filtered: List[List[int]] = []
    for p in pool:
        if not within_corr(p):
            continue
        if _edge_diversity_ok(p, filtered, min_edge_diff):
            filtered.append(p)
        if max_candidates_per_line is not None and len(filtered) >= int(max_candidates_per_line):
            break
    # Labels vorbereiten
    detour_set = set(map(tuple, _detour_candidates(model, adj, seed, max(0, detour_count)))) if detour_count > 0 else set()
    ksp_set    = set(map(tuple, _yen_ksp(model, adj, src, dst, max(0, ksp_count)))) if ksp_count > 0 else set()

    def plen(p): return sum(_arc_length(model, a) for a in p)

    # Kosten ggü. NOMINAL (immer)
    ref_set = nominal_set
    out: List[Dict] = []
    for p in filtered:
        pset = set(p)
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

    return out


def _mirror_candidates_for_line(
    model,
    s: int,
    cand_list_fwd: List[Dict],
    rev: List[Optional[int]],
    ell_bwd: int,
) -> List[Dict]:
    """
    Spiegelt Kandidaten einer Vorwärts-Linie auf die Rückwärts-Linie:
    - Arc-Sequenz wird rückwärts gelesen und jeder Arc via rev[a] invertiert.
    - Kosten (add/rem/delta) werden ggü. dem NOMINAL der Rückwärtslinie berechnet.
    - Falls ein rev[a] fehlt oder Arc im Szenario gesperrt ist, wird der betreffende Kandidat verworfen.
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
            "is_base": False,  # 'base' nur wenn man auch die base-Berechnung gespiegelt prüft
        })
    return mirrored


# --------------------------- Public Entry Point ---------------------------

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
    corr_eps: Optional[float] = None,   # <— NEU
) -> Dict[int, Dict[int, List[Dict]]]:
    """
    Erzeugt candidates[s][ell] = Liste von Kandidaten für Linie ell im Szenario s.

    Parameter:
      - ksp_weight_mode: "length" (nur Länge) oder "composite" (Länge + Replan-Penalty)
      - w_len, w_repl, gamma_ksp: Gewichte für die KSP-Gewichtsfunktion
      - only_when_blocked: nur Alternativen erzeugen, wenn nominal im Szenario verletzt
      - min_edge_diff: Diversitäts-Schwelle (# unterschiedlicher Kanten)
      - max_candidates_per_line: Obergrenze je Linie
      - mirror_backward: Kandidaten der Vorwärtslinie auf Rückwärtslinie spiegeln
    """
    rev = _rev_map(model)
    S   = len(model.p_s)
    results: Dict[int, Dict[int, List[Dict]]] = {}

    # Optional: Gruppensicht für mirror_backward
    groups = getattr(model, "line_group_to_lines", None)

    for s in range(S):
        per_line: Dict[int, List[Dict]] = {}

        if mirror_backward and isinstance(groups, dict):
            # Erzeuge pro Gruppe: erst FWD, dann spiegeln auf BWD (falls vorhanden)
            for g, pair in groups.items():
                ell_fwd, ell_bwd = pair

                # FWD, wenn vorhanden
                if ell_fwd is not None and ell_fwd >= 0:
                    cand_fwd = candidates_for_line_scenario(
                        model, s, int(ell_fwd),
                        detour_count, ksp_count, rev,
                        ksp_weight_mode=ksp_weight_mode,
                        w_len=w_len, w_repl=w_repl, gamma_ksp=gamma_ksp,
                        only_when_blocked=only_when_blocked,
                        min_edge_diff=min_edge_diff,
                        max_candidates_per_line=max_candidates_per_line,
                        corr_eps=corr_eps,    
                    )
                    per_line[int(ell_fwd)] = cand_fwd
                else:
                    cand_fwd = []

                # BWD: spiegeln, wenn möglich; sonst normale Generierung
                if ell_bwd is not None and ell_bwd >= 0:
                    if cand_fwd:
                        cand_bwd_mirror = _mirror_candidates_for_line(model, s, cand_fwd, rev, int(ell_bwd))
                        if cand_bwd_mirror:
                            per_line[int(ell_bwd)] = cand_bwd_mirror
                            continue  # erfolgreich gespiegelt
                    # Fallback: normale Generierung
                    per_line[int(ell_bwd)] = candidates_for_line_scenario(
                        model, s, int(ell_bwd),
                        detour_count, ksp_count, rev,
                        ksp_weight_mode=ksp_weight_mode,
                        w_len=w_len, w_repl=w_repl, gamma_ksp=gamma_ksp,
                        only_when_blocked=only_when_blocked,
                        min_edge_diff=min_edge_diff,
                        max_candidates_per_line=max_candidates_per_line,
                        corr_eps=corr_eps,    
                    )
        else:
            # Unabhängig pro Linie
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
    Bequemer Wrapper: nimmt CandidateConfig + main_cfg (dict oder Config-Objekt),
    löst Gewichte auf und ruft die bestehende build_candidates_all_scenarios_per_line
    mit den richtigen, 'flachen' Parametern auf.
    """
    # main_cfg kann dict oder dataclass sein:
    def _get(cfg, key, default):
        try:
            return getattr(cfg, key)
        except Exception:
            return cfg.get(key, default)

    w_len  = cand_cfg.w_len  if cand_cfg.w_len  is not None else float(_get(main_cfg, "line_operation_cost_mult", 1.0))
    w_repl = cand_cfg.w_repl if cand_cfg.w_repl is not None else float(_get(main_cfg, "cost_repl_line", 0.0))

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
        mirror_backward=(str(cand_cfg.mirror_backward).lower() != "off"),
        corr_eps=float(cand_cfg.corr_eps),
    )
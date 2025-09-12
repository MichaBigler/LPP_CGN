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
    tails, heads = _arc_endpoints(model)
    adj: Dict[int, List[Tuple[int, int, Optional[int], float]]] = {}

    for a in allowed:
        u, v = tails[a], heads[a]

        # u -> v (nur weil a erlaubt ist)
        w_uv = float(weight_fn(a))
        ab = rev[a] if (rev[a] is not None and rev[a] in allowed) else None
        adj.setdefault(u, []).append((v, a, ab, w_uv))

        # v -> u NUR wenn echte Gegenrichtung existiert und ebenfalls erlaubt
        if ab is not None:
            w_vu = float(weight_fn(ab))
            adj.setdefault(v, []).append((u, ab, a, w_vu))

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

def _mk_nominal_candidate(model, ell):
    arcs_nom = list(map(int, model.line_idx_to_arcs[ell]))
    L_nom = sum(float(model.len_a[a]) for a in arcs_nom)
    return {
        "arcs": arcs_nom,
        "len": L_nom,
        "add_len": 0.0, "rem_len": 0.0, "delta_len_nom": 0.0,
        "kind": "nominal", "is_nominal": True, "is_base": False
    }

def _nominal_ok(model, s, ell, allowed=None):
    if allowed is None:
        allowed = allowed_arcs_forward(model, s)
    return all(int(a) in allowed for a in model.line_idx_to_arcs[ell])

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

def _uniq_keep_order(cands):
    seen = set()
    out = []
    for c in cands:
        key = tuple(int(a) for a in c.get("arcs", []))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


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
    ksp_weight_mode: str = "length",
    w_len: float = 1.0,
    w_repl: float = 0.0,
    gamma_ksp: Optional[float] = None,
    only_when_blocked: bool = False,
    min_edge_diff: int = 1,
    max_candidates_per_line: Optional[int] = None,
    corr_eps: Optional[float] = None,
) -> List[Dict]:
    

    

    allowed = allowed_arcs_forward(model, s)
    src, dst, nominal = _line_endpoints(model, ell)
    nominal_ok = all(a in allowed for a in nominal)
    bad = [int(a) for a in nominal if a not in allowed]


    nominal_set = set(nominal)

    # Gewichtsfunktion
    if ksp_weight_mode.lower() == "composite":
        gamma = float(w_repl if gamma_ksp is None else gamma_ksp)
        def weight_fn(a: int) -> float:
            L = _arc_length(model, a)
            return w_len * L + (gamma if a not in nominal_set else 0.0) * L
    else:
        def weight_fn(a: int) -> float:
            return w_len * _arc_length(model, a)

    # gerichtete Sicht
    adj = _adj_directed_weighted(model, allowed, rev, weight_fn)
    edge_cnt = sum(len(nbrs) for nbrs in adj.values())
    

    # Referenzen
    base = _shortest_path(model, adj, src, dst)
    if not base:
     
        if nominal_ok and nominal:
            
            return [{
                "arcs": nominal,
                "len": sum(_arc_length(model,a) for a in nominal),
                "add_len": 0.0, "rem_len": 0.0, "delta_len_nom": 0.0,
                "kind": "nominal", "is_nominal": True, "is_base": False
            }]
        return []

    seed = nominal if nominal_ok else base

    # --- Pool aufbauen (nominal immer rein, wenn zulässig) ---
    pool: List[List[int]] = []
    if nominal_ok and nominal:
        pool.append(nominal)

    # base IMMER hinzufügen, wenn existent und verschieden
    if base and (not nominal_ok or base != nominal):
        pool.append(base)

    # Alternativen nur dann unterdrücken, wenn ausdrücklich gewünscht
    if not (only_when_blocked and nominal_ok):
        if detour_count > 0:
            dets = _detour_candidates(model, adj, seed, detour_count)
            
            pool += dets
        if ksp_count > 0:
            ksps = [p for p in _yen_ksp(model, adj, src, dst, ksp_count) if p and p != seed]
            
            pool += ksps

    before = len(pool)
    pool = _unique_paths(pool)
    

    if not pool:
        pool = [base]

    if nominal_ok:
        for i, p in enumerate(pool):
            if p == nominal and i != 0:
                pool.insert(0, pool.pop(i))
                break

    

    def plen(p): return sum(_arc_length(model, a) for a in p)

    ref_path = nominal if nominal_ok else (base if base else [])
    L_ref = plen(ref_path) if ref_path else float("inf")

    def within_corr(p: List[int]) -> bool:
        return True if corr_eps is None else (plen(p) <= (1.0 + float(corr_eps)) * L_ref)

    # --- EINZIGE Filterrunde: Korridor + Diversität + Limit ---
    filtered: List[List[int]] = []
    for p in pool:
        if within_corr(p) and _edge_diversity_ok(p, filtered, min_edge_diff):
            filtered.append(p)


    if not filtered:
        filtered = [nominal] if nominal_ok else [base]

    

    # Label-Sets (nur wenn Alternativen gesucht wurden)
    detour_set = set()
    ksp_set = set()
    if not (only_when_blocked and nominal_ok):
        if detour_count > 0:
            detour_set = set(map(tuple, _detour_candidates(model, adj, seed, max(0, detour_count))))
        if ksp_count > 0:
            ksp_set = set(map(tuple, _yen_ksp(model, adj, src, dst, max(0, ksp_count))))

    # Ausgabestruktur
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
    corr_eps: Optional[float] = None,
) -> Dict[int, Dict[int, List[Dict]]]:
    """
    Erzeugt candidates[s][ell] = Liste von Kandidaten für Linie ell im Szenario s.

    Parameter:
      - ksp_weight_mode: "length" (nur Länge) oder "composite" (Länge + Replan-Penalty)
      - w_len, w_repl, gamma_ksp: Gewichte für die KSP-Gewichtsfunktion
      - only_when_blocked: nur Alternativen erzeugen, wenn nominal im Szenario verletzt
      - min_edge_diff: Diversitäts-Schwelle (# unterschiedlicher Kanten)
      - max_candidates_per_line: Obergrenze je Linie
      - mirror_backward: Kandidaten der Vorwärtslinie auf Rückwärtslinie spiegeln und
                         den nominalen BWD (falls zulässig) zusätzlich behalten
      - corr_eps: Korridor (max. relative Längenzunahme ggü. Referenz)
    """
    rev = _rev_map(model)
    S   = len(model.p_s)
    results: Dict[int, Dict[int, List[Dict]]] = {}

    def _uniq_keep_order(cands: List[Dict]) -> List[Dict]:
        seen = set()
        out: List[Dict] = []
        for c in cands:
            t = tuple(int(a) for a in c.get("arcs", []))
            if t in seen:
                continue
            seen.add(t)
            out.append(c)
        return out

    groups = getattr(model, "line_group_to_lines", None)

    for s in range(S):
        per_line: Dict[int, List[Dict]] = {}

        if mirror_backward and isinstance(groups, dict):
            for g, pair in groups.items():
                ell_fwd, ell_bwd = pair

                allowed = allowed_arcs_forward(model, s)

                # 1) Beide Richtungen unabhängig generieren
                cand_fwd_gen = []
                cand_bwd_gen = []
                if ell_fwd is not None and ell_fwd >= 0:
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
                if ell_bwd is not None and ell_bwd >= 0:
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

                # 2) Beidseitig spiegeln
                cand_fwd_mir = []
                cand_bwd_mir = []
                if ell_fwd is not None and ell_fwd >= 0 and cand_bwd_gen:
                    cand_fwd_mir = _mirror_candidates_for_line(model, s, cand_bwd_gen, rev, int(ell_fwd))
                if ell_bwd is not None and ell_bwd >= 0 and cand_fwd_gen:
                    cand_bwd_mir = _mirror_candidates_for_line(model, s, cand_fwd_gen, rev, int(ell_bwd))

                # 3) Pro Richtung: nominal immer beibehalten, union + dedup + limit
                if ell_fwd is not None and ell_fwd >= 0:
                    out_fwd = []
                    if _nominal_ok(model, s, int(ell_fwd), allowed):
                        out_fwd.append(_mk_nominal_candidate(model, int(ell_fwd)))
                    out_fwd += cand_fwd_mir + cand_fwd_gen
                    out_fwd = _uniq_keep_order(out_fwd)
                    if max_candidates_per_line is not None and len(out_fwd) > int(max_candidates_per_line):
                        out_fwd = out_fwd[:int(max_candidates_per_line)]
                    per_line[int(ell_fwd)] = out_fwd

                if ell_bwd is not None and ell_bwd >= 0:
                    out_bwd = []
                    if _nominal_ok(model, s, int(ell_bwd), allowed):
                        out_bwd.append(_mk_nominal_candidate(model, int(ell_bwd)))
                    out_bwd += cand_bwd_mir + cand_bwd_gen
                    out_bwd = _uniq_keep_order(out_bwd)
                    if max_candidates_per_line is not None and len(out_bwd) > int(max_candidates_per_line):
                        out_bwd = out_bwd[:int(max_candidates_per_line)]
                    per_line[int(ell_bwd)] = out_bwd
        else:
            # Unabhängig je Linie
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
        total = sum(len(v) for v in per_line.values())
        empties = [ell for ell, L in per_line.items() if not L]
        print(f"[cands] s={s}: total_cands={total}, empty_lines={len(empties)} -> {empties[:10]}")

    return results

def build_candidates_all_scenarios_per_line_cfg(model, cand_cfg, main_cfg):
    def _get(cfg, key, default):
        try:
            return getattr(cfg, key)
        except Exception:
            return cfg.get(key, default)

    w_len  = cand_cfg.w_len  if cand_cfg.w_len  is not None else float(_get(main_cfg, "line_operation_cost_mult", 1.0))
    w_repl = cand_cfg.w_repl if cand_cfg.w_repl is not None else float(_get(main_cfg, "cost_repl_line", 0.0))

    corr = None if getattr(cand_cfg, "corr_eps", None) is None else float(cand_cfg.corr_eps)

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
        corr_eps=corr,   # <-- robust
    )
# cgn_candidates.py (oder in cgn.py neben make_cgn)
from typing import Dict, List, Tuple
from cgn import CGN

def make_cgn_with_candidates_per_line(model, cand_lines_s: Dict[int, List[dict]]) -> CGN:
    """
    Erzeuge ein CGN aus per-Linie-Kandidaten.
    cand_lines_s[ell] = Liste von Kandidaten-Dicts mit Schlüssel "arcs" (gerichtete infra-Arc-IDs).
    WICHTIG:
      - Keine automatische Spiegelung/Verknüpfung von Vor-/Rückrichtung.
      - Jede Linie (Richtung) verwendet nur ihre eigenen Kandidaten.
    """

    # --- Tail/Head je Infra-Arc im INDEXraum 0..N-1 ---
    nid2idx = model.node_id_to_idx
    tails: List[int] = []
    heads: List[int] = []
    for (u_id, v_id) in model.idx_to_arc_uv:
        tails.append(int(nid2idx[u_id]))
        heads.append(int(nid2idx[v_id]))

    # --- Ground Nodes ---
    cgn_nodes: List[Tuple[int, int, int]] = []     # (phys_node, line, variant); (-1,-1) = ground
    node_id: Dict[Tuple[int, int, int], int] = {}
    ground_of = [-1] * model.N
    for i in range(model.N):
        key = (i, -1, -1)
        node_id[key] = len(cgn_nodes)
        ground_of[i] = node_id[key]
        cgn_nodes.append(key)

    def get_node(i: int, ell: int, k: int) -> int:
        key = (i, ell, k)
        idx = node_id.get(key)
        if idx is None:
            idx = len(cgn_nodes)
            node_id[key] = idx
            cgn_nodes.append(key)
        return idx

    # Merker: welche (ell,k) liegen an physischem Knoten i (für change/board/alight)
    lines_at_node: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(model.N)}

    # --- Ride-Arcs je (ell,k) Kandidat ---
    arc_tail: List[int] = []; arc_head: List[int] = []
    arc_kind: List[str] = []; arc_line: List[int] = []
    arc_edge: List[int] = []; arc_variant: List[int] = []
    arc_line_to: List[int] = []

    for ell in range(model.L):
        cand_list = cand_lines_s.get(ell, []) or []
        for k, cand in enumerate(cand_list):
            arcs = [int(a) for a in cand.get("arcs", [])]
            if not arcs:
                continue

            # Knotenfolge entlang des Kandidaten
            u = tails[arcs[0]]
            path_nodes = [u]
            for a in arcs:
                path_nodes.append(heads[a])

            # Layer-Knoten registrieren
            for i in path_nodes:
                if (ell, k) not in lines_at_node[i]:
                    lines_at_node[i].append((ell, k))
                # get_node erzeugt bei Bedarf
                get_node(i, ell, k)

            # Ride-Arcs erzeugen
            u = tails[arcs[0]]
            for a in arcs:
                v = heads[a]
                arc_tail.append(get_node(u, ell, k))
                arc_head.append(get_node(v, ell, k))
                arc_kind.append("ride")
                arc_line.append(ell)      # Linie dieses Layers
                arc_edge.append(a)        # Infra-Arc-ID
                arc_variant.append(k)
                arc_line_to.append(-1)    # für ride nicht benutzt
                u = v

    # --- Change-Arcs zwischen allen (ell1,k1)->(ell2,k2) am selben phys. Knoten ---
    for i in range(model.N):
        L_i = lines_at_node[i]
        for (ell1, k1) in L_i:
            v_from = get_node(i, ell1, k1)
            for (ell2, k2) in L_i:
                if ell1 == ell2 and k1 == k2:
                    continue
                v_to = get_node(i, ell2, k2)
                arc_tail.append(v_from)
                arc_head.append(v_to)
                arc_kind.append("change")
                arc_line.append(ell1)     # Quelle (nur Diagnose)
                arc_edge.append(-1)
                arc_variant.append(k1)
                arc_line_to.append(ell2)  # Ziel-Linie (für Waiting-Frequenz)

    # --- Board/Alight an allen (i,ell,k) ---
    for i in range(model.N):
        v_ground = ground_of[i]
        for (ell, k) in lines_at_node[i]:
            v_line = get_node(i, ell, k)
            # board
            arc_tail.append(v_ground); arc_head.append(v_line)
            arc_kind.append("board");  arc_line.append(ell)
            arc_edge.append(-1);       arc_variant.append(k); arc_line_to.append(ell)
            # alight
            arc_tail.append(v_line);   arc_head.append(v_ground)
            arc_kind.append("alight"); arc_line.append(ell)
            arc_edge.append(-1);       arc_variant.append(k); arc_line_to.append(-1)

    V = len(cgn_nodes)
    A = len(arc_tail)
    in_arcs  = [[] for _ in range(V)]
    out_arcs = [[] for _ in range(V)]
    for a, (t, h) in enumerate(zip(arc_tail, arc_head)):
        out_arcs[t].append(a); in_arcs[h].append(a)

    node_line    = [ln for (_, ln, _) in cgn_nodes]
    node_variant = [kv for (_, _, kv) in cgn_nodes]

    return CGN(
        V, A, in_arcs, out_arcs, ground_of,
        arc_tail, arc_head, arc_kind, arc_line, arc_edge, arc_variant,
        node_line, node_variant, arc_line_to
    )

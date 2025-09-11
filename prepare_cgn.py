# cgn.py
from dataclasses import dataclass
from typing import List, Tuple, Dict

from cgn import CGN


def make_cgn(data) -> CGN:
    """Build CGN (ohne Kandidaten, Variante=0), inkl. Felder für waiting-time."""
    GROUND = -1

    # --- Knoten anlegen ---
    cgn_nodes: List[Tuple[int, int]] = []        # (phys_node, line) mit line=-1 für Ground
    node_id: Dict[Tuple[int, int], int] = {}

    ground_of: List[int] = [-1] * data.N
    # Ground-Knoten
    for i in range(data.N):
        key = (i, GROUND)
        node_id[key] = len(cgn_nodes)
        ground_of[i] = node_id[key]
        cgn_nodes.append(key)

    # Linien-Knoten: nur dort, wo Linie i bedient
    lines_at_node: Dict[int, List[int]] = {i: [] for i in range(data.N)}
    for ell in range(data.L):
        for i in data.line_idx_to_stops[ell]:
            key = (i, ell)
            if key not in node_id:
                node_id[key] = len(cgn_nodes)
                cgn_nodes.append(key)
            lines_at_node[i].append(ell)

    V = len(cgn_nodes)

    # --- Node-Attribute ---
    node_line: List[int] = [ell for (_i, ell) in cgn_nodes]
    node_variant: List[int] = [(-1 if ell == GROUND else 0) for ell in node_line]

    # --- Arcs anlegen ---
    arc_tail: List[int] = []; arc_head: List[int] = []
    arc_kind: List[str] = []; arc_line: List[int] = []
    arc_edge: List[int] = []; arc_variant: List[int] = []
    arc_line_to: List[int] = []

    # 1) ride arcs entlang jeder Linie
    for ell in range(data.L):
        stops = data.line_idx_to_stops[ell]
        arcs  = data.line_idx_to_arcs[ell]  # gleiche Länge wie stops-1
        for p, a in enumerate(arcs):
            u, v = stops[p], stops[p + 1]
            tail = node_id[(u, ell)]
            head = node_id[(v, ell)]
            arc_tail.append(tail);           arc_head.append(head)
            arc_kind.append("ride");         arc_line.append(ell)
            arc_edge.append(int(a));         arc_variant.append(0)
            arc_line_to.append(-1)           # ride erzeugt keine Wartezeit

    # 2) change arcs: i^l1 -> i^l2 (l1 != l2)
    for i in range(data.N):
        L_i = sorted(set(lines_at_node[i]))
        for l1 in L_i:
            for l2 in L_i:
                if l1 == l2: 
                    continue
                tail = node_id[(i, l1)]
                head = node_id[(i, l2)]
                arc_tail.append(tail);        arc_head.append(head)
                arc_kind.append("change");    arc_line.append(l1)     # Quelle (nur Diagnose)
                arc_edge.append(-1);          arc_variant.append(0)
                arc_line_to.append(l2)        # **Ziel-Linie für Waiting!**

    # 3) board/alight an jedem bedienten Knoten
    for i in range(data.N):
        for ell in sorted(set(lines_at_node[i])):
            v_line = node_id[(i, ell)]
            v_grnd = ground_of[i]
            # board: ground -> line
            arc_tail.append(v_grnd);          arc_head.append(v_line)
            arc_kind.append("board");         arc_line.append(ell)    # (nur Diagnose)
            arc_edge.append(-1);              arc_variant.append(0)
            arc_line_to.append(ell)           # **Ziel-Linie für Waiting!**
            # alight: line -> ground
            arc_tail.append(v_line);          arc_head.append(v_grnd)
            arc_kind.append("alight");        arc_line.append(ell)
            arc_edge.append(-1);              arc_variant.append(0)
            arc_line_to.append(-1)            # keine Waiting-Kopplung

    A = len(arc_tail)

    in_arcs  = [[] for _ in range(V)]
    out_arcs = [[] for _ in range(V)]
    for a, (t, h) in enumerate(zip(arc_tail, arc_head)):
        out_arcs[t].append(a)
        in_arcs[h].append(a)

    # *** WICHTIG: Rückgabe-Reihenfolge exakt wie im Dataclass! ***
    return CGN(
        V, A,
        in_arcs, out_arcs,
        ground_of,
        arc_tail, arc_head, arc_kind, arc_line, arc_edge, arc_variant,
        node_line, node_variant, arc_line_to
    )

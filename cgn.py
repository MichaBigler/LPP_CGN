# cgn.py
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class CGN:
    V: int; A: int
    in_arcs: List[List[int]]; out_arcs: List[List[int]]
    ground_of: List[int]
    arc_tail: List[int]; arc_head: List[int]
    arc_kind: List[str]      # "ride" | "change" | "board" | "alight"
    arc_line: List[int]      # line id for ride arcs, else -1
    arc_line_to: List[int]   # target line id for change/board, else -1
    arc_edge: List[int]      # directed infra arc index for ride arcs, else -1

def make_cgn(data) -> CGN:
    """Build CGN nodes and arcs (ground, per-line), with minimal metadata."""
    GROUND = -1
    cgn_nodes: List[Tuple[int,int]] = []
    cgn_id: Dict[Tuple[int,int], int] = {}

    # ground nodes for all physical nodes
    ground_of = [None] * data.N
    for i in range(data.N):
        key = (i, GROUND)
        cgn_id[key] = len(cgn_nodes)
        ground_of[i] = cgn_id[key]
        cgn_nodes.append(key)

    # line-layer nodes only where a line visits the node; also collect lines per node
    lines_at_node: Dict[int, List[int]] = {i: [] for i in range(data.N)}
    for ell in range(data.L):
        for i in data.line_idx_to_stops[ell]:
            key = (i, ell)
            if key not in cgn_id:
                cgn_id[key] = len(cgn_nodes)
                cgn_nodes.append(key)
            lines_at_node[i].append(ell)

    V = len(cgn_nodes)

    arc_tail: List[int] = []; arc_head: List[int] = []
    arc_kind: List[str] = []; arc_line: List[int] = []
    arc_line_to: List[int] = []; arc_edge: List[int] = []

    # ride arcs along each line chain (need arc_edge for time/length)
    for ell in range(data.L):
        stops = data.line_idx_to_stops[ell]
        arcs  = data.line_idx_to_arcs[ell]  # aligned with stops
        for p, a in enumerate(arcs):
            u, v = stops[p], stops[p+1]
            arc_tail.append(cgn_id[(u, ell)]); arc_head.append(cgn_id[(v, ell)])
            arc_kind.append("ride"); arc_line.append(ell); arc_line_to.append(-1); arc_edge.append(a)

    # change arcs: i^l1 -> i^l2 for all l1 != l2 that serve i
    for i in range(data.N):
        L_i = sorted(set(lines_at_node[i]))
        for l1 in L_i:
            for l2 in L_i:
                if l1 == l2: continue
                arc_tail.append(cgn_id[(i, l1)]); arc_head.append(cgn_id[(i, l2)])
                arc_kind.append("change"); arc_line.append(-1); arc_line_to.append(l2); arc_edge.append(-1)

    # board/alight arcs at each served node
    for i in range(data.N):
        for ell in sorted(set(lines_at_node[i])):
            # board: ground -> line
            arc_tail.append(ground_of[i]); arc_head.append(cgn_id[(i, ell)])
            arc_kind.append("board"); arc_line.append(-1); arc_line_to.append(ell); arc_edge.append(-1)
            # alight: line -> ground
            arc_tail.append(cgn_id[(i, ell)]); arc_head.append(ground_of[i])
            arc_kind.append("alight"); arc_line.append(-1); arc_line_to.append(-1); arc_edge.append(-1)

    A = len(arc_tail)
    in_arcs  = [[] for _ in range(V)]
    out_arcs = [[] for _ in range(V)]
    for a, (t, h) in enumerate(zip(arc_tail, arc_head)):
        out_arcs[t].append(a); in_arcs[h].append(a)

    return CGN(V, A, in_arcs, out_arcs, ground_of, arc_tail, arc_head, arc_kind, arc_line, arc_line_to, arc_edge)

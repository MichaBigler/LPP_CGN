# cgn.py  — NEU
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class CGN:
    V: int; A: int
    in_arcs: List[List[int]]; out_arcs: List[List[int]]
    ground_of: List[int]

    arc_tail: List[int]; arc_head: List[int]
    arc_kind: List[str]            # "ride" | "change" | "board" | "alight"
    arc_line: List[int]            # line id of the arc's layer (ride: that line; board/change: source line; -1 if N/A)
    arc_edge: List[int]            # infra arc id (only for ride, else -1)
    arc_variant: List[int]         # candidate index per layer arc; -1 if none

    # NEW:
    node_line: List[int]           # per CGN node: line id at node (-1 for ground)
    node_variant: List[int]        # per CGN node: candidate index (-1 for ground)
    arc_line_to: List[int]         # target line for waiting: board/change -> target line; else -1

def make_cgn_with_candidates(model, candidates_s: Dict[int, List[Dict]]) -> CGN:
    """
    Build a CGN with all (line, variant) layers per group using the candidate paths.
    candidates_s[g] = list of {"arcs":[...], "len":..., ...} with directed infra arc ids.
    Both directions of a group get proper forward/backward sequences.
    """

    # --- derive tail/head in index space + reverse map ---
    nid2idx = model.node_id_to_idx
    tails: List[int] = []
    heads: List[int] = []
    by_uv: Dict[Tuple[int,int], int] = {}
    for a, (u_id, v_id) in enumerate(model.idx_to_arc_uv):
        u, v = nid2idx[u_id], nid2idx[v_id]
        tails.append(u); heads.append(v)
        by_uv[(u, v)] = a
    rev = [ by_uv.get((v, u)) for (u, v) in zip(tails, heads) ]

    # --- ground nodes ---
    cgn_nodes: List[Tuple[int,int,int]] = []   # (phys_node, line, variant)
    node_id: Dict[Tuple[int,int,int], int] = {}

    ground_of = [-1] * model.N
    for i in range(model.N):
        key = (i, -1, -1)
        node_id[key] = len(cgn_nodes)
        ground_of[i] = node_id[key]
        cgn_nodes.append(key)

    # will fill after nodes created
    node_line: List[int] = []       # aligns with cgn_nodes
    node_variant: List[int] = []

    # helper to ensure (i,ell,k) node exists
    def get_node(i: int, ell: int, k: int) -> int:
        key = (i, ell, k)
        idx = node_id.get(key)
        if idx is None:
            idx = len(cgn_nodes)
            node_id[key] = idx
            cgn_nodes.append(key)
        return idx

    # collect which (ell,k) touches which physical node: for change arcs
    lines_at_node: Dict[int, List[Tuple[int,int]]] = {i: [] for i in range(model.N)}

    # --- register all (ell,k) layer nodes touched by candidates, for both fwd/bwd lines ---
    for g, (ell_fwd, ell_bwd) in model.line_group_to_lines.items():
        lines_g = [ell for ell in (ell_fwd, ell_bwd) if ell is not None and ell >= 0]
        cand_list = candidates_s.get(g, [])
        if not cand_list:
            continue

        for k, cand in enumerate(cand_list):
            arcs = [int(a) for a in cand.get("arcs", [])]
            if not arcs:
                continue

            # forward path node sequence:
            path_nodes_fwd: List[int] = [tails[arcs[0]]]
            for a in arcs:
                path_nodes_fwd.append(heads[a])

            # backward path arc sequence = reverse of fwd using reverse arcs
            arcs_bwd: List[int] = []
            for a in reversed(arcs):
                ra = rev[a]
                if ra is None:
                    # if reverse missing, we simply skip the backward line for this candidate
                    arcs_bwd = []
                    break
                arcs_bwd.append(ra)
            path_nodes_bwd: List[int] = []
            if arcs_bwd:
                path_nodes_bwd = [tails[arcs_bwd[0]]]
                for a in arcs_bwd:
                    path_nodes_bwd.append(heads[a])

            # create layer nodes for each line in group (forward/backward use respective sequences)
            for ell in lines_g:
                # decide direction: ell == ell_fwd -> fwd, ell == ell_bwd -> bwd
                if ell == ell_fwd:
                    used_nodes = path_nodes_fwd
                else:
                    used_nodes = path_nodes_bwd if path_nodes_bwd else path_nodes_fwd  # fallback if reverse missing

                for i in used_nodes:
                    v = get_node(i, ell, k)
                    if (ell, k) not in lines_at_node[i]:
                        lines_at_node[i].append((ell, k))

    V = len(cgn_nodes)
    # fill node_line / node_variant
    node_line = [ln for (_, ln, _) in cgn_nodes]
    node_variant = [kv for (_, _, kv) in cgn_nodes]

    # --- build arcs ---
    arc_tail: List[int] = []; arc_head: List[int] = []
    arc_kind: List[str] = []; arc_line: List[int] = []
    arc_edge: List[int] = []; arc_variant: List[int] = []
    arc_line_to: List[int] = []

    # 1) ride arcs for each (ell,k) path
    for g, (ell_fwd, ell_bwd) in model.line_group_to_lines.items():
        cand_list = candidates_s.get(g, [])
        if not cand_list:
            continue

        for k, cand in enumerate(cand_list):
            arcs_fwd = [int(a) for a in cand.get("arcs", [])]
            if not arcs_fwd:
                continue

            # build per line:
            # forward line:
            if ell_fwd is not None and ell_fwd >= 0:
                u = tails[arcs_fwd[0]]
                for a in arcs_fwd:
                    v = heads[a]
                    arc_tail.append(node_id[(u, ell_fwd, k)])
                    arc_head.append(node_id[(v, ell_fwd, k)])
                    arc_kind.append("ride")
                    arc_line.append(ell_fwd)     # source line layer
                    arc_edge.append(a)           # infra arc id
                    arc_variant.append(k)
                    arc_line_to.append(-1)       # not used for ride
                    u = v

            # backward line (if reverse exists)
            if ell_bwd is not None and ell_bwd >= 0:
                arcs_bwd: List[int] = []
                ok = True
                for a in reversed(arcs_fwd):
                    ra = rev[a]
                    if ra is None:
                        ok = False
                        break
                    arcs_bwd.append(ra)
                if ok and arcs_bwd:
                    u = tails[arcs_bwd[0]]
                    for a in arcs_bwd:
                        v = heads[a]
                        arc_tail.append(node_id[(u, ell_bwd, k)])
                        arc_head.append(node_id[(v, ell_bwd, k)])
                        arc_kind.append("ride")
                        arc_line.append(ell_bwd)
                        arc_edge.append(a)
                        arc_variant.append(k)
                        arc_line_to.append(-1)
                        u = v

    # 2) change arcs: between any (ell1,k1) and (ell2,k2) at same physical node
    for i in range(model.N):
        L_i = lines_at_node[i]
        for (ell1, k1) in L_i:
            v_from = node_id[(i, ell1, k1)]
            for (ell2, k2) in L_i:
                if ell1 == ell2 and k1 == k2:
                    continue
                v_to = node_id[(i, ell2, k2)]
                arc_tail.append(v_from)
                arc_head.append(v_to)
                arc_kind.append("change")
                arc_line.append(ell1)       # source line (not used by waiting)
                arc_edge.append(-1)
                arc_variant.append(k1)
                arc_line_to.append(ell2)    # **target line for waiting**

    # 3) board / alight arcs at all layer nodes
    for i in range(model.N):
        for (ell, k) in lines_at_node[i]:
            v_line = node_id[(i, ell, k)]
            v_ground = ground_of[i]
            # board
            arc_tail.append(v_ground); arc_head.append(v_line)
            arc_kind.append("board")
            arc_line.append(ell)           # source is ground; but keep line=ell for diagnostics
            arc_edge.append(-1)
            arc_variant.append(k)
            arc_line_to.append(ell)        # **target line = the one we board onto**
            # alight
            arc_tail.append(v_line); arc_head.append(v_ground)
            arc_kind.append("alight")
            arc_line.append(ell)
            arc_edge.append(-1)
            arc_variant.append(k)
            arc_line_to.append(-1)         # not a waiting point

    A = len(arc_tail)
    in_arcs  = [[] for _ in range(V)]
    out_arcs = [[] for _ in range(V)]
    for a, (t, h) in enumerate(zip(arc_tail, arc_head)):
        out_arcs[t].append(a); in_arcs[h].append(a)

    return CGN(
        V, A, in_arcs, out_arcs, ground_of,
        arc_tail, arc_head, arc_kind, arc_line, arc_edge, arc_variant,
        node_line, node_variant, arc_line_to
    )
# debug_cgn.py
from collections import deque

def cgn_path_exists(cgn, data, o, d):
    """BFS on CGN with our rules: board only at origin, alight only at destination."""
    start = cgn.ground_of[o]
    goal  = cgn.ground_of[d]
    seen = [False]*cgn.V
    Q = deque([start]); seen[start] = True
    while Q:
        v = Q.popleft()
        if v == goal: 
            return True
        for a in cgn.out_arcs[v]:
            k = cgn.arc_kind[a]
            w = cgn.arc_head[a]
            # enforce OD-specific rules
            if k == "board" and v != cgn.ground_of[o]:
                continue
            if k == "alight" and w != cgn.ground_of[d]:
                continue
            if not seen[w]:
                seen[w] = True
                Q.append(w)
    return False

def precheck_all_od_connectivity(cgn, data, K, max_report=20):
    bad = []
    for (o, d) in K:
        if not cgn_path_exists(cgn, data, o, d):
            bad.append((o, d))
            if len(bad) >= max_report:
                break
    if bad:
        print("[precheck] Unreachable OD pairs in CGN (first hits):", bad)
    else:
        print("[precheck] All OD pairs have at least one CGN path.")

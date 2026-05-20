#!/usr/bin/env python3
# scripts/convert_mumford_to_lintim.py
# -*- coding: utf-8 -*-
"""
Convert a Mumford TNDP instance (nodes.txt, links.txt, demand.txt) into the
LinTim/LPP_CGN file layout:

  lintim/<NETWORK>/{Stop.giv, Edge.giv, OD.giv}
  Data/<NETWORK>/{lines.csv, scenario_infra.csv, scenario_prob.csv, properties_general.csv}

Since Mumford ships only the bare network + OD demand (no line pool), this
script also synthesises a sensible line pool by computing shortest paths
between the top-N OD pairs (by demand). The resulting pool is a starting
point — feel free to replace lines.csv with a published route set.

Usage:
    python scripts/convert_mumford_to_lintim.py \
        --src ../TransitNetworkDesign/Mumford/Mumford0 \
        --network Mumford0 \
        --num-lines 30 \
        --infra-cap-std 10

The output is written relative to the current working directory (assumed to
be the LPP_CGN repo root).
"""
from __future__ import annotations

import argparse
import csv
import heapq
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Mumford readers (comma-separated, lowercase header)
# ---------------------------------------------------------------------------

def _read_nodes(path: str) -> List[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _read_links(path: str) -> List[Tuple[int, int, float]]:
    out: List[Tuple[int, int, float]] = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out.append((int(r["from"]), int(r["to"]), float(r["travel_time"])))
    return out


def _read_demand(path: str) -> List[Tuple[int, int, float]]:
    out: List[Tuple[int, int, float]] = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out.append((int(r["from"]), int(r["to"]), float(r["demand"])))
    return out


# ---------------------------------------------------------------------------
# Graph helpers for line-pool synthesis
# ---------------------------------------------------------------------------

def _build_adjacency(links: List[Tuple[int, int, float]]) -> Dict[int, List[Tuple[int, float]]]:
    """Undirected adjacency: node -> [(neighbour, travel_time), …]."""
    adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    seen = set()
    for u, v, t in links:
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        adj[u].append((v, t))
        adj[v].append((u, t))
    return adj


def _dijkstra_path(adj, src: int, dst: int,
                   forbidden_edges: Optional[set] = None) -> Optional[List[int]]:
    """Shortest path src→dst by travel time. Returns node list or None.

    `forbidden_edges` is a set of frozensets({u, v}); those undirected edges
    are skipped during expansion (used for k-shortest path generation).
    """
    if src == dst:
        return [src]
    pq: List[Tuple[float, int]] = [(0.0, src)]
    prev: Dict[int, int] = {src: src}
    dist: Dict[int, float] = {src: 0.0}
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        if u == dst:
            break
        for v, w in adj.get(u, []):
            if forbidden_edges and frozenset({u, v}) in forbidden_edges:
                continue
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dst not in prev:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def _k_shortest_paths(adj, src: int, dst: int, k: int) -> List[List[int]]:
    """Yen-lite: greedy edge-removal k-shortest paths.

    Generates up to k pairwise edge-disjoint-ish paths. Not exact Yen's,
    but works fine for diverse line synthesis: each successive path is the
    shortest path under removal of one edge from the previous best.
    """
    if k < 1:
        return []
    paths: List[List[int]] = []
    seen_keys: set = set()
    first = _dijkstra_path(adj, src, dst)
    if first is None:
        return []
    paths.append(first)
    seen_keys.add(tuple(first))

    # Generate up to k-1 more by removing one edge at a time from earlier paths
    candidates: List[Tuple[float, List[int]]] = []
    for base in paths:
        if len(paths) >= k:
            break
        for i in range(len(base) - 1):
            forbid = {frozenset({base[i], base[i + 1]})}
            alt = _dijkstra_path(adj, src, dst, forbidden_edges=forbid)
            if alt is None or tuple(alt) in seen_keys:
                continue
            # Score by length
            length = sum(_edge_w(adj, alt[j], alt[j + 1]) for j in range(len(alt) - 1))
            candidates.append((length, alt))
    candidates.sort(key=lambda x: x[0])
    for _, p in candidates:
        if tuple(p) in seen_keys:
            continue
        paths.append(p)
        seen_keys.add(tuple(p))
        if len(paths) >= k:
            break
    return paths


def _edge_w(adj, u: int, v: int) -> float:
    for nb, w in adj.get(u, []):
        if nb == v:
            return w
    return float("inf")


def _path_length(adj, path: List[int]) -> float:
    return sum(_edge_w(adj, path[i], path[i + 1]) for i in range(len(path) - 1))


def _synth_line_pool(demand: List[Tuple[int, int, float]],
                     adj: Dict[int, List[Tuple[int, float]]],
                     n_lines: int,
                     min_stops: int = 4,
                     strategy: str = "combined",
                     k_shortest: int = 1) -> List[List[int]]:
    """Pick OD pairs by `strategy` and generate k shortest paths per pair.

    strategy:
      demand    — sort by demand desc (favours high-demand OD)
      length    — sort by shortest-path length desc (favours long routes ≈ diameter pairs)
      combined  — sort by demand × shortest-path length (high-demand AND long)

    For each chosen OD, generate up to `k_shortest` paths and add as lines.
    Dedupe by path shape (in either direction). Stops once `n_lines` reached.
    """
    # Pre-compute shortest path lengths to enable the length-aware strategies.
    od_with_len: List[Tuple[int, int, float, float]] = []
    for o, d, w in demand:
        if o == d:
            continue
        sp = _dijkstra_path(adj, o, d)
        if sp is None or len(sp) < min_stops:
            continue
        L = _path_length(adj, sp)
        od_with_len.append((o, d, w, L))

    if strategy == "demand":
        od_with_len.sort(key=lambda x: -x[2])
    elif strategy == "length":
        od_with_len.sort(key=lambda x: -x[3])
    else:  # combined
        od_with_len.sort(key=lambda x: -(x[2] * x[3]))

    seen: set = set()
    lines: List[List[int]] = []
    for o, d, _w, _L in od_with_len:
        for path in _k_shortest_paths(adj, o, d, k=k_shortest):
            if len(path) < min_stops:
                continue
            key = tuple(path)
            rkey = tuple(reversed(path))
            if key in seen or rkey in seen:
                continue
            seen.add(key)
            lines.append(path)
            if len(lines) >= n_lines:
                return lines
    return lines


# ---------------------------------------------------------------------------
# Writers (LinTim / LPP_CGN format)
# ---------------------------------------------------------------------------

def _write_stop_giv(out_path: str, nodes: List[dict]) -> None:
    """LinTim Stop.giv: # stop-id; short-name; long-name; x; y"""
    lines = ["# stop-id; short-name; long-name; x-coordinate; y-coordinate"]
    for n in nodes:
        nid = int(n["id"])
        x = n.get("lon", n.get("x", 0))
        y = n.get("lat", n.get("y", 0))
        lines.append(f"{nid}; {nid}; Stop{nid}; {x}; {y}")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_edge_giv(out_path: str, links: List[Tuple[int, int, float]]) -> None:
    """LinTim Edge.giv: # edge-id; left-stop; right-stop; length; lower-bound; upper-bound

    Mumford has a single `travel_time` per link. We use it as length AND set
    lower==upper bound to the same value (uncrowded == crowded → no overdemand
    inflation by default; the user can later raise upper-bound to model
    crowding).

    Mumford's links file lists both directions of each undirected edge;
    we keep only one (the lexicographically smaller).
    """
    lines = ["# edge-id; left-stop-id; right-stop-id; length; lower-bound; upper-bound"]
    seen = set()
    eid = 1
    for u, v, t in links:
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        lines.append(f"{eid}; {a}; {b}; {t}; {t}; {t}")
        eid += 1
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_od_giv(out_path: str, demand: List[Tuple[int, int, float]]) -> None:
    """LinTim OD.giv: # left-stop-id; right-stop-id; customers"""
    lines = ["# left-stop-id; right-stop-id; customers"]
    for o, d, c in demand:
        if o == d:
            continue
        lines.append(f"{o}; {d}; {c}")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_lines_csv(out_path: str, line_paths: List[List[int]]) -> None:
    """LPP_CGN lines.csv format:
        property;line_group;value_1
        line;<group>;<stop-seq comma-separated>
    Each group gets two rows: +1 forward and -1 backward direction.
    """
    rows = ["property;line_group;value_1"]
    for g, path in enumerate(line_paths, start=1):
        fwd = ",".join(str(s) for s in path)
        bwd = ",".join(str(s) for s in reversed(path))
        rows.append(f"line;{g};{fwd}")
        rows.append(f"line;{g};{bwd}")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")


def _write_scenario_files(out_dir: str) -> None:
    """Write minimal placeholder scenario files. The VSS sweep replaces them
    with generated ones; this is only needed for sanity-test `procedure=one`."""
    infra = os.path.join(out_dir, "scenario_infra.csv")
    with open(infra, "w", encoding="utf-8") as fh:
        fh.write("scenario;left-stop;right-stop;infrastructure_capacity\n")
    prob = os.path.join(out_dir, "scenario_prob.csv")
    with open(prob, "w", encoding="utf-8") as fh:
        fh.write("property;value_1;value_2\nscenario;1;1.0\n")


def _write_properties_general(out_path: str, infra_cap_std: int) -> None:
    """properties_general.csv with sensible defaults for a fresh network.
    `exclude_list` left empty — adjust if you want to skip OD entries for
    specific nodes (e.g. shared/transit-only nodes).
    """
    lines = [
        "num_scenarios;line_cost_mult;infrastructure_capacity_standard;exclude_list",
        f"1;50;{infra_cap_std};",
    ]
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="Path to a Mumford instance folder (with *_nodes.txt, "
                         "*_links.txt, *_demand.txt)")
    ap.add_argument("--network", required=True,
                    help="Network name in the LPP_CGN tree (e.g. Mumford0)")
    ap.add_argument("--repo-root", default=".",
                    help="Path to LPP_CGN repo root (default: current dir)")
    ap.add_argument("--num-lines", type=int, default=30,
                    help="Number of lines to synthesise via shortest paths "
                         "between selected OD pairs")
    ap.add_argument("--min-stops", type=int, default=4,
                    help="Discard synthesised lines shorter than this")
    ap.add_argument("--strategy", default="combined",
                    choices=["demand", "length", "combined"],
                    help="OD selection strategy: 'demand' (top by demand), "
                         "'length' (top by shortest-path length, diameter-like), "
                         "'combined' (top by demand x length, default)")
    ap.add_argument("--k-shortest", type=int, default=1,
                    help="Number of distinct shortest paths per OD pair "
                         "(uses Yen-lite edge removal). Higher = more diverse, "
                         "overlapping lines for transfer-rich routing")
    ap.add_argument("--infra-cap-std", type=int, default=10,
                    help="Default infrastructure capacity per arc")
    ns = ap.parse_args()

    src = ns.src.rstrip("/\\")
    base = os.path.basename(src).lower()  # mumford0
    nodes_path = os.path.join(src, f"{base}_nodes.txt")
    links_path = os.path.join(src, f"{base}_links.txt")
    demand_path = os.path.join(src, f"{base}_demand.txt")
    for p in (nodes_path, links_path, demand_path):
        if not os.path.isfile(p):
            print(f"[ERR] missing input file: {p}", file=sys.stderr)
            sys.exit(1)

    nodes = _read_nodes(nodes_path)
    links = _read_links(links_path)
    demand = _read_demand(demand_path)

    lintim_dir = os.path.join(ns.repo_root, "lintim", ns.network)
    data_dir = os.path.join(ns.repo_root, "Data", ns.network)
    os.makedirs(lintim_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    _write_stop_giv(os.path.join(lintim_dir, "Stop.giv"), nodes)
    _write_edge_giv(os.path.join(lintim_dir, "Edge.giv"), links)
    _write_od_giv(os.path.join(lintim_dir, "OD.giv"), demand)

    adj = _build_adjacency(links)
    line_paths = _synth_line_pool(demand, adj, ns.num_lines, ns.min_stops,
                                  strategy=ns.strategy, k_shortest=ns.k_shortest)
    _write_lines_csv(os.path.join(data_dir, "lines.csv"), line_paths)
    _write_scenario_files(data_dir)
    _write_properties_general(os.path.join(data_dir, "properties_general.csv"),
                              ns.infra_cap_std)

    print(f"Wrote network: {ns.network}")
    print(f"  Nodes : {len(nodes)}")
    print(f"  Edges : {len({(min(u,v),max(u,v)) for u,v,_ in links})} (undirected)")
    print(f"  ODs   : {sum(1 for o,d,_ in demand if o != d)}")
    print(f"  Lines : {len(line_paths)} (synthesised by top-demand shortest paths)")
    print(f"Output : lintim/{ns.network}/  and  Data/{ns.network}/")


if __name__ == "__main__":
    main()

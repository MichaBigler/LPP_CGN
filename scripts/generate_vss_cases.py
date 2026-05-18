#!/usr/bin/env python3
# scripts/generate_vss_cases.py
# -*- coding: utf-8 -*-
"""
Expand a VSS sweep config (e.g. Data/config_vss.csv) into a flat run config
with one row per actual run, plus the matching scenario_infra/scenario_prob
files in Data/<Network>/generated/.

A sweep row declares HOW MANY cases of which kind to produce. The generator
materialises those cases deterministically so the same sweep config produces
the same files across runs.

Sweep row schema (extends the regular config row with):

    case_k          int   — edges per failure scenario
    case_per_run    int   — failure scenarios per run (in addition to the
                            implicit undisturbed scenario)
    case_variants   int   — number of runs (= bundles) to produce
    case_selection  str   — pool from which failures are drawn:
                              line_consecutive (default), line_all,
                              share_stop, random
    case_p_fail     float — probability mass on the failure side; split
                            equally over `case_per_run` failure scenarios.
                            The undisturbed scenario gets 1 − p_fail.

Generated artefacts per case_id:
    Data/<Network>/generated/scenario_infra_<case_id>.csv
    Data/<Network>/generated/scenario_prob_<case_id>.csv

Each generated config row carries:
    scenario_infra_id = case_id
    scenario_prob_id  = case_id
    case_k            = k
    case_per_run      = per_run
    case_edges        = "|"-separated list of failure variant edges per scenario
                        (e.g. "1-2,3-4|5-6" = scenario 2: edges 1-2 & 3-4;
                                              scenario 3: edge 5-6;
                                              scenario 1 implicit undisturbed)
    case_lines        = "|"-separated list of line IDs the failures touch
    case_selection    = passthrough
    case_p_fail       = passthrough

Usage:
    python scripts/generate_vss_cases.py \\
        --in Data/config_vss.csv \\
        --out Data/config_vss_expanded.csv \\
        --data-root .
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, FrozenSet, List, Set, Tuple

import pandas as pd


# Undirected edge representation: (min(u,v), max(u,v))
UndirEdge = Tuple[int, int]
FailureSet = FrozenSet[UndirEdge]   # one failure scenario = a set of undirected edges


# ---------------------------------------------------------------------------
# Network parsing helpers
# ---------------------------------------------------------------------------

def _parse_edge_giv(path: str) -> List[UndirEdge]:
    """Read lintim Edge.giv → list of undirected (u, v) tuples, u < v."""
    edges: List[UndirEdge] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # format: edge-id; u; v; len; lo; hi
            parts = [p.strip() for p in line.split(";")]
            try:
                u, v = int(parts[1]), int(parts[2])
            except (IndexError, ValueError):
                continue
            edges.append((min(u, v), max(u, v)))
    return edges


def _parse_lines_csv(path: str) -> Dict[int, List[List[int]]]:
    """Read Data/<Network>/lines.csv → {line_group: [stops...]} for the +1 direction.

    We deliberately only keep one direction per group because the undirected
    edge set is the same on the backward direction.
    """
    df = pd.read_csv(path, sep=";")
    out: Dict[int, List[List[int]]] = {}
    for _, row in df.iterrows():
        if str(row.get("property", "")).lower() != "line":
            continue
        try:
            g = int(row["line_group"])
        except (KeyError, ValueError):
            continue
        seq_str = str(row.get("value_1", ""))
        stops = [int(s) for s in seq_str.split(",") if s.strip()]
        out.setdefault(g, []).append(stops)
    return out


def _line_groups_to_edges(line_seqs: Dict[int, List[List[int]]]) -> Dict[int, List[UndirEdge]]:
    """Per line group, take the first stop sequence and turn it into consecutive
    undirected edges. Reverse direction yields the same undirected edges."""
    out: Dict[int, List[UndirEdge]] = {}
    for g, seqs in line_seqs.items():
        if not seqs:
            continue
        stops = seqs[0]
        edges = [(min(u, v), max(u, v)) for u, v in zip(stops[:-1], stops[1:])]
        out[g] = edges
    return out


# ---------------------------------------------------------------------------
# Failure-pool builders (one per selection mode)
# ---------------------------------------------------------------------------

def _pool_line_consecutive(
    line_edges: Dict[int, List[UndirEdge]], k: int,
) -> List[Tuple[FailureSet, List[int]]]:
    """Failures = consecutive k-edge windows on each line.

    Returns list of (failure_set, [line_ids it touches]).
    De-duplicates failure sets that appear on multiple lines.
    """
    seen: Dict[FailureSet, List[int]] = {}
    for g, edges in line_edges.items():
        if len(edges) < k:
            continue
        for i in range(len(edges) - k + 1):
            window = frozenset(edges[i:i + k])
            seen.setdefault(window, []).append(g)
    return [(fs, sorted(set(gs))) for fs, gs in seen.items()]


def _pool_line_all(
    line_edges: Dict[int, List[UndirEdge]], k: int,
) -> List[Tuple[FailureSet, List[int]]]:
    """All k-subsets of edges that lie on the same line (any positions)."""
    from itertools import combinations
    seen: Dict[FailureSet, List[int]] = {}
    for g, edges in line_edges.items():
        if len(edges) < k:
            continue
        for combo in combinations(edges, k):
            fs = frozenset(combo)
            seen.setdefault(fs, []).append(g)
    return [(fs, sorted(set(gs))) for fs, gs in seen.items()]


def _pool_share_stop(
    edges: List[UndirEdge], k: int,
) -> List[Tuple[FailureSet, List[int]]]:
    """k edges that pairwise share at least one stop (graph-adjacent cluster).

    Heuristic: build a graph where edges are nodes and two edges are linked
    iff they share a stop. Failures = connected k-cliques. For k=1 it is just
    all edges; for k=2 every adjacent pair; for k>=3 it gets expensive — we
    cap with a sampling fallback above 1000 candidates.
    """
    from itertools import combinations
    if k == 1:
        return [(frozenset({e}), []) for e in edges]
    stop_to_edges: Dict[int, Set[UndirEdge]] = {}
    for e in edges:
        u, v = e
        stop_to_edges.setdefault(u, set()).add(e)
        stop_to_edges.setdefault(v, set()).add(e)
    seen: Set[FailureSet] = set()
    for cluster in stop_to_edges.values():
        if len(cluster) < k:
            continue
        for combo in combinations(cluster, k):
            seen.add(frozenset(combo))
            if len(seen) > 5000:
                break
        if len(seen) > 5000:
            break
    return [(fs, []) for fs in seen]


def _pool_random(
    edges: List[UndirEdge], k: int, target: int, seed: int,
) -> List[Tuple[FailureSet, List[int]]]:
    """`target` random k-subsets, no line affiliation. Deterministic by seed."""
    from itertools import combinations
    rng = random.Random(seed)
    total = 1
    for i in range(k):
        total = total * (len(edges) - i) // (i + 1)
    if total <= target:
        return [(frozenset(combo), []) for combo in combinations(edges, k)]
    seen: Set[FailureSet] = set()
    while len(seen) < target:
        seen.add(frozenset(rng.sample(edges, k)))
    return [(fs, []) for fs in seen]


# ---------------------------------------------------------------------------
# Main expansion
# ---------------------------------------------------------------------------

def _format_edges(fs: FailureSet) -> str:
    """Render a failure set as 'u1-v1,u2-v2,...' sorted lexicographically."""
    return ",".join(f"{u}-{v}" for u, v in sorted(fs))


def _format_bundle_edges(bundle: List[FailureSet]) -> str:
    """Render a per-run bundle as 'scen2|scen3|...' (scenario 1 is undisturbed)."""
    return "|".join(_format_edges(fs) for fs in bundle)


def _format_bundle_lines(bundle_lines: List[List[int]]) -> str:
    return "|".join(",".join(str(g) for g in lines) for lines in bundle_lines)


def _build_pool(
    selection: str, k: int, line_edges: Dict[int, List[UndirEdge]],
    edges: List[UndirEdge], seed: int, target: int,
) -> List[Tuple[FailureSet, List[int]]]:
    sel = selection.strip().lower()
    if sel == "line_consecutive":
        return _pool_line_consecutive(line_edges, k)
    if sel == "line_all":
        return _pool_line_all(line_edges, k)
    if sel == "share_stop":
        return _pool_share_stop(edges, k)
    if sel == "random":
        return _pool_random(edges, k, max(target, 50), seed)
    raise ValueError(f"Unknown case_selection: {selection!r}")


def _write_scenario_infra(path: str, bundle: List[FailureSet]) -> None:
    """Scenario 1 is implicitly undisturbed (no rows). Scenarios 2..N+1 list
    the failed edges of bundle[0..N-1] with capacity 0 in both directions.

    The loader (`build_scenario_capacities`) writes capacity to the index of
    the reverse arc (v, u); emitting both (u, v) and (v, u) rows guarantees
    closure in either convention.
    """
    rows = []
    for i, fs in enumerate(bundle, start=2):
        for (u, v) in sorted(fs):
            rows.append({"scenario": i, "left-stop": u, "right-stop": v,
                         "infrastructure_capacity": 0})
            rows.append({"scenario": i, "left-stop": v, "right-stop": u,
                         "infrastructure_capacity": 0})
    df = pd.DataFrame(rows, columns=["scenario", "left-stop", "right-stop",
                                     "infrastructure_capacity"])
    df.to_csv(path, sep=";", index=False)


def _write_scenario_prob(path: str, n_failures: int, p_fail: float) -> None:
    """Scenario 1 = undisturbed; the rest = `n_failures` failure scenarios.

    Probabilities: undisturbed gets (1 - p_fail), each failure gets
    p_fail / n_failures. Sums to 1 by construction.
    """
    p_nom = max(0.0, 1.0 - float(p_fail))
    p_each = float(p_fail) / max(1, n_failures)
    rows = [{"property": "scenario", "value_1": 1, "value_2": p_nom}]
    for i in range(n_failures):
        rows.append({"property": "scenario", "value_1": 2 + i, "value_2": p_each})
    pd.DataFrame(rows, columns=["property", "value_1", "value_2"]).to_csv(
        path, sep=";", index=False
    )


def expand_sweep(in_path: str, out_path: str, data_root: str, *, base_case_id: int = 1000) -> int:
    """Read a sweep CSV, write scenario files and a flat expanded config.

    Returns the number of expanded rows.
    """
    sweep_df = pd.read_csv(in_path, sep=";")
    if sweep_df.empty:
        print("[ERR] sweep config is empty", file=sys.stderr)
        return 0

    required = {"network", "case_k", "case_per_run", "case_variants",
                "case_selection", "case_p_fail"}
    missing = required - set(sweep_df.columns)
    if missing:
        raise ValueError(f"Sweep CSV missing required columns: {sorted(missing)}")

    # Columns to copy verbatim from each sweep row into the expanded rows
    passthrough_cols = [c for c in sweep_df.columns if c not in {
        "case_k", "case_per_run", "case_variants",
        "case_selection", "case_p_fail",
        "scenario_infra_id", "scenario_prob_id",
    }]

    expanded: List[Dict] = []
    case_id = int(base_case_id)

    # Cache parsed networks
    networks_cache: Dict[str, Tuple[List[UndirEdge], Dict[int, List[UndirEdge]]]] = {}

    for sweep_idx, sweep_row in sweep_df.iterrows():
        network = str(sweep_row["network"]).strip()
        scen_dir = os.path.join(data_root, "Data", network)
        gen_dir = os.path.join(scen_dir, "generated")
        os.makedirs(gen_dir, exist_ok=True)

        # Parse network once per unique network
        if network not in networks_cache:
            edge_giv = os.path.join(data_root, "lintim", network, "Edge.giv")
            lines_csv = os.path.join(scen_dir, "lines.csv")
            edges = _parse_edge_giv(edge_giv)
            line_edges = _line_groups_to_edges(_parse_lines_csv(lines_csv))
            networks_cache[network] = (edges, line_edges)
        edges, line_edges = networks_cache[network]

        k = int(sweep_row["case_k"])
        per_run = int(sweep_row["case_per_run"])
        variants = int(sweep_row["case_variants"])
        selection = str(sweep_row["case_selection"])
        p_fail = float(sweep_row["case_p_fail"])

        pool = _build_pool(selection, k, line_edges, edges,
                           seed=int(sweep_idx), target=variants * per_run + 50)
        if not pool:
            print(f"[WARN] sweep row {sweep_idx}: empty failure pool "
                  f"(k={k}, selection={selection})", file=sys.stderr)
            continue
        if len(pool) < per_run:
            print(f"[WARN] sweep row {sweep_idx}: pool has {len(pool)} variants, "
                  f"per_run={per_run} truncated", file=sys.stderr)

        rng = random.Random(int(sweep_idx) * 1_000_003 + 17)

        for variant_idx in range(variants):
            actual_per_run = min(per_run, len(pool))
            picks = rng.sample(pool, actual_per_run)
            bundle = [fs for (fs, _gs) in picks]
            bundle_lines = [gs for (_fs, gs) in picks]

            infra_path = os.path.join(gen_dir, f"scenario_infra_{case_id}.csv")
            prob_path = os.path.join(gen_dir, f"scenario_prob_{case_id}.csv")
            _write_scenario_infra(infra_path, bundle)
            _write_scenario_prob(prob_path, actual_per_run, p_fail)

            # Build expanded config row: copy passthrough columns + case meta
            row: Dict = {c: sweep_row[c] for c in passthrough_cols}
            row["scenario_infra_id"] = case_id
            row["scenario_prob_id"] = case_id
            row["case_k"] = k
            row["case_per_run"] = actual_per_run
            row["case_variants"] = variants
            row["case_selection"] = selection
            row["case_p_fail"] = p_fail
            row["case_id"] = case_id
            row["case_sweep_row"] = int(sweep_idx)
            row["case_variant_idx"] = variant_idx
            row["case_edges"] = _format_bundle_edges(bundle)
            row["case_lines"] = _format_bundle_lines(bundle_lines)
            expanded.append(row)
            case_id += 1

    if not expanded:
        print("[ERR] no expanded rows produced", file=sys.stderr)
        return 0

    out_df = pd.DataFrame(expanded)
    # Stable column order: original passthrough first, then case meta
    case_cols = ["scenario_infra_id", "scenario_prob_id",
                 "case_k", "case_per_run", "case_variants",
                 "case_selection", "case_p_fail",
                 "case_id", "case_sweep_row", "case_variant_idx",
                 "case_edges", "case_lines"]
    ordered = passthrough_cols + [c for c in case_cols if c not in passthrough_cols]
    out_df = out_df.reindex(columns=ordered)
    out_df.to_csv(out_path, sep=";", index=False)
    print(f"Wrote {len(expanded)} expanded rows to {out_path}")
    return len(expanded)


def _cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True,
                    help="Path to sweep CSV (e.g. Data/config_vss.csv)")
    ap.add_argument("--out", dest="out_path", required=True,
                    help="Path to write expanded flat config")
    ap.add_argument("--data-root", default=".",
                    help="Root containing Data/ and lintim/")
    ap.add_argument("--base-case-id", type=int, default=1000,
                    help="Starting case_id, kept high to avoid collision with "
                         "existing scenario_infra_<id>.csv files")
    ns = ap.parse_args()
    n = expand_sweep(ns.in_path, ns.out_path, ns.data_root,
                     base_case_id=ns.base_case_id)
    sys.exit(0 if n > 0 else 1)


if __name__ == "__main__":
    _cli()

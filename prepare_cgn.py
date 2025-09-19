# prepare_cgn.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from data_model import CGN


# ---------------------------------------------------------------------------
# Shared internals
# ---------------------------------------------------------------------------

class _CGNBuilder:
    """
    Minimal builder for CGN graphs that works for both:
      - nominal lines (variant=0)
      - candidate-per-line layers (variant=k)

    Node representation (internal):
      tuple(phys_node, line, variant), where variant == -1 indicates ground layer.

    The ordering of fields in the final CGN dataclass strictly follows data_model.CGN.
    """

    GROUND_VARIANT = -1  # sentinel for ground layer

    def __init__(self, model):
        self.m = model

        # --- node bookkeeping ---
        self.nodes: List[Tuple[int, int, int]] = []               # (phys, line, variant)
        self._node_id: Dict[Tuple[int, int, int], int] = {}       # key -> node index
        self.ground_of: List[int] = [-1] * int(model.N)           # phys node idx -> ground node idx

        # For change/board/alight construction: which (line,variant) touches each phys node
        self.lines_at_node: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(int(model.N))}

        # --- arc lists ---
        self.arc_tail: List[int] = []
        self.arc_head: List[int] = []
        self.arc_kind: List[str] = []
        self.arc_line: List[int] = []
        self.arc_edge: List[int] = []
        self.arc_variant: List[int] = []
        self.arc_line_to: List[int] = []

        # Pre-compute tails/heads for infra arcs in *index* space 0..N-1
        nid2idx = model.node_id_to_idx
        self._infra_tail: List[int] = []
        self._infra_head: List[int] = []
        for (u_id, v_id) in model.idx_to_arc_uv:
            self._infra_tail.append(int(nid2idx[u_id]))
            self._infra_head.append(int(nid2idx[v_id]))

    # -------- node layer helpers --------

    def _get_or_create_node(self, phys: int, line: int, variant: int) -> int:
        """Return node index, creating it if necessary."""
        key = (phys, line, variant)
        idx = self._node_id.get(key)
        if idx is None:
            idx = len(self.nodes)
            self._node_id[key] = idx
            self.nodes.append(key)
        return idx

    def add_ground_layer(self) -> None:
        """Create one ground node per physical node."""
        for i in range(int(self.m.N)):
            idx = self._get_or_create_node(i, -1, self.GROUND_VARIANT)
            self.ground_of[i] = idx

    def touch_line_at_node(self, phys: int, line: int, variant: int) -> int:
        """
        Ensure a (line,variant) node exists at physical node `phys`,
        register its presence for later change/board/alight arcs, and return the node index.
        """
        if (line, variant) not in self.lines_at_node[phys]:
            self.lines_at_node[phys].append((line, variant))
        return self._get_or_create_node(phys, line, variant)

    # -------- arc builders --------

    def add_ride_path(self, line: int, variant: int, arc_ids: List[int]) -> None:
        """
        Add ride arcs along a directed infra-arc sequence for (line,variant).
        `arc_ids` must be infra-arc indices (directed).
        """
        if not arc_ids:
            return
        # Build the node sequence in phys-index space via infra arcs
        u = self._infra_tail[arc_ids[0]]
        path_nodes = [u]
        for a in arc_ids:
            path_nodes.append(self._infra_head[a])

        # Ensure layer nodes exist and are registered at every node on the path
        for phys in path_nodes:
            self.touch_line_at_node(phys, line, variant)

        # Emit ride arcs between consecutive layer nodes
        u = path_nodes[0]
        for a in arc_ids:
            v = self._infra_head[a]
            tail = self._get_or_create_node(u, line, variant)
            head = self._get_or_create_node(v, line, variant)

            self.arc_tail.append(tail)
            self.arc_head.append(head)
            self.arc_kind.append("ride")
            self.arc_line.append(int(line))
            self.arc_edge.append(int(a))
            self.arc_variant.append(int(variant))
            self.arc_line_to.append(-1)  # ride does not induce waiting target
            u = v

    def add_change_arcs(self) -> None:
        """
        For each physical node, connect every (line,variant) to every other (line,variant).
        """
        for phys in range(int(self.m.N)):
            lv = self.lines_at_node[phys]
            if not lv:
                continue
            for (l1, k1) in lv:
                v_from = self._get_or_create_node(phys, l1, k1)
                for (l2, k2) in lv:
                    if l1 == l2 and k1 == k2:
                        continue
                    v_to = self._get_or_create_node(phys, l2, k2)
                    self.arc_tail.append(v_from)
                    self.arc_head.append(v_to)
                    self.arc_kind.append("change")
                    self.arc_line.append(int(l1))      # source line (diagnostic)
                    self.arc_edge.append(-1)
                    self.arc_variant.append(int(k1))
                    self.arc_line_to.append(int(l2))   # target line for waiting

    def add_board_alight(self) -> None:
        """
        At each physical node, connect ground <-> (line,variant).
        - board: ground -> line (arc_line_to = line)
        - alight: line -> ground
        """
        for phys in range(int(self.m.N)):
            v_g = self.ground_of[phys]
            lv = self.lines_at_node[phys]
            if not lv:
                continue
            for (line, var) in lv:
                v_line = self._get_or_create_node(phys, line, var)
                # board
                self.arc_tail.append(v_g)
                self.arc_head.append(v_line)
                self.arc_kind.append("board")
                self.arc_line.append(int(line))
                self.arc_edge.append(-1)
                self.arc_variant.append(int(var))
                self.arc_line_to.append(int(line))
                # alight
                self.arc_tail.append(v_line)
                self.arc_head.append(v_g)
                self.arc_kind.append("alight")
                self.arc_line.append(int(line))
                self.arc_edge.append(-1)
                self.arc_variant.append(int(var))
                self.arc_line_to.append(-1)

    def add_bypass_arcs_if_enabled(self) -> None:
        """
        Optionally add ground-level bypass arcs for each directed infra arc:
          ground(u) -> ground(v), kind='bypass', arc_edge=a_idx, line=-1, variant=-1.
        Enabled if config['bypass_multiplier'] >= 0.0 (same policy as before).
        """
        bypass_mult = float(getattr(self.m, "config", {}).get("bypass_multiplier", -1.0))
        if bypass_mult < 0.0:
            return
        for a_idx, (u_id, v_id) in enumerate(self.m.idx_to_arc_uv):
            u = int(self.m.node_id_to_idx[u_id])
            v = int(self.m.node_id_to_idx[v_id])
            self.arc_tail.append(self.ground_of[u])
            self.arc_head.append(self.ground_of[v])
            self.arc_kind.append("bypass")
            self.arc_line.append(-1)
            self.arc_edge.append(int(a_idx))
            self.arc_variant.append(-1)
            self.arc_line_to.append(-1)

    def to_cgn(self) -> CGN:
        """Finalize adjacency and pack everything into the CGN dataclass."""
        V = len(self.nodes)
        A = len(self.arc_tail)

        in_arcs: List[List[int]] = [[] for _ in range(V)]
        out_arcs: List[List[int]] = [[] for _ in range(V)]
        for a, (t, h) in enumerate(zip(self.arc_tail, self.arc_head)):
            out_arcs[t].append(a)
            in_arcs[h].append(a)

        node_line    = [line for (_phys, line, _var) in self.nodes]
        node_variant = [var  for (_phys, _line, var)  in self.nodes]

        return CGN(
            V, A,
            in_arcs, out_arcs,
            self.ground_of,
            self.arc_tail, self.arc_head, self.arc_kind, self.arc_line, self.arc_edge, self.arc_variant,
            node_line, node_variant, self.arc_line_to
        )


# ---------------------------------------------------------------------------
# Public API: nominal CGN (variant=0)
# ---------------------------------------------------------------------------

def make_cgn(model) -> CGN:
    """
    Build a CGN from nominal lines only (no candidates).
    - One ground node per physical node.
    - For each directed line ℓ: add ride arcs along its nominal infra arcs (variant=0).
    - Add full change arcs between line layers at shared stops.
    - Add board/alight arcs between ground and line layers.
    - Optionally add bypass arcs on ground if bypass is enabled.

    Returns a data_model.CGN instance.
    """
    b = _CGNBuilder(model)
    b.add_ground_layer()

    # Ride arcs along nominal line definitions (variant=0)
    for ell in range(int(model.L)):
        arc_seq = list(map(int, model.line_idx_to_arcs[ell]))
        if not arc_seq:
            continue
        b.add_ride_path(line=int(ell), variant=0, arc_ids=arc_seq)

    # Transfers and access/egress
    b.add_change_arcs()
    b.add_board_alight()

    # Optional ground-level bypass
    b.add_bypass_arcs_if_enabled()

    return b.to_cgn()


# ---------------------------------------------------------------------------
# Public API: CGN from per-line candidates (variant = candidate index)
# ---------------------------------------------------------------------------

def make_cgn_with_candidates_per_line(model, cand_lines_s: Dict[int, List[dict]]) -> CGN:
    """
    Build a CGN from per-line candidate paths for a single scenario.

    Arguments:
      cand_lines_s[ell] = list of candidate dicts, each having at least:
        - "arcs": List[int] of directed infra arc IDs (in model index space)
        (other fields like "len"/"delta" are ignored by the builder)

    Behavior:
      - One ground node per physical node.
      - For each line ℓ and each candidate k: add ride arcs along the candidate
        arc sequence as (line=ℓ, variant=k).
      - Add full change arcs between all (line,variant) layers co-located at a stop.
      - Add board/alight arcs between ground and each (line,variant) layer.
      - Optionally add bypass arcs on ground if bypass is enabled.

    Returns a data_model.CGN instance.
    """
    b = _CGNBuilder(model)
    b.add_ground_layer()

    # Add a layered ride path per (line, variant=k)
    for ell in range(int(model.L)):
        cand_list = cand_lines_s.get(int(ell), []) or []
        for k, cand in enumerate(cand_list):
            arc_seq = [int(a) for a in cand.get("arcs", [])]
            if not arc_seq:
                continue
            b.add_ride_path(line=int(ell), variant=int(k), arc_ids=arc_seq)

    # Transfers and access/egress over all (line,variant) at each phys node
    b.add_change_arcs()
    b.add_board_alight()

    # Optional ground-level bypass (added exactly once)
    b.add_bypass_arcs_if_enabled()

    return b.to_cgn()

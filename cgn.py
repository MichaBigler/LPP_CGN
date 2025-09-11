from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

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
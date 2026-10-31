import pandas as pd

from load_data import load_and_build
from prepare_cgn import make_cgn

# 1. Load the first configuration row
cfg = pd.read_csv("Data/config.csv", sep=";")
cfg_row = cfg.iloc[0].to_dict()

# 2. Load the network and construct model data
domain, model = load_and_build(
    data_root=".",
    cfg_row=cfg_row,
    symmetrise_infra=False,
    zero_od_diagonal=False,
)

# 3. Construct the CGN (without solving anything)
cgn = make_cgn(model)

print(f"CGN vertices: {cgn.V}")
print(f"CGN arcs: {cgn.A}")

# 4. Verify the new data structures
assert isinstance(cgn.bypass_length_by_arc, dict)
assert isinstance(cgn.arc_line_to, list)

assert len(cgn.ground_in_of) == model.N
assert len(cgn.ground_out_of) == model.N

# 5. Verify ground vertex structure
assert all(
    len(cgn.in_arcs[v]) == 0
    for v in cgn.ground_in_of
)

assert all(
    len(cgn.out_arcs[v]) == 0
    for v in cgn.ground_out_of
)

# 6. Verify bypass arcs
bypass_arcs = [
    a for a in range(cgn.A)
    if cgn.arc_kind[a] == "bypass"
]

expected_bypasses = sum(
    model.D[o, d] > 0
    for o in range(model.N)
    for d in range(model.N)
)

assert len(bypass_arcs) == expected_bypasses

ground_in = set(cgn.ground_in_of)
ground_out = set(cgn.ground_out_of)

for a in bypass_arcs:
    assert cgn.arc_tail[a] in ground_in
    assert cgn.arc_head[a] in ground_out
    assert a in cgn.bypass_length_by_arc
    assert cgn.arc_edge[a] == -1

print("All CGN structural tests passed!")

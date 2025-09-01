

from data_model import LineDef, DomainData, ModelData


# ---------- inspection / debug prints (compact, readable) ----------

def _head(d: dict, n: int = 5):
    """Return first n (key, value) pairs of a dict as a list."""
    out = []
    for k, v in d.items():
        out.append((k, v))
        if len(out) >= n:
            break
    return out

def print_domain_summary(domain: DomainData, max_rows: int = 5, max_lines: int = 5, max_include: int = 5):
    """Print compact checks for all DomainData structures."""
    print("\n=== DomainData ===")

    # Stops
    print(f"Stops: {len(domain.stops_df)}")
    print(domain.stops_df.head(max_rows).to_string(index=False))

    # Links (undirected in file)
    print(f"\nLinks (file, undirected): {len(domain.links_df)}")
    print(domain.links_df.head(max_rows).to_string(index=False))

    # OD
    nnz = int((domain.od_df['demand'] != 0).sum())
    print(f"\nOD rows: {len(domain.od_df)} (nonzero={nnz})")
    print(domain.od_df.head(max_rows).to_string(index=False))

    # Lines
    print(f"\nLines (definitions): {len(domain.lines)}")
    for ld in domain.lines[:max_lines]:
        seq = ld.stops
        preview = seq[:6] + (["…"] if len(seq) > 6 else [])
        print(f"  group={ld.group} dir={ld.direction:+d} stops={preview}")

    # Include sets
    print(f"\nInclude nodes (from num_include={domain.config.get('num_include')}): {len(next(iter(domain.include_sets.values()), set()))}")
    inc_preview = sorted(next(iter(domain.include_sets.values()), set()))[:20]
    print(f"  nodes: {inc_preview}{' …' if len(inc_preview)==20 else ''}")

    # Scenario probabilities
    print(f"\nScenario probs (rows={len(domain.scen_prob_df)}):")
    print(domain.scen_prob_df.head(max_rows).to_string(index=False))

    # Scenario infra overrides
    print(f"\nScenario infra overrides (rows={len(domain.scen_infra_df)}):")
    print(domain.scen_infra_df.head(max_rows).to_string(index=False))

    # Properties
    print("\nProperties (general):")
    for k, v in domain.props.items():
        print(f"  {k}: {v}")

    # Config row
    print("\nConfig row (selected):")
    for k, v in _head(domain.config, 10):
        print(f"  {k}: {v}")

def print_model_summary(model: ModelData, domain: DomainData, max_items: int = 5, max_lines: int = 5):
    """Print compact checks for all ModelData structures."""
    import numpy as np

    print("\n=== ModelData ===")
    print(f"Sizes: N={model.N}  E_dir={model.E_dir}  L={model.L}  S={model.S}")

    # Node indexing + coordinates
    print("\nNodes/indexing:")
    print("  first id→idx:", _head(model.node_id_to_idx, max_items))
    if model.N:
        print(f"  coord_x[min,max]=({model.coord_x.min():.2f},{model.coord_x.max():.2f})",
              f"coord_y[min,max]=({model.coord_y.min():.2f},{model.coord_y.max():.2f})")

    # Arcs with attributes
    print(f"\nArcs (directed): {model.E_dir}")
    for a_idx, (u, v) in list(enumerate(model.idx_to_arc_uv))[:max_items]:
        print(f"  a={a_idx}: {u}->{v} len={model.len_a[a_idx]:.3f} t=[{model.t_min_a[a_idx]:.3f},{model.t_max_a[a_idx]:.3f}]")

    # OD matrix stats
    nnz = int(np.count_nonzero(model.D))
    total = float(model.D.sum())
    diag = float(np.trace(model.D))
    print(f"\nOD matrix: shape={model.D.shape}, nnz={nnz}, total={total:.2f}, diag={diag:.2f}")

    # Scenarios + capacity overrides
    print(f"\nScenarios: S={model.S}, p_s={model.p_s.tolist()}")
    std_cap = int(domain.props.get('infra_cap_std', -1))
    diff_s, diff_a = np.where(model.cap_sa != std_cap)
    print(f"Capacity matrix: shape={model.cap_sa.shape}, std={std_cap}, overrides={len(diff_s)}")
    for s_idx, a_idx in list(zip(diff_s, diff_a))[:max_items]:
        u, v = model.idx_to_arc_uv[a_idx]
        print(f"  s={s_idx} arc {u}->{v} cap={int(model.cap_sa[s_idx, a_idx])}")

    # Lines
    print(f"\nLines: L={model.L}, groups={len(model.line_group_to_lines)}")
    for g, (ell_fwd, ell_bwd) in list(model.line_group_to_lines.items())[:max_lines]:
        def line_preview(ell_idx: int) -> str:
            if ell_idx < 0:
                return "-"
            idxs = model.line_idx_to_stops[ell_idx]
            ids = [model.idx_to_node_id[i] for i in idxs]
            preview = ids[:6] + (["…"] if len(ids) > 6 else [])
            return f"ℓ={ell_idx} stops={preview}"
        print(f"  group {g}: {line_preview(ell_fwd)} | {line_preview(ell_bwd)}")

    # Adjacency degrees
    deg_out = np.array([len(x) for x in model.adj_out], dtype=int)
    deg_in  = np.array([len(x) for x in model.adj_in],  dtype=int)
    if len(deg_out):
        print(f"\nDegrees: out[min,avg,max]=({deg_out.min()},{deg_out.mean():.2f},{deg_out.max()})",
              f"in[min,avg,max]=({deg_in.min()},{deg_in.mean():.2f},{deg_in.max()})")

    # Incidence matrices
    def _density(mat) -> float:
        """Safe density for scipy CSR/COO/CSC; returns 0.0 for empty or missing shape."""
        shp = getattr(mat, "shape", None)
        if not shp:
            return 0.0
        rows, cols = shp
        if rows == 0 or cols == 0:
            return 0.0
        return float(mat.nnz) / float(rows * cols)

    dens_eL = _density(model.A_edge_line)
    dens_nL = _density(model.A_node_line)
    print(f"\nA_edge_line: shape={model.A_edge_line.shape}, nnz={model.A_edge_line.nnz}, density={dens_eL:.4f}")
    print(f"A_node_line: shape={model.A_node_line.shape}, nnz={model.A_node_line.nnz}, density={dens_nL:.4f}")

    # Include / exclude
    print(f"\nExclude nodes (IDs): {sorted(list(domain.props.get('exclude_nodes', [])))[:20]}",
          "(…)" if len(domain.props.get('exclude_nodes', [])) > 20 else "")
    print(f"Include sets: {len(model.include_sets)}")

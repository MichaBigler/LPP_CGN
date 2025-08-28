
# -*- coding: utf-8 -*-
"""
Compact pipeline:
- load_and_build(data_root, cfg_row, symmetrise_infra=False, zero_od_diagonal=False)
  -> DomainData, ModelData
All parsing + building helpers live INSIDE load_and_build for locality/clarity.
Comments in English.
"""
from typing import Dict, Tuple, List, Set
import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from data_model import LineDef, DomainData, ModelData  

def load_and_build(data_root: str, cfg_row: dict, *, symmetrise_infra: bool=False, zero_od_diagonal: bool=False) -> Tuple[DomainData, ModelData]:
    """Resolve strict nested paths from cfg_row (expects keys: source, network, scenario_line_data) and build structures."""

    def key(d: dict, name: str) -> str:
        for k in d.keys():
            if k.lower() == name.lower():
                return k
        raise KeyError(f"Missing '{name}' in config row")

    def must(p: str) -> str:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return p

    # ---------- parse the three path parameters ----------
    src, net, scen = str(cfg_row[key(cfg_row,'source')]), str(cfg_row[key(cfg_row,'network')]), str(cfg_row[key(cfg_row,'scenario_line_data')])
    net_dir  = os.path.join(data_root, src, net)
    scen_dir = os.path.join(data_root, "Data", scen)

    # ---------- parsers (scoped) ----------
    def read_stops(p: str) -> pd.DataFrame:
        return pd.read_csv(must(p), sep=';', comment='#', header=None, names=['id','short','long','x','y']).astype({'id':int,'short':str,'long':str,'x':float,'y':float})

    def read_links(p: str) -> pd.DataFrame:
        return pd.read_csv(must(p), sep=';', comment='#', header=None, names=['id','a','b','length','t_min','t_max']).astype({'id':int,'a':int,'b':int,'length':float,'t_min':float,'t_max':float})

    def read_od(p: str) -> pd.DataFrame:
        return pd.read_csv(must(p), sep=';', comment='#', header=None, names=['i','j','demand']).astype({'i':int,'j':int,'demand':float})

    def read_lines(p: str) -> List[LineDef]:
        raw = pd.read_csv(must(p), sep=';')
        raw = raw[raw['property'].str.lower()=='line']
        out: List[LineDef] = []
        for g, grp in raw.groupby('line_group', sort=True):
            rows = grp.sort_index()
            seqs = [[int(x) for x in str(rows.iloc[k]['value_1']).split(',') if str(x).strip()!=''] for k in range(len(rows))]
            if len(seqs)==1: out.append(LineDef(int(g), +1, seqs[0]))
            else: out += [LineDef(int(g), +1, seqs[0]), LineDef(int(g), -1, seqs[1])]
        return out

    def read_include_list(p: str) -> Dict[int, Set[int]]:
        df = pd.read_csv(must(p), sep=';')
        stop_cols = [c for c in df.columns if c!='num_include']
        include = {}
        for r, row in df.iterrows():
            allowed = set()
            for c in stop_cols:
                try: j = int(c)
                except: continue
                v = row[c]
                if pd.notna(v) and str(v).strip() not in ('','0','0.0','False','false'):
                    try:
                        if float(v)!=0.0: allowed.add(j)
                    except: allowed.add(j)
            if allowed: include[r+1] = allowed
        return include

    def read_scenario_prob(p: str) -> pd.DataFrame:
        df = pd.read_csv(must(p), sep=';')
        df = df[df['property'].str.lower()=='scenario'][['value_1','value_2']].rename(columns={'value_1':'id','value_2':'prob'})
        return df.astype({'id':int,'prob':float})

    def read_scenario_infra(p: str) -> pd.DataFrame:
        return pd.read_csv(must(p), sep=';').rename(columns={'left-stop':'u','right-stop':'v','infrastructure_capacity':'cap'}).astype({'scenario':int,'u':int,'v':int,'cap':int})

    def read_properties_general(p: str) -> dict:
        df = pd.read_csv(must(p), sep=';')
        row = df.iloc[0].to_dict()
        ex_key = [c for c in df.columns if c.lower().startswith('exclude')][0]
        cap_key = [c for c in df.columns if 'infrastructure' in c.lower() and ('std' in c.lower() or 'standard' in c.lower())][0]
        return {
            'num_scenarios': int(row[[c for c in df.columns if c.lower().startswith('num_scen')][0]]),
            'line_cost_mult': float(row[[c for c in df.columns if c.lower().startswith('line_cost')][0]]),
            'infra_cap_std': int(row[cap_key]),
            'exclude_nodes': {int(x) for x in str(row.get(ex_key,'')).split(',') if str(x).strip()!=''}
        }

    # ---------- read all ----------
    stops_df = read_stops(os.path.join(net_dir, "Stop.giv"))
    links_df = read_links(os.path.join(net_dir, "Edge.giv"))
    od_df    = read_od(   os.path.join(net_dir, "OD.giv"))
    lines    = read_lines(os.path.join(scen_dir, "lines.csv"))
    include  = read_include_list(os.path.join(scen_dir, "include_list.csv"))
    scen_p   = read_scenario_prob(os.path.join(scen_dir, "scenario_prob.csv"))
    scen_i   = read_scenario_infra(os.path.join(scen_dir, "scenario_infra.csv"))
    props    = read_properties_general(os.path.join(scen_dir, "properties_general.csv"))

    domain = DomainData(stops_df, links_df, od_df, lines, include, scen_p, scen_i, props, dict(cfg_row))

    # All helpers are local to keep scope tight and intent clear.
    def build_node_indexing(stops_df):
        """Map external stop IDs (file) -> dense 0-based indices (model)."""
        node_ids = stops_df['id'].astype(int).tolist()
        node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        idx_to_node_id = node_ids[:]  # inverse as list for O(1) decode
        return node_id_to_idx, idx_to_node_id, len(node_ids)

    def build_arcs(links_df):
        """Duplicate undirected links into directed arcs; collect arc attributes."""
        idx_to_arc_uv = []
        len_list, tmin_list, tmax_list = [], [], []
        for _, r in links_df.iterrows():
            a, b = int(r['a']), int(r['b'])
            for (u, v) in ((a, b), (b, a)):
                idx_to_arc_uv.append((u, v))
                len_list.append(float(r['length']))
                tmin_list.append(float(r['t_min']))
                tmax_list.append(float(r['t_max']))
        arc_uv_to_idx = {uv: i for i, uv in enumerate(idx_to_arc_uv)}
        return (
            np.array(len_list),
            np.array(tmin_list),
            np.array(tmax_list),
            idx_to_arc_uv,
            arc_uv_to_idx,
            len(idx_to_arc_uv),
        )

    def build_coordinates(stops_df, node_id_to_idx, N):
        """Pack node coordinates into dense arrays aligned with indices."""
        coord_x = np.zeros(N)
        coord_y = np.zeros(N)
        for _, r in stops_df.iterrows():
            i = node_id_to_idx[int(r['id'])]
            coord_x[i] = float(r['x'])
            coord_y[i] = float(r['y'])
        return coord_x, coord_y

    def build_od_matrix(od_df, node_id_to_idx, N, zero_od_diagonal=False):
        """Build dense OD matrix D[i,j]."""
        D = np.zeros((N, N))
        for _, r in od_df.iterrows():
            i = node_id_to_idx[int(r['i'])]
            j = node_id_to_idx[int(r['j'])]
            D[i, j] = float(r['demand'])
        if zero_od_diagonal:
            np.fill_diagonal(D, 0.0)
        return D

    def build_scenarios(scen_prob_df):
        """Scenario probabilities and an id->index map."""
        scen_ids = scen_prob_df['id'].astype(int).tolist()
        p_s = scen_prob_df['prob'].astype(float).to_numpy()
        scen_id_to_idx = {sid: s for s, sid in enumerate(scen_ids)}
        return p_s, scen_id_to_idx

    def build_capacities(scen_infra_df, scen_id_to_idx, arc_uv_to_idx, E_dir, cap_std, symmetrise_infra=False):
        """Capacity per (scenario, arc), initialised with standard cap and overridden by scenario entries."""
        S = len(scen_id_to_idx)
        cap_sa = np.full((S, E_dir), int(cap_std), dtype=int)
        for _, r in scen_infra_df.iterrows():
            s = scen_id_to_idx[int(r['scenario'])]
            u, v, cap = int(r['u']), int(r['v']), int(r['cap'])
            if (u, v) in arc_uv_to_idx:
                cap_sa[s, arc_uv_to_idx[(u, v)]] = cap
            if symmetrise_infra and (v, u) in arc_uv_to_idx:
                cap_sa[s, arc_uv_to_idx[(v, u)]] = cap
        return cap_sa

    def build_lines(lines, node_id_to_idx, arc_uv_to_idx):
        """Expand line definitions into node/arc index sequences; group forward/backward."""
        line_idx_to_group = []
        line_group_to_lines = {}
        line_idx_to_stops = []
        line_idx_to_arcs = []

        # group lines by group id
        by_group = {}
        for ld in lines:
            by_group.setdefault(ld.group, []).append(ld)

        ell = 0
        for g, items in sorted(by_group.items()):
            # prefer +1, then -1
            dir_order = sorted(items, key=lambda t: -t.direction)
            ids_here = []
            for ld in dir_order:
                # nodes on this line (indices)
                stops_idx = [node_id_to_idx[sid] for sid in ld.stops]
                # arcs on this line (indices), must exist
                arcs_idx = []
                for u_id, v_id in zip(ld.stops[:-1], ld.stops[1:]):
                    if (u_id, v_id) not in arc_uv_to_idx:
                        raise ValueError(f"Line group {g} missing link {(u_id, v_id)}")
                    arcs_idx.append(arc_uv_to_idx[(u_id, v_id)])
                line_idx_to_group.append(g)
                line_idx_to_stops.append(stops_idx)
                line_idx_to_arcs.append(arcs_idx)
                ids_here.append(ell)
                ell += 1
            # map group -> (forward, backward|-1)
            line_group_to_lines[g] = (ids_here[0], ids_here[1] if len(ids_here) > 1 else -1)

        L = len(line_idx_to_group)
        return line_idx_to_group, line_group_to_lines, line_idx_to_stops, line_idx_to_arcs, L

    def build_adjacency(idx_to_arc_uv, node_id_to_idx, N):
        """Adjacency lists of arc indices for fast flow balance."""
        adj_out = [[] for _ in range(N)]
        adj_in = [[] for _ in range(N)]
        for a_idx, (u, v) in enumerate(idx_to_arc_uv):
            ui, vi = node_id_to_idx[u], node_id_to_idx[v]
            adj_out[ui].append(a_idx)
            adj_in[vi].append(a_idx)
        return adj_out, adj_in

    def build_incidence_matrices(line_idx_to_arcs, line_idx_to_stops, E_dir, L, N):
        """CSR incidence matrices: arc–line (E×L), node–line (N×L)."""
        # Edge-Line
        rows, cols, data = [], [], []
        for ell_idx, arc_list in enumerate(line_idx_to_arcs):
            for a in arc_list:
                rows.append(a); cols.append(ell_idx); data.append(1.0)
        A_edge_line = csr_matrix((data, (rows, cols)), shape=(E_dir, L))
        # Node-Line
        rows, cols, data = [], [], []
        for ell_idx, stops in enumerate(line_idx_to_stops):
            for i in stops:
                rows.append(i); cols.append(ell_idx); data.append(1.0)
        A_node_line = csr_matrix((data, (rows, cols)), shape=(N, L))
        return A_edge_line, A_node_line

    # ----- build in order -----
    node_id_to_idx, idx_to_node_id, N = build_node_indexing(domain.stops_df)
    len_a, t_min_a, t_max_a, idx_to_arc_uv, arc_uv_to_idx, E_dir = build_arcs(domain.links_df)
    coord_x, coord_y = build_coordinates(domain.stops_df, node_id_to_idx, N)
    D = build_od_matrix(domain.od_df, node_id_to_idx, N, zero_od_diagonal=zero_od_diagonal)
    p_s, scen_id_to_idx = build_scenarios(domain.scen_prob_df)
    cap_sa = build_capacities(domain.scen_infra_df, scen_id_to_idx, arc_uv_to_idx, E_dir,
                            cap_std=domain.props['infra_cap_std'],
                            symmetrise_infra=symmetrise_infra)
    (line_idx_to_group, line_group_to_lines,
    line_idx_to_stops, line_idx_to_arcs, L) = build_lines(domain.lines, node_id_to_idx, arc_uv_to_idx)
    adj_out, adj_in = build_adjacency(idx_to_arc_uv, node_id_to_idx, N)
    A_edge_line, A_node_line = build_incidence_matrices(line_idx_to_arcs, line_idx_to_stops, E_dir, L, N)

    # ----- assemble model -----
    model = ModelData(
        N=N, E_dir=E_dir, L=L, S=len(p_s),
        node_id_to_idx=node_id_to_idx, idx_to_node_id=idx_to_node_id,
        arc_uv_to_idx=arc_uv_to_idx, idx_to_arc_uv=idx_to_arc_uv,
        line_idx_to_group=line_idx_to_group, line_group_to_lines=line_group_to_lines,
        line_idx_to_stops=line_idx_to_stops, line_idx_to_arcs=line_idx_to_arcs,
        coord_x=coord_x, coord_y=coord_y,
        len_a=len_a, t_min_a=t_min_a, t_max_a=t_max_a,
        D=D, p_s=p_s, cap_sa=cap_sa,
        exclude_nodes=set(domain.props.get('exclude_nodes', set())),
        include_sets=domain.include_sets,
        adj_out=adj_out, adj_in=adj_in,
        A_edge_line=A_edge_line, A_node_line=A_node_line
    )
    return domain, model

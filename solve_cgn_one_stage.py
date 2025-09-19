# solve_cgn_one_stage.py
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from prepare_cgn import make_cgn
from optimisation import (
    add_passenger_capacity, add_infrastructure_capacity,
    build_obj_invehicle, build_obj_waiting, build_obj_operating,
    add_frequency_grouped, build_obj_bypass, build_obj_invehicle_with_overdemand
)
from solve_utils import (
    _freq_values_from_config, _routing_is_aggregated,
    _waiting_mode, _add_flows
)


# ---------- ONE STAGE (nominal, with global infra capacity) ----------

def solve_one_stage(domain, model, *, gurobi_params=None):
    """
    One-stage LPP-CGN:
      - Build a single-layer CGN (no scenario branching).
      - Decide route flows (aggregated by origin or full OD, per config).
      - Pick one frequency per line-group (on/off via group binary).
      - Enforce vehicle capacity and global infrastructure capacity.
      - Objective = travel time (+ optional overdemand hinge) + bypass + waiting + operating.

    Returns:
      m          : gurobipy.Model (solved)
      solution   : dict with status, objective, chosen frequencies, cost breakdown, runtime
      artifacts  : dict with CGN and flow variables for downstream logging
    """
    m = gp.Model("LPP_ONE_STAGE")

    # Build nominal CGN (single layer, variant=0, includes optional bypass arcs)
    cgn = make_cgn(model)

    # Config-derived switches
    freq_vals = _freq_values_from_config(domain)     # discrete frequency choices per group
    aggregated = _routing_is_aggregated(domain)      # True → one commodity per origin
    wait_freq = _waiting_mode(domain)                # True → half-headway ~ 0.5 * 1/f

    # ----------------------------- Flow variables -----------------------------
    # x0, arc_to_keys: flows and a mapping arc -> [keys] where keys are (o,d) or origin ids
    x0, arc_to_keys = _add_flows(m, model, cgn, aggregated)

    # ----------------------------- Frequencies -----------------------------
    # Group-coupled frequency selection:
    #   z0[g] ∈ {0,1}, pick-or-off; delta0[(ell,r)] mirrors group choice per line for waiting-time code
    #   f0_expr[ell] = group frequency chosen for the line's group
    #   h0_expr[ell] = 1 / f_ell (as linear expression via delta)
    z0, delta0, f0_expr, h0_expr = add_frequency_grouped(m, model, freq_vals)

    # ----------------------------- Capacities -----------------------------
    # Vehicle capacity on ride arcs: sum(flow on arc) ≤ Q * f_ell
    Q = int(domain.config.get("train_capacity", 200))
    add_passenger_capacity(m, model, cgn, x0, f0_expr, arc_to_keys, Q=Q)

    # Global infrastructure capacity per directed infra-arc: sum(f_ell on that arc) ≤ cap_std
    cap_std = int(domain.config.get("infrastructure_capacity", 10))
    add_infrastructure_capacity(m, model, f0_expr, cap_std=cap_std)

    # ----------------------------- Cost buckets -----------------------------
    # Travel time with optional overdemand hinge (τ, μ):
    #   time0_base_raw = Σ t_a * flow_a
    #   time0_over_raw = Σ t_a * s_a, s_a ≥ flow_a − τ * Q * f_ell
    # total = base + (μ−1) * over
    tau = float(domain.config.get("overdemand_threshold", 1.0))
    mu  = float(domain.config.get("overdemand_multiplier", 1.0))
    time0_base_raw, time0_over_raw = build_obj_invehicle_with_overdemand(
        m, model, cgn, x0, arc_to_keys, f0_expr, Q,
        threshold=tau, multiplier=mu, use_t_min_time=True
    )
    time0_total = time0_base_raw + max(mu - 1.0, 0.0) * time0_over_raw

    # Bypass cost (already multiplied by bypass_multiplier if enabled)
    bypass0 = build_obj_bypass(m, model, cgn, x0, arc_to_keys)

    # Waiting cost:
    #   - change arcs (and optionally board at origin) split to R frequency bins via delta
    #   - or flat penalty if waiting_time_frequency=False
    wait0, y0 = build_obj_waiting(
        m, model, cgn, x0, arc_to_keys, freq_vals, delta0,
        include_origin_wait=True,
        waiting_time_frequency=wait_freq
    )

    # Operating cost: Σ f_ell * line_length(ell)
    oper0, line_len = build_obj_operating(model, f0_expr)

    # ----------------------------- Objective -----------------------------
    time_w = float(domain.config.get("travel_time_cost_mult", 1.0))
    wait_w = float(domain.config.get("waiting_time_cost_mult", 1.0))
    op_w   = float(domain.config.get("line_operation_cost_mult", 1.0))
    m.setObjective(time_w * time0_total + bypass0 + wait_w * wait0 + op_w * oper0, GRB.MINIMIZE)

    # ----------------------------- Solver params -----------------------------
    if gurobi_params:
        for k, v in gurobi_params.items():
            setattr(m.Params, k, v)
    if "time_limit" in domain.config:
        m.Params.TimeLimit = int(domain.config["time_limit"])
    if "threads" in domain.config:
        m.Params.Threads = int(domain.config["threads"])
    if "seed" in domain.config:
        m.Params.Seed = int(domain.config["seed"])
    if "mip_gap" in domain.config:
        m.Params.MIPGap = float(domain.config["mip_gap"])
    elif "gap" in domain.config:
        m.Params.MIPGap = float(domain.config["gap"])

    m.optimize()

    # ----------------------------- Decode decisions -----------------------------
    # Per-line frequency (inherits its group selection via delta0[(ell,r)])
    chosen_freq0 = {}
    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        for ell in range(model.L):
            f = 0
            for r, _ in enumerate(freq_vals):
                if delta0[(ell, r)].X > 0.5:
                    f = freq_vals[r]
                    break
            chosen_freq0[ell] = f

    # ----------------------------- Cost breakdown (weighted + raw) -----------------------------
    costs0 = {}
    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
        v_time0_base_raw = float(time0_base_raw.getValue())
        v_time0_over_raw = float(time0_over_raw.getValue())
        v_time0_raw = v_time0_base_raw + max(mu - 1.0, 0.0) * v_time0_over_raw
        v_bypass0_raw = float(bypass0.getValue())
        v_wait0_raw = float(wait0.getValue())
        v_oper0_raw = float(oper0.getValue())

        v_time0      = time_w * v_time0_raw
        v_time0_base = time_w * v_time0_base_raw
        v_time0_over = time_w * max(mu - 1.0, 0.0) * v_time0_over_raw
        v_bypass0    = v_bypass0_raw
        v_wait0      = wait_w * v_wait0_raw
        v_oper0      = op_w   * v_oper0_raw

        obj0 = v_time0 + v_bypass0 + v_wait0 + v_oper0
        costs0 = dict(
            time=v_time0, time_base=v_time0_base, time_over=v_time0_over,
            bypass=v_bypass0,
            wait=v_wait0, oper=v_oper0, objective=obj0,
            # raw components (unweighted), useful for diagnostics
            time_raw=v_time0_raw, time_base_raw=v_time0_base_raw, time_over_raw=v_time0_over_raw,
            bypass_raw=v_bypass0_raw,
            wait_raw=v_wait0_raw, oper_raw=v_oper0_raw
        )

    # ----------------------------- Return payloads -----------------------------
    solution = dict(
        status_code=int(m.Status),
        status=m.Status,
        objective=costs0.get("objective"),
        runtime_s=getattr(m, "Runtime", None),
        chosen_freq=chosen_freq0,
        costs_0=costs0
    )

    artifacts = dict(
        # stage-1 artifacts (used by downstream logging)
        cgn_stage1=cgn,
        x_stage1=x0,
        arc_to_keys_stage1=arc_to_keys,
        # placeholders to keep a consistent shape vs. two-stage solvers
        cgn_stage2_list=[],
        x_stage2_list=[],
        arc_to_keys_stage2_list=[],
        line_len=line_len
    )

    return m, solution, artifacts

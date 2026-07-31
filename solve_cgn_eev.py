# solve_cgn_eev.py
# -*- coding: utf-8 -*-
"""
EEV — Expected result of Expected Value solution.

Standard two-step stochastic-programming benchmark:
  Step 1. Replace random parameters by their probability-weighted mean
          (here: per-arc infrastructure capacity).
  Step 2. Solve the resulting *deterministic* one-stage problem on the
          mean surrogate → first-stage plan x̄.
  Step 3. Evaluate x̄ honestly against the real scenario distribution
          by solving the same stage-2 recourse as the `separated`
          procedure, but anchored on x̄'s frequencies.

The deterministic surrogate differs from the nominal-baseline used by
`solve_two_stage_separated`: instead of full capacity on every arc,
restricted arcs already see a fractional mean capacity, so x̄ tends to
"pre-warn" the first stage without seeing the actual distribution.

Use cases:
  - Reporting VSS_mean = EEV − RP alongside VSS_nominal = Separated − RP.
  - Quantifying how much "soft anticipation" by mean capacities buys
    relative to ignoring disruptions entirely.

Caveat: with binary closures the mean capacity may be a "phantom" state
that no scenario actually realises. With graduated reductions (e.g.,
80/60/40/20/0 %) it is physically much more plausible.

solve_eev:
    solves a deterministic expected-value surrogate and evaluates recourse

solve_two_stage_eev:
    solves the expected-value version of the two-stage model and evaluates
    the resulting first-stage decision under the original stochastic model

"""

import math
from typing import Any, Dict, Optional, Tuple
from dataclasses import replace
import numpy as np
import pandas as pd

from data_model import (
    LineDef, DomainData, ModelData, Config
)
from solve_cgn_one_stage import solve_one_stage
from solve_cgn_separated import _solve_stage2_given_first_stage
from solve_cgn_integrated import solve_two_stage_integrated
from solve_utils import _mean_scenario_capacity



def build_expected_value_scenario_infra(domain: DomainData) -> pd.DataFrame:
    """
    Build expected-value infrastructure scenario.

    Missing edge entries in a scenario mean:
        no disruption -> capacity factor 1.0
    """

    infra = domain.scen_infra_df.copy()

    prob_map = dict(
        zip(
            domain.scen_prob_df["id"],
            domain.scen_prob_df["prob"],
        )
    )

    scenarios = domain.scen_prob_df["id"].tolist()

    # all edges that appear in any disruption scenario
    edges = infra[["u", "v"]].drop_duplicates()

    rows = []

    for _, edge in edges.iterrows():
        u = edge["u"]
        v = edge["v"]

        expected_factor = 0.0

        edge_rows = infra[
            (infra["u"] == u) &
            (infra["v"] == v)
        ]

        edge_by_scenario = dict(
            zip(
                edge_rows["scenario"],
                edge_rows["cap"]
            )
        )

        for s in scenarios:

            prob = float(prob_map[s])

            if s in edge_by_scenario:
                cap_value = edge_by_scenario[s]

                if isinstance(cap_value, str) and cap_value.startswith("*"):
                    factor = float(cap_value[1:])
                else:
                    factor = float(cap_value)

            else:
                # no entry means no disruption
                factor = 1.0

            expected_factor += prob * factor

        rows.append(
            {
                "scenario": 1,
                "u": u,
                "v": v,
                "cap": f"*{expected_factor:g}",
            }
        )

    return pd.DataFrame(rows)


def build_expected_value_data(domain: DomainData,model: ModelData,) -> Tuple[DomainData, ModelData]:
    """
    Create the deterministic expected-value version of a stochastic model.

    Replaces the scenario-dependent capacity matrix by its probability-weighted
    mean:
        cap_EV[a] = sum_s p_s * cap_sa[s,a]

    The resulting model has one scenario with probability 1.
    """

    cap_mean = np.average(
        model.cap_sa,
        axis=0,
        weights=model.p_s,
    )

    model_ev = replace(
        model,
        S=1,
        p_s=np.array([1.0]),
        cap_sa=cap_mean.reshape(1, -1),
    )

    scen_prob_ev = pd.DataFrame({
        "id": [1],
        "prob": [1.0],
    })

    domain_ev = replace(
        domain,
        scen_prob_df=scen_prob_ev,
        scen_infra_df=build_expected_value_scenario_infra(domain),
    )

    return domain_ev, model_ev


def solve_eev(domain, model, *, gurobi_params: Optional[Dict[str, Any]] = None):
    """
    Solve the EEV bound.

    Returns the same (m, solution, artifacts) shape as `solve_two_stage_separated`,
    so the downstream logger and KPI aggregation in `run.py` work unchanged.

    The first-stage objective stored in `solution["obj_stage1"]` is the cost of x̄
    measured on the *mean* surrogate (not on the nominal network), and the
    expected stage-2 component reflects the recourse against the real scenarios.
    The total objective is therefore comparable to RP and to `separated`'s total.
    """
    # ----- Step 1+2: Deterministic surrogate on probability-weighted mean capacities -----
    mean_cap = _mean_scenario_capacity(model)
    m0, sol0, art0 = solve_one_stage(
        domain, model,
        gurobi_params=gurobi_params,
        cap_per_arc_override=mean_cap,
    )

    # If the surrogate is itself infeasible, abort early without stage 2.
    # NOTE: `gap or inf` is wrong — Gurobi returns 0.0 on optimal solves, which
    # is falsy and would short-circuit to inf and trip the early-return.
    gap = sol0.get("opt_gap")
    no_solution = (m0 is not None and m0.SolCount == 0)
    if no_solution or gap is None or math.isinf(float(gap)):
        print("[WARN] EEV stage-1 (mean surrogate) found no solution, skipping stage-2.")
        return m0, sol0, art0

    # ----- Step 3: Honest evaluation under the true scenario distribution -----
    return _solve_stage2_given_first_stage(
        domain, model, sol0, art0, gurobi_params=gurobi_params
    )


def solve_two_stage_eev(domain, model,*,gurobi_params: Optional[Dict[str, Any]] = None):
    """
    Solve the two-stage EEV procedure.

    Step 1:
        Solve the expected-value (EV) two-stage problem, i.e second stage has 1 scenario E[xi]
            min w1*C1(x, nominal) + w2*C2(x, E[xi])

        This gives x_EV.

    Step 2:
        Fix x_EV and evaluate it under the true scenario distribution:
            EEV = w1*C1(x_EV, nominal)
                  + w2*E[C2(x_EV, xi)]

    Returns:
        Same (m, solution, artifacts) shape as other two-stage solvers.
    """


    # ----------------------------------------------------
    # Step 1: Solve EV problem
    # ----------------------------------------------------
    domain_ev, model_ev = build_expected_value_data(domain, model)


    m_ev, sol_ev, art_ev = solve_two_stage_integrated(
        domain_ev,
        model_ev,
        gurobi_params=gurobi_params,
    )

    # ----------------------------------------------------
    # Check EV solution exists before evaluating
    # ----------------------------------------------------
    gap = sol_ev.get("opt_gap")

    if (
            m_ev is None
            or m_ev.SolCount == 0
            or gap is None
            or math.isinf(float(gap))
    ):
        print("[WARN] Expected-value solve found no valid solution; skipping EEV evaluation.")
        return m_ev, sol_ev, art_ev

    # ----- Step 2: Evaluate x_EV under true stochastic scenarios -----
    return _solve_stage2_given_first_stage(
        domain,
        model,
        sol_ev,
        art_ev,
        gurobi_params=gurobi_params,
    )

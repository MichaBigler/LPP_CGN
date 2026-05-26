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
"""

import math
from typing import Any, Dict, Optional

from solve_cgn_one_stage import solve_one_stage
from solve_cgn_separated import _solve_stage2_given_first_stage
from solve_utils import _mean_scenario_capacity


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

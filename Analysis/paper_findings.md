# Paper Findings — Two-Stage Stochastic Line Planning under Disruptions

SiouxFalls (SF) vs Mumford0. All findings below are from the **matched
program**: both networks run at identical operational parameters and the same
load factor ρ=0.10, so every within- and cross-network comparison is
controlled. Metric = relative value (VSS/RP or EVPI/RP); aggregates are the
**mean over cases** (the median is misleading where the VSS distribution is
bimodal, e.g. Mumford0 at low k). Status: 2026-06-11.

## Setup

- **Bounds:** RP (recourse/integrated two-stage), WS (wait-and-see),
  EEV_nom (expected-value-of-nominal-plan). EVPI = RP − WS, VSS = EEV_nom − RP.
- **Solver:** Gurobi 12.0.3, gap 1e-5, time_limit 12000 s/procedure. WS
  sub-solves pinned to Threads=1/Seed=42/NumericFocus=2 (a parallel-cut
  artefact had produced WS > RP; 0 violations after the fix).
- **Networks:** SF — 24 nodes / 38 edges / density 13.8 %, hierarchical;
  Mumford0 — 30 nodes / 90 edges / density 20.7 %, dense/grid-like.
- **Matched setup:** train_capacity=50, max_frequency=10,
  infrastructure_capacity=10, bypass_multiplier=50, num_od=50 for **both**
  networks; load factor ρ=0.10 (passenger-km / seat-km at max frequency) set
  by demand scaling (SF ×2.787, Mumford0 ×0.0368). Only remaining differences:
  topology and relative demand pattern. ~5025 cases total.

---

# Robust findings (matched, controlled)

## R1 — Replanning cost is the dominant lever, in both networks *(headline)*

VSS/RP mean (k=1), varying frequency-replanning cost:

| cost_repl_freq | 10 | 100 | 1000 |
|---|---:|---:|---:|
| SF | 0.02 % | 1.65 % | **19.13 %** |
| Mumford0 | 0.01 % | 1.50 % | **9.43 %** |

Spread ≈ 19 pp (SF) / 9 pp (Mumford0) — orders of magnitude larger than any
other lever (bypass ≈ 0.2–0.3 pp, overdemand ≈ 0.03–0.10 pp). **When
replanning is expensive, anticipating disruptions is worth a large fraction of
operating cost; when it is cheap, stochastic planning adds almost nothing.**
The effect is present in both topologies (stronger in the sparse SF).

## R2 — Disruption probability has a threshold effect, universal across topology

VSS/RP and EVPI/RP mean (k=1) vs failure probability:

| p_fail | ≤ 0.8 | 0.90 | 0.95 |
|---|---|---|---|
| SF — VSS / EVPI | ~0.4 / ~0.7 % | 1.68 / 0.22 % | 1.87 / 0.11 % |
| Mumford0 — VSS / EVPI | ~0.4 / ~0.4 % | 1.54 / 0.15 % | 1.77 / 0.08 % |

Flat over a wide range (p ≤ 0.8), then a sharp split at p ≥ 0.9: VSS jumps
(the nominal-day plan is wrong for an almost-certain disruption) while EVPI
collapses (RP and WS both converge to "always plan for failure"). **Same
pattern in both networks** — topology-independent.

## R3 — Load factor governs the value, more than topology

From the matched load-factor curve (ρ ∈ {0.05, 0.10, 0.15}, mean over k):
the value of stochastic planning **declines with load** in both networks
(SF 1.32 → 0.30 %, Mumford0 0.78 → 0.39 % VSS/RP). Stochastic planning pays
off most in lightly-loaded networks; under congestion, deterministic bypass
overflow dominates and no plan choice helps. SF declines more steeply (it
reaches its congestion onset earlier).

## R4 — Spatial failure pattern does not systematically matter

At matched conditions, selection mode (line_consecutive / line_all /
share_stop / random) produces no consistent ordering of VSS — all ~0.4–0.7 %
in both networks. It is the *probability* and *cost* structure (R1, R2), not
the *spatial pattern*, that drives the value.

## R5 — Topology does NOT produce a level difference in VSS *(key methodological result)*

At matched ρ=0.10, the VSS/RP k-profile is essentially identical:

| k | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| SF | 0.47 % | 0.60 % | 0.36 % | 0.53 % |
| Mumford0 | 0.46 % | 0.74 % | 0.44 % | 0.67 % |

Within ~1.2×, with Mumford0 often higher. **The dense and the sparse network
derive comparable value from stochastic planning once load and parameters are
equalized.** What a naïve native comparison reports as a ~30× topology effect
(see "Artefacts" below) is entirely explained by the two networks operating at
different load factors and capacities.

---

# Artefacts removed by the matched design

These appeared in the native-parameter sweeps and do **not** survive matching:

- **"~30× topology gap" (native):** SF native VSS/RP ≈ 0.86 %, Mumford0 ≈
  0.02 %. Pure load/parameter confound — SF ran at ρ=0.039, Mumford0 at ρ=0.27
  with 5× larger trains. At matched ρ the gap vanishes (R5). See
  `Analysis/matched/topology_collapse.png`.
- **"VSS → 0 collapse under scenario multiplicity" (native):** at SF's native
  light load, per_run = 3,5 drove VSS to exactly 0. At matched ρ=0.10 there is
  no collapse (SF VSS stays 0.29–0.50 % across per_run = 1…8). A light-load
  phenomenon, not a general property.

**Methodological takeaway for the paper:** cross-network comparisons of
stochastic-programming value are dominated by the operating regime (load
factor, capacity, replanning cost). Matching the load factor is essential;
without it, regime differences masquerade as topology effects.

---

# Measurement note — VSS vs EVPI

VSS (vs the deterministic mean-plan) and EVPI (vs perfect information) respond
differently and should be reported separately: near-certain disruption
(R2) makes VSS large but EVPI small. They are not interchangeable summaries of
"the value of stochastic planning".

---

# Figures

- `Analysis/matched/replcost_lever.png` — R1
- `Analysis/matched/pfail_threshold.png` — R2
- `Analysis/matched/topology_collapse.png` — R5 (+ native overlay)
- `Analysis/loadmatch/matched_curve_3point.png` — R3

# Open items

- Optional robustness: full matched program also at ρ=0.05 (compute budget is
  not a constraint — fully subsidised to CHF 1000/yr; ~CHF 90 used so far).
- Sensitivities currently k=1,2; extend to k=3,4 if a k-dependence of the
  levers is of interest.

# Paper Findings — Two-Stage Stochastic Line Planning under Disruptions

Consolidated interpretation of the SiouxFalls (SF) vs Mumford0 study.
All numbers are medians of the relative metric (value / RP) unless noted.
Status date: 2026-06-10.

## Setup

- **Bounds:** RP = recourse problem (integrated two-stage), WS = wait-and-see,
  EEV_nom = expected-value-of-nominal-plan, EEV_mean = mean-capacity surrogate.
  EVPI = RP − WS (value of perfect information), VSS = EEV − RP (value of the
  stochastic solution over the deterministic mean-plan).
- **Solver:** Gurobi 12.0.3, gap target 1e-5, time_limit 6000 s/procedure.
  WS sub-solves pinned to `Threads=1, Seed=42, NumericFocus=2` after a
  parallel-cut artefact produced WS > RP in ~12 % of Mumford0 cases
  (now 0 violations across 4108 cases).
- **Networks:** SF — 24 nodes, 38 edges, density 13.8 %, hierarchical/radial;
  Mumford0 — 30 nodes, 90 edges, density 20.7 %, dense/grid-like.

## Finding 1 — Native comparison: SF shows far higher VSS *(superseded, see Finding 8)*

Across all 6 main + sensitivity sweeps (native parameters):

| | n | EVPI/RP median | VSS/RP median | VSS/RP P95 |
|---|---:|---:|---:|---:|
| SF | 2198 | 0.367 % | 0.857 % | 5.91 % |
| Mumford0 | 1910 | 0.051 % | 0.022 % | 0.36 % |

Naively ~7× (EVPI) / ~30× (VSS) higher for SF. **This gap is largely a
parameter/load-regime artefact — see Finding 8.** Do not report the 30× as a
topology effect.

## Finding 2 — VSS and EVPI diverge under scenario multiplicity *(robust)*

SF, k=1 line_consecutive, increasing failure scenarios per case:

| per_run | VSS/RP | EVPI/RP |
|---:|---:|---:|
| 1 | 0.71 % | 0.75 % |
| 3 | 0.00 % | 1.55 % |
| 5 | 0.00 % | 1.61 % |

When failure mass spreads over many scenarios, no single one dominates → the
nominal-day plan *is* the right first-stage decision → VSS collapses to 0.
EVPI keeps rising because perfect information still permits per-scenario
adaptation. The two measures are not interchangeable.

## Finding 3 — p_fail acts as a threshold, not monotonically *(robust)*

SF, single-edge failure. Flat across p_fail = 0.05–0.8 (VSS ≈ 0.74 %,
EVPI ≈ 0.75 %), then at p_fail ≥ 0.9 they split sharply:

| p_fail | 0.05–0.8 | 0.90 | 0.95 |
|---|---|---|---|
| VSS/RP | 0.74 % | 3.23 % | 3.64 % |
| EVPI/RP | 0.75 % | 0.14 % | 0.07 % |

Near certainty, the nominal plan is wrong for an almost-never-occurring day
(VSS explodes), while RP and WS both converge to "always plan for failure"
(EVPI collapses). Reinforces Finding 2.

## Finding 4 — Replanning cost is the dominant operational lever *(robust)*

Spread of median VSS/RP across each sensitivity axis (percentage points):

| | replcost | p_fail | traincap | bypass | overdemand |
|---|---:|---:|---:|---:|---:|
| SF | 3.63 | 2.90 | 0.94 | 0.57 | 0.30 |
| Mumford0 | 1.00 | 0.15 | 0.04 | 0.04 | 0.02 |

Expensive frequency replanning makes anticipation most valuable, in both
networks.

## Finding 5 — k-trend *(solid for k=1–4)*

Native main map, VSS/RP median by disruption size k:

- SF: ~1 % flat for k=1–4 (k=5,6 noisy, n=11/4).
- Mumford0: constant ~0.03 % across all k.

## Finding 6 — Spatial failure pattern has no systematic effect *(negative result)*

SF, per_run=1, VSS/RP by selection mode × k — no monotone ordering
(`random` is highest at k=2). It is the *probability structure* of the
disruption (Findings 2/3), not its *spatial pattern*, that drives VSS.

## Finding 7 — In SF, load and vulnerability co-locate on hub edges *(belegt)*

Shortest-path edge-load concentration: SF peak/mean 2.65, busiest edges are
the hub corridors 2-6 / 10-16 / 11-14 — the same edges with the highest VSS.
NB: a previously suspected "Mumford0 congests at higher load" claim is
**confounded** (the load-test used unequal native parameters) and is dropped.
Static structural metrics do not support it: Mumford0 actually has higher
load concentration (peak/mean 4.76) and sparser line coverage (0.91 vs 3.42
lines/edge).

## Finding 8 — Load-matched: the topology gap collapses *(key result)*

Both networks run at **identical operational parameters** (train_capacity=50,
max_frequency=10, infrastructure_capacity=10, bypass_multiplier=50) and the
**same load factor ρ=0.10** (passenger-km / seat-km at max frequency),
achieved by demand scaling (SF ×2.54, Mumford0 ×0.037). The only remaining
differences are topology and relative demand pattern.

| k | SF VSS/RP | Mumford0 VSS/RP | SF EVPI/RP | Mumford0 EVPI/RP |
|---:|---:|---:|---:|---:|
| 1 | 0.46 % | 0.00 % | 0.55 % | 0.00 % |
| 2 | 0.85 % | 0.13 % | 1.02 % | 0.53 % |
| 3 | 0.83 % | 1.00 % | 1.18 % | 0.88 % |
| 4 | 0.90 % | 1.00 % | 1.34 % | 1.34 % |

The native ~30× gap **disappears**. Under equal conditions the networks are
comparable (~0.5–1.3 % VSS). The residual topology signal is qualitative, not
a level shift:

- **Small disruptions (k=1,2):** SF higher — the sparse/hierarchical network
  is already sensitive to single-edge failures.
- **Large disruptions (k=3,4):** Mumford0 catches up / exceeds — the dense
  network only benefits from anticipation once several edges fail.

**Revised topology narrative:** topology does not change the *level* of the
value of stochastic planning; it changes its *dependence on disruption size*.
In the sparse network even a single failure is worth anticipating; in the
dense network only larger multi-edge failures are.

Caveat: ρ=0.10 is SF's congestion-onset regime (hard MIPs; 7 of 139 hardest
high-k SF tasks did not finish within walltime). The picture is stable but a
second matched load factor (ρ=0.05, 0.15) is being run to confirm robustness.

## Open items

- Confirm Finding 8 at additional matched load factors (ρ=0.05, 0.15).
- Finalize SF ρ=0.10 high-k tasks (longer walltime).
- Optional: third network (rejected — Mumford1 too large/unrealistic).

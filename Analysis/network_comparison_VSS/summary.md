# Network comparison: SiouxFalls vs Mumford0

Primary metric: **VSS_nom**  (VSS_nom = EEV_nom − RP)

## Headline (across all sweeps)

| | SiouxFalls | Mumford0 |
|---|---:|---:|
| cases | 2198 | 1910 |
| median VSS_nom / RP | **0.857%** | **0.022%** |
| P95 VSS_nom / RP    | 5.912% | 0.360% |
| cases with VSS_nom<0 (solver noise) | 0 | 11 |

**Structural finding:** SF realises VSS_nom of ~0.86% of nominal cost on average, vs 0.022% for Mumford0 (~38.5x ratio). Sparse hierarchical networks (SF) benefit substantially more from explicit stochastic / perfect-info planning; dense redundant networks (Mumford0) already carry built-in robustness via alternative line paths.

## Sweep inventory

| slug | rows | source |
|---|---:|---|
| sf_main | 544 | `Results\vss_sf_main_redo\vss_map.csv` |
| sf_bypass | 400 | `Results\vss_sf_bypass_redo\vss_map.csv` |
| sf_overdemand | 342 | `Results\vss_sf_overdemand_redo\vss_map.csv` |
| sf_pfail | 418 | `Results\vss_sf_pfail_redo\vss_map.csv` |
| sf_replcost | 342 | `Results\vss_sf_replcost_redo\vss_map.csv` |
| sf_traincap | 152 | `Results\vss_sf_traincap_redo\vss_map.csv` |
| mumford0 | 402 | `Results\vss_mumford0_redo\vss_map.csv` |
| mumford0_bypass | 320 | `Results\vss_mumford0_bypass_redo\vss_map.csv` |
| mumford0_overdemand | 324 | `Results\vss_mumford0_overdemand_redo\vss_map.csv` |
| mumford0_pfail | 396 | `Results\vss_mumford0_pfail_redo\vss_map.csv` |
| mumford0_replcost | 324 | `Results\vss_mumford0_replcost_redo\vss_map.csv` |
| mumford0_traincap | 144 | `Results\vss_mumford0_traincap_redo\vss_map.csv` |

## Main map — VSS_nom by case_k (absolute)

| network | k | n | median | mean | P75 | P95 | max | n<sub><0</sub> |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SF | 1 | 194 | 529 | 971 | 1.4k | 3.7k | 4.7k | 0 |
| SF | 2 | 251 | 564 | 832 | 1.3k | 2.7k | 3.8k | 0 |
| SF | 3 | 61 | 435 | 521 | 884 | 1.3k | 1.8k | 0 |
| SF | 4 | 23 | 564 | 648 | 1.0k | 1.7k | 1.8k | 0 |
| SF | 5 | 11 | 46 | 323 | 375 | 1.2k | 1.4k | 0 |
| SF | 6 | 4 | 281 | 499 | 654 | 1.2k | 1.3k | 0 |
| Mumford0 | 1 | 188 | 1.4k | 1.5k | 2.0k | 3.6k | 5.0k | 0 |
| Mumford0 | 2 | 184 | 2.0k | 3.9k | 6.9k | 10.7k | 22.6k | 1 |
| Mumford0 | 3 | 19 | 1.6k | 1.6k | 2.1k | 2.7k | 3.6k | 0 |
| Mumford0 | 4 | 11 | 1.6k | 1.4k | 1.6k | 1.8k | 1.9k | 0 |

## Main map — VSS_nom / RP by case_k (relative, % of nominal cost)

| network | k | n | median VSS_nom/RP | P95 VSS_nom/RP |
|---|---|---:|---:|---:|
| SF | 1 | 194 | 0.965% | 6.400% |
| SF | 2 | 251 | 1.010% | 4.497% |
| SF | 3 | 61 | 0.766% | 2.390% |
| SF | 4 | 23 | 0.969% | 3.105% |
| SF | 5 | 11 | 0.089% | 2.348% |
| SF | 6 | 4 | 0.505% | 2.078% |
| Mumford0 | 1 | 188 | 0.023% | 0.067% |
| Mumford0 | 2 | 184 | 0.035% | 0.226% |
| Mumford0 | 3 | 19 | 0.027% | 0.050% |
| Mumford0 | 4 | 11 | 0.029% | 0.034% |

**Observation:** SF median VSS_nom = 529 vs Mumford0 median = 1.6k. SF P95 = 2.9k vs Mumford0 P95 = 9.2k.

**Solver noise check:** SF has 0 cases with negative VSS_nom, Mumford0 has 1. (Should be 0 if WS sub-solves are correct.)

## Top-10 highest VSS_nom per network (main map)

### SF

| k | edges | VSS_nom |
|---|---|---:|
| 1 | `23-24` | 4.7k |
| 1 | `14-23` | 4.7k |
| 1 | `13-24` | 4.6k |
| 1 | `15-22` | 4.5k |
| 1 | `10-15` | 4.4k |
| 1 | `23-24` | 4.3k |
| 1 | `14-23` | 4.3k |
| 1 | `13-24` | 4.1k |
| 1 | `10-15` | 4.1k |
| 1 | `15-22` | 3.9k |

### Mumford0

| k | edges | VSS_nom |
|---|---|---:|
| 2 | `6-7,19-20` | 22.6k |
| 2 | `14-19,19-20` | 19.8k |
| 2 | `6-16,13-18` | 15.9k |
| 2 | `11-30,15-25` | 15.8k |
| 2 | `9-13,28-30` | 14.4k |
| 2 | `2-25,11-22` | 14.3k |
| 2 | `5-25,21-24` | 11.4k |
| 2 | `6-7,7-17` | 11.3k |
| 2 | `6-7,14-19` | 11.1k |
| 2 | `12-15,15-24` | 10.8k |

## Sensitivity sweeps — median VSS_nom

| sweep | parameter | network | values | median range |
|---|---|---|---|---|
| sf_bypass | bypass_multiplier | SF | 5 pts | 408 … 783 |
| sf_overdemand | overdemand_multiplier | SF | 3 pts | 589 … 672 |
| sf_pfail | case_p_fail | SF | 11 pts | 413 … 2.0k |
| sf_replcost | cost_repl_freq | SF | 3 pts | 0 … 2.0k |
| sf_traincap | train_capacity | SF | 4 pts | 34 … 695 |
| mumford0_bypass | bypass_multiplier | Mumford0 | 5 pts | 470 … 1.6k |
| mumford0_overdemand | overdemand_multiplier | Mumford0 | 3 pts | 0 … 1.0k |
| mumford0_pfail | case_p_fail | Mumford0 | 11 pts | 1.5k … 8.4k |
| mumford0_replcost | cost_repl_freq | Mumford0 | 3 pts | 310 … 55.2k |
| mumford0_traincap | train_capacity | Mumford0 | 4 pts | 820 … 1.9k |

---

See `main_map_vss.png` and `sensitivity.png` in this directory.

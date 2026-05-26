# Network comparison: SiouxFalls vs Mumford0

Primary metric: **EVPI**  (EVPI = RP − WS)

## Headline (across all sweeps)

| | SiouxFalls | Mumford0 |
|---|---:|---:|
| cases | 2198 | 1910 |
| median EVPI / RP | **0.367%** | **0.051%** |
| P95 EVPI / RP    | 2.649% | 0.142% |
| cases with EVPI<0 (solver noise) | 0 | 0 |

**Structural finding:** SF realises EVPI of ~0.37% of nominal cost on average, vs 0.051% for Mumford0 (~7.1x ratio). Sparse hierarchical networks (SF) benefit substantially more from explicit stochastic / perfect-info planning; dense redundant networks (Mumford0) already carry built-in robustness via alternative line paths.

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

## Main map — EVPI by case_k (absolute)

| network | k | n | median | mean | P75 | P95 | max | n<sub><0</sub> |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SF | 1 | 194 | 167 | 364 | 630 | 1.1k | 1.7k | 0 |
| SF | 2 | 251 | 529 | 595 | 915 | 1.6k | 3.7k | 0 |
| SF | 3 | 61 | 526 | 515 | 908 | 1.3k | 1.7k | 0 |
| SF | 4 | 23 | 579 | 579 | 910 | 1.3k | 1.7k | 0 |
| SF | 5 | 11 | 131 | 205 | 319 | 612 | 633 | 0 |
| SF | 6 | 4 | 611 | 658 | 1.1k | 1.2k | 1.2k | 0 |
| Mumford0 | 1 | 188 | 3.0k | 3.0k | 3.8k | 5.0k | 10.4k | 0 |
| Mumford0 | 2 | 184 | 1.6k | 2.1k | 3.3k | 4.6k | 10.4k | 0 |
| Mumford0 | 3 | 19 | 3.4k | 3.5k | 4.3k | 5.7k | 10.4k | 0 |
| Mumford0 | 4 | 11 | 2.7k | 2.7k | 3.1k | 4.0k | 4.8k | 0 |

## Main map — EVPI / RP by case_k (relative, % of nominal cost)

| network | k | n | median EVPI/RP | P95 EVPI/RP |
|---|---|---:|---:|---:|
| SF | 1 | 194 | 0.325% | 1.972% |
| SF | 2 | 251 | 0.980% | 2.643% |
| SF | 3 | 61 | 0.893% | 2.200% |
| SF | 4 | 23 | 1.028% | 2.163% |
| SF | 5 | 11 | 0.255% | 1.168% |
| SF | 6 | 4 | 1.091% | 2.138% |
| Mumford0 | 1 | 188 | 0.056% | 0.093% |
| Mumford0 | 2 | 184 | 0.031% | 0.082% |
| Mumford0 | 3 | 19 | 0.062% | 0.101% |
| Mumford0 | 4 | 11 | 0.051% | 0.067% |

**Observation:** SF median EVPI = 319 vs Mumford0 median = 2.8k. SF P95 = 1.4k vs Mumford0 P95 = 4.8k.

## Top-10 highest EVPI per network (main map)

### SF

| k | edges | EVPI |
|---|---|---:|
| 2 | `2-6,3-4` | 3.7k |
| 2 | `2-6,16-17` | 3.2k |
| 2 | `2-6,17-19` | 2.9k |
| 2 | `11-12,19-20` | 2.5k |
| 2 | `6-8,11-12` | 2.4k |
| 2 | `2-6,6-8` | 2.1k |
| 2 | `4-11,23-24` | 1.9k |
| 1 | `2-6` | 1.7k |
| 2 | `6-8,8-16` | 1.7k |
| 3 | `10-11,10-16,11-12` | 1.7k |

### Mumford0

| k | edges | EVPI |
|---|---|---:|
| 1 | `4-12` | 10.4k |
| 2 | `12-18,18-23` | 10.4k |
| 3 | `2-15,8-15,8-17` | 10.4k |
| 1 | `9-27|21-24` | 10.4k |
| 1 | `12-15|4-12|8-21` | 10.4k |
| 2 | `1-23,18-23|17-28,17-29` | 10.4k |
| 1 | `18-23` | 10.4k |
| 1 | `18-23` | 10.4k |
| 2 | `1-14,1-23` | 10.4k |
| 1 | `7-17` | 5.2k |

## Sensitivity sweeps — median EVPI

| sweep | parameter | network | values | median range |
|---|---|---|---|---|
| sf_bypass | bypass_multiplier | SF | 5 pts | 131 … 526 |
| sf_overdemand | overdemand_multiplier | SF | 3 pts | 178 … 313 |
| sf_pfail | case_p_fail | SF | 11 pts | 42 … 423 |
| sf_replcost | cost_repl_freq | SF | 3 pts | 39 … 318 |
| sf_traincap | train_capacity | SF | 4 pts | 74 … 691 |
| mumford0_bypass | bypass_multiplier | Mumford0 | 5 pts | 3.0k … 3.3k |
| mumford0_overdemand | overdemand_multiplier | Mumford0 | 3 pts | 3.0k … 3.2k |
| mumford0_pfail | case_p_fail | Mumford0 | 11 pts | 490 … 3.1k |
| mumford0_replcost | cost_repl_freq | Mumford0 | 3 pts | 250 … 4.4k |
| mumford0_traincap | train_capacity | Mumford0 | 4 pts | 580 … 4.3k |

---

See `main_map_vss.png` and `sensitivity.png` in this directory.

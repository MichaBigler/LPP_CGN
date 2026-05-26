# Network comparison: SiouxFalls vs Mumford0

Primary metric: **VSS_nom**  (VSS_nom = EEV_nom − RP)

## Sweep inventory

| slug | rows | source |
|---|---:|---|
| sf_main | 544 | `Results\vss_544_complete\vss_map.csv` |
| sf_bypass | 400 | `Results\vss_bypass\vss_map.csv` |
| sf_overdemand | 342 | `Results\vss_overdemand\vss_map.csv` |
| sf_pfail | 418 | `Results\vss_pfail\vss_map.csv` |
| sf_replcost | 342 | `Results\vss_replcost\vss_map.csv` |
| sf_traincap | 152 | `Results\vss_traincap\vss_map.csv` |
| mumford0 | 402 | `Results\vss_mumford0\vss_map.csv` |
| mumford0_bypass | 320 | `Results\vss_mumford0_bypass\vss_map.csv` |
| mumford0_overdemand | 324 | `Results\vss_mumford0_overdemand\vss_map.csv` |
| mumford0_pfail | 396 | `Results\vss_mumford0_pfail\vss_map.csv` |
| mumford0_replcost | 324 | `Results\vss_mumford0_replcost\vss_map.csv` |
| mumford0_traincap | 144 | `Results\vss_mumford0_traincap\vss_map.csv` |

## Main map — VSS_nom by case_k

| network | k | n | median | mean | P75 | P95 | max | n<sub><0</sub> |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SF | 1 | 194 | 125 | 400 | 506 | 1.8k | 2.9k | 0 |
| SF | 2 | 245 | 420 | 734 | 1.2k | 2.7k | 3.6k | 2 |
| SF | 3 | 58 | 329 | 644 | 1.1k | 2.0k | 2.6k | 1 |
| SF | 4 | 22 | 842 | 1.1k | 1.8k | 2.4k | 2.7k | 0 |
| SF | 5 | 10 | 1.0k | 1.2k | 1.7k | 2.7k | 2.7k | 0 |
| SF | 6 | 4 | 1.4k | 1.5k | 1.8k | 2.5k | 2.7k | 0 |
| Mumford0 | 1 | 188 | 1.6k | 2.0k | 2.6k | 5.0k | 9.1k | 1 |
| Mumford0 | 2 | 184 | 4.2k | 4.7k | 7.5k | 11.3k | 22.6k | 3 |
| Mumford0 | 3 | 19 | 1.6k | 1.6k | 2.1k | 2.7k | 3.6k | 0 |
| Mumford0 | 4 | 11 | 1.6k | 1.4k | 1.6k | 1.8k | 1.9k | 0 |

**Observation:** SF median VSS_nom = 312 vs Mumford0 median = 2.0k. SF P95 = 2.3k vs Mumford0 P95 = 9.7k.

**Solver noise check:** SF has 3 cases with negative VSS_nom, Mumford0 has 4. (Should be 0 if WS sub-solves are correct.)

## Top-10 highest VSS_nom per network (main map)

### SF

| k | edges | VSS_nom |
|---|---|---:|
| 2 | `8-16,10-16` | 3.6k |
| 2 | `10-16,16-18` | 3.4k |
| 2 | `6-8,8-16` | 3.4k |
| 2 | `2-6,6-8` | 3.2k |
| 2 | `9-10,10-11` | 3.2k |
| 2 | `8-16,16-17` | 3.1k |
| 2 | `10-11,11-14` | 3.0k |
| 2 | `4-11,11-14` | 3.0k |
| 2 | `11-14,14-23` | 3.0k |
| 1 | `10-16` | 2.9k |

### Mumford0

| k | edges | VSS_nom |
|---|---|---:|
| 2 | `6-7,19-20` | 22.6k |
| 2 | `14-19,19-20` | 19.8k |
| 2 | `6-16,13-18` | 15.9k |
| 2 | `11-30,15-25` | 15.8k |
| 2 | `9-13,28-30` | 14.4k |
| 2 | `1-14,1-23` | 14.4k |
| 2 | `2-25,11-22` | 14.3k |
| 2 | `12-15,12-18` | 11.6k |
| 2 | `5-25,21-24` | 11.4k |
| 2 | `6-7,7-17` | 11.3k |

## Sensitivity sweeps — median VSS_nom

| sweep | parameter | network | values | median range |
|---|---|---|---|---|
| sf_bypass | bypass_multiplier | SF | 5 pts | 408 … 642 |
| sf_overdemand | overdemand_multiplier | SF | 3 pts | 566 … 794 |
| sf_pfail | case_p_fail | SF | 11 pts | 187 … 2.0k |
| sf_replcost | cost_repl_freq | SF | 3 pts | 0 … 3.3k |
| sf_traincap | train_capacity | SF | 4 pts | 34 … 695 |
| mumford0_bypass | bypass_multiplier | Mumford0 | 5 pts | 688 … 2.0k |
| mumford0_overdemand | overdemand_multiplier | Mumford0 | 3 pts | 0 … 1.9k |
| mumford0_pfail | case_p_fail | Mumford0 | 11 pts | 888 … 8.4k |
| mumford0_replcost | cost_repl_freq | Mumford0 | 3 pts | 310 … 79.8k |
| mumford0_traincap | train_capacity | Mumford0 | 4 pts | 820 … 1.9k |

---

See `main_map_vss.png` and `sensitivity.png` in this directory.

#!/usr/bin/env python3
# scripts/gen_matched_configs.py
# -*- coding: utf-8 -*-
"""
Generate the full *matched* experiment program for SF vs Mumford0.

All sweeps run at IDENTICAL operational parameters and the SAME load factor
rho=0.10 (achieved by demand scaling), so every within- and cross-network
comparison is controlled. The native-parameter sweeps are superseded by these.

Matched setup (both networks):
  train_capacity=50, max_frequency=10, infrastructure_capacity=10,
  bypass_multiplier=50 (baseline), num_od=50, gap=1e-5, time_limit=12000.
  demand_scale: SF=2.7866, Mumford0=0.0368  (both -> rho=0.10 at num_od=50)

Sweeps emitted per network:
  main      : k=1..4 consecutive (headline) + per_run {2,3,5,8} + selection
              modes {line_all, share_stop, random} at k=1..3
  bypass    : bypass_multiplier {10,20,50,100,200} x k {1,2}
  overdemand: threshold {0.5,0.75,1.0} x multiplier {1,2,5} x k {1,2}
  pfail     : case_p_fail {0.05..0.95} x k {1,2}
  replcost  : cost_repl_freq {10,100,1000} x cost_repl_line {50,500,5000} x k {1,2}

(traincap is intentionally dropped: varying train_capacity changes rho, i.e.
it IS the load-factor curve, covered separately by the rho=0.05/0.10/0.15 runs.)

Writes Data/config_matched_<sweep>_<net>.csv. Run from repo root:
    python scripts/gen_matched_configs.py
"""
from __future__ import annotations
import csv, os

COLS = ["source","network","scenario_line_data","procedure","routing_agg",
        "waiting_time_frequency","gap","time_limit","travel_time_cost_mult",
        "waiting_time_cost_mult","line_operation_cost_mult","cost_repl_freq",
        "cost_repl_line","repl_budget","bypass_multiplier","overdemand_threshold",
        "overdemand_multiplier","num_od","train_capacity","infrastructure_capacity",
        "max_frequency","threads","case_k","case_per_run","case_variants",
        "case_selection","case_p_fail","demand_scale"]

NETS = {
    "sf":       dict(network="SiouxFalls", demand_scale=2.7866),
    "mumford0": dict(network="Mumford0",   demand_scale=0.0368),
}

def base(net_key):
    n = NETS[net_key]
    return dict(
        source="lintim", network=n["network"], scenario_line_data=n["network"],
        procedure="bounds", routing_agg="TRUE", waiting_time_frequency="False",
        gap="0.00001", time_limit=12000, travel_time_cost_mult=1,
        waiting_time_cost_mult=2, line_operation_cost_mult=20,
        cost_repl_freq=50, cost_repl_line=200, repl_budget=-5,
        bypass_multiplier=50, overdemand_threshold=0.75, overdemand_multiplier=2,
        num_od=50, train_capacity=50, infrastructure_capacity=10, max_frequency=10,
        threads=16, case_k=1, case_per_run=1, case_variants=30,
        case_selection="line_consecutive", case_p_fail=0.5,
        demand_scale=n["demand_scale"],
    )

def row(net_key, **over):
    r = base(net_key); r.update(over); return r

def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path} ({len(rows)} rows)")

def sweep_main(nk):
    rows = []
    # headline k-sweep (deeper variants)
    for k, v in [(1,60),(2,60),(3,40),(4,30)]:
        rows.append(row(nk, case_k=k, case_variants=v))
    # scenario-count (per_run) at k=1
    for pr in [2,3,5,8]:
        rows.append(row(nk, case_k=1, case_per_run=pr, case_variants=30))
    # selection modes at k=1..3 (line_consecutive already in k-sweep)
    for sel in ["line_all","share_stop","random"]:
        for k in [1,2,3]:
            rows.append(row(nk, case_k=k, case_selection=sel, case_variants=30))
    return rows

def sweep_bypass(nk):
    return [row(nk, bypass_multiplier=b, case_k=k, case_variants=30)
            for b in [10,20,50,100,200] for k in [1,2]]

def sweep_overdemand(nk):
    return [row(nk, overdemand_threshold=t, overdemand_multiplier=m, case_k=k, case_variants=30)
            for t in [0.5,0.75,1.0] for m in [1,2,5] for k in [1,2]]

def sweep_pfail(nk):
    return [row(nk, case_p_fail=p, case_k=k, case_variants=30)
            for p in [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95] for k in [1,2]]

def sweep_replcost(nk):
    return [row(nk, cost_repl_freq=cf, cost_repl_line=cl, case_k=k, case_variants=30)
            for cf in [10,100,1000] for cl in [50,500,5000] for k in [1,2]]

SWEEPS = {"main":sweep_main, "bypass":sweep_bypass, "overdemand":sweep_overdemand,
          "pfail":sweep_pfail, "replcost":sweep_replcost}

def main():
    out = "Data"
    for nk in NETS:
        for sw, fn in SWEEPS.items():
            write(os.path.join(out, f"config_matched_{sw}_{nk}.csv"), fn(nk))

if __name__ == "__main__":
    main()

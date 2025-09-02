# -*- coding: utf-8 -*-
"""Run LPP-CGN optimisation row-by-row from Data/config.csv, logging via log.py."""

import os, time
import pandas as pd
from load_data import load_and_build
from solve_cgn import build_and_solve
from print import print_domain_summary, print_model_summary  # optional
from log import RunBatchLogger  # <- neues Modul

# --------- small parsers ----------
def _parse_freq_vals(s):
    if s is None or str(s).strip() == "":
        return None
    return [int(x) for x in str(s).replace(",", " ").split() if x.strip().isdigit()]

def _parse_bool(x, default=False):
    if x is None: return default
    s = str(x).strip().lower()
    if s in ("1","true","yes","y"): return True
    if s in ("0","false","no","n"): return False
    return default

def _parse_routing_agg(x):
    """Return 'od' or 'origin'."""
    if x is None: return "od"
    s = str(x).strip().lower()
    if s in ("origin","by_origin","o","true","1","yes","y"): return "origin"
    return "od"

def _status_name(code: int) -> str:
    mapping = {
        1:"LOADED", 2:"OPTIMAL", 3:"INFEASIBLE", 4:"INF_OR_UNBD", 5:"UNBOUNDED",
        6:"CUTOFF", 7:"ITERATION_LIMIT", 8:"NODE_LIMIT", 9:"TIME_LIMIT",
        10:"SOLUTION_LIMIT", 11:"INTERRUPTED", 12:"NUMERIC",
        13:"SUBOPTIMAL", 14:"INPROGRESS", 15:"USER_OBJ_LIMIT"
    }
    return mapping.get(int(code), f"STATUS_{code}")

def main():
    data_root = os.environ.get("DATA_ROOT", ".")
    cfg_path  = os.path.join(data_root, "Data", "config.csv")
    cfg       = pd.read_csv(cfg_path, sep=';')

    verbose = _parse_bool(os.environ.get("VERBOSE_PRINTS", "0"), default=False)

    # centralised logger (creates Results/<timestamp>/ and base_log.csv with header)
    logger = RunBatchLogger(data_root=data_root, cfg_df=cfg)
    print(f"Logging to: {logger.summary_paths()['out_dir']}")

    for i, row in cfg.iterrows():
        tag = f"{row['source']}/{row['network']}@{row['scenario_line_data']}"
        print(f"\n\n##### RUN {i}: {tag} #####")
        t0 = time.time()

        # prepare base row (config columns prefilled)
        base_row = logger.base_row_template(row)

        try:
            # ----- data -----
            domain, model = load_and_build(
                data_root=data_root,
                cfg_row=row.to_dict(),
                symmetrise_infra=False,
                zero_od_diagonal=False,
            )

            if verbose:
                print_domain_summary(domain, max_rows=5, max_lines=5, max_include=5)
                print_model_summary(model, domain, max_items=5, max_lines=5)

            # ----- solver settings -----
            freq_vals           = _parse_freq_vals(domain.config.get("freq_values"))
            include_origin_wait = _parse_bool(domain.config.get("include_origin_wait"), default=False)
            select_lines        = _parse_bool(domain.config.get("select_lines"), default=False)
            waiting_time        = float(domain.config.get("waiting_time", -1.0))
            routing_agg         = _parse_routing_agg(domain.config.get("routing_agg"))

            # Optional: extra Gurobi params via env
            grb_params = {}
            if "GRB_TimeLimit" in os.environ: grb_params["TimeLimit"] = int(os.environ["GRB_TimeLimit"])
            if "GRB_MIPGap"   in os.environ: grb_params["MIPGap"]    = float(os.environ["GRB_MIPGap"])

            # ----- build & solve -----
            m, art = build_and_solve(
                domain, model,
                freq_vals=freq_vals,
                include_origin_wait=include_origin_wait,
                select_lines=select_lines,
                waiting_time=waiting_time,
                routing_agg=routing_agg,
                time_w=1.0, wait_w=1.0, op_w=1.0,
                gurobi_params=grb_params or None
            )

            status_code = int(m.Status)
            status_name = _status_name(status_code)
            runtime     = getattr(m, "Runtime", None)
            costs       = art.get("costs") or {}
            objective   = costs.get("objective", getattr(m, "ObjVal", None))

            # fill KPI columns
            base_row.update({
                "status_code": status_code,
                "status": status_name,
                "objective": objective,
                "cost_time": costs.get("time"),
                "cost_wait": costs.get("wait"),
                "cost_oper": costs.get("oper"),
                "runtime_s": runtime,
                "N": model.N, "E_dir": model.E_dir, "L": model.L, "S": model.S,
            })

            # frequencies file
            chosen_freq = art.get("chosen_freq", {})
            logger.write_freq_file(i, chosen_freq)

            # console summary
            print(f"Status={status_name}  Obj={objective}  "
                  f"Time={costs.get('time')}  Wait={costs.get('wait')}  Oper={costs.get('oper')}  "
                  f"Runtime={runtime}s")

        except Exception as e:
            base_row.update({
                "status_code": -1,
                "status": "ERROR",
                "objective": None,
                "runtime_s": round(time.time() - t0, 3)
            })
            print(f"[ERROR] run {i} ({tag}): {e}")

        finally:
            # append immediately
            logger.append_base_row(base_row)
            dt = time.time() - t0
            print(f"Run {i} finished in {dt:.2f}s")

    print(f"\nBase log: {logger.summary_paths()['base_log']}")

if __name__ == "__main__":
    main()

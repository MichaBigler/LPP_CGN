# -*- coding: utf-8 -*-
"""Run LPP-CGN optimisation row-by-row from Data/config.csv, logging via log.py."""

import os
import time
import traceback
import pandas as pd

from load_data import load_and_build
from solve_cgn import solve_one_stage, solve_two_stage_integrated, solve_two_stage_separated
from print import print_domain_summary, print_model_summary  # optional
from log import RunBatchLogger                    # zentraler Logger


def _as_bool(x, default=False):
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y"): return True
    if s in ("0", "false", "no", "n"): return False
    return default

def _status_name(code):
    # map the common Gurobi status codes to names
    MAP = {
        2: "OPTIMAL",
        3: "INFEASIBLE",
        4: "INF_OR_UNBD",
        5: "UNBOUNDED",
        9: "TIME_LIMIT",
        13: "SUBOPTIMAL",
    }
    try:
        if isinstance(code, str):
            return code
        c = int(code)
        return MAP.get(c, str(c))
    except Exception:
        return str(code)

def main():
    data_root = os.environ.get("DATA_ROOT", ".")
    cfg_path  = os.path.join(data_root, "Data", "config.csv")
    cfg_df    = pd.read_csv(cfg_path, sep=';')

    # optional hübsche Konsolenprints
    verbose = _as_bool(os.environ.get("VERBOSE_PRINTS", "0"), default=False)

    # zentraler Logger: legt Results/<YYYY_MM_DD_HH_MM>/ an und schreibt base_log.csv (Header)
    logger = RunBatchLogger(data_root=data_root, cfg_df=cfg_df)
    print(f"Logging to: {logger.out_dir}")

    for i, cfg_row in cfg_df.iterrows():
        tag = f"{cfg_row['source']}/{cfg_row['network']}@{cfg_row['scenario_line_data']}"
        print(f"\n\n##### RUN {i}: {tag} #####")
        t0 = time.time()

        # Basizeile fürs Base-Log: enthält bereits alle config-Spalten
        base_row = logger.base_row_template(cfg_row)

        try:
            domain, model = load_and_build(
                data_root=data_root,
                cfg_row=cfg_row.to_dict(),
                symmetrise_infra=False,
                zero_od_diagonal=False,
            )

            if verbose:
                print_domain_summary(domain, max_rows=5, max_lines=5, max_include=5)
                print_model_summary(model, domain, max_items=5, max_lines=5)

            proc = str(domain.config.get("procedure", "one")).lower()
            if proc in ("one", "one_stage"):
                m, solution, artifacts = solve_one_stage(domain, model)
            else:
                if proc in ("integrated", "joint"):
                    m, solution, artifacts = solve_two_stage_integrated(domain, model)
                elif proc in ("separated", "sequential"):
                    m, solution, artifacts = solve_two_stage_separated(domain, model)
                else:
                    print(f"[WARN] unknown procedure '{proc}', falling back to one_stage")
                    m, solution, artifacts = solve_one_stage(domain, model)


            if proc in ("one", "one_stage"):
                costs = solution.get("costs_0") or {}
                base_row.update({
                    "status": _status_name(solution.get("status")),
                    "objective":   solution.get("objective"),
                    "runtime_s":   solution.get("runtime_s"),
                    "cost_time":   costs.get("time"),
                    "cost_wait":   costs.get("wait"),
                    "cost_oper":   costs.get("oper"),
                    "obj_stage1":     None,
                    "obj_stage2_exp": None,
                    "repl_cost_exp":  None,
                })
                logger.write_freq_file(i, solution.get("chosen_freq") or artifacts.get("chosen_freq", {}))
            else:
                # aggregate stage-1 + expected stage-2 component costs
                nom = solution.get("costs_0") or {}
                scen_list = solution.get("scenarios") or []
                agg_time = (float(nom.get("time") or 0.0)
                            + sum(float(s.get("prob") or 0.0) * float(s.get("cost_time") or 0.0) for s in scen_list))
                # cost_wait ist bereits GEWICHTET (mit waiting_time_cost_mult) in beiden Stages
                agg_wait = (float(nom.get("wait") or 0.0)
                            + sum(float(s.get("prob") or 0.0) * float(s.get("cost_wait") or 0.0) for s in scen_list))
                agg_oper = (float(nom.get("oper") or 0.0)
                            + sum(float(s.get("prob") or 0.0) * float(s.get("cost_oper") or 0.0) for s in scen_list))

                base_row.update({
                    "status": _status_name(solution.get("status")),
                    "objective":     solution.get("objective"),
                    "runtime_s":     solution.get("runtime_s"),
                    "cost_time":     agg_time,
                    "cost_wait":     agg_wait,
                    "cost_oper":     agg_oper,
                    "obj_stage1":    solution.get("obj_stage1"),
                    "obj_stage2_exp": solution.get("obj_stage2_exp"),
                    "repl_cost_exp":  solution.get("repl_cost_exp"),
                })

                # breite Frequenz-CSV inkl. Nominal-Kosten
                logger.write_freqs_two_stage(
                    i, model,
                    nominal=solution.get("chosen_freq_stage1", {}),
                    scenarios=solution.get("scenarios", []),
                    nominal_costs=solution.get("costs_0", {})
                )

            print(f"Status={_status_name(solution.get('status'))}  Obj={solution.get('objective')}  "
                f"Stage1={base_row.get('obj_stage1')}  Stage2_exp={base_row.get('obj_stage2_exp')}  "
                f"Repl_exp={base_row.get('repl_cost_exp')}  Runtime={solution.get('runtime_s')}s")

        except Exception as e:
            # Im Fehlerfall trotzdem eine sinnvolle Zeile loggen
            traceback.print_exc()
            base_row.update({
                "status_code": -1,
                "status": "ERROR",
                "objective": None,
                "runtime_s": round(time.time() - t0, 3),
                "cost_time": None,
                "cost_wait": None,
                "cost_oper": None,
            })
            print(f"[ERROR] run {i} ({tag}): {e}")

        finally:
            # Sofort anhängen (Streaming-Log)
            base_row.setdefault("status_code", solution.get("status_code") if 'solution' in locals() else -1)
            logger.append_base_row(base_row)

    print(f"\nBase log: {logger.base_log_path}")


if __name__ == "__main__":
    main()
